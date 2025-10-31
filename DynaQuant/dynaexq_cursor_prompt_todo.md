
# DynaExQ – Cursor Implementation Prompt & To-Do List

> Goal: Implement a system-level runtime for dynamic expert precision management in MoE inference that adapts quantization to workload dynamics without pausing inference. This project does not invent new quantization algorithms; it orchestrates when to use prebuilt W4A4 vs W2A2 experts and how to swap them efficiently across GPU/CPU/SSD.

---

## Context & Constraints

- Models: MoE LLMs (e.g., Qwen3-A3B / Mixtral-8x7B). Router top-k per layer, k in {2,4}.
- Hardware: RTX 5090 / 4090 / A6000, 24-32 GB HBM; 32 GB host DRAM; 1 TB NVMe SSD.
- Precision options: W4A4 (hot), W2A2 (cold). Quantizers come from existing toolchains (e.g., AWQ/MoEQuant/EAQuant).
- Non-goals: No router retraining; no algorithmic quantization changes.
- Success criteria:
  - Near-FP16 accuracy vs static W2A2;
  - No visible stalls when experts change precision;
  - Throughput/TTFT improvements vs static W4A4 baselines at the same memory budget.

---

## High-Level Architecture

```
 +----------------------------------------------+
 | Inference Driver (e.g., SGLang/DeepSpeed)    |
 +-------------------+--------------------------+
                     |
           +---------v---------+
           | Expert Monitor    |  Collect per-batch top-k and logits
           +---------+---------+
                     |  S_i = EWMA(logit mass) over epoch windows
           +---------v---------+
           | Precision Ctrl    |  tau_h / tau_c thresholding  -> target {W4, W2}
           +---------+---------+
                     |  diff(target, resident) -> swap plan
           +---------v---------+
           | Swap Engine       |  async prefetch/evict, pinned buffers, streams
           +---------+---------+
                     |  place into memory pools
           +---------v---------+
           | Memory Manager    |  Hot / Cold / Transient pools, LRU/FIFO
           +-------------------+
```

---

## Key Design Requirements (from Challenges)

1. Workload-aware dynamics: couple expert precision to runtime hotness.
2. Non-blocking swaps: expert upgrades/downgrades must not pause inference.
3. Fragmentation control: stable pool allocator to avoid HBM churn.
4. Resource-aware thresholds: adapt tau_h / tau_c using workload + pool pressure.

---

## Core Concepts

- Hotness score:
  Let S_i = EWMA_t( mean_{x in batch} g_i(x) ) where g_i(x) is router weight.
  Epoch windowing: global epoch increments every 5 min, resets EWMA span.
- Thresholds: promote if S_i > tau_h -> W4A4; demote if S_i < tau_c -> W2A2.
  Hysteresis: tau_h > tau_c to avoid oscillation.
- Pools:
  - HotPool (W4 slots)
  - ColdPool (W2 slots)
  - Transient (staging for DMA, conversion)
- Storage tiers: HBM <-> DRAM <-> SSD with an optional SSD index mapping expertID -> byte range(s).

---

## Directory Layout (proposed)

```
dynaexq/
  runtime/
    monitor.py
    controller.py
    swap_engine.py
    memmgr.py
    prefetch.py
    ssd_index.py
    kernels/
      group_gemm.cu
      packbits.cu
  integration/
    hooks_sglang.py
    hooks_deepspeed.py
  tests/
    test_monitor.py
    test_swap_engine.py
    test_memmgr.py
    perf/
      bench_throughput.py
      bench_ttft.py
  configs/
    default.yaml
  scripts/
    run_qwen_a3b_demo.sh
    trace_router.py
  README.md
```

---

## Interfaces & Data Structures

### Expert identity and residency
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class ExpertID:
    layer: int
    idx: int  # expert index within layer

@dataclass
class Residency:
    bitwidth: Literal["W4","W2"]
    location: Literal["HBM","DRAM","SSD"]
    hbm_ptr: Optional[int] = None   # device pointer / handle
    bytes: int = 0
    last_used_ts: float = 0.0
```

### Hotness statistics
```python
import numpy as np

class ExpertMonitor:
    def update_batch(self, layer:int, topk_idx:np.ndarray, logits:np.ndarray) -> None: ...
    def score(self, expert:ExpertID) -> float: ...
    def epoch_tick(self) -> None: ...  # every 5 minutes
```

### Precision controller
```python
class PrecisionController:
    def __init__(self, tau_h:float, tau_c:float, max_w4_slots:int):
        ...
    def plan(self, active_set:list[ExpertID], monitor:ExpertMonitor) -> dict[ExpertID, str]:
        """Return target bitwidth per expert with hysteresis and pool limits."""
```

### Memory pools & allocator
```python
class MemoryManager:
    def reserve_hot(self, expert:ExpertID, nbytes:int) -> bool: ...  # LRU evict if needed
    def place(self, expert:ExpertID, residency:Residency) -> None: ...
    def evict_hot(self) -> list[ExpertID]: ...
```

### Swap engine (async)
```python
class SwapEngine:
    def upgrade(self, expert:ExpertID):
        """Bring W4 into HBM; pipeline SSD->DRAM->HBM; double-buffered pinned memory."""
    def downgrade(self, expert:ExpertID):
        """Convert/pack to W2 and move to DRAM or SSD if needed (non-blocking)."""
    def wait_ready(self, expert:ExpertID) -> None: ...
```

### Prefetch planner (layer-wise pipeline)
```python
class PrefetchPlanner:
    def lookahead(self, layer:int, next_active:list[ExpertID]) -> None:
        """Trigger upgrades of next layer experts while current layer computes."""
```

---

## Execution Flow (pseudocode)

```python
# At each MoE layer L, per batch
active = router.topk_indices(L)             # GPU -> CPU lightweight copy
monitor.update_batch(L, active, router.logits(L))

targets = controller.plan(active, monitor)  # decide W4/W2 targets
plan = diff(targets, residency_map)         # list of upgrades/downgrades

# Launch non-blocking swaps
for e in plan.upgrades:   swap_engine.upgrade(e)
for e in plan.downgrades: swap_engine.downgrade(e)

prefetch.lookahead(L+1, predict_next_active(L+1))

# Ensure required experts are ready just-in-time
for e in active:
    swap_engine.wait_ready(e)  # should be already done if prefetch effective

launch_group_gemm(active, bitwidth_map=targets)  # mixed-precision GroupGEMM
```

---

## Kernels

- group_gemm.cu: batched GEMM across experts; supports per-expert bitwidth.
  - Inputs: packed W2 weights, nibble-packed W4 weights, FP16 activations.
  - Constraint: use default CUDA math.
- packbits.cu: device/host packing for W2/W4 formats; zero-copy into HBM slots.

---

## Configuration (configs/default.yaml)

```yaml
hotness:
  window: 5m_epoch
  ewma_alpha: 0.2
thresholds:
  tau_h: 0.65   # promote
  tau_c: 0.45   # demote
pool:
  hot_w4_slots:  N_HOT_SLOTS   # per-layer max W4 experts
  transient_mb:  2048
storage:
  ssd_path: /mnt/ssd/experts.bin
  index_path: /mnt/ssd/experts.index
streams:
  memcpy_h2d: 2
  memcpy_d2h: 1
  compute: 2
```

---

## Telemetry & Safety

- Metrics: TTFT, TPOP, tokens/s, HBM usage, upgrade/downgrade latency, ready-before-use ratio, eviction rate, SSD read bandwidth.
- Tracing: per-layer timeline of swaps and kernels; CUDA events & NVTX ranges.
- Safety: if ready-before-use < 99%, temporarily widen (tau_h - tau_c) or increase hot_w4_slots by one step.

---

## To-Do List (prioritized)

### P0 — MVP (single process, single GPU)
- [ ] Implement ExpertMonitor with EWMA hotness; epoch tick every 5 minutes.
- [ ] Implement PrecisionController.plan() with hysteresis and max_w4_slots.
- [ ] Implement MemoryManager with three pools and LRU; fixed-size slots per expert.
- [ ] Implement SwapEngine with pinned buffers, two H2D streams, one D2H stream.
- [ ] Implement PrefetchPlanner.lookahead(); simple next-layer peek heuristic.
- [ ] Integrate a dummy group_gemm path: call separate W2 and W4 GEMMs first.
- [ ] End-to-end smoke test on a tiny MoE (2 layers, 8 experts, k=2).
- [ ] Telemetry: log JSONL per batch; expose Prometheus or CSV dump.

### P1 — Performance
- [ ] Implement real GroupGEMM that mixes W2 and W4 matmuls in one launch.
- [ ] SSD indexer: memory-mapped file + sparse index for expert byte ranges.
- [ ] Layer-wise pipeline: ensure L+1 upgrades overlap with L compute.
- [ ] Double buffering: transient pool for pack/convert while compute runs.
- [ ] Adaptive thresholds: auto-tune tau_h/tau_c based on ready-before-use and HBM pressure.

### P2 — Robustness & UX
- [ ] Config loader + runtime CLI: --tau_h, --tau_c, --hot-slots N.
- [ ] Failure policy: on missed deadline, fallback to W2 for that expert, record event.
- [ ] Warm-start preheat list per domain (text/math/code) to avoid cold spikes.
- [ ] Unit tests for monitor/controller/memmgr/swap_engine.
- [ ] Benchmark scripts: TTFT/TPOP/throughput/memory footprint.

---

## Acceptance Tests

1. No-Stall Upgrade: During a synthetic workload with periodic expert flips, zero kernel gaps > 1 ms when an expert promotes to W4.
2. Ready-Before-Use >= 99%: For 10k tokens workload with lookahead, swaps complete before compute in >= 99% of uses.
3. Throughput Gain: +1.3x over static W4A4 at the same HBM cap; TTFT reduced vs static W4A4.
4. Fragmentation Bounded: HBM free space oscillation < 10% over 30-minute run.

---

## Integration Notes

- Hook points for SGLang/DeepSpeed: capture per-layer top-k and logits; call wait_ready once before expert matmul; pass bitwidth_map into GroupGEMM.
- Prebuilt quantized weights: ensure both W2 and W4 exist per expert; store in DRAM/SSD; keep pointer table.

---

## Nice-to-Have (later)

- Multi-GPU sharding of HotPool (NVLink copies); per-GPU controllers with global coordinator.
- Mixed FP16 + INT8 variant on dual A6000s (extend bitwidth_map to include W8A8/FP16).
- Lightweight heuristic model to predict hot experts from prompt domain tags.
