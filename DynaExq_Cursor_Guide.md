# DynaExq Implementation & Experiment Guide (for Cursor)

This document turns the paper into an **engineering specification**: a concrete module breakdown, data contracts, and an experiment harness plan. It is intended to be used as a “product requirements + design doc” to guide Cursor-assisted implementation.

**Paper reference:** `ICML_2026_DyMoE.pdf` (DynaExq).  
Core framing: **online, budget-constrained precision allocation under single-GPU device-memory limits**, realized by a **budget-feasible stable scheduler** + a **non-blocking transition pipeline** + a **deterministic mixed-precision memory manager**.

---

## 0) Goals, non-goals, and acceptance criteria

### Goals (must satisfy)
1. **Budget feasibility (hard constraint).** Total resident expert footprint never exceeds the configured device-memory budget for experts, accounting for reserved memory (KV cache, activations, dense weights).  
2. **Non-blocking forward path (hard constraint).** Forward always runs on the **last stable registered** expert representation; promotions/demotions must never stall forward.  
3. **Stability (soft constraint with measurable bounds).** Avoid thrashing under transient routing fluctuations using **EMA hotness** and **periodic updates** (and optional rate limit).

### Non-goals (explicitly out of scope for v1)
- Training-time router modification or retraining.
- Perfect optimality of the online allocation (we implement **budget-feasible projection** described in the paper).
- Distributed multi-GPU placement/precision co-optimization (single-GPU focus).

### Acceptance criteria (testable)
- No OOM for expert-memory pool under adversarial routing with worst-case transitions (within configured bounds).
- Forward thread never blocks on any transition stage (verified by tracing forward step duration vs transition activity).
- Under workload shift benchmark, hot-set overlap changes and the scheduler adapts; switch rate remains bounded; tail latency increases are bounded and explainable.

---

## 1) Conceptual model and invariants (translate paper into contracts)

### 1.1 Entities
- **Layer** `l ∈ [0..L-1]`: MoE layers.
- **Expert** `e ∈ [0..E_l-1]`: experts per layer.
- **Tier**: `{HI, LO}` (two-tier in v1).  
  - Example: HI=FP16 or INT4; LO=INT4 or INT2 depending on model.

### 1.2 Invariants (must hold at all times)
**I1 Feasibility by construction.** For each layer `l`, the number of HI experts `n_hi[l]` is fixed by budget initialization. Runtime scheduling only changes **which experts** occupy HI slots, not the slot count.  
**I2 Provenance-consistent execution.** Forward binds to the **last stable registered** representation. A promotion is visible only after “register” completes.  
**I3 Bounded switching frequency.** Scheduler updates every `T_u` steps; EMA smoothing prevents oscillations; optional rate limit `R` per period.

### 1.3 Key equations to implement
**EMA hotness update** (paper Eq. 2):  
`S[l,e] ← α * S[l,e] + (1-α) * g[l,e](x_t)`  
Where `g` is router probability or selection indicator/weight.

**Budget-feasible projection** (paper Eq. 3):  
`H_l(t) = Top-n_hi[l] experts ranked by S[l,e](t)`  
Assign tier HI to `H_l`, LO otherwise.

**Budget initialization** (paper Eq. 4, summarized):  
Choose `n_hi[l]` such that total expert memory fits:
`Σ_l ( n_hi[l]*m_l(HI) + (E_l - n_hi[l])*m_l(LO) ) ≤ M_exp`  
Where `M_exp` is device-memory budget available for expert weights (after reserving for KV cache, activations, shared weights, transient buffers).

---

## 2) High-level architecture (modules and responsibilities)

### 2.1 Module graph
1. **RouterObserver** → provides per-step `g[l,e]` signals.
2. **HotnessTracker** → maintains `S[l,e]` (EMA).
3. **PrecisionScheduler** → every `T_u` computes `H_l` and emits transition requests.
4. **TransitionEngine** → executes promotion/demotion asynchronously via stages.
5. **MemoryManager** → provides deterministic allocation from tier pools + transient buffers.
6. **ExpertRegistry** → atomic “last stable representation” pointer/handle per expert used by forward.
7. **ExperimentHarness** → workload streams, metrics logging, and analysis outputs.

### 2.2 Integration points in inference
- **During forward step**:
  - Get router outputs (top-k / probs).
  - Update `HotnessTracker` (cheap).
  - Lookup expert handle from `ExpertRegistry` and execute expert kernel.
  - Scheduler/TransitionEngine do **not** block forward.
- **Background threads/streams**:
  - Scheduler tick every `T_u` steps.
  - TransitionEngine processes queued promotions/demotions.
  - MemoryManager serves allocations without cudaMalloc.

---

## 3) Data contracts (schemas you must log and exchange)

### 3.1 Core structs (Python typing style)
```python
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple

class Tier(Enum):
    HI = auto()
    LO = auto()

@dataclass(frozen=True)
class ExpertKey:
    layer: int
    expert: int

@dataclass
class HotnessState:
    alpha: float
    scores: "np.ndarray"  # shape [L, maxE], masked per-layer if E varies

@dataclass
class TierAssignment:
    # bitmask per layer or list of HI experts
    hi_sets: List["np.ndarray"]  # each array of expert ids length n_hi[layer]

@dataclass
class TransitionReq:
    key: ExpertKey
    src: Tier
    dst: Tier
    reason: str  # e.g., "enter_hi_topn" / "leave_hi_topn"
    issued_step: int
```

### 3.2 Expert representation handle (used by forward)
```python
@dataclass
class ExpertHandle:
    tier: Tier
    device_ptr: int          # or torch.Tensor / storage handle
    format: str              # "fp16" / "int4" / "int2"
    bytes: int
    version: int             # monotonically increasing per expert
```

### 3.3 Required logs (minimum for paper experiments)
**Per scheduler update (every `T_u`):**
- `step`, `timestamp`
- `hi_set[layer]` (expert ids)
- `delta_promotions`, `delta_demotions`
- `switch_rate` (count / minute)
- `hotness_mass_covered[layer] = sum_{e in hi} S / sum_e S`
- Optional: `score_gap_stats` (min/median/max of boundary gaps)

**Per transition event:**
- `ExpertKey`, `src->dst`
- stage timings: fetch, h2d, register, reclaim
- bytes moved, source tier (SSD/DRAM cache hit/miss)
- in-flight queue length at issue/start

**Per request / per batch:**
- TTFT, TPOP, prefill tokens/s, decode tokens/s
- p50/p95/p99 across window
- optional: GPU utilization, H2D bandwidth counters if available

---

## 4) Core implementation design (code-level abstraction)

### 4.1 RouterObserver
**Purpose:** convert model/router outputs into `g[l,e](x_t)` signals.  
**Contract:** must run in forward and be cheap.

Implementation options:
- If router already returns top-k expert ids per token, use `indicator` counts:
  - `g[l,e] = (#times expert e selected in this step)/ (tokens*topk)`
- If you can access gating probabilities, use probability mass:
  - `g[l,e] = mean probability of expert e across tokens` (more stable).

### 4.2 HotnessTracker (EMA)
**Inputs:** `g[l,e]` for each step.  
**State:** `S[l,e]`, float32 on CPU (or GPU if cheap).  
**Update:** vectorized EMA per layer.

Important edge cases:
- Varying `E_l` across layers: maintain per-layer slices.
- If using indicator counts, normalize carefully (avoid scale drift across batch sizes).

### 4.3 BudgetInitializer
**Inputs:**
- Total device memory `M_device` (bytes).
- Reserved memory budgets: `M_kv`, `M_act`, `M_dense`, safety margin `M_margin`.
- Tier footprint models: `m_l(HI)`, `m_l(LO)` in bytes per expert for each layer.
- Optional weighting: layer proportional allocation or sensitivity-based weights.

**Outputs:**
- `n_hi[l]` per layer (integer), fixed during run.
- Pool sizes: hi_pool_bytes, lo_pool_bytes, transient_bytes.

Implementation approach (v1):
1. Compute `M_exp = M_device - (M_kv + M_act + M_dense + M_margin)`.
2. Choose initial `n_hi[l]` by proportional rule (e.g., uniform or proportional to expert bytes).
3. Verify feasibility: evaluate Eq. (4). If infeasible, reduce `n_hi` greedily from largest layers or globally until feasible.
4. Save the result and allocate pools accordingly.

### 4.4 PrecisionScheduler
**Inputs:** `S[l,e]`, `n_hi[l]`, `T_u`, optional `rate_limit_R`.  
**Outputs:** a list of `TransitionReq`.

Algorithm (paper Algorithm 1):
1. Every `T_u` steps:
2. For each layer `l`, compute `H_l = top_n(S[l,:], n_hi[l])`.
3. Compare with current tier assignment `tier[l,e]`.
4. Enqueue promotions for `e in H_l` with current LO; enqueue demotions for `e not in H_l` with current HI.
5. Optional: if rate limiting `R`, prioritize the largest score gaps:
   - promote candidates with largest `S_candidate - S_boundary`
   - demote incumbents with smallest `S_incumbent - S_boundary`

**Determinism requirement:** given `S` and fixed tie-breaker (e.g., expert id), selection must be deterministic.

### 4.5 TransitionEngine (non-blocking pipeline)
**Queue:** `promotion_queue`, `demotion_queue`.  
**Concurrency:** background CPU thread + CUDA streams.

Stages for promotion LO→HI:
1. **Fetch:** ensure HI weights exist in DRAM (or CPU pinned) — load from SSD if needed.
2. **H2D Transfer:** copy into HI pool block using a dedicated CUDA stream.
3. **Register:** atomically update `ExpertRegistry[key]` to new `ExpertHandle` (version++).
4. **Reclaim:** return old LO block to LO pool.

Demotion HI→LO mirrors the process.

**Non-blocking guarantee:** forward always reads `ExpertRegistry[key]` handle without waiting.  
If a promotion is in-flight, it continues using old handle until register completes.

### 4.6 ExpertRegistry
A thread-safe/atomic mapping from `ExpertKey` → `ExpertHandle`.
- Implement via:
  - CPU-side atomic pointer swap (if handles are small objects, use locks carefully), or
  - GPU-side indirection table if kernel reads pointers directly (more complex).

V1 recommendation:
- Keep `ExpertHandle` as a Python object referencing a `torch.Tensor` for weights; swap under a per-expert lock.  
- Ensure the lock is only taken by the TransitionEngine, not by forward. Forward reads a stable reference (e.g., using RCU-like pattern: store to an array and read without lock, accepting eventual consistency).

### 4.7 MemoryManager (deterministic pools)
Two pools: **HI pool** and **LO pool**, plus a small **transient pool** for in-flight dual residency.

Design:
- Each pool is a fixed set of blocks of identical size per tier.
- Allocation: pop a free block id from a lock-free queue (or a mutex-protected list).
- Free: push back.

Block size computation:
- For each layer, experts may have different tensor sizes; simplest approach:
  - Choose per-layer pools (HI_pool[l], LO_pool[l]) with blocks sized for that layer’s expert format.
  - This avoids internal fragmentation from one global block size.

Transient buffer:
- Bounded by `max_inflight` promotions (config).
- Ensure budget initialization accounts for transient bytes.

---

## 5) Reference implementation plan (files and classes)

Recommended repo layout:
```
dynaexq/
  core/
    router_observer.py
    hotness_tracker.py
    budget_init.py
    scheduler.py
    registry.py
    memory_pool.py
    transition_engine.py
    config.py
  integration/
    moe_wrapper.py           # wraps model forward w/ observer + registry handles
    kernels/                 # optional: int2/int4 kernels integration points
  experiments/
    workloads.py             # shift stream, request generator
    metrics.py               # TTFT/TPOP + percentiles + traces
    baselines.py             # static INT2/INT4, MP-Offline, MP-Window
    run_shift.py
    run_tail.py
    run_ablation.py
    analyze_shift.ipynb
    analyze_tail.ipynb
  scripts/
    download_models.sh
    run_all.sh
```

Config (YAML) example:
```yaml
model:
  name: qwen3-moe-30b-a3b
  layers: 48
  experts_per_layer: 128
  topk: 8
precision:
  hi: fp16
  lo: int4
scheduler:
  alpha: 0.9
  update_period_steps: 200   # or time-based
  rate_limit: null           # or integer R
memory:
  device_mem_bytes: 48_000_000_000
  reserve_kv_bytes: 10_000_000_000
  reserve_act_bytes: 2_000_000_000
  reserve_dense_bytes: 5_000_000_000
  safety_margin_bytes: 1_000_000_000
  max_inflight: 4
experiments:
  concurrency: 64
  input_tokens: 2048
  output_tokens: 256
  shift:
    phases: ["wikitext", "gsm8k_aime", "humaneval"]
    phase_duration_s: 180
    cycles: 3
```

---

## 6) Experiment harness specification (what to run, what to log)

### 6.1 New Experiment #1: Workload shift (must-have)
**Purpose:** validate “online adaptation under routing shift” + “stability” + “non-blocking”.

#### Workload stream
- Phase A: WikiText-like prompts
- Phase B: GSM8K + AIME-style math prompts
- Phase C: HumanEval-style code prompts
- Repeat A→B→C for `cycles`.

#### Required outputs
1. **Hot-set overlap matrix / time series**
   - Jaccard `J(H_l^A, H_l^B)` per layer, plus mean±std across layers.
2. **Hotness mass coverage**
   - `coverage_l(t) = sum_{e in H_l(t)} S[l,e] / sum_e S[l,e]`
3. **Quality proxy per phase**
   - LM: perplexity or NLL on a fixed sample
   - Math: accuracy on a small fixed subset
   - Code: pass@1 on a small fixed subset
4. **Switching behavior**
   - promotions/min, bytes/s, max in-flight
5. **Latency traces**
   - TTFT/TPOP time series; mark phase boundaries and high transition windows.

#### Baselines to include
- Static INT4 / INT2
- MP-Offline: fixed expert-wise assignment from a calibration mixture under same budget
- MP-Window: dynamic but short-window (no EMA), same `n_hi[l]`

### 6.2 New Experiment #2: Tail latency & transition overhead (must-have)
**Purpose:** validate “non-blocking realization” beyond averages.

#### Protocol
- Choose a window where transitions are frequent (right after phase changes) and a steady window.
- Report TTFT/TPOP percentiles (P50/P95/P99) separately.

#### Required outputs
- TTFT: P50/P95/P99 steady vs transition
- TPOP: P50/P95/P99 steady vs transition
- Transition overhead: promotions/min, bytes/s, stage timing breakdown

### 6.3 Ablations (recommended)
- w/o EMA (replace EMA with short window)
- w/o async (synchronous promotion; should worsen tail)
- w/o pools (use cudaMalloc/torch allocator; should add jitter)
- w/o budget init (best-effort; should risk OOM or instability)

---

## 7) Practical guidance for Cursor (task decomposition)

### 7.1 Implementation milestones (in order)
1. **Scaffolding & configs**: load model, expose router outputs, create config system.
2. **HotnessTracker**: implement EMA update; unit test with synthetic routing.
3. **BudgetInitializer**: compute `M_exp`, `n_hi[l]`, allocate pools; add feasibility assertions.
4. **Registry + Forward binding**: forward reads `ExpertHandle` per expert.
5. **Scheduler**: top-n projection; deterministic tie-breaking; emits TransitionReq.
6. **TransitionEngine**: implement LO→HI promotions first; then HI→LO demotions.
7. **Metrics & logging**: TTFT/TPOP, percentiles, transition logs.
8. **Shift harness**: phased workload runner.
9. **Baselines**: static, MP-Offline, MP-Window.

### 7.2 Unit/integration tests (minimum)
- **Feasibility test:** worst-case schedule changes do not exceed pool capacity; transient buffers enforce bounds.
- **Non-blocking test:** forward loop time does not stall when transitions are triggered; confirm no synchronization in forward path.
- **Determinism test:** given same hotness scores, scheduler outputs identical `H_l`.

### 7.3 Common failure modes and how to detect
- **Thrashing**: promotions/min spikes; P99 TTFT increases sharply. Fix: increase `α`, increase `T_u`, add rate limit.
- **Hidden blocking**: accidental `cuda.synchronize()` or stream waits in forward. Fix: isolate all waits to TransitionEngine thread.
- **Allocator jitter**: cudaMalloc appears in traces. Fix: pre-allocate pools; avoid allocating tensors inside transition path beyond the pool.
- **Incorrect visibility semantics**: forward uses partially copied weights. Fix: register only after copy completion event.

---

## 8) Reporting templates (tables/figures to generate)

### Shift benchmark figures
- Fig A: hot-set overlap (Jaccard) across phases (layer-mean ± std)
- Fig B: quality recovery curve after each phase boundary
- Fig C: TTFT/TPOP trace with transition markers
- Table: promotions/min, bytes/s, max in-flight for each method

### Tail latency
- Table: TTFT P50/P95/P99 steady vs transition (per method)
- Table: TPOP P50/P95/P99 steady vs transition (per method)
- Bar/line: stage timing breakdown per promotion (fetch/h2d/register/reclaim)

---

## 9) “Done” definition (what you should be able to claim)
You are ready to write the final Evaluation section when you can produce:
1. Shift benchmark: overlap + recovery + switch rate + traces for DynaExq vs MP-Offline vs MP-Window.
2. Tail latency: P99 comparisons steady vs transition and bounded spikes.
3. Ablation table showing each component supports a paper constraint (feasibility / non-blocking / stability).

---

## Appendix: Minimal pseudocode snippets (Cursor-friendly)

### Scheduler tick
```python
def scheduler_tick(step: int):
    if step % Tu != 0:
        return
    for l in range(L):
        hi = top_n(S[l], n_hi[l])  # deterministic tie-breaker
        for e in range(E[l]):
            want_hi = e in hi
            have_hi = (tier[l,e] == HI)
            if want_hi and not have_hi:
                enqueue_promotion(l,e)
            elif (not want_hi) and have_hi:
                enqueue_demotion(l,e)
```

### Promotion pipeline (conceptual)
```python
def promote(key):
    w_hi_host = fetch_hi_weights(key)           # SSD/CPU → DRAM (async if possible)
    dst_block = hi_pool.alloc(layer=key.layer)  # deterministic
    h2d_copy_async(dst_block, w_hi_host, stream=copy_stream)
    stream_wait_event(copy_stream, copy_done_event)
    registry.register(key, dst_block.handle())  # atomic visibility point
    old = registry.old_handle(key)
    lo_pool.free(old.block_id)
```

---

## References / Paper anchors
- EMA hotness update and top-n projection: \S3.2, Eq.(2)(3).
- Non-blocking transition semantics: \S3.3.
- Memory pools and feasibility invariant: \S3.4--3.5, Eq.(4).
- Evaluation questions and metrics: \S4.
