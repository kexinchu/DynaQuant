# Mixed-Precision Evaluation Runtime Walkthrough

This note walks through the control flow behind `scripts/evaluate_mmlu_perplexity_mixed.py`, showing how the runtime fulfils the requirements for expert monitoring, periodic precision scheduling, asynchronous swaps, resource-aware budgeting, and pre-allocated memory pools.

## 1. CLI Setup (`scripts/evaluate_mmlu_perplexity_mixed.py`)

1. Parse flags including `--use-ssd`, which optionally points to a cache directory. 【`scripts/evaluate_mmlu_perplexity_mixed.py`】
2. Load the activation ranking file and select tail experts for low precision via `select_tail_experts`.
3. Resolve device/dtype, note whether SSD storage is requested, and forward that into `load_mixed_precision_model`.

## 2. Model / Runtime Bootstrap (`dynaexq/runtime/hf_runtime.py`)

1. Build a dual-precision repository from W4/W2 checkpoints. A metadata pass (`_collect_layer_metadata`) records per-layer expert counts and tensor sizes.
2. Compute `HFRuntimeConfig` with `infer_runtime_config`, which:
   - Calculates available HBM after subtracting non-expert weights.
   - Distributes hot-slot budgets across layers to respect memory limits.
   - Records per-layer hot/cold tensor sizes and total expert counts.  
3. Choose weight storage:
   - In-memory (`InMemoryWeightStore`) if `--use-ssd` is unset.
   - SSD-backed (`SSDWeightStore`) otherwise; `_build_ssd_repository` writes a contiguous payload plus JSON index and drops in-RAM expert tensors.
4. Instantiate `HuggingFaceDynaExQ`, passing the weight store, expert roster, and runtime config.
5. Load the Transformer config/tokenizer, materialise the mixed-precision state dict onto the module, then attach runtime hooks.

## 3. Live Expert Monitoring & Scheduling (`HuggingFaceDynaExQ`)

1. Router forward-hooks (discovered via `_locate_moe_layers`) stream top-k indices/logits into `ExpertMonitor`; no precision change happens in-line.
2. A global epoch timer (10 min by default) runs inside a lightweight scheduler thread. When it fires:
   - The controller is reset and re-planned for the full expert roster while respecting per-layer hot-slot limits.
   - `_synchronize_experts` enqueues alternating upgrades/downgrades so the swap pipeline can execute asynchronously.
   - `ExpertMonitor.reset_all()` clears accumulated hotness, starting the next epoch with a clean slate.

This satisfies “update activations continuously, exchange experts only on epoch rollover”.

## 4. Asynchronous Prefetch & Swap

1. `_synchronize_experts` builds alternating upgrade/downgrade sequences.
2. If the weight store supports prefetch (SSD path), `ExpertPrefetchWorker` stages payloads from SSD → DRAM, signalling readiness events.
3. `ExpertSwapWorker` consumes the queue, waits on any prefetch events, then calls `SwapEngine.upgrade`/`downgrade`.
4. `SwapEngine` loads tensors, tags the resulting `Residency` with the pre-reserved slot ID, and lets `MemoryManager.place` copy them into live parameters once HBM residency is confirmed.

The swaps are fully asynchronous; the evaluation loop only waits when the runtime asks `wait_ready`.

## 5. Slot-Based Memory Pools (`dynaexq/runtime/memmgr.py`)

1. `PoolConfig` now predefines `(hot_slots + 1)` and `(cold_slots + 1)` along with fixed slot sizes (largest tensor per precision).
2. `MemoryManager.reserve_hot/reserve_cold` hands out slot IDs; if necessary, `_ensure_hot_slot` evicts the LRU expert, ensuring no fragmentation.
3. `SwapEngine` propagates the slot ID through the swap future so `MemoryManager.place` can update `Residency.tags["slot_id"]` and track occupancy.
4. Slots are returned to the free pool when an expert migrates or a swap fails (`cancel_*_reservation`), guaranteeing the requested `n+1`/`m+1` buffers.

## 6. Resource-Aware Budgeting

1. `infer_runtime_config` keeps the total hot allocation within the measured HBM budget after subtracting non-expert weights (with a 20 % reserve), and records per-layer hot/cold sizes plus expert counts.
2. The runtime exports these numbers (`runtime_config` block in the JSON output) along with the epoch interval, enabling validation of per-layer budgets and SSD usage.

## 7. Evaluation Loop

1. After setup, `evaluate_mmlu_perplexity_mixed.py` streams the dataset, calling either the reference-aware or plaintext perplexity path.
2. Each call into the model benefits from the runtime: experts are upgraded/downgraded asynchronously, hot storage stays within the pre-allocated slots, and SSD payloads are prefetched transparently when configured.
3. Upon completion, the script prints aggregated metrics plus `runtime` counters and `runtime_config` diagnostics (including SSD usage).

This end-to-end flow fulfils all five requested behaviours while keeping the evaluation script straightforward for future calibration runs.

