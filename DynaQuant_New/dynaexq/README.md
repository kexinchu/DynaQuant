# DynaExQ

DynaExQ orchestrates expert precision transitions for Mixture-of-Experts
language models. It monitors router activations, selects precision tiers
with hysteresis, manages multi-tier memory pools, and performs non-blocking
upgrades/downgrades across HBM/DRAM/SSD.

This directory contains a from-scratch implementation aligned with the
`dynaexq_cursor_prompt_todo.md` design brief.

## Layout

- `runtime/` – Core runtime components (monitor, controller, memory manager, swap engine, prefetch, telemetry).
- `integration/` – Hook classes to integrate with SGLang and DeepSpeed.
- `configs/` – Default YAML configuration scaffold.
- `tests/` – Pytest unit tests for the runtime components.
- `scripts/` – Utility scripts (e.g., `run_smoke_test.py`) demonstrating end-to-end flows.

## Quickstart

```bash
cd /workspace/DynaQuant/DynaQuant_New
python -m pytest dynaexq/tests
python scripts/run_smoke_test.py --w4 /path/to/int4_checkpoint --prompt "你好"
```

### Dual-precision weight loading

The runtime can ingest paired weight checkpoints (e.g. INT4/INT2) and keep
both precisions in DRAM. Non-expert parameters default to the higher precision,
while expert parameters retain both versions for dynamic swapping:

```python
from dynaexq.runtime import Bitwidth, DualPrecisionWeights, ParameterIndex

weights = DualPrecisionWeights.from_files(
    "/path/to/model_int4.pt",
    "/path/to/model_int2.pt",
)

# Fetch a CPU tensor for an expert parameter, or move it directly to CUDA.
expert_idx = ParameterIndex(
    name="layers.0.experts.3.weight",
    bitwidth=Bitwidth.W4,
    expert=ExpertID(layer=0, idx=3),
)
tensor = weights.get_tensor(expert_idx, device="cuda")

# Build a state_dict that can be loaded into a torch.nn.Module without
# touching disk again (experts default to W4 if available).
state_dict = weights.materialize_state_dict()
model.load_state_dict(state_dict, strict=False)
```

To feed the swap engine, wrap the repository with
`dynaexq.runtime.InMemoryWeightStore`; the engine will read expert bundles from
memory instead of reloading from files.  The CLI in `run_smoke_test.py` accepts
`--w4` and optional `--w2` arguments pointing at HuggingFace-style directories
or standalone `.bin`/`.pt`/`.safetensors` files and performs a single prompt
generation after loading the model entirely from memory.

## Next Steps

- Implement real CUDA kernels for mixed-precision GroupGEMM in `runtime/kernels`.
- Connect storage backends to actual SSD/DRAM mappings.
- Wire telemetry sink to Prometheus or a structured logging pipeline.

