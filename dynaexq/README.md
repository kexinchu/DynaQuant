# DynaExQ Python package

The supported implementation path is:

- `core/`: hotness tracking, scheduling, quantization, exact byte accounting,
  fixed device pools, versioned handles, and transition execution;
- `integration/`: handle-aware model dispatch and router observation;
- `models/`: Qwen3-MoE and Phi-MoE model adapters;
- `experiments/`: formal quality, performance, calibration, ablation,
  sensitivity, overhead, and routing-profile entry points;
- `configs/`: the three paper-model contracts;
- `tests/`: CPU-safe correctness tests plus optional model/GPU integration
  tests.

Validated DynaExQ experiments run through `dynaexq.experiments.eval_dynamic`,
normally via:

```bash
bash scripts/reproduce_paper.sh dynamic --help
```

Run the default suite from the repository root:

```bash
python -m pytest
```

See the repository-level [`README.md`](../README.md) for installation and
experiment commands. A passing unit test is not a substitute for a clean,
checkpoint-backed experiment artifact.
