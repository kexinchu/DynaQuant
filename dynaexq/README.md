# DynaExQ Python package

The journal artifact uses the following implementation path:

- `core/`: hotness tracking, scheduling, quantization, exact byte accounting,
  fixed device pools, versioned handles, and transition execution;
- `integration/`: handle-aware model dispatch and router observation;
- `models/`: Qwen3-MoE and Phi-MoE model adapters;
- `experiments/`: formal quality, performance, calibration, ablation,
  sensitivity, overhead, and routing-profile entry points;
- `configs/`: the three paper-model contracts;
- `tests/`: CPU-safe correctness tests plus optional model/GPU integration
  tests.

The older `runtime/` and SGLang/DeepSpeed hook modules are compatibility
prototypes. They are not used as evidence for the IEEE Transactions on
Computers manuscript. Formal DynaExQ results must run through
`dynaexq.experiments.eval_dynamic`, normally via:

```bash
bash scripts/reproduce_paper.sh dynamic --help
```

Run the default suite from the repository root:

```bash
python -m pytest
```

See the repository-level [`README.md`](../README.md) for installation and
artifact commands, and
[`ICCAD_2026_DynExq/RESULT_PROVENANCE.md`](../ICCAD_2026_DynExq/RESULT_PROVENANCE.md)
for the submission evidence gate. A passing unit test is not a substitute for
a clean, checkpoint-backed, registered experiment artifact.
