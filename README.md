# DynaExQ

DynaExQ is a research runtime for budget-safe online precision residency in
Mixture-of-Experts (MoE) inference. It observes router traffic, selects a
bounded set of high-precision experts, and changes active expert
representations through versioned handles and admission-controlled GPU-memory
pools.

The open-source package contains:

- `dynaexq/core/`: scheduling, quantization, handles, memory pools, and
  asynchronous transition machinery;
- `dynaexq/integration/`: handle-aware MoE dispatch and Qwen3-Next integration;
- `dynaexq/models/`: Qwen3-MoE and Phi-3.5-MoE adapters;
- `dynaexq/experiments/`: quality, latency, memory, and workload harnesses;
- `dynaexq/baselines/`: baseline identity checks and offload baselines;
- `scripts/`: reproducible model-manifest, benchmark, telemetry, rendering, and
  artifact-audit tools;
- `ICCAD_2026_DynExq/`: manuscript source, figures, submission package, and
  compiled paper;
- `results/`: model manifests, registered paper artifacts, raw performance
  records, and smoke-check provenance.

The repository intentionally excludes model checkpoints, downloaded benchmark
data, raw user prompts, generated logs, and credentials.

## Installation

Create an isolated Python 3.10+ environment and install the package in editable
mode:

```bash
python -m pip install -e '.[test]'
```

AutoRound is optional:

```bash
python -m pip install -e '.[test,autoround]'
```

Loading legacy GPTQ-compatible static baselines requires a separate disposable
environment:

```bash
python -m pip install -e '.[gptq]'
```

Large-model experiments additionally require compatible model checkpoints and
a CUDA stack matching the installed PyTorch build.

## Verify the runtime

The default suite is CPU-safe and does not load model checkpoints:

```bash
python -m pytest
```

GPU/model integration tests can be selected explicitly after their optional
dependencies and checkpoints are available:

```bash
python -m pytest dynaexq/tests/test_handle_mode_forward.py
python -m pytest dynaexq/tests/test_phimoe_integration.py
python -m pytest dynaexq/tests/test_qwen3_moe_integration.py
```

## Core API

The public package entry point exposes the runtime core:

```python
from dynaexq.core import (
    BudgetTracker,
    DynaExqConfig,
    ExpertRegistry,
    HotnessTracker,
    MemoryPool,
    PrecisionScheduler,
    TransitionEngine,
)
```

Model integrations should dispatch expert calls through versioned
`ExpertHandle` snapshots. Transitions reserve exact bytes before allocation,
publish a new handle version only after its CUDA event is ready, and release
the previous representation after readers have drained.

## Reproducible experiments

The wrapper runs one explicit experiment at a time:

```bash
bash scripts/reproduce_paper.sh test
bash scripts/reproduce_paper.sh quality --help
bash scripts/reproduce_paper.sh perf --help
bash scripts/reproduce_paper.sh dynamic --help
bash scripts/reproduce_paper.sh activation-density --help
bash scripts/reproduce_paper.sh routing-trace --help
bash scripts/reproduce_paper.sh offload-waiting --help
bash scripts/reproduce_paper.sh audit
bash scripts/reproduce_paper.sh render-figures
bash scripts/reproduce_paper.sh paper
```

To publish the verified local AutoRound checkpoints to Hugging Face after
logging in with a write-enabled token:

```bash
export DYNAEXQ_MODEL_ROOT=/path/to/Models
bash scripts/upload_huggingface_models.sh
```

The upload uses the `Kris2017` namespace by default, creates public model
repositories, and can be resumed safely after interruption. Set
`HF_NAMESPACE` to override the target namespace.

Published checkpoints:

- [Phi-3.5-MoE-instruct W4A16 AutoRound](https://huggingface.co/Kris2017/Phi-3.5-MoE-instruct-W4A16-AutoRound)
- [Qwen3-30B-A3B-Instruct-2507 W4A16 AutoRound](https://huggingface.co/Kris2017/Qwen3-30B-A3B-Instruct-2507-W4A16-AutoRound)
- [Qwen3-Next-80B-A3B-Instruct W4A16 AutoRound](https://huggingface.co/Kris2017/Qwen3-Next-80B-A3B-Instruct-W4A16-AutoRound)
- [Qwen3-Next-80B-A3B-Instruct W2A16 AutoRound-derived](https://huggingface.co/Kris2017/Qwen3-Next-80B-A3B-Instruct-W2A16-AutoRound-derived)

The immutable Hub revisions and byte-level verification summary are recorded
in `release/huggingface/manifest.json`.

For example, a static-checkpoint performance run can be launched with:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh perf \
  --model /path/to/checkpoint \
  --paper-model qwen30b \
  --method quantized_checkpoint \
  --quantization int4 \
  --batch-size 32 \
  --input-length 2048 \
  --output-length 256 \
  --paper-protocol \
  --hash-model-files \
  --output results/qwen30b_int4_bs32.json
```

The performance harness samples current-process GPU device-memory use through
NVML, including native CUDA allocations that PyTorch allocator counters omit.
It records foreign GPU processes separately and rejects formal samples with
nonzero foreign-process utilization. Latency results cover isolated model
execution, not request queues or network overhead.

For a locally derived checkpoint, first register the immutable source, build a
non-test calibration corpus, quantize it, and record the resulting checkpoint:

```bash
python scripts/build_model_manifest.py \
  --model-dir /path/to/pinned-bf16 \
  --provider huggingface --repository org/model \
  --revision <immutable-commit> \
  --output results/model_bf16.json

python scripts/build_independent_calibration.py \
  --output /tmp/wikitext103_train_256x2048.jsonl \
  --manifest /tmp/wikitext103_train_256x2048.manifest.json

CUDA_VISIBLE_DEVICES=0 python scripts/quantize_with_autoround.py \
  --model-path /path/to/pinned-bf16 \
  --output-path /path/to/model-int4-mixed-AutoRound \
  --source-manifest results/model_bf16.json \
  --scheme W4A16 \
  --calibration-jsonl /tmp/wikitext103_train_256x2048.jsonl \
  --nsamples 256 --seqlen 2048 --seed 42 \
  --low-gpu-mem-usage --output-format auto_round

python scripts/build_quantized_model_manifest.py \
  --model-dir /path/to/model-int4-mixed-AutoRound \
  --output results/model_w4a16_autoround.json
```

The manifest builder rejects moving revisions such as `master`. The
quantization loader rejects test-split calibration rows and existing output
directories. Generated artifacts should record the method, immutable model and
dataset revisions, dependency versions, seed, command, and hardware identity.

HumanEval execution requires the explicit `--allow-code-execution` option and
should run in a disposable sandbox. A unit-test pass is not evidence for an
end-to-end GPU performance or accuracy claim.

## Paper and result artifacts

The manuscript source and its checked-in PDF are under
`ICCAD_2026_DynExq/`. Build the paper with:

```bash
bash scripts/reproduce_paper.sh paper
```

Machine-readable artifacts are separated by role:

- `results/paper/`: registered motivation artifacts and their manifest;
- `results/paper/performance/`: preserved formal and smoke performance JSON;
- `results/paper/audits/`: point-in-time audit reports;
- `results/model_manifests/`: immutable model/checkpoint provenance;
- `results/smoke_checks/`: checkpoint loading and generation checks.

Run `python scripts/audit_paper_results.py` before treating a manuscript value
as reproducible. Raw or smoke artifacts are retained for transparency but are
not automatically promoted into the strict claim manifest.

## License and citation

DynaExQ is released under the Apache License 2.0. Adapted upstream model files
retain their original copyright and license headers. See `NOTICE` for details
and `CITATION.cff` for software citation metadata.
