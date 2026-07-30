# DynaExQ

DynaExQ is a research runtime for budget-safe, online precision residency in
Mixture-of-Experts inference. It observes router traffic, selects a bounded set
of high-precision experts, and changes the active representation through
versioned handles and admission-controlled memory pools.

The repository contains:

- `dynaexq/core/`: scheduling, quantization, handles, memory pools, and
  transition machinery;
- `dynaexq/models/`: Qwen3-MoE, Phi-3.5-MoE, and DeepSeek-V2 adapters;
- `dynaexq/experiments/`: dataset, quality, and latency harnesses;
- `scripts/`: large-model experiments and artifact utilities;
- `ICCAD_2026_DynExq/`: the IEEE Transactions on Computers manuscript.

## Install

Create an isolated Python 3.10+ environment, then install the package in
editable mode:

```bash
python -m pip install -e '.[test]'
```

AutoRound is optional:

```bash
python -m pip install -e '.[test,autoround]'
```

Loading legacy GPTQ-compatible static baselines additionally requires a
separate disposable environment:

```bash
python -m pip install -e '.[gptq]'
```

The pinned GPTQModel stack requires Protobuf 7, which can conflict with
OpenTelemetry packages in a general development environment. The formal
AutoRound checkpoint below uses the AutoRound Triton backend and does not
require GPTQModel.

For a locally derived checkpoint, first register the immutable source and
build a non-test calibration corpus, then pass that corpus explicitly:

```bash
python scripts/build_model_manifest.py \
  --model-dir /path/to/pinned-bf16 \
  --provider huggingface --repository org/model \
  --revision <immutable-commit> \
  --output results/model_manifests/model_bf16.json
python scripts/build_independent_calibration.py \
  --output calibration_datasets/formal/wikitext103_train_256x2048.jsonl \
  --manifest calibration_datasets/formal/wikitext103_train_256x2048.manifest.json
CUDA_VISIBLE_DEVICES=0 python scripts/quantize_with_autoround.py \
  --model-path /path/to/pinned-bf16 \
  --output-path /path/to/model-int4-mixed-AutoRound \
  --source-manifest results/model_manifests/model_bf16.json \
  --scheme W4A16 --calibration-jsonl \
  calibration_datasets/formal/wikitext103_train_256x2048.jsonl \
  --nsamples 256 --seqlen 2048 --seed 42 --low-gpu-mem-usage \
  --output-format auto_round
python scripts/build_quantized_model_manifest.py \
  --model-dir /path/to/model-int4-mixed-AutoRound \
  --output results/model_manifests/model_w4a16_autoround.json
```

The manifest builder rejects moving revisions such as `master`, and the
quantization loader rejects test-split calibration rows and existing output
directories. A completed export must pass layer, terminal-tensor,
safetensors-index, and byte-count checks before the script writes
`quantization_provenance.json`.

Qwen3-Next uses Intel's official
`Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound` checkpoint for the static
INT4 reference. It has W4/group-128 expert weights, W8 non-expert projections,
and FP16 gates. Because ModelScope exposes a moving `master` ref, the
registration command derives a stable content-set SHA-256 from the complete
remote file catalog and verifies every downloaded byte against it.

The formal INT2 checkpoint is locally requantized from that registered INT4
parent: reconstructed W4 values are converted to W2/group-64 with
deterministic symmetric RTN, while the parent's W8/FP16 overrides are
preserved. This is explicitly recorded as INT4-to-INT2 requantization, not
direct BF16-to-INT2 calibration. Fetch/register the parent, then derive INT2:

```bash
bash scripts/quantize_qwen3_next_static.sh int4
bash scripts/quantize_qwen3_next_static.sh int2
```

The DynaExQ dynamic INT4/INT2 tiers are independently packed from the pinned
BF16 checkpoint because the runtime consumes native three-dimensional expert
tensors rather than static AutoRound QuantLinear modules.

Formal static quality and performance commands must add
`--autoround-backend triton`; the CLI rejects a paper-protocol quantized run
whose backend is left to environment-dependent automatic selection.

For inference, explicitly request a compatible backend instead of accepting
an environment-dependent automatic choice:

```python
from transformers import AutoRoundConfig

quantization = AutoRoundConfig(
    bits=4,
    group_size=128,
    sym=True,
    backend="triton",
    packing_format="auto_round:auto_gptq",
)
```

Large-model experiments additionally require the model checkpoints and a CUDA
stack compatible with the installed PyTorch build.

## Verify the runtime

The default test suite contains CPU-safe unit tests and does not collect
checkpoint-loading scripts:

```bash
python -m pytest
```

GPU/model integration tests can be selected explicitly after their
dependencies are available:

```bash
python -m pytest dynaexq/tests/test_handle_mode_forward.py
python -m pytest dynaexq/tests/test_phimoe_integration.py
python -m pytest dynaexq/tests/test_qwen3_moe_integration.py
```

## Paper artifact

Before using a number in the manuscript, run the provenance audit:

```bash
python scripts/audit_paper_results.py
```

The audit intentionally fails when a manuscript value disagrees with its raw
JSON result, when samples were skipped, or when a claimed configuration has no
machine-readable source. A successful unit test run is not evidence for the
paper's end-to-end performance claims.

Build the manuscript with:

```bash
cd ICCAD_2026_DynExq
pdflatex -interaction=nonstopmode -halt-on-error main_sc.tex
bibtex main_sc
pdflatex -interaction=nonstopmode -halt-on-error main_sc.tex
pdflatex -interaction=nonstopmode -halt-on-error main_sc.tex
```

Before an IEEE Transactions on Computers upload, copy
`ICCAD_2026_DynExq/TC_SUBMISSION_METADATA.example.json` to
`TC_SUBMISSION_METADATA.json`, replace every author/prior-version field, pass
the result-provenance audit, set `\artifactverifiedtrue`, rebuild, and run:

```bash
python scripts/audit_tc_submission.py
```

This final gate rejects anonymous or inconsistent author metadata, concurrent
submission, an under-disclosed conference extension, an unverified result
gate, rendered `TBD`/draft text, more than 12 Letter pages, LaTeX
overfull/undefined-reference failures, or an incomplete result manifest.
`TC_COVER_LETTER.example.md` provides the corresponding disclosure template.

## Reproducibility boundary

Several scripts in `scripts/` are exploratory model-loading programs rather
than unit tests. They may require local checkpoint paths and substantial GPU
memory. Do not label an output as FP16, INT4, INT2, an external offload
system, or DynaExQ
unless the invoked runtime actually activates that method and records the
method, checkpoint, commit, dependency versions, seed, and hardware in the
result file.

The safe reproduction wrapper forwards one explicit experiment at a time:

```bash
bash scripts/reproduce_paper.sh quality \
  --model /path/to/checkpoint \
  --paper-model qwen30b \
  --method quantized_checkpoint \
  --quantization int4 \
  --benchmarks wikitext,mmlu_pro,gpqa,aime25,gsm8k,humaneval \
  --paper-protocol \
  --allow-code-execution \
  --hash-model-files \
  --output results/paper/qwen30b_int4_quality.json

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
  --output results/paper/qwen30b_int4_bs32.json

bash scripts/reproduce_paper.sh dynamic \
  --config dynaexq/configs/qwen30b.yaml \
  --model-path /path/to/a/compatible-source-checkpoint \
  --device cuda:0 \
  --output results/paper/qwen30b_dynaexq_quality.json \
  --hash-model-files \
  quality --benchmarks wikitext,mmlu_pro,gpqa,aime25,gsm8k,humaneval \
  --paper-protocol \
  --allow-code-execution

CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh moe-infinity \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --repo /path/to/clean/pinned/MoE-Infinity \
  --offload-dir /local-ssd/moe-infinity-cache/qwen30b \
  --batch-size 32 \
  --output results/paper/qwen30b_moe_infinity_bs32.json
```

HumanEval requires the explicit `--allow-code-execution` flag and should run
inside a disposable container. Performance artifacts report isolated-model
latency; they do not include request queues or network overhead. The dynamic
entry point preloads both expert tiers, releases native expert parameters,
materializes every handle, reserves native-layout conversion workspaces, drains
transition workers before serializing final counters, and aborts if the
checkpoint cannot satisfy that contract. Benchmark loaders pin immutable
dataset revisions and record repository/config/split/fingerprint metadata.
Formal DynaExQ artifacts are rejected unless the scheduler ran, accepted
transitions completed exactly, per-stream event fences avoided every global
device-sync fallback, and shutdown left no active transition or pending
reservation.
The `moe-infinity` command accepts only the official pinned source identity,
the declared single RTX A6000 protocol, and Qwen3-30B; it also requires
positive external offload state and measured prefetch telemetry.
Formal performance producers sample the selected GPU's current-process HBM
use through NVML every 2 ms, including native CUDA allocations that PyTorch's
allocator counters omit, and retain all raw samples. Idle foreign processes
may keep HBM resident; their process count and bytes are recorded separately
and excluded from the reported current-process peak. Any nonzero foreign
NVML process utilization invalidates the sample. On a multi-GPU host, set
`CUDA_VISIBLE_DEVICES` to one A6000 as shown above; a model is never split
across two GPUs in the single-GPU protocol. Use `--paper-protocol`
for manuscript accuracy rows; it applies the complete
MMLU-Pro, GSM8K, GPQA-Diamond, AIME25, and HumanEval splits, plus at most 128
WikiText windows. A single global `--n-samples` is only for exploratory runs.
The `shift` command is retained for exploratory workload streams; formal
quality, performance, calibration, ablation, sensitivity, overhead,
perplexity, and routing-hotset artifacts use the validated `dynamic` entry
point.

The motivation section has separate executable collectors for activation
density and cold-cache blocking offload:

```bash
bash scripts/reproduce_paper.sh activation-density --help
bash scripts/reproduce_paper.sh routing-trace --help
bash scripts/reproduce_paper.sh offload-waiting --help
```

The waiting-time benchmark consumes the hashed routing trace rather than a
handwritten curve. See `ICCAD_2026_DynExq/RESULT_PROVENANCE.md` for the exact
protocol and registration requirements.

After producing a non-accuracy artifact, register its exact bytes and command:

```bash
bash scripts/reproduce_paper.sh register \
  --group performance \
  --claim-id performance:qwen30b:dynaexq:bs32 \
  --artifact results/paper/qwen30b_dynaexq_bs32.json \
  --command 'CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh dynamic ... perf ...'
```

Registration refuses legacy JSON, paths outside the repository, claim/group
mismatches, and duplicate claim IDs unless `--replace` is explicit.

After two complete quality artifacts exist, compare matched predictions rather
than treating their aggregate accuracies as independent:

```bash
bash scripts/reproduce_paper.sh compare-quality \
  --paper-model qwen30b \
  --left results/paper/qwen30b_int4_quality.json \
  --right results/paper/qwen30b_dynaexq_quality.json \
  --output results/paper/qwen30b_int4_vs_dynaexq_significance.json

bash scripts/reproduce_paper.sh register \
  --group quality_significance \
  --claim-id quality_significance:qwen30b:static_ptq_vs_dynaexq \
  --artifact results/paper/qwen30b_int4_vs_dynaexq_significance.json \
  --command 'bash scripts/reproduce_paper.sh compare-quality ...'
```

The comparison verifies identical dataset revisions, fingerprints, and sample
IDs, reports the paired accuracy delta and exact McNemar test per benchmark,
and applies Holm correction across the five tests. The strict manifest
requires the corresponding comparison for all three paper models.
