# Result-provenance gate

The manuscript is an internal draft until:

```bash
python scripts/audit_paper_results.py
```

exits successfully and `\artifactverifiedtrue` is set in `main_sc.tex`.

That result audit is necessary but not sufficient for upload. After filling
`TC_SUBMISSION_METADATA.json` from the checked-in example and rebuilding the
PDF, the final command is:

```bash
python scripts/audit_tc_submission.py
```

It independently reruns this result audit and rejects missing author/ORCID
metadata, concurrent or under-disclosed prior work, draft markers, placeholders,
an invalid abstract, non-Letter or over-12-page output, and unresolved LaTeX
layout/reference failures.

The audit also requires `results/paper/manifest.json`. Its schema is:

```json
{
  "schema_version": 2,
  "groups": {
    "quality_significance": [],
    "performance": [
      {
        "claim_id": "performance:qwen30b:dynaexq:bs32",
        "path": "results/paper/...",
        "sha256": "...",
        "command": "..."
      }
    ],
    "ablation": [],
    "runtime_overhead": [],
    "budget_sensitivity": [],
    "activation_density": [],
    "offload_waiting": [],
    "routing_hotset": [],
    "perplexity_curve": [],
    "figure_bundle": []
  }
}
```

The manifest enumerates every reported operating point rather than merely one
file per result family. It requires three paired quality-significance claims,
42 performance claims (static PTQ and DynaExQ on three models, plus the
pinned official MoE-Infinity runtime on Qwen3-30B, at six batch sizes),
eight ablation claims, two runtime overhead claims, fourteen
budget-sensitivity claims, six activation-density rows, three
blocking-offload waiting curves, three workload hot-set claims, two
perplexity curves, and one rendered-figure bundle: 84 claims in total.
Paths are
repository-relative; SHA-256 values must match the files; `command` is the
exact reproduction command. Duplicate, missing, or unrecognized claims fail
the audit.

The artifact itself must use schema version 2 or newer, identify an immutable
checkpoint, record a clean Git commit and environment, and include the raw
evidence required by its claim. Performance claims require 2,048 input tokens,
256 output tokens, five warmups, all 100 raw samples, the isolated-performance
protocol marker, and summaries that the audit can recompute exactly. Each
timed generation also carries a 2 ms NVML high-water trace for the current
process on the selected physical GPU. This covers native CUDA-extension and
external-runtime allocations omitted by PyTorch's peak allocated and peak
reserved counters; those allocator counters remain diagnostic fields. The
audit rejects missing monitor metadata, fewer than two memory or utilization
polls, unavailable process accounting, any nonzero foreign-process NVML
utilization, or a peak beyond physical device memory. Idle foreign HBM
residency is recorded separately and excluded from the paper's current-process
high-water mark. Environment provenance also includes process peak RSS.
DynaExQ claims require router/handle/transition telemetry. The audit requires
an active scheduler and completed transitions (except the zero-capacity
sensitivity endpoint), exact accepted/completed counts, zero global-sync
reclaims, zero active workers, and zero pending/staging reservations after
shutdown. MoE-Infinity claims are accepted only for Qwen3-30B and require the
official repository at commit
`ba5651897a80d9c9b7a1500cef2c68adaa63db0f`, a clean recursive source-tree-manifest hash,
the imported module path, positive offloaded-tensor state, and measured-interval
prefetch counters. The adapter enables activation-aware caching and overlapping
speculative prefetch while using the isolated-model path shared by all methods.
It also records that the current public runtime differs from the paper
implementation. The repository's blocking single-LRU helper remains named
`LRUOffloadCache` and is not accepted as an external-system result.
Runtime-overhead claims require final transition/budget state;
ablation claims require both raw quality and performance evidence, recompute
the three displayed table metrics, and prove that the requested runtime
switches were active; sensitivity claims bind the model and exact
high-precision ratio. This prevents an orphan PDF plot, summary-only file, or
unrelated run from being treated as evidence.

Each core quality comparison (all-low static PTQ vs. DynaExQ) is also a
registered claim. Its producer accepts only the model-specific baseline
(`INT4` for Qwen3-30B and Phi-3.5-MoE, `INT2` for Qwen3-Next-80B), reopens two
clean schema-v2 quality artifacts, and requires identical dataset revisions,
fingerprints, and sample IDs. The significance artifact stores the complete
paired 2-by-2 counts, accuracy delta, two-sided exact McNemar p-value, and
Holm-adjusted p-value for each of the five tasks. The audit independently
rehashes both sources and recomputes every field; the cross-task average is
never treated as one pooled hypothesis test.

The motivation section is gated as strictly as the main evaluation.
Activation-density rows require raw per-layer active-expert counts at all six
batch sizes; waiting-latency curves require every raw blocking-offload sample
and cannot be labeled as an overlapping offload system; the layer-15 routing bundle must recompute
each top-10 set and prove pairwise disjointness; and every perplexity point
must recompute first from raw per-window boundaries, target-token counts,
mean losses, and NLLs and then from total NLL/token counts under the pinned
WikiText protocol. A curve also names and hashes all eight clean single-point
artifacts; the audit reopens them and checks their checkpoint, calibrated
ranking, frozen scheduler, explicit low-expert sets, and copied raw windows.
Conceptual architecture/procedure drawings do not require experimental
registration.

After all figure-driving claims are registered, regenerate the 17 empirical
PDFs and their input/output hash bundle:

```bash
bash scripts/reproduce_paper.sh render-figures
bash scripts/reproduce_paper.sh register \
  --group figure_bundle \
  --claim-id figure_bundle:main \
  --artifact results/paper/figure_provenance.json \
  --command 'bash scripts/reproduce_paper.sh render-figures'
```

The renderer refuses missing groups or artifact hash mismatches. The audit
then requires the bundle's 64 input claim hashes to equal the current
manifest and verifies the hashes of all 17 PDFs used by the manuscript.

## Confirmed problems in the current draft

- A dirty-tree exploratory run of the executable activation-density collector
  invalidated the old Qwen3-30B decode row, which had accumulated activity
  across more than one decode step. The draft now uses the raw per-layer
  one-step values (6.3/7.9/11.8/16.1/20.8/26.0 for decode and
  59.8/70.0/82.9/89.7/92.4/94.0 for prefill), but they still require a clean
  registered rerun before submission. The Qwen3-Next and DeepSeek rows remain
  unverified.
- Qwen3-30B FP16 and INT4 manuscript values for GSM8K and HumanEval disagree
  with `results/full_fp16_qwen30b.json` and
  `results/full_int4_qwen30b.json`.
- Those two source files also report skipped MMLU-Pro/GPQA examples.
- The available Phi-3.5-MoE INT4 result skipped every example in four
  benchmarks and cannot support a paper row.
- The available Phi-3.5-MoE FP16 values do not match the manuscript row.
- No registered machine-readable source currently supports the
  Qwen3-Next-80B
  accuracy rows, any DynaExQ accuracy row, the ablation table, the overhead
  table, or the budget-sensitivity figures.
- Several legacy performance JSON files mix vLLM, Transformers, one GPU, and
  two-GPU tensor parallelism. They are not directly comparable.
- Immutable BF16 source snapshots are now locally available and fully
  content-addressed for Qwen3-30B, Phi-3.5-MoE, and Qwen3-Next-80B. The
  original local Phi mixed-AutoRound directory is incomplete (only layers
  0--23 of 32) and is explicitly rejected. Its complete locally quantized
  W4A16 replacement now has structural, per-file, source, calibration, and
  execution provenance and has passed real generation with the AutoRound
  Triton backend. Intel's official Qwen3-Next mixed INT4 checkpoint is now
  pinned by a complete ModelScope content-set catalog, and the locally derived
  mixed INT2 checkpoint is bound to that parent manifest and has passed
  independent structural and per-file verification. Real Qwen3-Next Triton
  generation, clean-commit calibration rankings, and registered formal runs
  are still required.

## Required evidence for each reported operating point

Each result artifact must contain:

1. model/checkpoint identity and precision method;
2. Git commit plus dirty-worktree flag;
3. Python, PyTorch, Transformers, CUDA, and GPU versions;
4. dataset repository, configuration, split, pinned commit, loader fingerprint,
   deterministic subset, prompt protocol, and seed;
5. total, evaluated, failed/unparsed, and correct counts;
6. raw latency samples, warmup count, and the exact metric scope;
7. peak allocated/reserved GPU memory and host-memory use where claimed;
8. DynaExQ scheduler configuration, high/low pool capacities, transition
   counts, and transferred bytes for dynamic runs.

Expected or interpolated data must not be registered as evidence.

## Validated dynamic entry point

First create a ranking from an independent JSONL corpus. Every row must have
`dataset`, a `train`/`validation`/`dev`/`calibration` split, a stable `id`,
and `prompt`; test splits are refused.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh dynamic \
  --config dynaexq/configs/qwen30b.yaml \
  --model-path /path/to/compatible/source-checkpoint \
  --device cuda:0 \
  --output results/paper/qwen30b_initial_map.json \
  --hash-model-files \
  calibrate --prompts /path/to/independent_calibration.jsonl \
  --max-prompts 256 --max-input-tokens 2048
```

Run calibration from a clean commit. The artifact contains the checkpoint,
source and selected-ID hashes, clean code revision, and a full permutation of
experts for every layer. Calibration forces the uniform all-low
representation, disables online reassignment, and ranks by mean
token-normalized routed probability mass across prompts. The separate FP64
accumulator is order-invariant, so neither an arbitrary expert-ID prefix nor
prompt order biases the map. Formal runs refuse a checkpoint/model mismatch,
fewer than 128 prompts, test-split input, a dirty calibration revision, or a
tampered ranking hash.

Dynamic quality and performance runs must use:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh dynamic \
  --config dynaexq/configs/qwen30b.yaml \
  --model-path /path/to/compatible/source-checkpoint \
  --device cuda:0 \
  --output results/paper/qwen30b_dynaexq_quality.json \
  --hash-model-files \
  --initial-map results/paper/qwen30b_initial_map.json \
  quality --benchmarks wikitext,mmlu_pro,gpqa,aime25,gsm8k,humaneval \
  --paper-protocol \
  --allow-code-execution
```

For each performance batch size, use the `perf` subcommand and its exact
formal protocol:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh dynamic \
  --config dynaexq/configs/qwen30b.yaml \
  --model-path /path/to/compatible-source-checkpoint \
  --device cuda:0 \
  --output results/paper/qwen30b_dynaexq_bs32.json \
  --hash-model-files \
  --initial-map results/paper/qwen30b_initial_map.json \
  perf --batch-size 32 --input-length 2048 --output-length 256 \
  --warmup 5 --repeats 100 --paper-protocol
```

The entry point preloads both tiers into host memory, releases native expert
parameters, moves the remaining dense model to exactly one selected GPU,
materializes every expert handle from fixed resident/staging pools, validates
router observation and handle dispatch, and records runtime telemetry. It also
uses measured dense bytes and checks actual free device memory before pool
allocation. INT4 conversion workspaces are reserved before the resident
allocation is solved. The output is serialized only after transition workers
drain, so its failure and pool counters are final. Failure at any stage aborts
the run.

## Official MoE-Infinity performance baseline

The external offloading baseline is deliberately limited to Qwen3-30B, the
manuscript checkpoint supported by the pinned public runtime. Prepare a clean
official checkout at the exact commit named below; do not install a moving
branch or copy its benchmark numbers:

```bash
git clone https://github.com/EfficientMoE/MoE-Infinity.git \
  /local-ssd/moe-infinity
git -C /local-ssd/moe-infinity checkout \
  ba5651897a80d9c9b7a1500cef2c68adaa63db0f

CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh moe-infinity \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --repo /local-ssd/moe-infinity \
  --offload-dir /local-ssd/moe-infinity-cache/qwen30b \
  --batch-size 32 \
  --output results/paper/qwen30b_moe_infinity_bs32.json

bash scripts/reproduce_paper.sh register \
  --group performance \
  --claim-id performance:qwen30b:moe_infinity:bs32 \
  --artifact results/paper/qwen30b_moe_infinity_bs32.json \
  --command 'CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh moe-infinity ... --batch-size 32 ...'
```

Repeat the producer and registrar for batch sizes 1, 2, 4, 8, 16, and 32.
The producer first resolves the Hugging Face model ID to an immutable local
snapshot, verifies a single 48 GB RTX A6000 and a clean DynaExQ tree, then
measures exact 2,048-token inputs and 256 generated tokens with five warmups
and 100 repeats. It hashes the committed recursive Git tree manifest, proves the
imported module came from that checkout, and rejects a run unless the external
engine marks expert tensors offloaded and invokes its prefetch path during the
measured interval. The figures and text call this implementation
``MoE-Infinity (open source)'' because its own README says the current code was
redesigned and differs from the paper version. No MoE-Infinity claim is
registered for Qwen3-Next-80B or Phi-3.5-MoE.
On a host with multiple GPUs, expose only the reserved A6000 through
`CUDA_VISIBLE_DEVICES`; the producer rejects multiple visible devices, and
the shared timer invalidates any sample if another compute process appears on
the selected GPU.

For the 80B model, use
`dynaexq/configs/qwen3_next_80b.yaml` with the exact
`Qwen/Qwen3-Next-80B-A3B-Instruct` source checkpoint. Its 512 routed experts
are handle-managed; the additional shared expert remains fixed and is charged
to the measured dense reservation.

`--paper-protocol` is mandatory for manuscript accuracy rows. It evaluates
the complete pinned MMLU-Pro, GSM8K, GPQA-Diamond, AIME25, and HumanEval
splits and caps WikiText at 128 windows, rather than applying one ambiguous
global sample count. Non-accuracy artifacts are added to the manifest with:

The component-ablation artifact is produced by a dedicated combined mode:

```bash
bash scripts/reproduce_paper.sh dynamic \
  --config dynaexq/configs/qwen30b.yaml \
  --model-path /path/to/compatible/source-checkpoint \
  --device cuda:0 \
  --output results/paper/qwen30b_ablation_blocking.json \
  --hash-model-files \
  --initial-map results/paper/qwen30b_initial_map.json \
  ablation --ablation-config blocking --allow-code-execution
```

Valid configurations are `full`, `static`, `blocking`, and
`no_hysteresis`. They respectively select normal asynchronous operation,
freeze the bootstrap precision map, run transitions inline without a
migration stream, and set the scheduler margin to zero. The full Qwen
configurations explicitly use a nonzero margin. The combined run fixes the
quality-task order and then executes the exact batch-32 latency protocol, so
all Table IV values are derived from one auditable state trace.

One budget-sensitivity point uses:

```bash
bash scripts/reproduce_paper.sh dynamic \
  --config dynaexq/configs/qwen30b.yaml \
  --model-path /path/to/compatible/source-checkpoint \
  --device cuda:0 \
  --output results/paper/qwen30b_ratio20.json \
  --hash-model-files \
  --initial-map results/paper/qwen30b_initial_map.json \
  sensitivity --hi-ratio-pct 20 --allow-code-execution
```

The accepted ratios are 0, 5, 10, 15, 20, 25, and 30 percent. The runtime
sets every layer to `floor(E * ratio)`, records requested and realized
ratios plus resident bytes, and fails if that exact quota cannot fit the
declared envelope. The audit recomputes the five-task average and checks the
complete per-layer quota vector; it never accepts a silently reduced point.

The runtime-overhead table uses:

```bash
bash scripts/reproduce_paper.sh dynamic \
  --config dynaexq/configs/qwen30b.yaml \
  --model-path /path/to/compatible/source-checkpoint \
  --device cuda:0 \
  --output results/paper/qwen30b_overhead.json \
  --hash-model-files \
  --initial-map results/paper/qwen30b_initial_map.json \
  overhead --allow-code-execution
```

This mode executes the same five-task trace and exact batch-32 performance
protocol. It derives peak reserved GPU memory from the 100 raw samples
(while retaining peak allocated bytes),
resident and transient pool bytes from initialization, migrations and copied
bytes from final transition counters, scheduler mean/P99 from raw
control-plane timing samples, and the pinned-cache payload from the packed
host representations. Fixed resident pools are full by construction, so the
paper no longer reports a misleading average ``pool utilization.'' Empirical
cells remain `TBD` until the two registered artifacts exist.

The WikiText perplexity curve is collected one frozen point at a time:

```bash
bash scripts/reproduce_paper.sh dynamic \
  --config dynaexq/configs/qwen30b.yaml \
  --model-path /path/to/compatible/source-checkpoint \
  --device cuda:0 \
  --output results/paper/qwen30b_ppl_ratio0.json \
  --hash-model-files \
  --initial-map results/paper/qwen30b_initial_map.json \
  perplexity-point --low-ratio-pct 0
```

Repeat this command for exactly `0, 15, 30, 45, 60, 75, 90, 100`. Each run
disables online scheduling and demotes the calibrated coldest suffix, so the
curve changes only the low-precision quota. Aggregate the eight point files
with repeated `--point` arguments:

```bash
bash scripts/reproduce_paper.sh build-ppl-curve \
  --paper-model qwen30b \
  --point results/paper/qwen30b_ppl_ratio0.json \
  --point results/paper/qwen30b_ppl_ratio15.json \
  --point results/paper/qwen30b_ppl_ratio30.json \
  --point results/paper/qwen30b_ppl_ratio45.json \
  --point results/paper/qwen30b_ppl_ratio60.json \
  --point results/paper/qwen30b_ppl_ratio75.json \
  --point results/paper/qwen30b_ppl_ratio90.json \
  --point results/paper/qwen30b_ppl_ratio100.json \
  --output results/paper/qwen30b_perplexity_curve.json
```

The builder refuses inconsistent checkpoints, configurations, calibration
maps, code revisions, ratios, or scheduler state. Register only the aggregate
curve under `perplexity_curve:qwen30b`; its source hashes retain the full
single-run chain of custody.

The layer-15 routing bundle uses a dedicated profiler:

```bash
bash scripts/reproduce_paper.sh dynamic \
  --config dynaexq/configs/qwen30b.yaml \
  --model-path /path/to/compatible/source-checkpoint \
  --device cuda:0 \
  --output results/paper/qwen30b_routing_hotset.json \
  --hash-model-files \
  routing-hotset --allow-code-execution
```

This formal run fixes every expert at INT4, disables the scheduler, evaluates
the complete pinned WikiText, GSM8K, and HumanEval workloads in that order,
and resets an exact selected token--expert dispatch counter between tasks.
Ordinary modes leave this counter disabled, so routing profiling cannot
inflate the paper's latency measurements. The same bundle is registered for
the three `routing_hotset:qwen30b:*:layer15` claims; the audit recomputes each
top-10 list from all 128 raw counts and checks the reported disjointness.

Activation-density rows are collected as one two-stage artifact per model:

```bash
bash scripts/reproduce_paper.sh activation-density \
  --paper-model qwen30b \
  --model-path /path/to/pinned/qwen3-30b-checkpoint \
  --prompts calibration_datasets/requests/mmlu_pro_200.jsonl \
  --device cuda:0 \
  --hash-model-files \
  --output results/paper/qwen30b_activation_density.json
```

The prompt JSONL must live in the repository and contain at least 160 unique
`id`/`prompt` rows. Five ordered blocks of 32 are reused as nested prefixes
for batches 1--32. For every block and routed layer, the collector records
the number of unique experts selected over nonpadding prefill tokens and over
the immediately following single-token decode. The audit rehashes the prompt
source, enforces model-specific expert/top-k/layer contracts, recomputes all
36 percentages from the raw count tensors, and compares them with Table I.
Use `--device auto` only when a pinned checkpoint must be sharded.

The blocking-offload motivation curve is a separate cold-cache
microbenchmark. It consumes a clean, registered routing-active-set trace with
exactly 17 nested input lengths (`16, 32, ..., 1536, 2048`), two warmup
trials, and ten measured trials per point:

```bash
bash scripts/reproduce_paper.sh routing-trace \
  --paper-model qwen30b \
  --model-path /path/to/pinned/qwen3-30b-checkpoint \
  --prompts calibration_datasets/requests/mmlu_pro_200.jsonl \
  --device cuda:0 \
  --hash-model-files \
  --output results/paper/qwen30b_routing_active_set_trace.json

bash scripts/reproduce_paper.sh offload-waiting \
  --paper-model qwen30b \
  --trace results/paper/qwen30b_routing_active_set_trace.json \
  --device cuda:0 \
  --output results/paper/qwen30b_blocking_offload_waiting.json
```

The collector concatenates an EOS-separated, repository-local prompt corpus
and takes 12 disjoint 2,048-token blocks. Each curve point reuses a nested
prefix of those same blocks, so input length changes without changing trial
identity. For each traced layer the source artifact records the sorted unique
expert IDs plus every routed-expert tensor's name, shape, dtype, element size,
and stored bytes. The audit sums those raw parameter/buffer records and
recomputes the per-expert payload. The benchmark starts each replay from an
empty logical cache, copies every missed payload from pinned host memory to
the selected GPU, synchronizes before reporting exposed wall time, and
retains CUDA-event time, miss count, transferred bytes, and trial ID for every
measured replay. The audit rehashes and reopens both the trace and prompt
source, enforces the model layer/expert contract and exact trial grid, and
recomputes every miss and transferred-byte count before accepting a curve.
This is deliberately labeled `blocking_on_demand`; it is not evidence for
ExpertFlow or for a system that overlaps communication with computation.

Non-accuracy artifacts are added to the manifest with:

```bash
bash scripts/reproduce_paper.sh register \
  --group performance \
  --claim-id performance:qwen30b:dynaexq:bs32 \
  --artifact results/paper/example.json \
  --command 'the exact command that produced example.json'
```

The registrar hashes the artifact, stores a repository-relative path, refuses
legacy schemas or a claim/group mismatch, and replaces an existing claim only
with explicit `--replace`.

Paired accuracy comparisons use:

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

Repeat with `qwen80b` (INT2 vs. DynaExQ) and `phi35` (INT4 vs. DynaExQ).
The command refuses dirty or ambiguously labeled quality artifacts,
mismatched dataset revisions, fingerprints, incomplete samples, or different
sample IDs. It reports the paired effect and exact McNemar p-value for each
benchmark with Holm correction across the five comparisons; it does not apply
an unpaired test to aggregate percentages.
