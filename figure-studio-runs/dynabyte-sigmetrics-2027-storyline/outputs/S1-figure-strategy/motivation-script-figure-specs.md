# Script-only motivation figure specifications

These figures must be rendered by deterministic plotting scripts from registered raw artifacts. Image generation is prohibited for their data-bearing content.

## O1 — transfer-pressure timeline

Suggested output: `SIGMETRICS_2027/figures/motivation_transfer_pressure_timeline.pdf`.

Required synchronized records per request/control window:

- request/run ID, model/checkpoint, runtime commit, device;
- phase and layer/control-window timestamp;
- demanded experts and actual miss experts;
- miss bytes by representation;
- effective H2D bandwidth and available compute-overlap duration;
- per-expert use deadline and arrival time or late bytes;
- exposed wait, demand-fetch wait, eviction/pollution events;
- current residency and precision map;
- budget, staging bytes, KV/activation reserve, and process peak memory.

Panels:

1. miss bytes over the aligned request timeline;
2. available overlap capacity `C × T_overlap` plus deadline markers;
3. late bytes/exposed wait, using the same x-axis and same run IDs.

Comparison: one fixed policy versus one adaptive policy, matched run protocol. Do not combine blocking trace replay with overlapping-runtime telemetry.

Falsification rule: if active-set changes do not produce meaningful changes in late bytes/exposed wait after overlap is accounted for, shrink O1 to workload context.

## O2 — precision quota × prefetch horizon response surface

Suggested outputs:

- `motivation_joint_surface_latency.pdf`;
- `motivation_joint_surface_quality.pdf` or a registered quality-feasibility mask;
- machine-readable grid artifact containing every attempted point, including refused/infeasible points.

Grid keys:

- expert-memory budget `B`;
- high-precision bytes or realized quota, not only requested ratio;
- horizon `S`;
- P95 TPOT and/or P95 exposed wait;
- quality metric and reference delta;
- peak memory including staging/transient copies;
- miss bytes, migration bytes, cache evictions, predictor configuration;
- raw samples and confidence intervals.

Overlays:

- quality-feasible boundary;
- memory-infeasible mask;
- precision-first sequential point;
- prefetch-first sequential point;
- best static joint point, labeled oracle if test-space searched;
- online joint point, only after it is actually measured.

Falsification rule: if joint/static/sequential differences are within uncertainty, remove the “joint control is necessary” claim and retain only configuration/operating-region characterization.

## O3 — quality value and migration amortization

Suggested output: `motivation_quality_amortization.pdf`.

Panel A:

- per-expert marginal NLL/loss/task-score effect of promotion;
- rank correlation/precision@k for hotness, offline sensitivity, and combined score;
- uncertainty across prompts/tasks and explicit calibration/test separation.

Panel B:

- top-k/hot-set overlap or identity survival versus window distance;
- promote+demote transferred bytes and measured time;
- break-even duration band derived from measured migration cost and observed benefit rate.

Falsification rules:

- weak hotness-to-quality relation → add sensitivity or drop hotness-only precision identity claims;
- identity lifetime shorter than break-even → do not claim slow-loop gains from online precision replacement.

## Existing-figure registration work

- DynaExQ Fig. 2: add three `routing_hotset:qwen30b:<workload>:layer15` raw bundles.
- DynaExQ Fig. 3: add `perplexity_curve:qwen30b` and `perplexity_curve:qwen80b` plus all point artifacts.
- Activation-density table: add the four missing Qwen model-stage claims only if it remains in the paper.
- DynaMoE legacy empirical plots: prefer unified-stack remeasurement over retrofitting rendered PDFs into the evidence manifest.

## Plotting and provenance contract

Each script-generated figure must register model/checkpoint, actual packed bytes, hardware/effective bandwidth, budget and reserves, workload protocol, raw samples, runtime commit, plotting command, input hashes, output hashes, and environment. Asset hashes alone are insufficient.
