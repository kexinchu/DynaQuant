# Qwen3-30B / A6000 priority-ordered evaluation plan

This file is the authoritative execution plan for the local Qwen3-30B lane.
It supersedes the earlier requirement to finish every member of the
152-artifact legacy matrix. Completed artifacts remain valid when they pass
the provenance checks below. Other checkpoints and accelerators are populated
separately and must not be inferred from this lane.

For the active Goal, “complete the missing Qwen3-30B/A6000 experiments” means
completing every `required` artifact and only those `conditional` artifacts
promoted by the rules in this file. It does not mean exhausting the legacy
152-item superset. Plotting, manuscript filling, and final audit begin only
after these requirements pass.

The immediate execution target is P2A, the first RQ1 headline screen under
the narrowed claim. P1A and the promoted P1B switch pair have passed
validation. P1B selects Seq-RF as the strongest deployable switch baseline, so
P2 must compare DynaByte against Seq-RF before spending GPU time on additional
component policies, regimes, or seeds. This rule is also the active Goal's
execution contract: finish the smallest priority-complete set that can support
the paper claim, not every artifact in the old 152-item matrix.

## Current priority queue for the active Goal

As of 2026-10-06, the active Goal is interpreted through the following
priority queue. The queue is ordered by how directly an experiment can change
the paper's main claims. Lower-priority rows must not run while a higher row is
unfinished. An idle GPU may be used only for another artifact in the same
active rank or for a later rank that has already been promoted by the stated
rule.

| Rank | Experiments | Status / trigger | Paper role |
|---:|---|---|---|
| 1a | P2A-1, \(\Phi=1.30\) recovery: `rq1_phi130_mixed_dynabyte_seed42.json` and `rq1_phi130_mixed_seq_rf_seed42.json` | Required now. \(\Phi=0.85\) DynaByte and Seq-RF are valid. Seq-RF \(\Phi=1.30\) is running on GPU0 and must not be interrupted. DynaByte \(\Phi=1.30\) failed at request-boundary quality-risk repair and must be debugged, retested, and validated before any lower rank starts. | Completes the strongest-baseline screen at the quality-feasible primary budget. |
| 1b | P2A-1 audit | Run immediately after Rank 1a. Validate all four P2A-1 artifacts and record any source-hash namespace change caused by runtime fixes. | Prevents carrying a mixed-version or invalid screening set into RQ1. |
| 2 | P2A-2: Static-Joint at the same two mixed-trace cells | Run only after all four P2A-1 artifacts validate. | Preserves the joint-online vs fixed-calibration comparison without delaying the strongest-baseline result. |
| 3 | P2A summary and promotion decision | Run immediately after P2A-2 validates. | Selects the mixed-regime winner, strongest baseline, and any static policy that must stay in the main RQ1 table. |
| 4 | P2C-screen-to-confirm: choose the minimum confirmation set | Use the P2A summary to select only the mixed-regime winner and strongest deployable baseline at each primary budget. If the two budgets select the same pair, reuse that pair across both budgets and do not add secondary policies. | Prevents five-seed confirmation from expanding into the old full matrix. |
| 5 | P2C: five paired seeds, 100 measured requests, selected pair only | Promote only for cells that will appear in the main RQ1 table. | Converts screening numbers into headline latency/goodput evidence with uncertainty. |
| 6 | P2D: prefill-heavy and decode-heavy winner-vs-baseline confirmation | Promote after P2C selects the pair. Run seed 42 screening first; add confirmation seeds only if the result is close or contradicts the mixed trace. | Shows the result is not specific to the mixed trace. |
| 7 | P3 mechanism removals and real shared-arena/private-arena executor checks | Run after the RQ1 headline path is stable. Use seed 42 screening unless a mechanism result changes the main explanation. | Explains why the selected policy wins. |
| 8 | P4 accounting and predictor evidence | Prefer off-GPU or low-cost collection when possible. Run GPU work only for fields that cannot be recovered from retained artifacts. | Supports proxy accuracy, memory accounting, and controller overhead claims. |
| 9 | P5 boundary sweeps | Run one axis at a time, only around policy transitions or unresolved ties. | Defines where the method stops helping. |

The old 152-item matrix remains a legacy superset. A missing legacy cell is not
a blocker unless one of the promotion rules above explicitly pulls it back into
the active queue. The expected active set is intentionally small: P2A-1 has
four screening artifacts, P2A-2 adds two, and later ranks promote only the
winner-vs-baseline cells needed for the paper's main table and mechanism
claims.

## 1. Objective and scheduling rule

The experiment queue is ordered by the value of the evidence to the paper,
not by the order in which configurations were added to the harness. The next
GPU job must belong to the highest unfinished priority whose code and data
gates pass.

| Priority | Question answered | Action |
|---|---|---|
| P0 | Is the measurement path sound? | Finish the currently active all-high quality reference and the seed-46 batch sweep; retain the completed calibration and runtime gates. |
| P1 | Is joint, online allocation necessary? | Run P1A first: sparse allocation anchors and DynaByte-vs-Static switch. Expand only when those runs cannot identify the optimum or the strongest online baseline. |
| P2 | Does DynaByte improve the quality-constrained latency frontier? | Run the paired RQ1 headline comparisons at the two primary budgets. |
| P3 | Which mechanisms cause the improvement? | Run the RQ3 removals and the real private-arena executor. |
| P4 | Are the proxy, predictor, allocator, and controller measurements credible? | Collect RQ4 prediction/outcome and memory-accounting evidence; run controller scaling off-GPU where possible. |
| P5 | Where does the method stop helping? | Screen RQ5 one-axis-at-a-time sweeps and confirm only policy transitions and statistically unresolved boundary points. |
| P6 | Does a result need reinforcement? | Run optional seeds or secondary cells only when the prespecified interval or validation test fails. |

For the active Goal, priority starts at the highest unfinished gate, not at
the first row of the table. P2 is currently that gate. P0 and P1 artifacts are
reused when they pass audit; they are not rerun unless provenance or
validation fails. P1 has already narrowed the online claim: DynaByte beats
Static-Joint on the switch trace, but Seq-RF is faster on that trace. More
repetitions must not be used to conceal that result.

### Active Goal execution order

The active Qwen3-30B/A6000 Goal follows this order until a priority is either
complete or explicitly pruned by its decision rule:

1. Preserve P1A as the completed sparse screen and do not add dense-grid cells.
2. Preserve the completed P1B Seq-FR and Seq-RF switch artifacts.
3. Treat Seq-RF as the strongest deployable Qwen3-30B baseline until RQ1
   evidence supersedes it.
4. Run P2A first: DynaByte and Seq-RF on the mixed trace at
   \(\Phi\in\{0.85,1.30\}\), seed 42, screening length.
5. Add Static-Joint at the same two P2A cells only after the DynaByte-vs-Seq-RF
   pair is valid; this preserves the joint-vs-static comparison without
   delaying the strongest-baseline check.
6. Run a P2A summary before any seed expansion. Promote five-seed headline
   confirmation only for the selected mixed-regime winner and the strongest
   deployable baseline at each primary budget.
7. Run prefill-heavy and decode-heavy regimes only after the mixed-regime
   winner and strongest baseline are known. Start with seed 42 screening and
   add confirmation seeds only when the regime changes the conclusion or the
   result is statistically close.
8. Add no secondary policies, regimes, boundary budgets, or extra seeds unless
   the trigger rules below promote them.

This ordering deliberately spends GPU time first on claim-critical evidence:
the completed P1A anchors establish that dense grid cells would mainly improve
visual smoothness, while the promoted P1B switch pair can still change the
strongest-baseline conclusion. Repeated seeds, dense grid cells, and secondary
policies remain lower priority unless the switch comparison is ambiguous.

At the current boundary, P1B has finished and validated. The next GPU work is
P2A. Do not backfill old dense-grid or full-switch cells.

### P2 trigger rules

The P2 queue is deliberately staged to avoid the 152-artifact legacy pattern.
Each stage starts only after the previous stage validates.

| Stage | Required first artifacts | Promotion rule |
|---|---|---|
| P2A-1 | `rq1_phi085_mixed_dynabyte_seed42.json`, `rq1_phi085_mixed_seq_rf_seed42.json`, `rq1_phi130_mixed_dynabyte_seed42.json`, `rq1_phi130_mixed_seq_rf_seed42.json` | Always run. These four cells decide whether DynaByte can beat the strongest deployable baseline at the two primary budgets. |
| P2A-2 | `rq1_phi085_mixed_static_joint_seed42.json`, `rq1_phi130_mixed_static_joint_seed42.json` | Run after P2A-1 validates. These cells retain the joint-online vs fixed-calibration comparison. |
| P2B | `seq_fr` and `fixed_partition` on the same two mixed cells | Run only if P2A is close, if Static-Joint beats one of the online policies, or if the frontier needs a component winner. |
| P2C | five paired seeds and 100 measured requests for the mixed-regime winner and strongest deployable baseline | Run only for the final main-table pair at each primary budget. If both budgets select the same pair, do not duplicate policy exploration. |
| P2D | prefill-heavy and decode-heavy winner-vs-baseline confirmation | Run only after P2C selects the pair. Start with seed 42 screening; add confirmation seeds only when the conclusion changes or the interval is unresolved. Add a component policy only if P2B shows it wins mixed or ties the winner. |
| P2E | \(\Phi=1.94656\), boundary budgets, or extra policies | Optional. Run only for a visible frontier transition or an unresolved statistical tie. |

The P2A-1 \(\Phi=1.30\) cells use the joint 20/10/70 share. Under the paper
budget mapping, that share is memory-infeasible at \(\Phi=0.85\); the
\(\Phi=0.85\) screen therefore uses the feasible 0/20/80 boundary point and is
reported as a tight-budget latency screen, not as a quality-feasible headline
point.

### Minimum-repeat rule for the active Goal

The queue is minimized around decision power, not visual completeness. A new
GPU run is allowed only if it can change one of these decisions:

1. whether the best feasible point uses more than one byte class;
2. whether DynaByte beats the strongest matched non-online policy under a
   workload switch;
3. whether a feasibility boundary, proxy/direct-quality disagreement, or
   statistical tie changes the selected policy;
4. whether the final winner and the strongest deployable baseline still hold
   under the prespecified confirmation seeds.

This rule replaces repeated dense sweeps. Aesthetic heatmap density, unused
baseline variants, and unpromoted seeds are not sufficient reasons to run
additional Qwen3-30B jobs.

## 2. Preservation and safe-boundary rule

The active runtime source hash is
`8277f77eefb7389d218790a9d38017e07a64d6916c3bef4d88c0c6405ac57e88`.
Do not interrupt an artifact in progress. Under this hash:

1. GPU0 finishes `all_high_reference_quality_seed42.json`;
2. GPU1 finishes the remaining seed-46 batch points;
3. the legacy runners stop at an artifact boundary;
4. the optimized manifest and priority queue replace the 152-item queue;
5. runtime changes start in a new source-hash namespace.

No completed artifact is deleted. The former 152-item list is a legacy
superset, not a completion target. A skipped member is recorded as
`not_required_by_optimized_plan`, rather than as a missing result.

Changes to policy semantics, allocation, copying, prediction, or measurement
instrumentation require a new source hash. Additive audit, manifest, and plot
code does not change the runtime hash.

## 3. Minimum retained evidence from the legacy matrix

The optimized legacy subset contains 76 or 77 artifacts, depending on the
number of distinct fidelity-map hashes. It removes repeated measurements that
do not strengthen a claim.

| Evidence | Required artifacts | Count |
|---|---|---:|
| Calibration and runtime gates | Calibration plus seven gates | 8 |
| All-high reference | One item/window-level quality artifact | 1 |
| Policy quality | One artifact per distinct fidelity-map hash | 4--5 |
| Component screening | Five component policies, seed 42, three regimes | 15 |
| Joint headline traces | Five paired seeds, three regimes | 15 |
| Batch scaling | Seed 46 at batches 1, 2, 4, 8, 16, 32; seeds 42--45 at batches 1, 8, 32 | 18 |
| Routing characterization | Frozen hot-set artifact | 1 |
| Existing mechanism variants | Eight seed-42 artifacts | 8 |
| Fidelity sensitivity | Ratios 0%, 10%, 15%, 20%, and 30% | 5 |
| Runtime accounting | One overhead artifact | 1 |
| **Total** |  | **76--77** |

Quality is never rerun merely because residency or lookahead placement
differs. Reuse is permitted only when the validator proves the frozen
fidelity-map hash is identical. SLO attainment, goodput, quantiles, confidence
intervals, and arena peaks are derived from retained per-request samples and
do not require new inference runs.

## 4. Repetition tiers

Three repetition tiers replace a uniform five-seed rule.

- Headline confirmation: five paired seeds, five warmups, and 100 measured
  requests. This tier is reserved for DynaByte and the strongest deployable
  baseline in the main result, plus any comparison directly supporting the
  joint/online claim.
- Mechanism and boundary confirmation: three paired seeds, five warmups, and
  100 measured requests. Use it for selected ablations, policy transitions,
  and boundary points.
- Screening and descriptive evidence: seed 42, two warmups, and 20 measured
  requests. Use it to select cells and policies. Screening numbers are not
  reported as headline latency.

All comparisons share request IDs. A screening cell is promoted only if it is
an optimum, a feasibility boundary, a policy transition, or the strongest
baseline. No full Cartesian product is confirmed.

## 5. Frozen setup

- Checkpoint: `Qwen3-30B-A3B-Instruct-2507`, 48 layers, 128 experts per
  layer, top-8 routing.
- Device: one exclusive NVIDIA RTX A6000 per timed process. GPU0 and GPU1 are
  interchangeable only after the same NVML isolation gate passes.
- High and low representations: FP16 and INT4.
- Measured expert footprints: \(P_{lo}=15,401,484,288\) bytes and
  \(P_{hi}=57,982,058,496\) bytes.
- Primary budgets: \(\Phi=0.85\) and \(1.30\). Boundary budgets are 0.70,
  1.00, and the existing A6000 maximum 1.94656.
- For Qwen3-30B runtime experiments, \(\Phi\) maps to expert-arena budget
  \(B=\Phi P_{lo}\). The runner then adds KV, activation, dense-weight,
  workspace, and safety reserves to form `device_mem_bytes`. It must not scale
  the whole device cap by \(\Phi\).
- Latency workload: fixed, length-stratified ShareGPT request IDs for
  prefill-heavy, mixed, and decode-heavy regimes.
- Adaptation workload: a frozen prefill-to-decode-to-mixed sequence with
  explicit change points.
- Quality workloads: WikiText-2, MMLU-Pro, GPQA, AIME25, GSM8K, and HumanEval
  at the pinned revisions and item IDs.

The default service endpoint requires TPOT at most 500 ms and TTFT at most
`max(2 s, 8 ms * input tokens)`. Goodput includes output tokens only from
requests that satisfy both conditions.

Direct test-set feasibility uses the preregistered limits: default allows a
WikiText perplexity increase of 0.5 and no task decrease above 1.0 percentage
point; strict uses 0.2 and 0.5; loose uses 1.5 and 2.0. Runtime quality risk is
an admission proxy, not direct \(\Delta Q\).

## 6. Priority experiments

### P1: RQ2 joint and online necessity

P1 uses a staged decision set instead of a full 5-by-5 grid. The first stage
answers the paper-critical question with the fewest GPU runs: can one byte use
explain the best point, and does online reallocation beat a fixed calibration
allocation after the workload changes?

Run these eight P1A artifacts first, in order:

| Order | Artifact | Why it runs first |
|---:|---|---|
| 1 | `rq2_grid_phi085_fid0_look0_seed42.json` | Residency-only anchor; already in flight and needed as the baseline for marginal byte value. |
| 2 | `rq2_grid_phi085_fid40_look0_seed42.json` | Fidelity-heavy anchor; tests whether quality bytes dominate under the tight budget. |
| 3 | `rq2_grid_phi085_fid0_look20_seed42.json` | Lookahead-heavy anchor; tests whether transfer hiding dominates. |
| 4 | `rq2_grid_phi085_fid20_look10_seed42.json` | Interior joint point; the smallest direct test of the three-use thesis. |
| 5 | `rq2_grid_phi085_fid20_look0_seed42.json` | Fidelity-residency edge; separates fidelity value from lookahead value. |
| 6 | `rq2_grid_phi085_fid0_look10_seed42.json` | Lookahead-residency edge; separates lookahead value from fidelity value. |
| 7 | `rq2_switch_dynabyte_seed42.json` | Online policy under the frozen prefill-decode-mixed workload switch. |
| 8 | `rq2_switch_static_joint_seed42.json` | Strongest non-online comparison for the same switch trace. |

Each grid cell stores the three byte uses, p95 TPOT, exposed wait, migration
bytes, direct and proxy feasibility, and memory feasibility. Direct quality
is shared across cells with the same fidelity-map hash. The six sparse cells
are enough to identify a clear axis winner, a clear interior winner, or an
ambiguous boundary. They are not presented as a dense heatmap unless the
conditional expansion below is triggered.

Run P1B grid expansion only if P1A is ambiguous. Add at most six cells, in
this order:

1. one neighbor on each side of the best feasible P1A point, if that neighbor
   has not already run;
2. the first direct-quality or memory infeasible boundary nearest to the best
   point;
3. any point where proxy risk and direct quality disagree;
4. one tie-breaker only if two feasible policies remain statistically
   unresolved after the first three checks.

Stop as soon as the feasible optimum and the policy boundary are clear. Do not
complete the old 25-cell grid unless the sparse expansion still cannot identify
the feasible optimum.

Run Seq-FR and Seq-RF switch screens only if Static-Joint is worse than
DynaByte or if the grid optimum cannot be explained by one dominant byte use.
Confirm DynaByte and the strongest matched deployable baseline with five
seeds after P1A/P1B identifies that baseline. Confirm another online policy
only if it changes the interpretation. Record applied fidelity, residency,
and lookahead bytes, rolling p95 TPOT, quality state, convergence time, and
migration bytes after each change. Oracle-Grid-Test is a held-out upper
bound, not an end-to-end baseline.

### P2: RQ1 quality-constrained frontier

P2 is staged by claim value. The first screen compares DynaByte with Seq-RF,
the strongest deployable Qwen3-30B baseline selected by P1B, on the mixed
trace at \(\Phi\in\{0.85,1.30\}\). These four artifacts are the highest
priority unfinished experiments because they decide whether the headline RQ1
frontier can show an online joint-allocation benefit under the local
Qwen3-30B/A6000 setting.

After the four strongest-baseline cells validate, add Static-Joint at the same
two budgets to keep the online-versus-fixed-calibration comparison. Add
Seq-FR and Fixed-partition only if the first screen is close, if Static-Joint
beats an online policy, or if the frontier needs a component winner. Confirm
only the final mixed-regime winner and strongest deployable baseline with five
paired seeds and 100 measured requests. Prefill-heavy and decode-heavy regimes
are run only after the mixed winner is known, and only as winner-versus-baseline
confirmation unless a promoted component policy wins the mixed screen.

The existing \(\Phi=1.94656\) component and joint data are retained as
secondary evidence. New \(\Phi=1.94656\), boundary-budget, and extra-policy
runs are optional and require a visible frontier transition or an unresolved
statistical tie. Report per-task \(\Delta Q\), p95/p99 TPOT, p95 TTFT, SLO
attainment, goodput, migration bytes, eviction-reload bytes, and arena and
NVML peaks for accepted cells.

Latency differences use paired bootstrap over seed clusters and shared
request IDs. Binary task scores use Wilson intervals and exact McNemar tests;
WikiText uses paired window bootstrap. Task scores are not pooled.

### P3: RQ3 mechanism attribution

Screen the full system, the seven existing removals, and the real
Fixed-partition executor with seed 42 on the mixed trace. Confirm the full
system, Fixed-partition, and at most two removals with the largest distinct
effects using three paired seeds. Existing seed-42 results are reused when the
cross-version bridge passes.

Every accepted exchange records request use, donor use, bytes, predicted
benefit, realized exposed-wait benefit, and eviction-reload cost. Zero-count
exchange types remain in the output.

### P4: RQ4 measurement validity and overhead

Collect individual prediction/outcome pairs for exposed wait, quality risk,
and predictor hits. Report calibration error, unsafe acceptances,
conservative rejections, recall by envelope depth, and a reliability diagram
for \(\omega_j\). Report fast- and slow-controller latency, fragmentation,
published and reserved bytes, reservation failures by cause, and every budget
violation.

Run \(K\in\{8,16,32,64\}\) by \(W\in\{1,2,4,8\}\) controller scaling as
an off-GPU replay with five CPU repetitions. It does not justify 80 new GPU
runs. Reuse request samples for SLO, goodput, and quantile calculations.

### P5: RQ5 operating boundaries

Screen one axis at a time around \(\Phi=0.85\), the default quality limit,
batch 8, and measured unthrottled bandwidth:

- \(\Phi\): 0.70, 0.85, 1.00, 1.30, and 1.94656;
- copy rate: 4, 8, 12, 16, and unthrottled GB/s;
- batch: 1, 2, 4, 8, 16, and 32, reusing the retained grid;
- predictor-noise probability: 0, 0.05, 0.10, and 0.20;
- quality limit: strict, default, and loose;
- byte inflation: 1.0, 1.25, and 1.5 with real padded allocations and copies.

Confirm only a policy transition, the first infeasible point on each side, or
a point where DynaByte and the best static policy are statistically
unresolved. Use three seeds unless that point appears in the headline claim,
in which case use five. If adjacent screen points select the same policy and
have monotone outcomes, do not run an interior confirmation.

## 7. Code, test, and provenance gates

The extension requires policy modes for Seq-FR, Seq-RF, Static-Joint, and
Fixed-partition; byte-accurate private caps; a \(\Phi\) override; copy-rate
throttling; deterministic predictor-noise injection; representation padding;
workload-switch logging; SLO/goodput derivation; prediction/outcome logging;
and RQ-specific statistical renderers.

Before GPU execution, tests must prove that:

- every policy obeys the common budget and transition interface;
- Fixed-partition cannot borrow across its three caps;
- throttling reaches its target within 5% after warmup;
- injected noise is reproducible and does not alter ground-truth routing;
- padded bytes appear in arena and NVML accounting;
- reset leaves no active transition, held donor, or reserved byte;
- SLO calculations match hand-checked samples;
- each artifact records source, harness, config, checkpoint, calibration,
  request-stream, and policy-state hashes.

The optimized manifest records `required`, `conditional`, `reused`, and
`not_required_by_optimized_plan`. Strict audit fails on a missing required
artifact, a promoted conditional artifact that is absent, a hash mismatch,
contamination, or an invalid measurement. It does not fail because an
unselected legacy-superset cell was skipped.

## 8. Cross-version bridge

Before an old result enters a new-source comparison, rerun bridge points with
the same checkpoint, configuration, requests, and seed. The all-high reference
must reproduce item/window outcomes. DynaByte and the strongest component
policy use the mixed trace; batch 1, 8, and 32 cover scaling; each existing
policy receives a runtime smoke.

An old family is reusable only when inactive new features preserve request
outputs and its paired latency difference is statistically unresolved.
Otherwise rerun that family under the new hash. Failed bridge data remains
historical evidence and cannot enter a headline comparison.

## 9. Completion criteria

The Qwen/A6000 lane is complete when all required and promoted conditional
artifacts pass strict audit; headline comparisons are paired by seed and
request ID; quality-infeasible points are labeled; published plus reserved
bytes never exceed \(B\); RQ2 contains the grid and adaptation trace; RQ3 uses
real private caps; RQ4 retains raw prediction/outcome pairs; RQ5 reports
negative and unresolved regions; and every manuscript number resolves to raw
artifact hashes. HOBBIT-M and DyMoE-M remain absent unless their semantics and
checkpoint behavior pass dedicated validation.
