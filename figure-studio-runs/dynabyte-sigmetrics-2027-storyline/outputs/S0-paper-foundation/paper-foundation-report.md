# S0-PAPER-FOUNDATION report: DynaByte motivation and figure reuse

## Problem and merged-paper claim boundary

The merged paper studies a single resource-allocation problem: under a fixed expert-memory budget and a quality constraint, jointly choose expert precision, current residency, and prefetch timing to reduce tail token latency. Precision affects both resident capacity and transfer bytes; residency affects avoidable misses; prefetch affects whether unavoidable transfers are exposed. A publishable merged-paper claim requires showing that independently or sequentially tuned controls lose to a shared-budget joint choice in a nontrivial operating region.

The current source material supports the two component stories but does not yet supply the decisive joint evidence. The P0 missing evidence is a precision-quota × prefetch-horizon response surface under a unified model, runtime, hardware, memory accounting, and quality protocol.

## Reusable motivation evidence

1. DynaExQ Fig. 2 `expert-active` can support only: task identity changes the frequently dispatched experts. It cannot by itself show that those experts have greater precision sensitivity, that promotion improves task quality, or that a hot set persists long enough to repay migration.
2. DynaExQ Fig. 3 `ppl` can support only: under a frozen ranking and protocol, precision quota has a continuous quality tradeoff. It cannot show that online hotness ranking is optimal or that precision should be coupled to prefetch.
3. DynaExQ Fig. 1 `waiting-vs-prompt` is evidence for blocking transfer pressure under its stated direct/replay protocols. It is not evidence for an overlapping joint runtime.
4. DynaMoE Fig. 2 `M1` provides the strongest reusable research question and visual grammar: horizon can exhibit an overlap--pollution tradeoff. The old plot should not be used as merged-paper evidence because model, budget, and runtime differ and the current repository lacks its raw claim manifest.
5. The activation-density table is a workload description, not transfer-pressure evidence. Only two of six expected model-stage claims are currently registered.

The canonical source-item disposition and asset hashes are recorded in `../../SIGMETRICS_2027/FIGURE_PROVENANCE.md`.

## Motivation figures still required

- P0: high-precision bytes/quota × prefetch horizon heatmap, with P95 TPOT or exposed wait, quality-feasible boundary, memory-infeasible mask, and four operating-point families: precision-first, prefetch-first, best static joint, online joint.
- P1/O1: one-run timeline aligning actual miss bytes, available overlap capacity/deadlines, and late bytes/exposed wait across prefill and decode/control windows.
- P1/O3: two-timescale evidence pairing per-expert marginal quality benefit/ranking quality with hot-set persistence and measured promote+demote break-even.
- P2 optional teaser: “one byte, three uses” concept paired with a measured slice from P0; the measured panel must not be invented before P0 exists.

## Method and semantic foundation

Ordered system logic supported by the merged draft:

1. Materialize/measure low- and high-precision expert representations and reserve non-expert/KV/activation headroom.
2. Observe or predict upcoming expert demand and per-layer use deadlines.
3. Estimate action effects: retain/evict, promote/demote, and prefetch change bytes, expected exposed wait, and a quality proxy.
4. Admit actions under one byte budget that includes transient migration/staging copies.
5. Apply fast prefetch decisions and slower precision-identity changes only when expected benefit exceeds migration cost.
6. Execute transitions using reserve--copy--publish--reclaim with versioned handles and safe delayed reclamation.

Core terms that must remain distinct in any later figure:

| Term | Meaning | Forbidden compression |
|---|---|---|
| precision | representation width/format of an expert and its quality/byte cost | Do not equate with residency. |
| residency | which expert representation is currently present on device | Do not equate resident with high precision. |
| prefetch horizon `S` | how far ahead requests are issued | Do not present `S` as the only source of transfer pressure. |
| miss bytes | required expert bytes absent from current residency | Do not replace with activation density. |
| overlap window | compute time/bandwidth available before use deadline | Aggregate capacity is not proof that each expert meets its deadline. |
| hotness | routed traffic/usage proxy | Do not label it “quality importance” without O3 evidence. |
| oracle static joint | test-space search/reference point | Must not look like an online deployable policy. |

## Artifact and arrow lineage relevant to later framework work

| Producer/artifact | Consumers | Supported relation | Risk |
|---|---|---|---|
| Router observations / demand predictor | fast prefetch loop; slow hotness estimator | shared observation source | A common input does not imply the two decisions are independent. |
| Packed low/high expert versions | residency pool; migration/prefetch transfers | representation choice changes both bytes and quality | Do not draw both copies as simultaneously resident unless staging is explicit. |
| Unified budget tracker | admission of residency, precision transitions, prefetch staging | shared feasibility gate | Old controllers cannot each be shown owning the whole budget. |
| Versioned published handle | forward dispatch; reclaimer | publish after copy, reclaim after leases/events drain | “forward never waits” applies only when a usable representation already exists. |
| Raw experiment artifacts | figure renderer; evidence manifest | data-to-plot provenance | A hash of the PDF alone is not raw evidence. |

## Source exclusions and claim risks

- Old empirical plots from different models/devices/protocols cannot be merged into one joint curve or heatmap.
- Existing `fig:phase` is a model schematic, not empirical motivation; its lexicographic allocation rule remains a hypothesis to validate.
- DynaMoE empirical figures have rendered assets but no discovered strict raw-artifact manifest in this repository.
- The current strict manifest leaves `routing_hotset`, `perplexity_curve`, `figure_bundle` (except the separate waiting-only bundle), performance, ablation, runtime-overhead, and budget-sensitivity evidence incomplete.
- Conceptual diagrams require semantic/source registration but not experimental raw samples.

## Readiness

`S0_FOUNDATION_READY_WITH_RISK` for S1 figure strategy. The source is sufficient to design figure roles and provenance-safe layouts, but not to treat the P0/P1 figures as measured results. Any later image prompts must visibly distinguish measured evidence, conceptual explanation, infeasible regions, oracle points, and deployable decisions.
