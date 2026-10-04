# S2 prompt-package audit summary

## Cycle 1 findings

| Candidate | Issue | Repair applied |
|---|---|---|
| C03 | Comparison framing could duplicate a full pipeline or reverse allocation arrows. | Replaced two-pipeline before/after with one continuous bridge; conflict uses non-directional grouping at the budget boundary; directed consequences retain source-supported direction. |
| C04 | Radial geometry could be misread as a quantitative ternary/simplex. | Explicitly forbade coordinates, scales, gradients, contours, shares, and optimum markers. |
| C05 | Deadline path could be rendered as a latency/time-series plot. | Prohibited ticks, numeric time, curves, and measured arrival; defined the line as conceptual ordering only. |
| C06 | Capacity container could become a measured stacked bar. | Prohibited proportional widths, percentages, legends, and numeric capacity; marked regions illustrative. |
| C07 | Cycle layout could add unsupported feedback/return arrows. | Limited to exactly one prefetch arc and one precision arc with source-supported decision effects; prohibited generic return arrows and repeated cycles. |
| C08 | Constraint frame could resemble a measured feasible region. | Prohibited contours, gradients, axes, data points, and optimum stars; defined boundaries as conceptual gates. |

## Cycle 2 verdict

All eight packages pass source-faithfulness, symbol-disambiguation, edge-support, direction, module-input, edge-cardinality, artifact-block, edge-label-first, internal-motif, process-instance, layout-divergence, empirical-content exclusion, and prompt-contradiction checks.

Common locks:

- exactly one shared expert-memory budget;
- exactly one canonical process and one instance per primary use;
- one allocation fork/bus and no cross-links among uses;
- no experimental plot primitives or numeric claims;
- `bytes` and outcome phrases are line/port/tag labels, not peer boxes;
- all O1/O2/O3 figures remain script-only.

Final status: C01--C08 `PROMPT_READY`. Residual generation risk is low for C01/C02 and medium-but-mitigated for C03--C08 because their metaphors could otherwise resemble comparison, ternary, timeline, stacked-bar, loop, or feasible-region plots. S3 must audit those visible failure modes if S2 is run.
