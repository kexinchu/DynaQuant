# C02 — Byte-lineage funnel/hourglass

Create one 16:9 raster research-paper conceptual schematic. Reviewer takeaway: the same bounded bytes fan into three competing uses and reconverge as one constrained operating choice. This is conceptual motivation only, not experimental evidence.

## Narrative and layout grammar

Use a top-to-bottom funnel/hourglass. Top center: `Expert-memory budget` as a bounded capacity vessel (20% height). A single vertical `bytes` bus forks into a wide middle band with `Residency`, `Precision upgrade`, and `Prefetch staging` arranged left/center/right (42% height). Their consequences converge through one lower merge into `Coupled operating point` (20% height). Put `memory bound` and `quality constraint` as small side gates on the lower merge. The main route is vertical; branch lines remain inside the middle corridor; no side panels.

## Non-visual graph contract

Never draw internal IDs: N_BUDGET, N_RESIDENCY, N_PRECISION, N_PREFETCH, N_POINT, N_MEMORY_BOUND, N_QUALITY_BOUND and E_ALLOC_R/P/F, E_R_OUT, E_P_OUT, E_F_OUT, E_MEM_GATE, E_QUAL_GATE.

Allowed visible edges: one allocation bus from budget that forks to all three uses; one consequence line from each use to the lower merge: `fewer avoidable misses`, `potential quality benefit`, `timely arrival / overlap`; two small feasibility gates into the coupled point. No other connector.

## Visible text and motifs

Node labels: `Expert-memory budget`, `Residency`, `Precision upgrade`, `Prefetch staging`, `Coupled operating point`. Edge labels: `bytes`, `fewer avoidable misses`, `potential quality benefit`, `timely arrival / overlap`. Tags: `memory bound`, `quality constraint`.

Show compact motifs: bounded bytes at the top; a few resident expert tiles; one representation-width change; one early-arrival tile and deadline tick; one lower merge/selector. Labels annotate motifs; no bullet lists.

## Hard constraints

No axes, plot frames, curves, heatmaps, gradients implying measured magnitude, numeric values, percentages, performance claims, error bars, or pseudo-data. Do not imply equal allocation merely because the middle modules are aligned. One shared budget, one conceptual process, one fork, one merge. Consequences stay on connectors, not standalone cards. No cross-links among the three uses, no reverse arrows, no decorative arrows, no internal IDs.

Surface: formal publication schematic, clean line art, restrained color accents, large whitespace and precise alignment; no infographic poster or experimental dashboard.
