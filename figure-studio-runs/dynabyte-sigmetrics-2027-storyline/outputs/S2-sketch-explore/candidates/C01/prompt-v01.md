# C01 — Linear shared-budget backbone

Create one 16:9 raster research-paper conceptual schematic. Reviewer takeaway: one bounded expert-memory budget must be divided among residency, precision upgrades, and prefetch staging, so the three decisions are coupled. This is a conceptual motivation teaser, not an experiment figure.

## Narrative and layout grammar

Use a strict left-to-right scan. Place `Expert-memory budget` as the large left anchor (22% width). From one labeled `bytes` fork/bus, route to three vertically stacked middle modules: `Residency`, `Precision upgrade`, `Prefetch staging` (45% width total). Merge their three distinct consequences into `Coupled operating point` at right (23% width). Put `memory bound` and `quality constraint` as two small gate tags beside the final merge. Reserve clear horizontal routing corridors; no crossed lines; no inset or legend.

Primary path: budget → three uses → coupled point. Secondary/context: feasibility tags. Do not repeat the process.

## Non-visual graph contract

Internal IDs are layout controls and must not be drawn: N_BUDGET, N_RESIDENCY, N_PRECISION, N_PREFETCH, N_POINT, N_MEMORY_BOUND, N_QUALITY_BOUND; E_ALLOC_R, E_ALLOC_P, E_ALLOC_F, E_R_OUT, E_P_OUT, E_F_OUT, E_MEM_GATE, E_QUAL_GATE. Draw only listed visible labels.

Allowed edges: one visible allocation bus from Expert-memory budget branching to the three use modules; Residency → Coupled operating point labeled `fewer avoidable misses`; Precision upgrade → Coupled operating point labeled `potential quality benefit`; Prefetch staging → Coupled operating point labeled `timely arrival / overlap`; memory and quality tags gate the coupled point. No other edges, reverse arrows, or cross-links.

## Visible text and motifs

Node labels only: `Expert-memory budget`, `Residency`, `Precision upgrade`, `Prefetch staging`, `Coupled operating point`. Edge/port labels only: `bytes`, `fewer avoidable misses`, `potential quality benefit`, `timely arrival / overlap`. Tags only: `memory bound`, `quality constraint`.

Inside modules, use tiny visual mechanisms rather than bullet lists: capacity bar for budget; resident expert tiles; one narrow-to-wide representation change for precision; one incoming expert tile approaching a deadline marker for prefetch; small merge/balance motif for coupled point.

## Hard constraints

Conceptual only. No axes, plots, curves, measured bars, heatmaps, response surfaces, numeric labels, percentages, speedups, error bars, or fake data. Do not claim optimality or measured superiority. One shared budget only. Variables and consequences live on lines/tags, never peer boxes. Exactly one canonical process. Use one bundled allocation bus. Only primary modules may be large boxes. Draw no decorative arrows and no hotness-to-quality arrow. Do not show internal IDs.

Surface: clean formal publication schematic, restrained vector-like line art, light scientific palette, crisp hierarchy, generous whitespace, double-column readability; no poster or dashboard style.
