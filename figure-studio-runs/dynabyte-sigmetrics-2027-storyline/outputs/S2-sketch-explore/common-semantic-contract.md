# Common source-grounded contract for C01--C08

This contract is instantiated verbatim in every image prompt. It governs only a conceptual motivation teaser.

## Semantic graph

Internal nodes: `N_BUDGET`, `N_RESIDENCY`, `N_PRECISION`, `N_PREFETCH`, `N_POINT`, `N_MEMORY_BOUND`, `N_QUALITY_BOUND`. Internal group: `G_THREE_USES`. Internal IDs must never be rendered.

Allowed directed semantic edges:

- `E_ALLOC_R`: `N_BUDGET` → `N_RESIDENCY`, byte allocation.
- `E_ALLOC_P`: `N_BUDGET` → `N_PRECISION`, byte allocation.
- `E_ALLOC_F`: `N_BUDGET` → `N_PREFETCH`, byte allocation.
- `E_R_OUT`: `N_RESIDENCY` → `N_POINT`, fewer avoidable misses.
- `E_P_OUT`: `N_PRECISION` → `N_POINT`, potential quality benefit.
- `E_F_OUT`: `N_PREFETCH` → `N_POINT`, timely arrival / overlap.
- `E_MEM_GATE`: `N_MEMORY_BOUND` → `N_POINT`, memory-feasibility gate.
- `E_QUAL_GATE`: `N_QUALITY_BOUND` → `N_POINT`, quality-feasibility gate.

Visual bundling: `E_ALLOC_R/P/F` must be one visible fork/bus from the shared budget. `E_MEM_GATE` and `E_QUAL_GATE` are compact boundary tags/gates, not peer process boxes. No other edges are allowed.

## Visible text contract

Node labels: `Expert-memory budget`, `Residency`, `Precision upgrade`, `Prefetch staging`, `Coupled operating point`.

Edge/port labels: `bytes`, `fewer avoidable misses`, `potential quality benefit`, `timely arrival / overlap`.

Boundary tags: `memory bound`, `quality constraint`. Candidate C05 may additionally use `use deadline`; C07 may additionally use `fast` and `slow` as timing tags.

Internal micro-motifs:

- Budget: one bounded capacity bar or pool outline.
- Residency: a few compact expert tiles already inside the device boundary.
- Precision upgrade: one expert tile changing from narrow/low representation to wider/high representation; no accuracy number.
- Prefetch staging: one incoming expert tile moving toward a small deadline marker; no latency number.
- Coupled point: a small merge/selector/balance motif, never a chart or measured optimum marker.

## Hard constraints

- Use only these modules, relations, directions, and transferred meanings.
- This is a conceptual schematic, not experimental evidence.
- Do not draw axes, plots, curves, bars representing measured values, heatmaps, response surfaces, numeric labels, percentages, speedups, error bars, legends implying data, or synthetic operating-point measurements.
- Do not claim optimality or that joint control already outperforms any baseline.
- One shared budget only. Do not give each use its own budget.
- Draw one canonical conceptual process. Repeated experts are compact markers, never cloned full workflows.
- Between modules use one bundled connector unless the contract names a distinct line family.
- Put `bytes` and consequences on connectors/ports/tags, never in peer boxes.
- Primary modules are the only large boxes/containers. Draw compact internal micro-motifs, not bullet lists.
- Draw only contracted edges; no decorative arrows, reverse arrows, cross-links among the three uses, or hotness-to-quality arrow.
- Internal IDs and schema terms are blacklisted visible text.
- Keep context under 20% of the canvas and preserve large whitespace.

Surface: clean formal publication schematic, restrained vector-like line art, light scientific colors, crisp hierarchy, readable at double-column scale; no glossy poster or dashboard treatment.
