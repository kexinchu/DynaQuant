# C05 — Deadline-centered scoped mechanism

Create one 16:9 raster research-paper conceptual schematic. Reviewer takeaway: expert bytes affect timely availability in three different ways—already resident, smaller/larger representation, or staged earlier—under one memory and quality boundary. No latency measurement is shown.

## Narrative and layout grammar

Use one horizontal upcoming-expert path across the center, ending at a small `use deadline` marker on the right. Above the path, place one shared `Expert-memory budget` bar. Along the path, use three compact callout modules in sequence: `Residency` near the start, `Precision upgrade` in the middle, `Prefetch staging` before the deadline. Place `Coupled operating point` as a small selector immediately before the deadline. Put `memory bound` above and `quality constraint` below the selector. This is one representative expert path with three factor callouts, not three parallel workflows.

## Non-visual graph contract

Internal IDs N_BUDGET, N_RESIDENCY, N_PRECISION, N_PREFETCH, N_POINT, N_MEMORY_BOUND, N_QUALITY_BOUND and edge IDs must never be visible. The shared budget allocation is one thin bus across the top with three short downward branches. Each callout connects to the selector with the allowed consequence label. The central timeline is a conceptual ordering line, not a metric axis; it has no ticks or numbers.

## Visible text and motifs

Allowed labels: `Expert-memory budget`, `Residency`, `Precision upgrade`, `Prefetch staging`, `Coupled operating point`, `bytes`, `fewer avoidable misses`, `potential quality benefit`, `timely arrival / overlap`, `memory bound`, `quality constraint`, `use deadline`.

Use expert tiles already inside a device boundary, one representation-width change, one incoming tile moving before a deadline, and one selector. No bullets.

## Hard constraints

The horizontal path must not become a plotted time series: no numeric time, tick marks, curves, measured bars, latency values, axes, heatmaps, percentages, speedups, or data markers. Do not claim the transfer meets the deadline in measured operation; show only the conceptual requirement. One canonical expert example only; no repeated full lanes. No unsupported arrows or hotness-to-quality link. No internal IDs.

Surface: formal publication schematic with a precise deadline marker and restrained scientific colors; sparse, readable, non-photorealistic.
