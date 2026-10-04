# C07 — Fast/slow timescale loop

Create one 16:9 raster research-paper conceptual schematic. Reviewer takeaway: prefetch staging changes quickly while precision replacement changes more slowly, but both consume the same budget and interact with residency. This is a hypothesis-explaining schematic, not proof that the decomposition is beneficial.

## Narrative and layout grammar

Use one central `Expert-memory budget` pool with `Residency` embedded as the stable inner state. Draw a short inner arc from `Prefetch staging` to the pool, tagged `fast`, and one larger outer arc from `Precision upgrade` to the pool, tagged `slow`. Both arcs must terminate at distinct budget ports and must not form a generic feedback loop. Place `Coupled operating point` to the right, receiving one bundled consequence route. Add `memory bound` and `quality constraint` as gate tags. Keep the inner and outer corridors visually separate and uncrossed.

## Non-visual graph contract

Internal IDs are invisible. The arcs represent decision timescale/dependency, not measured recurrence and not data return. Arrow directions remain from the prefetch/precision decision toward budgeted residency/operating choice, consistent with allocation effects. Allowed consequence meanings remain `fewer avoidable misses`, `potential quality benefit`, and `timely arrival / overlap`. No other loop or shortcut.

## Visible text and motifs

Allowed labels: `Expert-memory budget`, `Residency`, `Precision upgrade`, `Prefetch staging`, `Coupled operating point`, `fast`, `slow`, `bytes`, `fewer avoidable misses`, `potential quality benefit`, `timely arrival / overlap`, `memory bound`, `quality constraint`.

Use a resident-tile pool, one incoming staged tile, one representation replacement motif, and one selector/merge. No bullet lists.

## Hard constraints

Do not draw repeated cycles, periodic tick marks, iteration counts, quantitative time scales, axes, curves, heatmaps, numeric values, performance claims, or optimization claims. `fast` and `slow` are qualitative tags only. One pool, one prefetch arc, one precision arc, one outcome. No hotness-to-quality arrow, no decorative return arrows, no internal IDs.

Surface: formal academic schematic with two clearly coded but restrained line styles, ample whitespace, and double-column readability.
