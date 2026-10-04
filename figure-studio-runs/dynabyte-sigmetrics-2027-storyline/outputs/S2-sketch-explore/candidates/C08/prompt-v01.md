# C08 — Constraint-first inside-out map

Create one 16:9 raster research-paper conceptual schematic. Reviewer takeaway: a feasible operating point is bounded simultaneously by memory and quality while residency, precision, and prefetch compete inside those boundaries. No measured feasible region is depicted.

## Narrative and layout grammar

Use an inside-out composition. Center: a compact `Coupled operating point` selector. Around it, arrange `Residency`, `Precision upgrade`, and `Prefetch staging` as three medium modules connected inward by their distinct consequence labels. Surround the core with one non-numeric boundary frame labeled `Expert-memory budget`; attach `memory bound` to the frame and `quality constraint` as a second thin boundary/gate. Put a single `bytes` allocation cue along the outer frame, using one three-way fork to the inner modules. Ensure clear radial corridors and no triangle axes.

## Non-visual graph contract

Internal IDs N_BUDGET, N_RESIDENCY, N_PRECISION, N_PREFETCH, N_POINT, N_MEMORY_BOUND, N_QUALITY_BOUND and all edge IDs must remain invisible. Directed relations are the shared allocation fork from the budget frame to the three modules and the three consequences from modules to the central selector. Memory/quality relations are gates/boundaries. No other relation.

## Visible text and motifs

Node labels: `Expert-memory budget`, `Residency`, `Precision upgrade`, `Prefetch staging`, `Coupled operating point`. Edge labels: `bytes`, `fewer avoidable misses`, `potential quality benefit`, `timely arrival / overlap`. Boundary tags: `memory bound`, `quality constraint`.

Use a bounded pool outline, resident tiles, narrow-to-wide representation cue, staged incoming expert/deadline cue, and a small central selector. No bullets.

## Hard constraints

The surrounding frames are conceptual constraints, not a plotted feasible region. No axes, contours, heatmaps, gradients, numeric boundaries, data points, percentages, speedups, curves, error bars, optimum star, or measured operating point. Do not suggest equal tradeoffs from geometric symmetry. One canonical core only; no repeated pipelines; one allocation fork; only contracted consequence lines; no internal IDs or decorative arrows.

Surface: restrained formal publication line art, clean concentric hierarchy, high whitespace, short labels, no poster/dashboard appearance.
