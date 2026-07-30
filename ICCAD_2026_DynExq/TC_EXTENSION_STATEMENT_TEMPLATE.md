# TC conference-to-journal extension statement

> Use this document only if an earlier DynaExQ paper was accepted or
> published. Replace all bracketed text, attach the prior paper, and verify
> every percentage before submission.

## Prior publication

- Title: [conference-paper title]
- Authors: [authors]
- Venue and year: [venue, year]
- DOI / archival URL: [identifier]
- Relationship to this manuscript: [conference paper / workshop paper /
  other]

The submitted journal manuscript, “DynaExQ: Budget-Safe Dynamic Expert
Precision for Single-GPU MoE Inference,” extends the work above. The prior
paper is cited as Reference [X] in the journal manuscript and is included with
the submission.

## Quantified extension

- Estimated new impactful technical/scientific material: [XX]%.
- Estimated verbatim textual similarity: [XX]%.
- Tool and settings used for the similarity check: [tool/version/settings].
- Person and date of the manual section-level audit: [name/date].

The IEEE Computer Society threshold is at least 40% new impactful material
and less than 30% verbatim similarity. Do not infer either value from changed
page count alone.

## Section-by-section difference matrix

| Journal section | Closest prior section | Status | Specific new material | Evidence |
|---|---|---|---|---|
| Introduction | [section] | Revised | TC scope, explicit invariant boundary, audited claims | [pages/commit] |
| Background | [section] | Extended | One-token activation protocol; trace-driven cold-cache benchmark | [artifacts] |
| Architecture | [section] | Extended | Versioned immutable handles, leases, per-stream last-use fences | [code/tests] |
| Memory management | [section] | New/extended | Partitioned resident pools, global staging, exact-byte admission | [code/tests] |
| Implementation | [section] | New/extended | Fused INT4 and Triton INT2 dispatch; fail-closed adapters | [code/tests] |
| Evaluation protocol | [section] | New | Full pinned splits, raw samples, manifest and independent audit | [artifacts] |
| Statistical analysis | [section] | New | Paired exact McNemar with Holm correction | [artifacts] |
| Model coverage | [section] | Extended | Qwen3-Next-80B and/or Phi-3.5-MoE additions | [artifacts] |
| Ablation/overhead | [section] | New/extended | [describe only audited additions] | [artifacts] |
| Limitations | [section] | New/extended | Memory-invariant scope and validity threats | [pages] |

“New” means the mechanism, analysis, experiment, or scientific conclusion was
not present in the prior publication. Renaming, reformatting, additional
background, and repeated experiments do not by themselves count as impactful
new material.

## Main technical extensions

### 1. [Extension name]

Prior paper: [precise description and page/section].

Journal manuscript: [precise new mechanism or analysis].

Why it is impactful: [new capability, invariant, result, or scientific
conclusion].

Evidence: [implementation files, tests, artifact claim IDs, manuscript
pages].

### 2. [Extension name]

Prior paper: [...]

Journal manuscript: [...]

Why it is impactful: [...]

Evidence: [...]

### 3. [Extension name]

Prior paper: [...]

Journal manuscript: [...]

Why it is impactful: [...]

Evidence: [...]

## Reused material

List all reused figures, tables, algorithms, datasets, and substantial text.
For each item, state whether permission is required and how the prior source
is cited.

| Reused item | Prior location | Journal location | Verbatim/adapted | Citation/permission |
|---|---|---|---|---|
| [item] | [page] | [page] | [status] | [status] |

## Author declaration

We confirm that:

1. the prior publication is disclosed and cited;
2. this statement identifies the substantive differences accurately;
3. the reported new-material and similarity percentages were checked rather
   than estimated from formatting changes; and
4. no overlapping manuscript is simultaneously under review.

[All authors and date]
