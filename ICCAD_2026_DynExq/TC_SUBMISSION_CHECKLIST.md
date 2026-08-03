# IEEE Transactions on Computers submission checklist

Checked against the IEEE Computer Society author guidance and the TC general
submission call on 2026-07-29:

- https://www.computer.org/publications/author-resources
- https://www.computer.org/digital-library/journals/tc/call-for-papers-general-submissions

## Format and scope

- [x] IEEE journal template (`IEEEtran`, journal mode).
- [x] Regular-paper length: 12 formatted double-column pages, including
  references. Verified with `pdfinfo`.
- [x] Abstract length: 169 words, within the 100--200 word range.
- [x] Abstract contains no citations or displayed mathematics.
- [x] Index terms are present.
- [x] TC scope is explicit: runtime systems, software--hardware interaction,
  accelerator memory management, performance evaluation, and emerging
  machine-learning computing.
- [x] Real author list and affiliations are present (single-anonymous).
  TC does not use the optional double-anonymous review route for this
  submission (`review_model: single_anonymous` in metadata).

## Originality and prior versions

- [ ] Confirm that no version is simultaneously under review elsewhere
  (human confirmation at upload time).
- [x] Prior version declared: ICCAD submission was **rejected and unpublished**.
  No journal-extension / 40%-new-material statement is required. Keep
  `TC_EXTENSION_STATEMENT_TEMPLATE.md` unused unless a published conference
  version appears later. Cover letter: `TC_COVER_LETTER.md`.
- [ ] Add any required preprint submission notice if a public preprint exists.

## Technical evidence gate

- [x] The PDF displays an `Internal draft` warning while evidence is
  incomplete.
- [x] `scripts/audit_paper_results.py` enumerates 84 empirical/statistical
  claims, including three paired quality comparisons.
- [x] Figure generation is bound to registered input and output hashes.
- [x] Full local unit/integration suite passes.
- [ ] Produce `results/paper/manifest.json` and make the strict audit pass.
- [ ] Obtain compatible source checkpoints or verified dual-tier packed
  artifacts for every DynaExQ model.
- [ ] Replace all legacy, skipped-sample, mismatched, and missing quality
  results.
- [ ] Produce complete raw performance, ablation, overhead, sensitivity,
  activation-density, blocking-offload, routing-hotset, and perplexity
  artifacts from clean commits.
- [ ] Set `\artifactverifiedtrue` only after the strict audit succeeds.

## Submission package

- [ ] Run the IEEE LaTeX Analyzer on the final source bundle.
- [ ] Include high-resolution figure files individually.
- [x] Cover letter with TC scope fit and ICCAD rejection disclosure
  (`TC_COVER_LETTER.md`). Optional longer draft: `TC_COVER_LETTER_DRAFT.md`.
- [ ] After the evidence gate passes, insert only audited headline results
  into the cover letter and state artifact availability.
- [x] No conference PDF / difference statement attachment (rejected, unpublished).
- [ ] Check PDF metadata, author metadata, acknowledgments, funding,
  conflicts, and artifact availability immediately before submission.
- [x] Master readiness plan: `TC_READY_VERSION.md`.
