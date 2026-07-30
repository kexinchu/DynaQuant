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
- [ ] Replace `Anonymous Authors` with the real author list and affiliations.
  TC does not offer the optional double-anonymous review route described by
  the Computer Society guidance.

## Originality and prior versions

- [ ] Confirm that no version is simultaneously under review elsewhere.
- [ ] Confirm whether the `ICCAD_2026_DynExq` directory represents a submitted
  or published conference paper. If it does, cite that paper, provide the
  detailed difference statement required at submission, demonstrate at least
  40% new impacting technical/scientific material, and check that verbatim
  similarity is below 30%.
- [x] Prepare an internal section-by-section extension-statement template
  (`TC_EXTENSION_STATEMENT_TEMPLATE.md`).
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
- [x] Prepare an internal cover-letter template explaining TC scope fit
  (`TC_COVER_LETTER_DRAFT.md`).
- [ ] Replace its author/originality/artifact placeholders and insert only
  audited headline results.
- [ ] If applicable, attach the conference paper and journal-extension
  difference statement.
- [ ] Check PDF metadata, author metadata, acknowledgments, funding,
  conflicts, and artifact availability immediately before submission.
