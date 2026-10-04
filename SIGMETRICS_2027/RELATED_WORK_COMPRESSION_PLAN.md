# Related Work compression plan

## 1. Current assessment

The current Related Work contains 791 words, five subsections, eleven citation commands, and 28 citation occurrences covering 27 unique bibliography keys. In the compiled paper it begins in the lower half of page 20, fills page 21, and continues into page 22. Its effective length is roughly 1.5--1.7 pages.

The coverage is adequate, especially the direct treatment of HOBBIT and DyMoE. The excess length comes from structure and repetition rather than too many cited works:

- five subsection headings consume vertical space;
- each category repeats the pattern “prior work controls one decision; DynaByte also considers the other two”;
- the distinction between component composition and joint allocation is stated in the opening, prefetch discussion, and mixed-precision offloading discussion;
- serving-framework limitations repeat material already consolidated in Discussion, Section~6.5;
- several sentences enumerate implementation details without sharpening the novelty boundary.

## 2. Hard constraints

The rewrite will satisfy all of the following:

1. Preserve all 27 unique citation keys.
2. Preserve all 28 current citation occurrences, including the second use of `duanmu2025mxmoe` for mixed-precision kernels.
3. Do not move citations into footnotes or hide them in a table.
4. Keep HOBBIT and DyMoE as individually named closest systems.
5. Keep the distinction between the two ExpertFlow papers.
6. Preserve the statement that component coexistence is not DynaByte's novelty claim.
7. Target 500--560 words and approximately one ACM page of rendered space.

The citation multiset before and after editing will be compared automatically rather than checked by eye.

## 3. Target structure

Use four run-in paragraphs under the Related Work section instead of five numbered subsections. Run-in headings preserve the taxonomy without spending a separate line on each heading.

### Paragraph 1: Precision assignment

Target: 100--120 words.

- Group general PTQ methods in one sentence.
- Group MoE-specific sensitivity, frequency, and router-based allocation in one sentence.
- State their shared fixed-placement or fixed-transfer assumption once.
- Retain the second MxMoE citation to distinguish representation selection from mixed-precision kernel execution.
- End with one compact contrast: DynaByte prices the residency and lookahead displaced by fidelity bytes.

Citations retained:

- `frantar2022gptq`
- `lin2024awq`
- `xiao2023smoothquant`
- `dettmers2022gptint8`
- `kim2023mixturequantizedexpertsmoqe`
- `duanmu2025mxmoe` twice
- `zhang2025moqe`
- `chitty2025mopeq`
- `chowdhury2026efficient`

### Paragraph 2: Residency and placement

Target: 90--110 words.

- Describe locality-aware caching as one family rather than naming several policy variations in prose.
- State that these methods estimate residency under a fixed expert representation.
- Fold multi-device placement into the same paragraph as a neighboring but out-of-scope decision.
- Contrast both families with donor-aware residency leases in one sentence.

Citations retained:

- `xue2024moe`
- `li2023adaptive`
- `yi2023edgemoe`
- `zhong2024adapmoe`
- `kong2024swapmoe`
- `cao2024moe`
- `yao2024exploiting`
- `dai2024deepseekmoe`

### Paragraph 3: Prefetch and mixed-precision offloading

Target: 190--220 words. This is the longest paragraph because it contains the nearest work.

- Group cross-layer prefetch systems and their fixed/adaptive horizons.
- Preserve both ExpertFlow citations and disambiguate their decisions in one sentence.
- Introduce HOBBIT and DyMoE individually, stating what each combines.
- State the novelty boundary once: DynaByte does not claim that mixed precision, caching, or prefetching is new, either separately or together.
- Give the exact distinction in one compact sequence: byte leases, explicit donors, one deadline replay, hard quality and memory accounts, unchanged top-k semantics.
- End with the requirement for matched-policy evaluation rather than terminology-based differentiation.

Citations retained:

- `song2024promoe`
- `eliseev2023fast`
- `hwang2024pre`
- `shen2025expertflow`
- `he2024expertflow`
- `tang2024hobbit`
- `huang2026dymoe`

### Paragraph 4: Serving and memory management

Target: 60--80 words.

- Group vLLM, SGLang, and FlexGen in one sentence.
- State that DynaByte operates on a declared expert-memory budget after other reserves.
- Keep one sentence about runtime-ledger versus process-peak reporting.
- Remove the repeated prototype-integration limitation because it now belongs in Discussion, System Scope.

Citations retained:

- `kwon2023vllm`
- `zheng2024sglang`
- `sheng2023flexgen`

## 4. Sentence-level compression rules

- Open the section with at most two sentences explaining the decision-based taxonomy.
- Give each family one shared assumption, not one assumption per cited system.
- Describe DynaByte's complete distinction once in the closest-work paragraph; elsewhere use only the local contrast needed for that category.
- Remove phrases that restate a cited paper's title or repeat “these systems establish the value of.”
- Keep named descriptions for HOBBIT, DyMoE, and the two ExpertFlow papers; group the remaining work by mechanism.
- Avoid a closing summary paragraph. The closest-work comparison should already establish the boundary.

## 5. Verification procedure

1. Extract the citation-key multiset from the current file and save it before editing.
2. Rewrite to 500--560 words using four `\paragraph{}` headings and no `\subsection{}` headings.
3. Compare the before/after citation multisets. The edit fails if any key or duplicate occurrence changes.
4. Compile the paper twice and check for undefined citations or references.
5. Inspect the rendered pages. Related Work should occupy no more than approximately one page of vertical space even if it begins midway through a page.
6. Confirm that HOBBIT and DyMoE receive more explanatory space than general PTQ or serving frameworks.

## 6. Expected reviewer outcome

The compressed section should retain breadth while making the closest-work boundary easier to find. A reviewer should leave with three points: prior work separately optimizes precision, residency, or timing; HOBBIT and DyMoE already combine those mechanisms; DynaByte's claimed distinction is donor-explicit, counterfactual allocation under shared quality and byte constraints. This is the structure needed for a roughly 9/10 Related Work section, subject to the accuracy of the individual comparisons and the matched-policy evidence in Evaluation.
