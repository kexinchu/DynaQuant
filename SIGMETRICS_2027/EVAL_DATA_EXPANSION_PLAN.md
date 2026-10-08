# Evaluation data expansion and sentence-level revision plan

Date: 2026-10-07.  
Scope: `05_evaluation.tex` and the sentences outside it that still quote class-P TPOT. The opening of `05_evaluation.tex` was rewritten after the 2026-10-06 draft of this plan, so the old line numbers are not a map. Sentence targets below use the lines in the file as of this update.

`A6000_WIKITEXT_REPLAY_CONCLUSIONS.md` is stale. Do not cite its 125 ms / 114-exchange figures. Do not overwrite `results/paper/a6000_wikitext_replay/summary.json`.

None of the sentence replacements below have been applied to the `.tex` files. This update records the artifacts and the edits they now require.

## 0. Evidence rule

| Class | Meaning | May appear as a result? |
|---|---|---|
| M | A named file under `results/paper/`. Exposed wait and TTFT probes are not TPOT. | Yes, with the file and the window source in the same sentence. |
| D | Arithmetic on registered inputs. | Yes, labeled derived. |
| P | Unregistered manuscript draft, or a prior with no artifact. Red in the PDF. | No. |

A cell changes class only when a new file is named and the citing sentence is rewritten together. Do not convert exposed wait into TPOT.

Registered files after this round:

| File | What it is |
|---|---|
| `results/paper/a6000_wikitext_replay/summary.json` | 2026-10-03 deadline replay. 2,048-token layer window 38.58 ms is measured. 128- and 512-token windows in that file are token-linear scalings. |
| `e2_layer_window.json` | Measured resident-low TTFT at 128 / 512 / 2048 tokens, GPU 1, output length 1, 2 warmup, 10 repeats. |
| `e2_measured_window_replay.json` | Static-Joint, Seq-PQ, and Oracle-Grid replayed on the E2 windows. Online policies were not rerun. |
| `e7_e6a.json` | Copy-rate sweep on the registered 38.58 ms window, plus admit-flip counts against 20 GB/s. |
| `e6a_short_window.json` | Same flip test on the scaled 128- and 512-token windows. |
| `remaining_cpu_tests.json` | E4 exchange attribution, E6c predictor score, paired exposed-wait bootstrap. Also the blocked list for E1, E1b, E3, E5, E6. |
| `e8_host_arena.json`, `e8_pack_timing.json` | 21 lease tests, CPU arena stress, and the pack timing in section 4. |
| `e2_measured_window_online.json` | Counterfactual and independent policies on the E2 windows at 8 and 24.86 GB/s. |

## 1. Runs that landed

### 1.1 E2 rejects the linear layer window

`e2_layer_window.json`. Uniform-low resident prefill, not the offload-inflated 1851 ms TTFT. TTFT is nearly flat, so TTFT/48 is not a pure per-layer compute time. The minimum 128-token TTFT is 1594 ms, which already rejects a linear scale of the registered 38.58 ms window (that scale is about 2.41 ms per layer, or about 116 ms of TTFT).

| Tokens | TTFT mean (ms) | TTFT min (ms) | TTFT p95 (ms) | TTFT/48 (ms) |
|---|---:|---:|---:|---:|
| 128 | 1790.37 | 1594.14 | 2130.81 | 37.30 |
| 512 | 2300.60 | 1816.32 | 3679.20 | 47.93 |
| 2048 | 2317.92 | 1826.07 | 3421.74 | 48.29 |

The registered 38.58 ms window stays the 2,048-token source in `summary.json`. Do not replace it with 48.29 ms. They are different runs.

`e2_measured_window_replay.json` replays those three windows at 8 / 16 / 24.86 / 40 GB/s. At 16 GB/s and above, every length and every Φ ∈ {1.0, 1.3, 1.931} has Static-Joint, Seq-PQ, and Oracle-Grid p95 exposed wait 0 and 0 slots. The only nonzero cells are 8 GB/s, Φ = 1.0, and the oracle still uses 0 slots:

| Tokens | Static-Joint p95 (ms) | Oracle p95 (ms) | Slots |
|---|---:|---:|---:|
| 128 | 22.14 | 12.88 | 0 / 0 |
| 512 | 20.89 | 17.92 | 0 / 0 |
| 2048 | 81.20 | 45.58 | 0 / 0 |

The scaled 138.4 vs 106.0 ms, 8-slot gap does not survive a measured window. Delete it from result sentences. Do not paste 37 ms into the paper as the overlap window without splitting fixed overhead from token-scaled compute.

### 1.2 E7: the only separated cell on the registered window is 8 GB/s, Φ = 1.0

`e7_e6a.json`, layer time 38.58 ms, ε = 2.1, 10 trials. At 16, 24.86, and 40 GB/s, Φ ∈ {1.0, 1.3, 1.931} all have 0 exposed wait. Φ = 1.3 and Φ = 1.931 are already 0 at 8 GB/s.

At 8 GB/s, Φ = 1.0, lookahead slots stay 0. The gap is which experts are resident, not a slot count:

| Policy | p95 exposed wait (ms) | Exchanges | Slots |
|---|---:|---:|---:|
| Static-Joint and Seq-PQ | 239.50 | — | 0 |
| Counterfactual | 232.03 | 36, 0 expired | — |
| Independent | 219.53 | 116, 3 expired | — |
| Oracle-Grid | 178.08 | — | 0 |

The scaled short windows inside the same file still move oracle slots (128 tokens, Φ = 1.0: 2 / 4 / 8 / 12 slots as the rate goes 8 / 16 / 24.86 / 40). Those rows are the old linear window. They are not E2.

`e2_measured_window_online.json` fills the online half that the static-only replay left empty. Same E2 windows, 8 GB/s and 24.86 GB/s, Φ ∈ {1.0, 1.3, 1.931}. At 24.86 GB/s every online policy has 0 exposed wait, including where independent still admits exchanges (128 tokens, Φ = 1.0: 361 admits, 25 expired, p95 = 0). At 8 GB/s the only nonzero waits are Φ = 1.0:

| Tokens | Static / Seq p95 | Counterfactual | Independent | Oracle |
|---|---:|---|---|---:|
| 128 | 22.14, 0 exchanges for counterfactual | 22.14 | 22.10, 361 admits, 25 expired | 12.88 |
| 512 | 20.89, counterfactual admits 0 | 20.89 | 20.04, 214 admits, 5 expired | 17.92 |
| 2048 | 81.20 | 78.22, 12 admits, 0 expired | 67.04, 116 admits, 3 expired | 45.58 |

All of those rows still have 0 lookahead slots. On the measured short windows the counterfactual policy does not move exposed wait off the static point. Independent trims it by a fraction of a millisecond at 128 tokens and by about 14 ms at 2048 tokens. Neither reaches the oracle.

### 1.3 E6a: the 20 vs 24.86 GB/s bias flips no admit bit

`e7_e6a.json` has 18 comparisons and `e6a_short_window.json` has 12. Every `admit_flips` value is 0, including cells where exposed wait is nonzero and including the scaled short windows. Examples on the scaled windows: 128 tokens, Φ = 1.0, counterfactual 60 = 60 and independent 361 = 361. The +24% transfer-time bias does not change the exchange set on this trace.

### 1.4 E4 and the paired exposed-wait interval

`remaining_cpu_tests.json`. Same cell as the E7 split: 8 GB/s, Φ = 1.0, 2,048 tokens, registered 38.58 ms window. The walk calls the same settlement as `run_online`, and the exchange totals match E7 (36 and 116).

Every recorded exchange is `lookahead`, low tier, and has a donor.

| Policy | Exchanges | Hits | Misses | Mean gain (ms) | Miss-path demand |
|---|---:|---:|---:|---:|---:|
| Counterfactual | 36 | 36 | 0 | 0.681 | 74.15 GB |
| Independent | 116 | 113 | 3 | 0.070 | 73.80 GB |

Admit-set difference, not a duplicate-credit counter: 109 lookaheads are in the independent set only, 29 are in the counterfactual set only, 7 are in both. The controller does not log a duplicate-credit flag, and this file does not invent one. Realized wait is still the forward total. Thirty-six hits do not cover the miss path: counterfactual demand stays about 74 GB, and 36 low blocks are about 90 MB.

Paired bootstrap over the same 10 trials, 10,000 resamples, seed 0. This is exposed wait, not TPOT. Negative means the first policy waits less.

| Contrast | Mean Δ (ms) | Mean 95% interval | Δp95 (ms) | Δp95 95% interval |
|---|---:|---|---:|---|
| Counterfactual − Static-Joint | −8.31 | [−11.34, −5.09] | −7.47 | [−14.29, −5.42] |
| Independent − Static-Joint | −16.47 | [−20.12, −11.78] | −19.97 | [−21.11, −13.09] |
| Independent − counterfactual | −8.15 | [−10.75, −5.43] | −12.51 | [−12.51, −4.28] |

All three intervals exclude 0 on this cell. At the machine rate, 24.86 GB/s, the same Φ has 0 exposed wait for every feasible policy, so this interval is not the operating-point result. n = 10 is not the ≥1,000-request TPOT test.

### 1.5 E6c is a prefill-union score, not the manuscript recall

Same file. Width is the mean warmup active-set size, 105 experts per layer. The trace stores one expert set per layer for the whole block, not a token-level top-8. On the measured trials: precision 0.942, recall 0.946, 2,909 false-positive expert occurrences (7.292 GB of low blocks), 2,700 false-negative occurrences (6.768 GB). Do not replace the class-P triple 0.71 / 0.58 / 0.46 with these numbers. They answer a different question.

## 2. What was still open on 2026-10-07, and what started

Both A6000s were empty (about 48 GB free each). Three jobs started. The others stay blocked because the input is still missing.

| Gap | Status |
|---|---|
| E8 pack seconds and startup RSS | Done. See section 4. |
| E2 online half | Done. `e2_measured_window_online.json`. See section 1.2. |
| E1 calibration gate | Done. `e1_calibration_map.json` is a schema-v2 ranking, 48 layers, 256 WikiText-103 train prompts, seed 42. Forward time averaged 2.42 s per prompt. This is not TPOT. The first ShareGPT cell is `trace-perf`, policy `joint`, regime `mixed`, seed 42, 100 requests and 5 warm-ups, writing `e1_trace_perf_joint_mixed_seed42.json`. Earlier attempts were rejected by the NVML exclusivity check and wrote no file. A rerun started once both A6000s were idle (5 MiB used, no compute process). It is one cell, not the five-seed paired test. |
| E1b | Still blocked. The timed path has no arrival process or queue. |
| E3 | Still blocked. No HOBBIT or DyMoE implementation. |
| E5 | Still blocked. `run_shift.py` asks for `calibration_datasets/requests/wikitext2_128x2048.jsonl`, `gsm8k_200.jsonl`, and `humaneval_200.jsonl`. Those files are not in the tree. The only routing trace is one WikiText-103 stream. |
| E6 | Still blocked. The replay still admits from the offline perplexity curve. A full benchmark run would not produce the confusion matrix. |

## 3. Sentence edits these runs now require

Apply these in the commit that cites the artifact. Do not leave the scaled pair in a result sentence.

| Location | Current claim | Replacement |
|---|---|---|
| `05_evaluation.tex` around the 138.4 / 106.0 sentence and the following "E2 replaces…" sentence | Scaled oracle gap, pending E2 | Delete both. E2 measured windows have 0 slots. At ≥16 GB/s exposed wait is 0. |
| RQ2 sentence "The scaled 128-token row is the draft picture…" | 138.4 vs 106.0 | Delete the pair. Seq-PQ still matches Static-Joint on the measured windows. |
| RQ2 figure sentence that assigns panel (a) to a future E2 | Future measurement | Panel (a) cannot be drawn from TTFT/48 until fixed overhead is split out. |
| RQ5 "A simpler frozen assignment is not sufficient in the scaled 128-token…" | 32 ms and 8 slots | Delete. The measured-window oracle does not open slots. |
| `tables/rq_replay_shortwindow.tex` and the matching row in `tables/rq_placeholders.tex` | Scaled 138.4 / 106.0 | Caption must say the table is the retired linear scale. Do not cite it as a result. |
| `05_evaluation.tex` "Whether that bias changes an admission… `\tbd{E6a}`" | Prior that the decision delta is zero | Replace with admit-flip count 0 on `e7_e6a.json` and `e6a_short_window.json`. |
| RQ5 and `06_discussion.tex` "the gap narrows toward 40 GB/s… pending E7" | Class-P figure | On the registered window the gap is already 0 at 16 GB/s. The only nonzero online-vs-static cell is 8 GB/s, Φ = 1.0, and it is exposed wait with 0 slots. |
| RQ3 "Decision-level duplicate-credit counts are `\tbd{E4}`" | Missing count | Do not write 109 as duplicate credit. Write the admit-set split (109 / 29 / 7) and that no duplicate-credit flag is logged. |
| RQ4 predictor triple 0.71 / 0.58 / 0.46 | Class P | Stays class P. E6c's 0.94 / 0.95 is the prefill-union score and needs its own sentence if it is cited at all. |
| Abstract, introduction contribution 1, 458 / 518 ms | Class P | Unchanged. E1 did not run. |
| Contribution 3, replay 0.15 ms | Class P | Still class P. E8 measured 482.9 s of initialization and 121.5 GB max RSS, which is not the 0.15 ms replay cost. |

## 4. E8

`e8_host_arena.json` and `e8_pack_timing.json`. GPU 1 was empty.

- Lease invariants: `dynaexq/tests/test_lease_invariants.py`, 21 passed.
- CPU arena stress, one segment, capacity 15,394,406,400 bytes. That is 48 layers at the Φ = 1.0 static point (30 high, 15 low), not the deployed 29.74 GB pool. 400 allocate/publish steps, free every third live extent. Allocate failures: 0. Final external fragmentation \(1.13 \times 10^{-4}\). Peak \(8.10 \times 10^{-4}\). 267 extents still published. This does not reproduce the class-P \(F_{ext} = 0.47\).
- Checkpoint `/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507`, uniform-low config. Weight load 737.9 s. `initialize_dynaexq` 482.9 s, which includes packing both tiers and the device arena, not packing alone. Process max RSS 121.5 GB (`ru_maxrss` 127,404,292 KB), from 0.51 GB before the load. A second process was reading the same checkpoint on GPU 0 during this measurement, so 737.9 s is an upper bound on uncontended load time.

Do not replace the class-P 0.15 ms replay cost with 482.9 s. One is a claimed controller replay, the other is process initialization. The four class-P runs, seed 46, and \(F_{ext} = 0.47\) stay class P.

## 5. Order from here

1. Cite E2, E6a, and E7 in the sentences in section 3. Delete the scaled 32 ms claim in the same edit.
2. If a TPOT run is started, preregister the cell first. The short-window Φ = 1.0 target is withdrawn. The 8 GB/s, Φ = 1.0 replay cell is a bandwidth stress of exposed wait, not a substitute for TPOT at 24.86 GB/s.
3. E5, E3, E1b, and E6 stay blocked until the missing input exists. Do not split the WikiText trace and call it a workload switch. Do not draft a HOBBIT millisecond.
4. E4's paired interval may be cited only as exposed wait at 8 GB/s, n = 10, with the native-rate zero beside it.
