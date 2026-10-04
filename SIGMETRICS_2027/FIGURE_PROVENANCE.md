# DynaByte Figure Provenance Ledger

> Snapshot: 2026-10-02  
> Scope: figure assets referenced by the two source manuscripts, plus the empirical table reused by the merged draft. Author photos, template examples, and unused working exports are excluded.  
> Evidence rule: a PDF/PNG hash proves the identity of a rendered asset; it does **not** prove the underlying experimental claim. Empirical figures are publication-ready only when their raw artifacts and reproduction command are also registered in `results/paper/manifest.json` or a successor joint-experiment manifest.

## Status vocabulary

| Status | Meaning |
|---|---|
| `EVIDENCE_REGISTERED` | Rendered asset and its raw claim artifacts/reproduction route are registered. |
| `ASSET_REGISTERED / EVIDENCE_MISSING` | The existing rendered file is hash-registered here, but raw evidence is absent or incomplete. |
| `PARTIAL_EVIDENCE` | Only part of a multi-model/multi-panel claim is present in the strict manifest. |
| `CONCEPTUAL_ASSET` | Diagram identity is registered; no experimental claim registration is required. Any semantic reuse still needs a source-faithfulness review. |
| `REMEASURE` | Preserve only the question, protocol idea, or visual grammar; do not reuse the old plot as merged-paper evidence. |

## A. DynaExQ / dynamic-precision manuscript

Source PDF: `ICCAD_2026_DynExq/main_sc.pdf` (`sha256 ed7c14433adb8bdd08f94918a6abcf1127ff5ca16165cc762d5bd0d7a04c0728`).

| Source item | Canonical asset(s) | Current status | Merged-paper disposition | Missing registration/action |
|---|---|---|---|---|
| Fig. 1 `waiting-vs-prompt` | `ICCAD_2026_DynExq/figures/waiting_latency_vs_prompt_length.pdf` | `EVIDENCE_REGISTERED`, narrow protocol | Appendix/background only | Keep caption explicit: Qwen direct blocking measurements; Phi cold-cache trace replay; not an overlapping runtime result. |
| Fig. 2 `expert-active` | `wikitext_thinking_on_layer_15.pdf`; `gsm8k_thinking_off_layer_15.pdf`; `humaneval_thinking_on_layer_15.pdf` | `ASSET_REGISTERED / EVIDENCE_MISSING` | Conditional direct reuse as O3 supporting panel | Add three `routing_hotset:qwen30b:<workload>:layer15` artifacts with raw dispatch counts, pinned workload/checkpoint, frozen scheduler, command, commit, and hashes; regenerate/register the figure bundle. |
| Fig. 3 `ppl` | `wiki_ppl_qwen30b.pdf`; `wiki_ppl_qwen80b.pdf` | `ASSET_REGISTERED / EVIDENCE_MISSING` | Conditional direct reuse in Background/Scope | Add `perplexity_curve:qwen30b` and `perplexity_curve:qwen80b`, including all single-point artifacts, raw windows/NLL/token counts, frozen ranking, command, commit, and hashes; regenerate/register bundle. |
| Fig. 4 `architecture_overview` | `dynaexq_arch_submission.pdf` | `CONCEPTUAL_ASSET` | Modify, do not paste unchanged | Retain versioned handle and budgeted transition semantics; add absent state, prefetch admission, unified byte budget, and deadline path. |
| Fig. 5 `procedure` | `dynaexq_procedure.pdf` | `CONCEPTUAL_ASSET` | Reuse internal transaction motif | Preserve reserve--copy--publish--reclaim ordering; connect it to demand fetch/prefetch without implying that missing experts never block. |
| Fig. 6 `end2end` | six Avg/P99 PDFs for Qwen3-30B, Qwen3-Next-80B, Phi-3.5-MoE | `ASSET_REGISTERED / EVIDENCE_MISSING` | Do not reuse as merged-paper main result | `performance` manifest group is empty; rerun on the unified joint stack. |
| Fig. 7 `throughput` | three `*_p99_latency_throughput.pdf` files | `ASSET_REGISTERED / EVIDENCE_MISSING` | Do not reuse as merged-paper main result | Same unified-stack rerun requirement as Fig. 6. |
| Fig. 8 `budget_sensitivity` | `budget_sensitivity_qwen30b.pdf`; `budget_sensitivity_qwen80b.pdf` | `ASSET_REGISTERED / EVIDENCE_MISSING` | Protocol/axis inspiration only | `budget_sensitivity` manifest group is empty; rerun with residency/prefetch competition and a quality constraint. |
| Table I `expert_activation_ratio` | LaTeX table in `02_background.tex` | `PARTIAL_EVIDENCE` | Reuse structure; remeasure values | Strict manifest has only Phi-3.5 prefill/decode (2 of 6 expected model-stage claims). Add Qwen3-30B and Qwen3-Next-80B prefill/decode claims, then repeat on the joint stack if used in O1. |

## B. DynaMoE / dynamic-prefetch manuscript

Source PDF: `MOE_ICPP/main.pdf` (`sha256 0fd44c6467062e2e440a5b603ec750ed0140d4630a55512b2cbc13ef001ccf86`). No raw-experiment claim manifest for this manuscript was found in the current repository; all empirical plots below are therefore legacy assets, even when a rendered PDF exists.

| Source item | Canonical asset(s) | Current status | Merged-paper disposition | Missing registration/action |
|---|---|---|---|---|
| Fig. 1 `overview` | `Candidate_prefetch_manners.png` | `CONCEPTUAL_ASSET` | Visual-language reference only | Redraw around precision/residency/prefetch competition; remove unsupported “tracks the per-period optimum” implication unless measured. |
| Fig. 2 `M1` | `M1_numstep_latency.pdf` | `ASSET_REGISTERED / EVIDENCE_MISSING`, `REMEASURE` | Highest-value motivation prototype | Register raw per-request waiting/cache-miss samples, model/device/cap, horizon settings, commit, and plot command if preserving the legacy claim; for the merged paper rerun as the precision-quota × horizon scan. |
| Fig. 3 `M3` | `M3_batchsize_latency.pdf` | `ASSET_REGISTERED / EVIDENCE_MISSING`, `REMEASURE` | Fold into O1 only after rerun | Resolve the plotted batch range versus prose mismatch; collect miss bytes, overlap window, and exposed wait on the same run. |
| Fig. 4 `M5` | `M5_DeepSeek_Qwen1.5_Qwen2.pdf` | `ASSET_REGISTERED / EVIDENCE_MISSING`, `REMEASURE` | Optional routing-diversity supplement | Register raw embeddings/dispersion, activated-expert counts, latency linkage, protocol, and plot command; otherwise do not use for a latency claim. |
| Fig. 5 `architecture` | `architecture.png` | `CONCEPTUAL_ASSET` | Modify into the unified architecture | Merge predictor/horizon/prefetch path with precision/residency coordination rather than placing two old systems side by side. |
| Fig. 6 `codesign` | `codesign.pdf` | `CONCEPTUAL_ASSET` | Reuse forecaster/controller relation selectively | Revalidate every connector against the merged method; precision and unified budget paths are currently absent. |
| Fig. 7 `E2` | `E2.pdf`; `E2_H20.pdf`; `E2_910B.pdf` | `ASSET_REGISTERED / EVIDENCE_MISSING` | Sanity-check/reference only | Register raw latency samples and full model/device/budget/configuration provenance or rerun; not joint evidence. |
| Fig. 8 `predictor-accuracy` | `E1_pregate_model_with_markers.pdf` | `ASSET_REGISTERED / EVIDENCE_MISSING` | Evaluation planning only | Register prediction labels/scores/splits/model version and plotting command; rerun if the merged predictor differs. |
| Fig. 9 `E1_latency` | `E1_Group1_Latency.pdf`; `E1_Group2_Latency.pdf`; `E1_Group3_Latency.pdf` | `ASSET_REGISTERED / EVIDENCE_MISSING` | Evaluation planning only | Register raw cache-miss latency trials and configurations; rerun on unified stack. |
| Fig. 10 `E3_latency` | `E3_Group1_Latency.pdf`; `E3_Group2_Latency.pdf`; `E3_Group3_Latency.pdf` | `ASSET_REGISTERED / EVIDENCE_MISSING` | Evaluation planning only | Register raw trials and two-tier policy settings; rerun on unified stack. |
| Fig. 11 `cache-aware-routing` | `E4.pdf`; `E4_H20.pdf`; `E4_910B.pdf` | `ASSET_REGISTERED / EVIDENCE_MISSING` | Evaluation planning only | Register raw end-to-end trials, routing policy, device/model configuration, and plotting command; rerun on unified stack. |

## C. Rendered-asset hash registry

These hashes freeze the existing source assets as provenance inputs. They do not upgrade empirical evidence status.

```text
94e715e2e5c7ba35c913fac17e3d795141b27f9a915c827f2023566ee8b3d7fb  ICCAD_2026_DynExq/figures/waiting_latency_vs_prompt_length.pdf
1b4e5daf8d76d75093ee9b80e31ce23fd65a4cf090b416d96033f3f60c982f20  ICCAD_2026_DynExq/figures/wikitext_thinking_on_layer_15.pdf
f475597dcb653ba0a0f4ea3f309311d375a5d45e235c71b0c482c4749e548210  ICCAD_2026_DynExq/figures/gsm8k_thinking_off_layer_15.pdf
e31af780a54f872cda552e84d97d8367662d4f7b5d8ddee5ebc99c90588153c5  ICCAD_2026_DynExq/figures/humaneval_thinking_on_layer_15.pdf
d41e2a85cd7abc3b9fcbd3e8e45daf75d26239fa06cd7820e203ccb611a7be64  ICCAD_2026_DynExq/figures/wiki_ppl_qwen30b.pdf
5bb77551d498d4b5986da0b68628eee60bb51d918dd6fef7b4d0d2faa9002411  ICCAD_2026_DynExq/figures/wiki_ppl_qwen80b.pdf
35faec63f770ab27eceafbc02e12bd503a452beadb5bb4a1bb8c4d3c11f7e659  ICCAD_2026_DynExq/figures/dynaexq_arch_submission.pdf
42190c02dcdb9a076447d9151d1a63de6b5cc467e5bdf9040d778f8f41355861  ICCAD_2026_DynExq/figures/dynaexq_procedure.pdf
12dce4e61fffbf65330130ff569d0bb10771e5eafbc243eb50f0271dc807b646  ICCAD_2026_DynExq/figures/Qwen3-30B_avg_latency_end2end_vs_batch_size.pdf
30e30b692dafc4519f5e44df917c069fe9fa01488e4a7799507e6ff7a2c71cef  ICCAD_2026_DynExq/figures/Qwen3-80B_avg_latency_end2end_vs_batch_size.pdf
b2414493e2bb7bc0d8697b61c05ab731e3c5fa993ce69f6dfe989578a45578f5  ICCAD_2026_DynExq/figures/Phi-3.5-MoE_avg_latency_end2end_vs_batch_size.pdf
ef427b3da0541fdb6d24410cb926a8e18f56037810a4d12e4a47330177bcfd37  ICCAD_2026_DynExq/figures/Qwen3-30B_p99_latency_end2end_vs_batch_size.pdf
395526239d95c164384bbbccf30aeddd90a4449ca02f62e4b46f9c762c42a49c  ICCAD_2026_DynExq/figures/Qwen3-80B_p99_latency_end2end_vs_batch_size.pdf
4af6c7596aebe2f0f79efb23d69acf50e3c941ad4df97d57c8a242b7cb439149  ICCAD_2026_DynExq/figures/Phi-3.5-MoE_p99_latency_end2end_vs_batch_size.pdf
9ee979a4f3474e73774e2f1055f253d5abdcac7dcf8986c483f0c385f4c43a9d  ICCAD_2026_DynExq/figures/Qwen3-30B_p99_latency_throughput.pdf
03fc4f537f3f65a743690813e7d15cd1adbb6ae9f00002f2b817b774146d39cd  ICCAD_2026_DynExq/figures/Qwen3-80B_p99_latency_throughput.pdf
7e10c2dba1f89a54449c4fa25f9d88502ea031c7c88684277991466ee26b08e5  ICCAD_2026_DynExq/figures/Phi-3.5-MoE_p99_latency_throughput.pdf
65b638f9386b2d865cfbb7aa13bea17299c7b08aee1e749dacaefe3e6cd6bed5  ICCAD_2026_DynExq/figures/budget_sensitivity_qwen30b.pdf
245a62295c0d2b36888513c73f898a1f8fe209b524eb02c6bf43f550955647bf  ICCAD_2026_DynExq/figures/budget_sensitivity_qwen80b.pdf
1ccfd9c22867d680ed35585c546edeeb57eccdd4fe3649b68d95768732cd121d  MOE_ICPP/figures/Candidate_prefetch_manners.png
7d24c24fe3949b43527af1881209470bd85d090ba1cae01b9747299c0e46eb45  MOE_ICPP/figures/M1_numstep_latency.pdf
2fffe7f16d2a290a7c52fb70328d9e53befd7c4ef857bd89e2944c52415217b1  MOE_ICPP/figures/M3_batchsize_latency.pdf
53c623e19ae034a7234c6f481242744f6cc5d5878737c34e25ace35bad3903cf  MOE_ICPP/figures/M5_DeepSeek_Qwen1.5_Qwen2.pdf
7f2496e1e4b85814f2be1ffa78b17b59dc14925d5aadf56dbbebc2dbbcdb7322  MOE_ICPP/figures/architecture.png
bd5fe842a08c4cbdc67e80170365932efbbd0907eee3a5f9175c2bb8ac652b3a  MOE_ICPP/figures/codesign.pdf
4c91286fb34249d891241086e444f3e7204961af3f507d5f0b358abfb7be69ae  MOE_ICPP/figures/E2.pdf
c2ac88eed439c1a1a0c885e5471deba5720a3ebf24dc489362e4fe58fb04c9b4  MOE_ICPP/figures/E2_H20.pdf
e999ef29f646c8f1e21e37bda9c3ee4a72d57c1166170353fb8cc456af4f648b  MOE_ICPP/figures/E2_910B.pdf
39c453a10dda3f6aaf67c00c65b8864b4d683f6794e67446f0345b9df8137a54  MOE_ICPP/figures/E1_pregate_model_with_markers.pdf
8f1affcdaffd49488a6c4b96c4c95307ba8240209f55dbeacb111005d7018783  MOE_ICPP/figures/E1_Group1_Latency.pdf
4bb7163b0ecef0aa263c66aa4b40cb5e4f6a5dc09d18340b7bf7c3aa527c91e5  MOE_ICPP/figures/E1_Group2_Latency.pdf
6b52d5ab56b8cf4948d34ceab1811b41c4613d9a8619b1d1f0f39a21133d1d84  MOE_ICPP/figures/E1_Group3_Latency.pdf
48419dd08e4ece7ccf702f3b8a972ace48ce98fdd2eccebaae46441b04299aa4  MOE_ICPP/figures/E3_Group1_Latency.pdf
fbe8892c0cfeba67f85a548cc7791ffb92fe62c93e7d4d0c5a1d197a4d987496  MOE_ICPP/figures/E3_Group2_Latency.pdf
6ac0b39aefdd347cdb0047ce84f92b98931b4604194d2028b1b35876c732e291  MOE_ICPP/figures/E3_Group3_Latency.pdf
dc6238b708ba3156fad29772aef956ec08231594b9254e1fd24f2e635ac60d74  MOE_ICPP/figures/E4.pdf
1134fc0e66286e65f03a31d3f62aaf8bdb2c94f67ce54b9104e0d1ffa624bafc  MOE_ICPP/figures/E4_H20.pdf
cdb1a06fce39405a1d264826d8333de78ca3e44f9eeeeaa60e8e1eaba0b92825  MOE_ICPP/figures/E4_910B.pdf
```

## D. New Background characterization figures

| Manuscript figure | Derived artifact | Rendered asset | Evidence status | Interpretation boundary |
|---|---|---|---|---|
| `fig:bg-ppl` | Digitized in `scripts/plot_background_perplexity.py` from the vector paths of `ICCAD_2026_DynExq/figures/wiki_ppl_qwen30b.pdf` and `wiki_ppl_qwen80b.pdf` | `SIGMETRICS_2027/figures/wiki_ppl_qwen30b.pdf`, `SIGMETRICS_2027/figures/wiki_ppl_qwen80b.pdf` | `ASSET_REGISTERED / EVIDENCE_MISSING` | Points match the vertices drawn in the legacy PDFs. Raw windows, NLL, and the frozen ranking are still unregistered, so this is not a new measurement. |
| `fig:bg-transfer-pressure` | `results/paper/background/qwen30b_transfer_pressure.json` | `SIGMETRICS_2027/figures/background_transfer_pressure_legend.pdf`, `SIGMETRICS_2027/figures/background_transfer_pressure_prefill.pdf`, `SIGMETRICS_2027/figures/background_transfer_pressure_decode.pdf` | `MEASURED_INPUTS / DERIVED_UPPER_BOUND` | Demand, TTFT/TPOT, bytes, and blocking-copy timing are measured. Overlap capacity is an aggregate upper bound, not a deadline-aware overlapping-runtime result. |
| `fig:bg-joint` | `results/paper/background/qwen30b_joint_allocation_replay.json` | `SIGMETRICS_2027/figures/background_joint_allocation.pdf` | `MEASURED_INPUTS / TRACE_REPLAY` | Routing sets, byte footprints, transfer timing, and perplexity inputs are measured. Residency and oracle-lookahead outcomes are replayed and must not be reported as end-to-end DynaByte performance. |

```text
8c041e44e5beeec9302dfd3037a18cdb37d30094c2fbd572fb1d07fd9ed6d335  SIGMETRICS_2027/figures/wiki_ppl_qwen30b.pdf
1969d1a7bc0f7de221b088bec47bfda14a6bf8010ba68cff13f430d1e01a7b34  SIGMETRICS_2027/figures/wiki_ppl_qwen80b.pdf
f43df50c8d47617cb4ab3b824bfc33ffa87c2fdabad0097df78eafd01295254d  results/paper/background/qwen30b_transfer_pressure.json
6d5e1cde480213272202f81b931c74db5f77b75390230b8cea5e9b1c6ad47fc4  results/paper/background/qwen30b_joint_allocation_replay.json
ed5b70c22c7459198b1916e4c8a2a4cf0ec2462882ec68a9cc48134188156074  SIGMETRICS_2027/figures/background_transfer_pressure_legend.pdf
7e044acad44b7440ce0a1b557f3775dab75c1056c53fefb43916c28d6f99df49  SIGMETRICS_2027/figures/background_transfer_pressure_prefill.pdf
8541076d58ee98db7b10467d82f4beb7a3bf20a311ee1e8215220ec60d91862c  SIGMETRICS_2027/figures/background_transfer_pressure_decode.pdf
aaf2cd955e79d2955cead77a9e41fb48e7d1ad5606102f60016aa6095c902cd7  SIGMETRICS_2027/figures/background_joint_allocation.pdf
```

## E. Provenance completion order

1. Register Fig. 2 `expert-active` and Fig. 3 `ppl` raw artifacts first; these are the only conditional direct-reuse motivation candidates.
2. Complete the four missing Qwen activation-density claims only if the table remains in Background/O1.
3. Do not spend effort reconstructing DynaMoE legacy result provenance before the P0 joint scan. For merged-paper claims, rerunning on the unified stack is more defensible than retrofitting old plots.
4. Give every new O1--O3 figure its own raw-artifact entries and render bundle before inserting it into the manuscript.
5. Keep `results/paper/manifest.json` strict. Do not insert asset-only rows into its recognized experimental groups; this ledger is the asset registry, while the manifest remains the evidence registry.
