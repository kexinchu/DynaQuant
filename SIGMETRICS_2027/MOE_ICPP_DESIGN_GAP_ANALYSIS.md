# MOE_ICPP design retention analysis

> Integration status: the recommended Design changes have now been incorporated into `03_design.tex` and `figures/dynabyte_overview.tex`. The paper design includes stall/pollution feedback, an adaptive candidate envelope, an explicit predictor contract, protected/reclaimable donor indexes, resident-first execution, and a priced-reload rule. The current runtime code still needs corresponding implementation work before these mechanisms can be described as implemented and evaluated. The gap analysis below records the pre-integration state.

## 1. Verdict

The new Design does not discard all of the MOE_ICPP design, but it does discard most of its concrete adaptive-prefetch policy and part of its memory-management policy.

The new paper preserves the general goals of adaptive lookahead and safe residency management through a broader byte-lease abstraction. It also improves the formulation by making fidelity compete with residency and lookahead. However, the abstraction currently hides several mechanisms that made MOE_ICPP operational and testable:

- the EMA stall-versus-pollution controller;
- the bounded \(\pm1\) horizon update and analytical initialization;
- the CPU delta-adjusted forecaster and confidence-aware fallback;
- the two-tier LRU that protects demonstrated reuse;
- the explicit prefetch-cache feedback loop;
- the claim that misprediction causes at most one extra DMA rather than an eviction-reload cascade.

The result is a design that is more general on paper but less specific about how lookahead requests and residency donors are produced online.

## 2. Mechanism-by-mechanism comparison

| MOE_ICPP mechanism | New Design counterpart | Status | Main gap |
|---|---|---|---|
| Scalar adaptive horizon \(S_t\) | Maximum lookahead window plus per-expert lookahead leases | Replaced in principle | No online rule adapts the candidate horizon or controls prediction breadth. |
| EMA exposed-wait signal | Late bytes and exposed wait returned to the State Monitor | Partially retained | The new text does not define an EMA, a regime signal, or how feedback changes lookahead generation. |
| EMA pollution/eviction signal | Unused lookahead bytes and counterfactual donor loss | Partially retained | Eviction-reload pressure is not explicitly measured, so repeated pollution may not shrink the search window. |
| Bounded \(\pm1\) update with hysteresis | Receding-horizon exchange selection plus fidelity hysteresis | Mostly lost | Hysteresis applies to precision replacement, not to lookahead depth or prediction breadth. |
| Analytical \(S_0=\lceil N_eE_s/(C_sT_l)\rceil\) | Aggregate bound \(M_{\mathrm{miss}}(S)\le ST_lC\) | Retained as a bound | The bound prunes candidates but does not initialize or adapt the window. |
| CPU random-forest forecaster | “Predicted expert uses” supplied by the State Monitor | Lost | The source, features, confidence, training path, and fallback for the prediction are unspecified. |
| Pregate-delta correction | No explicit counterpart | Lost | The paper assumes prediction output without defining how router-local noise is corrected. |
| Confidence widens the predicted set and makes control conservative | Occurrence probability \(\omega_j\) | Possible but unspecified | There is no mapping from confidence to \(\omega_j\), candidate-set width, or admission. |
| Two-tier high/low LRU | Residency leases and donor selection by counterfactual loss | Replaced in principle | No concrete data structure protects reused experts or bounds donor-search overhead. |
| Prefetch-cache feedback | Revaluation after each exchange and execution feedback | Partially retained | The feedback variables and their effect on future candidate generation are not defined. |
| Cache-aware token routing | Published work proceeds while missing experts wait | Semantics retained | It should be named “resident-first execution order” to avoid implying a top-\(k\) routing change. |
| At most one extra DMA per prediction error | No equivalent invariant | Lost | A bad lookahead can still evict a future resident and trigger an eviction-reload chain unless donor eligibility forbids it. |
| Fixed memory cap | Live-plus-reserved byte ledger | Strengthened | The new transition-aware ledger is more precise than the old occupancy constraint. |

## 3. Adaptive prefetch assessment

### What the new design improves

The deadline replay is more expressive than a single scalar horizon. It can assign different urgency to experts within the same window, account for queued transfers, and expose the fact that changing precision changes transfer completion time. A per-expert lookahead lease is therefore a better final decision unit than \(S_t\).

### What was lost

The current text starts after the hardest online question has already been answered: it assumes a set of predicted uses, deadlines, and probabilities. MOE_ICPP supplied a concrete path from observations to that set. Its controller also bounded how quickly speculation expanded or contracted under workload changes. Without an equivalent mechanism, “adaptive lookahead” in the new Design means that values are recomputed, not that the prediction window itself adapts.

The implementation confirms this gap:

- `dynaexq/lease/exchange.py` receives `horizon` as an input and performs open-loop prefetch inside that fixed horizon.
- `dynaexq/runtime/prefetch.py` is explicitly a simple lookahead heuristic and does not update a horizon from stall or pollution.
- no current module implements the MOE_ICPP random-forest forecaster, pregate-delta correction, or confidence-aware widening.

Adaptive prefetch is therefore **conceptually subsumed but not operationally retained**.

## 4. Memory-management assessment

### What the new design preserves or strengthens

- The live-plus-reserved ledger is stronger than the old final-occupancy check.
- Reserve-copy-publish-reclaim makes precision transitions and prefetch destinations memory safe.
- Donor-aware exchanges express the opportunity cost of evicting a resident more directly than an independent LRU.
- The demand-miss path still allows published expert work to proceed while another expert finishes transferring.

### What was weakened

MOE_ICPP used two queues to separate demonstrated reuse from speculative or cold residency. The new coordinator says that it scans donors by counterfactual loss, but it does not define a practical index for those donors or a protection rule for experts predicted to recur. The current `dynaexq/runtime/memmgr.py` maintains one hot LRU rather than the documented `LRU_high` and `LRU_low` pair.

More importantly, the new byte ledger prevents oversubscription but does not by itself prevent cache pollution. Memory safety and replacement quality are different properties. A legal exchange can still evict an expert needed later, prefetch a false positive, and reload the evicted expert. The counterfactual replay should detect this only when the future use lies inside its finite window and the predictor is correct. MOE_ICPP's protected tier supplied an additional guard outside that condition.

Memory management is therefore **partially retained and made safer, but its anti-thrashing policy is missing**.

## 5. Recommended integration

The old design should not be copied back as three additional Design subsections. Its mechanisms fit inside the four current components.

### State Monitor

Add two fast-loop signals:

- an EMA of exposed deadline wait;
- an EMA of pollution pressure, measured as donor evictions that cause a reload inside a later planning window, plus unused lookahead bytes.

Keep measured in-overlap bandwidth and cache occupancy. These observations should drive candidate generation rather than create a separate memory budget.

### Lease Generator and Valuator

Restore an explicit prediction contract:

- pregate or recent-routing predictions provide the base expert set;
- an optional CPU correction model refines that set without using GPU cycles;
- confidence determines occurrence probabilities and the breadth of the candidate set;
- a cold-start or missed-deadline fallback uses the model's own routing signal.

Use the old analytical \(S_0\) as the initial candidate-window envelope. Adapt that envelope with the old bounded update, but do not let \(S_t\) directly reserve memory. It should only decide how far ahead the generator is allowed to propose leases. The exchange coordinator remains responsible for actual allocation.

### Exchange Coordinator

Use the stall-versus-pollution signal to update the candidate envelope by at most one routed layer per fast period. This restores stable adaptation while preserving the new per-expert decision model:

- \(S_t\) controls which lookahead requests are generated;
- deadline replay controls which generated requests are valuable;
- byte exchange controls which requests receive memory.

Donor eligibility should protect any residency lease with predicted reuse inside the current envelope unless an exchange explicitly charges the resulting reload. This is the lease-based equivalent of the old high-reuse tier.

### Safe Executor

Retain resident-first execution order and name it explicitly. Tokens keep their original top-\(k\) experts; only the order of ready expert groups changes.

To recover the old anti-cascade claim, add an enforceable rule: a speculative lookahead lease cannot evict a published expert that is predicted to execute before the speculative destination, unless the complete replay contains and charges that later reload. Without this rule, the paper should not claim that one prediction error costs at most one DMA.

## 6. What should not be restored unchanged

Three parts of MOE_ICPP conflict with the new paper if copied literally:

1. The old objective \(\tilde L_{\mathrm{wait}}+\lambda\tilde L_{\mathrm{miss}}\) should not replace the new quality-constrained objective. Stall and pollution are feedback for candidate-window adaptation, not the paper's global objective.
2. The two-tier LRU should not become an independent cache controller with a private quota. Its protected/reclaimable classification should accelerate donor construction inside the exchange coordinator.
3. “Cache-aware routing” should not suggest changing expert choice. The retained mechanism is resident-first execution order after top-\(k\) routing.

The old stability argument also needs weaker wording. A bounded update and hysteresis prevent large one-step changes and suppress noise near a stationary operating point; they do not guarantee convergence when the workload and regime signal change adversarially.

## 7. Priority and impact

### P0: required for a credible unified design

1. Define how predicted uses, deadlines, and probabilities are produced.
2. Restore adaptive candidate-window control using stall and pollution feedback.
3. Define donor protection against eviction-reload cascades.
4. State resident-first execution order explicitly.
5. Align the runtime implementation with these mechanisms or narrow the paper claim to trace-driven policy evaluation.

### P1: useful after P0

1. Add confidence-aware prediction breadth and cold-start fallback.
2. Maintain protected and reclaimable donor indexes for low control overhead.
3. Re-establish the one-extra-DMA property if it can be stated and tested as an invariant.

Without P0, the new Design is strong as an allocation formulation but weak as an adaptive prefetch system. With P0 integrated inside the current components, the paper can preserve MOE_ICPP's implementable control loop while keeping byte exchange as the higher-level novelty.
