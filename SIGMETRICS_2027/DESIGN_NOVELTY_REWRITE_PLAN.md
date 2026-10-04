# Design novelty review and rewrite plan

## 1. Reviewer assessment of the current design

The current section has a coherent resource-management story, but its technical novelty is not yet strong enough for a top review score.

| Dimension | Current score | Main issue |
|---|---:|---|
| Problem formulation | 8/10 | The shared expert-memory budget is clear and important. |
| Mechanism distinctiveness | 5/10 | A monitor, value estimator, greedy coordinator, and safe executor are generic system components. |
| Separation from HOBBIT and DyMoE | 5/10 | Both already combine mixed precision, caching, and prefetching. A component list does not separate DynaByte. |
| Mathematical consistency | 6/10 | Quality is both a hard constraint and a term converted into latency through a shadow price. Lease duration is mentioned but absent from the score. |
| Treatment of coupling | 6/10 | Independent action values can double-count the same avoided transfer and do not identify which resident bytes an action displaces. |
| Executability | 8/10 | Reservation, publication, and reclamation provide a credible path to a hard memory invariant. |

Overall assessment: **about 6/10 for novelty as currently written**. The design reads as a sensible composition of known mechanisms rather than a new allocation primitive.

## 2. Revised technical thesis

DynaByte should not claim novelty from combining quantization, caching, and prefetching. Its technical thesis is:

> DynaByte represents fidelity, residency, and lookahead as time-bounded claims on one expert-memory budget, values each claim by counterfactually replaying the same deadline schedule, and reallocates memory through feasible exchanges that name both the requested claim and the claims displaced by it.

This thesis creates three concrete mechanisms that can be implemented and ablated.

### 2.1 Byte leases

Every use of expert memory becomes a lease with an owner, byte count, and lifetime.

- A residency lease owns the low-precision footprint of a retained expert.
- A fidelity lease owns the incremental bytes between its high- and low-precision representations.
- A lookahead lease owns the destination footprint from transfer issue until the predicted use; after use, it either expires or converts to residency.

This representation exposes temporary occupancy and prevents a permanent cache quota from being confused with staging capacity.

### 2.2 Counterfactual value

The allocator should not estimate residency and lookahead independently. For a proposed state change, it replays the predicted transfer queue and compares exposed deadline wait before and after the change. The same replay captures three interactions:

- retaining an expert removes its future transfer, so prefetching it has no additional value;
- prefetching an expert reduces only the portion of its transfer that remains after available overlap;
- changing precision changes the bytes and completion time of any future transfer.

Quality remains a feasibility condition. It is not converted into milliseconds. Offline sensitivity and online routing weight form a calibrated quality-risk account used to reject infeasible exchanges.

### 2.3 Feasible byte exchanges

A capacity-consuming request is evaluated together with the leases it would displace. An exchange records its change in predicted deadline wait, quality risk, peak live-plus-reserved bytes, and migration cost. The coordinator accepts only exchanges that preserve both ledgers and have positive net latency value.

This makes opportunity cost explicit. The relevant comparison is not “is prefetch C useful?” but “is prefetching C more useful than retaining B or keeping A at high precision for the same constrained bytes?”

## 3. Target section structure

### 3.1 Overview

Use a general-to-specific structure:

1. State the allocation problem and the byte-lease insight.
2. Walk through monitor, lease valuator, exchange coordinator, and executor.
3. Give one concrete three-way exchange example before the architecture figure.
4. State the three invariants that the following subsections establish: quality feasibility, deadline-aware value, and peak-memory safety.

### 3.2 Objective and expert state

- Keep tail latency as the objective and quality plus expert memory as hard constraints.
- Define absent, resident-low, resident-high, and in-transition states.
- Decompose resident-high into a residency footprint and a fidelity increment.
- Define the planning window without claiming an online task-quality guarantee.

### 3.3 Byte leases

- Define a lease tuple containing use, expert, bytes, and lifetime.
- Explain when each lease starts, expires, or converts.
- Define the time-indexed memory ledger, including destinations and unreclaimed sources.

### 3.4 Estimating quality risk and deadline wait

- Retain the routing-weight update and offline sensitivity.
- Define the aggregate risk estimate and a calibrated proxy limit.
- Build a deadline-ordered transfer replay from predicted uses, representation sizes, queued bytes, and measured overlap bandwidth.
- Define counterfactual marginal latency value as the difference between two replays.

### 3.5 Constructing exchanges

- Pair every request with explicit donor leases.
- Include promotion/demotion redistribution, eviction/retention replacement, and prefetch admission.
- Reject exchanges that violate the risk ledger, the time-indexed byte ledger, or per-expert state compatibility.
- Rank positive exchanges by net saved stall per peak byte requested and recompute after every acceptance.

### 3.6 Two-timescale coordination

- Fast events refresh deadlines, reuse, bandwidth, and lookahead/residency leases.
- Slow events refresh routing weights and fidelity exchanges.
- Both use the same exchange evaluator and ledger.
- Add minimum tenure and hysteresis only as churn controls.

### 3.7 Safe execution

- Reserve all destinations before issuing any copy in an exchange.
- Use reserve, copy, publish, and reclaim.
- Keep optional precision transitions nonblocking when an old representation remains published.
- Preserve a demand-fetch path for an absent expert that misses its deadline.

### 3.8 Control algorithm and scope

- Present one receding-horizon control algorithm.
- State that greedy recomputation is an online approximation, not a global optimum.
- Bound the current design to two prebuilt representations, one accelerator, and host-memory backing.

## 4. Claims required for a 9/10 novelty case

The prose can make the mechanism legible, but the following evidence is required before claiming a 9/10 result:

1. A matched comparison against HOBBIT-style miss-path mixed precision and DyMoE-style importance-aware mixed cache/prefetch under the same byte budget and quality bound.
2. An ablation replacing counterfactual valuation with independent action scores, showing double counting or the wrong donor choice.
3. An ablation replacing exchange admission with fixed partitioning, precision-first, and prefetch-first policies.
4. A trace showing at least one accepted exchange of each important form: fidelity-to-residency, fidelity-to-lookahead, and residency-to-lookahead.
5. Prediction error for deadline wait and quality risk, including cases where the proxy rejects a configuration that would pass and admits one that fails.
6. Controller overhead and a proof-by-log that live plus reserved bytes never exceed the declared expert-memory budget.

If these comparisons do not show a stable advantage, the defensible claim is a shared accounting and execution mechanism, not a new joint allocator.

## 5. Terminology contract

Use the following terms consistently throughout the section:

- **expert-memory budget** for (B);
- **quality-risk estimate** for the online proxy and **quality-degradation limit** for the evaluated constraint;
- **byte lease** for a time-bounded memory claim;
- **request lease**, **donor leases**, and **exchange** for an allocation change;
- **predicted exposed wait** for the latency model;
- **published state** and **reserved state** for executable and in-flight representations;
- **fidelity**, **residency**, and **lookahead** for the three memory uses.

Avoid using “quality score,” “cache quota,” “prefetch quota,” or “global optimum.”

## 6. Post-draft Overview audit

The first rewrite establishes the right mechanism, but the Overview still introduces byte lease, exchange, and counterfactual replay before showing one complete decision. The final pass should use the following general-to-specific order:

1. Open with the whole decision: DynaByte reallocates a shared budget through exchanges rather than partitioning it among three controllers.
2. Define a byte lease and immediately show one complete exchange, such as funding a lookahead request by evicting a resident and, when feasible, releasing a fidelity increment.
3. Extract the three rules from that example: decompose expert state into claims, measure latency only on the resulting joint state, and keep quality plus peak bytes as hard feasibility checks.
4. Introduce the four implementation blocks after the decision is understood.
5. Redraw Figure~4's design counterpart so its visual center is “request + donors, replay before/after,” not three independently scored actions.

The Overview should not repeat the detailed equations or end with a prose summary. Its job is to make the allocation primitive understandable before the formal subsections.
