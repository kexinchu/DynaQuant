# Design component restructuring plan

## 1. Structural diagnosis

The current Design has the right mechanism but exposes it through ten subsections. Several headings describe concepts rather than executable components: Objective and Expert State, Byte Leases, Quality Risk, Counterfactual Value, Two Control Time Scales, and Control Period. A reader must reconstruct which runtime component owns each equation and when it runs.

The revised section will use five subsections in total:

1. Overview
2. State Monitor
3. Lease Generator and Valuator
4. Exchange Coordinator
5. Safe Executor

Overview establishes the contract. Each remaining subsection answers four questions for one component: what state it receives, what computation it performs, what state it emits, and what invariant or approximation bounds its behavior.

## 2. Revised section map

### 3.1 Overview

Keep the byte-exchange example and architecture figure. Move the optimization objective and expert state model into this subsection so that the reader knows the system contract before entering any component.

Order:

1. Shared-budget conflict and byte exchange.
2. One concrete request-and-donor example.
3. Byte lease definition at an intuitive level.
4. Architecture figure and component responsibilities.
5. Formal objective, expert states, and fidelity-byte decomposition.

Do not introduce detailed estimators or algorithms here.

### 3.2 State Monitor

Combine the former Runtime State and Estimating Quality Risk material.

- Published and reserved expert state.
- Routing-weight update.
- Offline sensitivity and calibrated quality-risk estimate.
- Predicted uses, deadlines, reuse, effective bandwidth, and observed transfer outcomes.
- Output contract: a snapshot consumed by the generator and coordinator.

Quality risk remains a hard feasibility account. The subsection must not convert it into latency.

### 3.3 Lease Generator and Valuator

Combine Byte Leases and Deadline-Aware Counterfactual Value.

- Define the lease tuple once.
- Explain fidelity, residency, and lookahead requests in three compact paragraphs.
- Define deadline-ordered replay and predicted exposed wait.
- Define the value of a complete exchange, not an isolated action.
- Retain the aggregate overlap equation only as a candidate-generation bound.

The component emits typed lease requests plus the state needed to revalue them after donor selection.

### 3.4 Exchange Coordinator

Combine Exchange Coordination, Two Control Time Scales, and Control Period.

- Maintain the provisional byte and quality-risk accounts.
- Attach explicit donor leases to each request.
- Check state compatibility, risk, transition peak, and deadline feasibility.
- Rank positive exchanges by net saved wait per requested byte.
- Recompute affected values after admission.
- Explain fast and slow refresh rates as scheduling of the same coordinator, not separate controllers.
- End with the control algorithm.

The subsection should clearly state that this is a receding-horizon approximation rather than a global optimizer.

### 3.5 Safe Executor

Combine Safe Admission and Publication.

- Fixed representation pools and staging capacity.
- Atomic reservation of the complete transition schedule.
- Dependency between donor reclamation and destination allocation.
- Reserve, copy, publish, and reclaim.
- Optional precision transitions versus blocking demand misses.
- Feedback returned to the monitor.

The executor owns the live-plus-reserved byte invariant because it is the component that can enforce it during concurrency.

## 3. Material moved out of Design

Remove Scope of Adaptation from Design. Move its non-goals into Discussion, System Scope:

- no router or top-k changes;
- no expert skipping;
- two offline-built representations per checkpoint;
- calibration-only sensitivities and constants;
- one accelerator with host-memory backing.

This avoids ending the mechanism section with limitations and keeps all scope boundaries in one place.

## 4. Readability rules

- Use one term for each concept: fidelity, residency, lookahead; byte lease; exchange; request lease; donor lease; quality-risk estimate; predicted exposed wait; published state; reserved state.
- Keep equations next to the component that consumes them.
- Use paragraphs inside a component for local distinctions, not additional subsections.
- Introduce variables before using them in an equation.
- Follow every equation with its operational meaning.
- Avoid chapter-end summaries and repeated statements of the paper thesis.
- Preserve existing equation and section labels needed by Evaluation.

## 5. Reviewer target

The revised structure should allow a reviewer to answer the following after one pass:

1. What is the unit of allocation? A byte lease.
2. What is compared? A complete request-plus-donors exchange.
3. Where do values come from? The monitor snapshot and deadline replay.
4. Who makes the decision? The exchange coordinator under two hard accounts.
5. Who guarantees memory safety? The executor through transition-aware reservation.

Meeting these conditions raises organization and mechanism legibility to the intended 9/10 level. Experimental novelty remains conditional on the matched-policy and ablation evidence listed in `DESIGN_NOVELTY_REWRITE_PLAN.md`.
