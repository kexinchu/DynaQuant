# DynaByte terminology contract

This file is an editing contract for the paper, not a section of the manuscript.
Use the canonical term in prose, equations, captions, and figures unless an
explicitly named baseline uses different terminology.

| Concept | Canonical term | Operational wording | Avoid for DynaByte |
|---|---|---|---|
| Memory objective | fidelity | promote/demote; precision transition | precision as the name of the objective |
| Persistent placement | residency | retain/evict | cache quota, pin, pinned expert |
| Temporary early placement | lookahead | prefetch/cancel | horizon as the name of the resource use |
| Available expert capacity | expert-memory budget \(B\) | remaining/uncommitted bytes | GPU-memory budget, pool capacity |
| Device allocation substrate | shared expert arena | bind/reclaim an arena extent | high-/low-tier pool, per-layer pool |
| Accounted memory claim | byte lease | acquire/renew/release | slot, pin |
| Forecast search depth | adaptive candidate envelope \(S_t\) | expand/shrink the envelope | planning window, DynaByte horizon |
| Visible representation | published state | publish a representation | active/current/live state |
| State used during selection | provisional state | apply an exchange provisionally | provisional target, candidate state |
| Accepted but incomplete change | reserved transition | reserve/cancel/complete | in-flight state when the transition is meant |
| Deployment quality bound | quality-degradation limit \(\epsilon\) | measure against the reference | quality budget, risk budget |
| Online quality proxy | quality-risk account \(\widehat R(X)\leq\widehat\epsilon\) | calibrated feasibility check | quality guarantee, quality limit |
| Latency surrogate | predicted exposed wait \(\widehat L\) | saved exposed wait | latency score, deadline value |
| Scheduling rule | resident-first execution | issue ready resident groups first | cache-aware issue order |
| Measured copy rate | in-overlap bandwidth | bytes/time for copies overlapping compute | effective bandwidth after first definition |
| Forward/reclaim synchronization | reader reference | acquire/release a reference | read lease |

Additional rules:

1. `prefetch horizon` is permitted only for a baseline that exposes a fixed
   horizon. DynaByte uses an adaptive candidate envelope.
2. `pool` may denote a logical collection of model experts in cited work, but
   not DynaByte's device allocator.
3. `device memory` is the hardware-neutral term. Use `GPU`, `CUDA`, or `NVML`
   only for an implementation or testbed that specifically uses NVIDIA GPUs.
4. Define a nonstandard term once at first use. Later occurrences use the term
   without restating its definition.
