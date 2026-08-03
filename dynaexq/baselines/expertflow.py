"""Guard against relabeling the blocking LRU baseline as ExpertFlow.

The full ExpertFlow system described in arXiv:2510.26730 is not vendored in
this repository. Paper experiments must supply a pinned external
implementation and record its repository commit plus runtime telemetry.
"""

raise ImportError(
    "The full ExpertFlow implementation is not available in this repository. "
    "Use dynaexq.baselines.lru_offload only when reporting an LRU-offload "
    "baseline; do not label it ExpertFlow."
)
