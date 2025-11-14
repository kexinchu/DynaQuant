from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .monitor import ExpertMonitor
from .types import Bitwidth, ExpertID


@dataclass
class PrecisionController:
    """Decide per-expert precision targets based on hotness scores."""

    tau_h: float
    tau_c: float
    max_w4_slots: int
    allow_new_w4: bool = True
    _targets: Dict[ExpertID, Bitwidth] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau_c < self.tau_h <= 1.0:
            raise ValueError("Require 0 <= tau_c < tau_h <= 1.0")
        if self.max_w4_slots <= 0:
            raise ValueError("max_w4_slots must be > 0")

    def plan(
        self,
        active_set: Iterable[ExpertID],
        monitor: ExpertMonitor,
    ) -> Dict[ExpertID, Bitwidth]:
        """Return target bitwidths for the supplied active experts."""
        active_list = list(dict.fromkeys(active_set))
        candidates = set(active_list)
        candidates.update(
            expert for expert, bitwidth in self._targets.items() if bitwidth is Bitwidth.W4
        )

        for expert in candidates:
            score = monitor.score(expert)
            current = self._targets.get(expert, Bitwidth.W2)

            if current is Bitwidth.W4:
                if score < self.tau_c:
                    self._targets[expert] = Bitwidth.W2
            else:
                if self.allow_new_w4 and score > self.tau_h:
                    self._targets[expert] = Bitwidth.W4
                else:
                    self._targets.setdefault(expert, Bitwidth.W2)

        self._enforce_cap(monitor)
        return {expert: self._targets.get(expert, Bitwidth.W2) for expert in candidates}

    def _enforce_cap(self, monitor: ExpertMonitor) -> None:
        w4_experts = [
            (expert, monitor.score(expert))
            for expert, bitwidth in self._targets.items()
            if bitwidth is Bitwidth.W4
        ]

        excess = len(w4_experts) - self.max_w4_slots
        if excess <= 0:
            return

        w4_experts.sort(key=lambda item: item[1])
        for expert, _ in w4_experts[:excess]:
            self._targets[expert] = Bitwidth.W2

    def reset(self) -> None:
        """Forget all previously assigned targets."""
        self._targets.clear()


__all__ = ["PrecisionController"]

