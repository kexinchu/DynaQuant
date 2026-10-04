"""Quality-risk feasibility for the shared expert state.

The account deliberately keeps quality risk dimensionless.  It never assigns
milliseconds to a low-precision execution; latency valuation and quality
feasibility remain separate decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from dynaexq.core.registry import ExpertKey


@dataclass
class QualityRiskStats:
    evaluations: int = 0
    infeasible_states: int = 0
    repair_plans: int = 0
    unrepaired_states: int = 0


class QualityRiskAccount:
    """Evaluate ``sum_i w_i q_i (1-h_i)`` under a calibrated limit."""

    def __init__(
        self,
        sensitivities: Mapping[ExpertKey, float],
        *,
        limit: float,
        tolerance: float = 1e-6,
    ) -> None:
        if limit < 0.0:
            raise ValueError("quality-risk limit must be non-negative")
        if tolerance < 0.0:
            raise ValueError("quality-risk tolerance must be non-negative")
        normalized = {}
        for key, value in sensitivities.items():
            number = float(value)
            if number < 0.0 or number == float("inf"):
                raise ValueError(
                    f"invalid sensitivity for {key}: {number}"
                )
            normalized[key] = number
        if not normalized:
            raise ValueError("quality-risk account requires sensitivities")
        self.sensitivities = normalized
        self.limit = float(limit)
        self.tolerance = float(tolerance)
        self.stats = QualityRiskStats()

    def contribution(
        self,
        key: ExpertKey,
        routing_weight: float,
    ) -> float:
        weight = float(routing_weight)
        if weight < 0.0:
            raise ValueError("routing weights must be non-negative")
        return weight * self.sensitivities.get(key, 0.0)

    def risk(
        self,
        routing_weights: Mapping[ExpertKey, float],
        high_experts: Iterable[ExpertKey],
    ) -> float:
        high = set(high_experts)
        self.stats.evaluations += 1
        value = sum(
            self.contribution(key, weight)
            for key, weight in routing_weights.items()
            if key not in high
        )
        if value > self.limit + self.tolerance:
            self.stats.infeasible_states += 1
        return value

    def feasible(
        self,
        routing_weights: Mapping[ExpertKey, float],
        high_experts: Iterable[ExpertKey],
    ) -> bool:
        return self.risk(routing_weights, high_experts) <= (
            self.limit + self.tolerance
        )

    def repair_order(
        self,
        routing_weights: Mapping[ExpertKey, float],
        high_experts: Iterable[ExpertKey],
        eligible: Iterable[ExpertKey],
    ) -> list[ExpertKey]:
        """Return the minimum prefix of highest-risk experts needed to fit."""
        high = set(high_experts)
        current = self.risk(routing_weights, high)
        if current <= self.limit + self.tolerance:
            return []
        ranked = sorted(
            (key for key in set(eligible) if key not in high),
            key=lambda key: (
                -self.contribution(key, routing_weights.get(key, 0.0)),
                key.layer,
                key.expert,
            ),
        )
        repair = []
        for key in ranked:
            reduction = self.contribution(
                key, routing_weights.get(key, 0.0)
            )
            if reduction <= 0.0:
                continue
            repair.append(key)
            current -= reduction
            if current <= self.limit + self.tolerance:
                self.stats.repair_plans += 1
                return repair
        self.stats.unrepaired_states += 1
        return repair

    def snapshot(self) -> dict[str, float | int]:
        return {
            "limit": self.limit,
            "tolerance": self.tolerance,
            "sensitivity_count": len(self.sensitivities),
            **vars(self.stats),
        }


__all__ = ["QualityRiskAccount", "QualityRiskStats"]
