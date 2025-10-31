"""
PrecisionController - Decide W4/W2 targets with hysteresis and pool limits
"""

import logging
from typing import Dict, List, Set
from collections import defaultdict

from .types import ExpertID
from .monitor import ExpertMonitor

logger = logging.getLogger(__name__)


class PrecisionController:
    """
    Determine target precision for each expert based on hotness scores.

    Uses hysteresis to prevent oscillation:
    - Promote to W4 if score > tau_h (hot threshold)
    - Demote to W2 if score < tau_c (cold threshold)
    - Maintain current state if tau_c <= score <= tau_h

    Also enforces pool capacity limits (max_w4_slots per layer).
    """

    def __init__(
        self,
        tau_h: float = 0.65,
        tau_c: float = 0.45,
        max_w4_slots: int = 16,
        num_layers: int = 32,
        num_experts_per_layer: int = 64,
    ):
        """
        Args:
            tau_h: Hot threshold for promotion to W4
            tau_c: Cold threshold for demotion to W2
            max_w4_slots: Maximum W4 experts per layer (pool capacity)
            num_layers: Number of MoE layers
            num_experts_per_layer: Number of experts per layer
        """
        assert tau_h > tau_c, "Hot threshold must be greater than cold threshold"

        self.tau_h = tau_h
        self.tau_c = tau_c
        self.max_w4_slots = max_w4_slots
        self.num_layers = num_layers
        self.num_experts_per_layer = num_experts_per_layer

        # Current precision assignments: ExpertID -> "W4" or "W2"
        # Initialize all to W2 (cold start)
        self.current_precision: Dict[ExpertID, str] = {}

        logger.info(
            f"PrecisionController initialized: tau_h={tau_h}, tau_c={tau_c}, "
            f"max_w4_slots={max_w4_slots}/layer"
        )

    def plan(
        self,
        active_set: List[ExpertID],
        monitor: ExpertMonitor
    ) -> Dict[ExpertID, str]:
        """
        Compute target precision for each expert in the active set.

        Args:
            active_set: List of experts that will be used in upcoming inference
            monitor: ExpertMonitor with hotness scores

        Returns:
            Dictionary mapping ExpertID to target bitwidth ("W4" or "W2")
        """
        # Get all hotness scores
        scores = monitor.get_all_scores()

        # Group active experts by layer
        layer_experts: Dict[int, List[ExpertID]] = defaultdict(list)
        for expert in active_set:
            layer_experts[expert.layer].append(expert)

        target_precision: Dict[ExpertID, str] = {}

        # Process each layer independently
        for layer_id, experts in layer_experts.items():
            target_precision.update(
                self._plan_layer(layer_id, experts, scores)
            )

        return target_precision

    def _plan_layer(
        self,
        layer_id: int,
        experts: List[ExpertID],
        scores: Dict[ExpertID, float]
    ) -> Dict[ExpertID, str]:
        """
        Plan precision for experts in a single layer.

        Enforces max_w4_slots constraint by prioritizing highest-scoring experts.
        """
        # Sort experts by score (descending)
        expert_scores = [(e, scores.get(e, 0.0)) for e in experts]
        expert_scores.sort(key=lambda x: x[1], reverse=True)

        layer_targets: Dict[ExpertID, str] = {}
        w4_count = 0

        for expert, score in expert_scores:
            current = self.current_precision.get(expert, "W2")

            # Apply hysteresis
            if score > self.tau_h and w4_count < self.max_w4_slots:
                # Promote to W4
                target = "W4"
                w4_count += 1
            elif score < self.tau_c:
                # Demote to W2
                target = "W2"
            elif current == "W4" and w4_count < self.max_w4_slots:
                # Maintain W4 (within hysteresis band)
                target = "W4"
                w4_count += 1
            else:
                # Default to W2 (either in hysteresis band as W2, or no W4 slots left)
                target = "W2"

            layer_targets[expert] = target

        # Update current precision
        self.current_precision.update(layer_targets)

        logger.debug(
            f"Layer {layer_id}: {w4_count}/{self.max_w4_slots} W4 experts, "
            f"{len(experts) - w4_count} W2 experts"
        )

        return layer_targets

    def get_diff(
        self,
        target_precision: Dict[ExpertID, str],
        current_residency: Dict[ExpertID, str]
    ) -> Dict[str, List[ExpertID]]:
        """
        Compute the diff between target and current state.

        Args:
            target_precision: Target precision mapping
            current_residency: Current precision mapping

        Returns:
            Dictionary with "upgrades" and "downgrades" lists
        """
        upgrades = []
        downgrades = []

        for expert, target in target_precision.items():
            current = current_residency.get(expert, "W2")

            if target == "W4" and current == "W2":
                upgrades.append(expert)
            elif target == "W2" and current == "W4":
                downgrades.append(expert)

        return {
            "upgrades": upgrades,
            "downgrades": downgrades
        }

    def adapt_thresholds(
        self,
        ready_ratio: float,
        hbm_pressure: float
    ) -> None:
        """
        Adaptively adjust tau_h and tau_c based on system feedback.

        Args:
            ready_ratio: Fraction of swaps that complete before use (0-1)
            hbm_pressure: HBM utilization (0-1, 1 = full)
        """
        # If too many misses, widen hysteresis band to reduce swap frequency
        if ready_ratio < 0.99:
            delta = 0.01
            self.tau_h = min(1.0, self.tau_h + delta)
            self.tau_c = max(0.0, self.tau_c - delta)
            logger.info(
                f"Widening thresholds due to low ready ratio {ready_ratio:.3f}: "
                f"tau_h={self.tau_h:.3f}, tau_c={self.tau_c:.3f}"
            )

        # If HBM pressure is high, increase tau_h to be more selective
        if hbm_pressure > 0.90:
            self.tau_h = min(1.0, self.tau_h + 0.02)
            logger.info(
                f"Increasing tau_h due to HBM pressure {hbm_pressure:.3f}: "
                f"tau_h={self.tau_h:.3f}"
            )

    def get_statistics(self) -> Dict:
        """Get controller statistics"""
        w4_count = sum(1 for p in self.current_precision.values() if p == "W4")
        w2_count = sum(1 for p in self.current_precision.values() if p == "W2")

        return {
            "tau_h": self.tau_h,
            "tau_c": self.tau_c,
            "max_w4_slots": self.max_w4_slots,
            "current_w4_experts": w4_count,
            "current_w2_experts": w2_count,
        }
