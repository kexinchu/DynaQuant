"""
PrefetchPlanner - Layer-wise prefetching to overlap transfers with compute
"""

import logging
from typing import Dict, List, Optional, Set
from collections import defaultdict

from .types import ExpertID
from .monitor import ExpertMonitor
from .swap_engine import SwapEngine

logger = logging.getLogger(__name__)


class PrefetchPlanner:
    """
    Prefetch experts for the next layer while current layer is computing.

    Uses simple heuristics:
    1. Predict next layer's active experts based on current batch patterns
    2. Trigger upgrades with high priority for predicted hot experts
    3. Overlap prefetch with current layer's computation
    """

    def __init__(
        self,
        swap_engine: SwapEngine,
        monitor: ExpertMonitor,
        num_layers: int = 32,
        lookahead_layers: int = 1,
        prefetch_top_k: int = 8,
    ):
        """
        Args:
            swap_engine: SwapEngine instance
            monitor: ExpertMonitor instance
            num_layers: Total number of MoE layers
            lookahead_layers: How many layers ahead to prefetch (usually 1)
            prefetch_top_k: Number of top experts to prefetch per layer
        """
        self.swap_engine = swap_engine
        self.monitor = monitor
        self.num_layers = num_layers
        self.lookahead_layers = lookahead_layers
        self.prefetch_top_k = prefetch_top_k

        # Track recent activation patterns for prediction
        # layer -> expert -> activation count
        self.recent_patterns: Dict[int, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Statistics
        self.prefetch_count = 0
        self.hit_count = 0
        self.miss_count = 0

        logger.info(
            f"PrefetchPlanner initialized: lookahead={lookahead_layers}, "
            f"prefetch_top_k={prefetch_top_k}"
        )

    def update_pattern(self, layer: int, active_experts: List[ExpertID]):
        """
        Update recent activation patterns.

        Args:
            layer: Current layer
            active_experts: Experts activated in current batch
        """
        for expert in active_experts:
            if expert.layer == layer:
                self.recent_patterns[layer][expert.idx] += 1

    def lookahead(
        self,
        current_layer: int,
        next_active: Optional[List[ExpertID]] = None
    ) -> None:
        """
        Trigger prefetch for next layer experts.

        Args:
            current_layer: Current layer being computed
            next_active: Optional list of known next-layer experts
                        If None, predicts based on hotness scores
        """
        target_layer = current_layer + self.lookahead_layers

        if target_layer >= self.num_layers:
            return  # No next layer

        # Determine which experts to prefetch
        if next_active is not None:
            # Use provided list (e.g., from router lookahead)
            experts_to_prefetch = [
                e for e in next_active if e.layer == target_layer
            ]
        else:
            # Predict based on hotness scores
            experts_to_prefetch = self._predict_next_experts(target_layer)

        # Trigger prefetch upgrades with high priority
        for expert in experts_to_prefetch[:self.prefetch_top_k]:
            if not self.swap_engine.is_ready(expert):
                self.swap_engine.upgrade(expert, priority=10)  # High priority
                self.prefetch_count += 1
                logger.debug(f"Prefetching {expert} for layer {target_layer}")

    def _predict_next_experts(self, layer: int) -> List[ExpertID]:
        """
        Predict which experts will be active in the given layer.

        Uses hotness scores from monitor and recent patterns.
        """
        # Get hotness scores for this layer's experts
        expert_scores = []

        for expert_idx in range(64):  # Assume 64 experts per layer
            expert = ExpertID(layer=layer, idx=expert_idx)
            score = self.monitor.score(expert)

            # Boost score based on recent patterns
            recent_count = self.recent_patterns[layer].get(expert_idx, 0)
            boosted_score = score + 0.1 * min(recent_count / 10.0, 1.0)

            expert_scores.append((expert, boosted_score))

        # Sort by score descending
        expert_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top experts
        return [e for e, _ in expert_scores[:self.prefetch_top_k * 2]]

    def record_hit(self, expert: ExpertID, was_prefetched: bool):
        """
        Record whether a prefetch was successful.

        Args:
            expert: Expert that was used
            was_prefetched: Whether it was ready before use
        """
        if was_prefetched:
            self.hit_count += 1
        else:
            self.miss_count += 1

    def get_statistics(self) -> Dict:
        """Get prefetch statistics"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0.0

        return {
            "prefetch_count": self.prefetch_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "lookahead_layers": self.lookahead_layers,
            "prefetch_top_k": self.prefetch_top_k,
        }

    def get_layer_patterns(self, layer: int) -> Dict[int, int]:
        """Get recent activation patterns for a layer"""
        return dict(self.recent_patterns[layer])
