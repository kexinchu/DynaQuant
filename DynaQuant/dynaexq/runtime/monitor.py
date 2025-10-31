"""
ExpertMonitor - Track expert hotness with EWMA and epoch windowing
"""

import time
import threading
from collections import defaultdict
from typing import Dict, Optional
import numpy as np
import logging

from .types import ExpertID

logger = logging.getLogger(__name__)


class ExpertMonitor:
    """
    Monitor expert usage and compute hotness scores using EWMA.

    Hotness score S_i = EWMA_t(mean_{x in batch} g_i(x))
    where g_i(x) is the router weight/logit for expert i.

    Epoch windowing: Global epoch increments every epoch_duration seconds,
    resets EWMA span to prevent stale data from dominating.
    """

    def __init__(
        self,
        ewma_alpha: float = 0.2,
        epoch_duration: float = 300.0,  # 5 minutes
        num_layers: int = 32,
        num_experts_per_layer: int = 64,
    ):
        """
        Args:
            ewma_alpha: EWMA smoothing factor (0 < alpha <= 1)
            epoch_duration: Duration of each epoch in seconds
            num_layers: Number of MoE layers
            num_experts_per_layer: Number of experts per layer
        """
        self.ewma_alpha = ewma_alpha
        self.epoch_duration = epoch_duration
        self.num_layers = num_layers
        self.num_experts_per_layer = num_experts_per_layer

        # Hotness scores: ExpertID -> EWMA score
        self.hotness: Dict[ExpertID, float] = defaultdict(float)

        # Statistics for current epoch
        self.batch_count: Dict[ExpertID, int] = defaultdict(int)
        self.total_logit_mass: Dict[ExpertID, float] = defaultdict(float)

        # Epoch management
        self.current_epoch = 0
        self.epoch_start_time = time.time()

        # Thread safety
        self.lock = threading.RLock()

        logger.info(
            f"ExpertMonitor initialized: alpha={ewma_alpha}, "
            f"epoch={epoch_duration}s, layers={num_layers}, experts={num_experts_per_layer}"
        )

    def update_batch(
        self,
        layer: int,
        topk_idx: np.ndarray,
        logits: Optional[np.ndarray] = None
    ) -> None:
        """
        Update expert statistics for a batch.

        Args:
            layer: Layer index
            topk_idx: Array of shape (batch_size, k) with selected expert indices
            logits: Optional array of shape (batch_size, k) with router weights/logits
                   If None, uses uniform weights (1/k for each selected expert)
        """
        with self.lock:
            # Check if we need to tick the epoch
            self._maybe_tick_epoch()

            batch_size, k = topk_idx.shape

            # If no logits provided, use uniform weights
            if logits is None:
                logits = np.ones_like(topk_idx, dtype=np.float32) / k

            # Normalize logits per token (softmax across k)
            if logits.ndim == 2:
                # Softmax normalization
                logits_exp = np.exp(
                    logits - np.max(logits, axis=1, keepdims=True))
                logits = logits_exp / np.sum(logits_exp, axis=1, keepdims=True)

            # Aggregate statistics per expert
            for batch_idx in range(batch_size):
                for k_idx in range(k):
                    expert_idx = topk_idx[batch_idx, k_idx]
                    expert_id = ExpertID(layer=layer, idx=int(expert_idx))
                    weight = float(logits[batch_idx, k_idx])

                    self.total_logit_mass[expert_id] += weight
                    self.batch_count[expert_id] += 1

    def score(self, expert: ExpertID) -> float:
        """
        Get current hotness score for an expert.

        Returns:
            EWMA hotness score (0.0 if never seen)
        """
        with self.lock:
            return self.hotness.get(expert, 0.0)

    def get_all_scores(self) -> Dict[ExpertID, float]:
        """Get all hotness scores as a dictionary"""
        with self.lock:
            return dict(self.hotness)

    def epoch_tick(self) -> None:
        """
        Manual epoch tick - update EWMA scores and reset statistics.
        Normally called automatically by _maybe_tick_epoch().
        """
        with self.lock:
            logger.info(
                f"Epoch tick: {self.current_epoch} -> {self.current_epoch + 1}")

            # Update EWMA for all experts that were active this epoch
            for expert_id in self.total_logit_mass:
                if self.batch_count[expert_id] > 0:
                    # Mean logit mass per batch
                    mean_logit = self.total_logit_mass[expert_id] / \
                        self.batch_count[expert_id]

                    # EWMA update
                    old_score = self.hotness[expert_id]
                    new_score = self.ewma_alpha * mean_logit + \
                        (1 - self.ewma_alpha) * old_score
                    self.hotness[expert_id] = new_score

            # Decay scores for inactive experts
            for expert_id in list(self.hotness.keys()):
                if expert_id not in self.total_logit_mass:
                    # Expert was not active this epoch, decay toward zero
                    self.hotness[expert_id] *= (1 - self.ewma_alpha)

            # Reset epoch statistics
            self.batch_count.clear()
            self.total_logit_mass.clear()
            self.current_epoch += 1
            self.epoch_start_time = time.time()

    def _maybe_tick_epoch(self) -> None:
        """Check if epoch duration has elapsed and tick if needed"""
        current_time = time.time()
        if current_time - self.epoch_start_time >= self.epoch_duration:
            self.epoch_tick()

    def get_statistics(self) -> Dict:
        """Get monitoring statistics"""
        with self.lock:
            return {
                "current_epoch": self.current_epoch,
                "epoch_elapsed": time.time() - self.epoch_start_time,
                "total_experts_tracked": len(self.hotness),
                "active_experts_this_epoch": len(self.total_logit_mass),
                "mean_hotness": np.mean(list(self.hotness.values())) if self.hotness else 0.0,
                "max_hotness": max(self.hotness.values()) if self.hotness else 0.0,
            }
