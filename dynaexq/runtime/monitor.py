from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from .types import ExpertID


@dataclass(slots=True)
class _LayerState:
    """Internal structure tracking scores for a single MoE layer."""

    scores: Dict[int, float]
    counts: Dict[int, int]

    def decay(self, factor: float) -> None:
        for expert, score in list(self.scores.items()):
            self.scores[expert] = score * factor
        for expert in list(self.counts.keys()):
            self.counts[expert] = 0


class ExpertMonitor:
    """Track expert hotness via an EWMA over router activations."""

    def __init__(
        self,
        ewma_alpha: float = 0.2,
        epoch_decay: float = 0.5,
    ) -> None:
        if not 0.0 < ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1].")
        if not 0.0 < epoch_decay <= 1.0:
            raise ValueError("epoch_decay must be in (0, 1].")

        self._ewma_alpha = ewma_alpha
        self._epoch_decay = epoch_decay
        self._layers: Dict[int, _LayerState] = defaultdict(
            lambda: _LayerState(scores=defaultdict(float), counts=defaultdict(int))
        )
        self._lock = threading.Lock()

    def update_batch(
        self,
        layer: int,
        topk_idx: np.ndarray,
        logits: np.ndarray,
    ) -> None:
        """Update hotness statistics for a batch."""
        indices = np.asarray(topk_idx, dtype=np.int32)
        weights = np.asarray(logits, dtype=np.float32)

        if indices.shape != weights.shape:
            raise ValueError(
                f"topk_idx shape {indices.shape} does not match logits {weights.shape}"
            )
        if indices.ndim != 2:
            raise ValueError("Expected 2-D arrays shaped (batch, k).")

        with self._lock:
            layer_state = self._layers[layer]

            for expert_idx, score in self._aggregate_batch(indices, weights):
                prev_score = layer_state.scores.get(expert_idx, score)
                new_score = prev_score + self._ewma_alpha * (score - prev_score)
                layer_state.scores[expert_idx] = float(new_score)
                layer_state.counts[expert_idx] += 1

    def _aggregate_batch(
        self,
        indices: np.ndarray,
        weights: np.ndarray,
    ) -> Iterable[Tuple[int, float]]:
        flat_idx = indices.reshape(-1)
        flat_weight = weights.reshape(-1)

        accumulator: Dict[int, Tuple[float, int]] = defaultdict(lambda: (0.0, 0))
        for expert_id, weight in zip(flat_idx.tolist(), flat_weight.tolist()):
            total, count = accumulator[expert_id]
            accumulator[expert_id] = (total + float(weight), count + 1)

        for expert_id, (total, count) in accumulator.items():
            yield expert_id, total / max(count, 1)

    def score(self, expert: ExpertID) -> float:
        """Return the latest EWMA score for an expert."""
        with self._lock:
            layer_state = self._layers.get(expert.layer)
            if layer_state is None:
                return 0.0
            return float(layer_state.scores.get(expert.idx, 0.0))

    def epoch_tick(self) -> None:
        """Decay statistics to avoid stale hotness dominating future decisions."""
        with self._lock:
            for layer_state in self._layers.values():
                layer_state.decay(self._epoch_decay)


__all__ = ["ExpertMonitor"]
