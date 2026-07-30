"""
HotnessTracker: maintains S[l,e] EMA scores.

Implements EMA hotness update: S[l,e] ← α * S[l,e] + (1-α) * g[l,e](x_t)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import Tier


@dataclass
class HotnessState:
    """Hotness state for all experts."""
    alpha: float
    scores: np.ndarray  # shape: [L, maxE], masked per-layer if E varies
    num_layers: int
    experts_per_layer: list[int]  # E_l for each layer
    cumulative_scores: np.ndarray
    update_counts: np.ndarray


class HotnessTracker:
    """
    Tracks expert hotness using EMA (Exponential Moving Average).
    
    Implements: S[l,e] ← α * S[l,e] + (1-α) * g[l,e](x_t)
    """

    def __init__(
        self,
        num_layers: int,
        experts_per_layer: list[int] | int,
        alpha: float = 0.9,
    ):
        """
        Args:
            num_layers: Number of MoE layers L
            experts_per_layer: List of expert counts per layer, or single int if uniform
            alpha: EMA smoothing factor (0 <= alpha < 1)
        """
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        
        self.num_layers = num_layers
        if isinstance(experts_per_layer, int):
            self.experts_per_layer = [experts_per_layer] * num_layers
        else:
            if len(experts_per_layer) != num_layers:
                raise ValueError(
                    f"experts_per_layer length {len(experts_per_layer)} != num_layers {num_layers}"
                )
            self.experts_per_layer = list(experts_per_layer)
        
        self.alpha = alpha
        max_experts = max(self.experts_per_layer)
        
        # Initialize scores to zero
        self._scores = np.zeros((num_layers, max_experts), dtype=np.float32)
        self._cumulative_scores = np.zeros(
            (num_layers, max_experts),
            dtype=np.float64,
        )
        self._update_counts = np.zeros(num_layers, dtype=np.int64)
        self._lock = threading.Lock()

    def update(self, layer: int, g_values: Dict[int, float]) -> None:
        """
        Update hotness scores for a layer.
        
        Args:
            layer: Layer index
            g_values: Dictionary mapping expert_id -> g[l,e] value
        """
        if layer < 0 or layer >= self.num_layers:
            raise ValueError(f"Layer {layer} out of range [0, {self.num_layers})")
        
        with self._lock:
            # Unselected experts receive g=0 and must decay as well. Updating
            # only selected IDs leaves old hot experts permanently hot after
            # a workload shift.
            self._scores[layer, : self.experts_per_layer[layer]] *= self.alpha
            for expert_id, g_val in g_values.items():
                if expert_id < 0 or expert_id >= self.experts_per_layer[layer]:
                    continue  # Skip invalid expert IDs
                
                # EMA update: S[l,e] ← α * S[l,e] + (1-α) * g[l,e]
                self._scores[layer, expert_id] += (1.0 - self.alpha) * g_val
                self._cumulative_scores[layer, expert_id] += float(g_val)
            self._update_counts[layer] += 1

    def get_score(self, layer: int, expert: int) -> float:
        """Get current hotness score for an expert."""
        if layer < 0 or layer >= self.num_layers:
            return 0.0
        if expert < 0 or expert >= self.experts_per_layer[layer]:
            return 0.0
        
        with self._lock:
            return float(self._scores[layer, expert])

    def get_layer_scores(self, layer: int) -> np.ndarray:
        """Get all scores for a layer (returns copy)."""
        if layer < 0 or layer >= self.num_layers:
            return np.array([], dtype=np.float32)
        
        with self._lock:
            num_experts = self.experts_per_layer[layer]
            return self._scores[layer, :num_experts].copy()

    def get_cumulative_layer_scores(self, layer: int) -> np.ndarray:
        """Return order-invariant mean routing mass across all updates."""
        if layer < 0 or layer >= self.num_layers:
            return np.array([], dtype=np.float64)

        with self._lock:
            num_experts = self.experts_per_layer[layer]
            count = int(self._update_counts[layer])
            if count == 0:
                return np.zeros(num_experts, dtype=np.float64)
            return (
                self._cumulative_scores[layer, :num_experts] / count
            ).copy()

    def get_state(self) -> HotnessState:
        """Get current hotness state (returns copy)."""
        with self._lock:
            return HotnessState(
                alpha=self.alpha,
                scores=self._scores.copy(),
                num_layers=self.num_layers,
                experts_per_layer=self.experts_per_layer.copy(),
                cumulative_scores=self._cumulative_scores.copy(),
                update_counts=self._update_counts.copy(),
            )

    def reset(self) -> None:
        """Reset all scores to zero."""
        with self._lock:
            self._scores.fill(0.0)
            self._cumulative_scores.fill(0.0)
            self._update_counts.fill(0)
