"""
RouterObserver: extracts g[l,e](x_t) signals from router outputs.

This module converts model/router outputs into hotness signals for tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class RouterSignal:
    """Router output signal for a single step."""
    layer: int
    expert_indices: np.ndarray  # shape: (batch, topk) or (tokens, topk)
    expert_weights: Optional[np.ndarray] = None  # shape: same as indices, or None
    num_tokens: int = 0


class RouterObserver:
    """
    Observes router outputs and extracts g[l,e] signals.
    
    Supports two modes:
    1. Indicator mode: counts expert selections (uses top-k indices)
    2. Probability mode: uses gating probabilities (more stable)
    """

    def __init__(self, use_probabilities: bool = True):
        """
        Args:
            use_probabilities: If True, use probability mass; if False, use indicator counts.
        """
        self.use_probabilities = use_probabilities

    def extract_signal(
        self,
        layer: int,
        topk_indices: np.ndarray | torch.Tensor,
        logits: Optional[np.ndarray | torch.Tensor] = None,
        topk: Optional[int] = None,
        selected_weights: Optional[np.ndarray | torch.Tensor] = None,
    ) -> RouterSignal:
        """
        Extract g[l,e] signal from router outputs.
        
        Args:
            layer: Layer index
            topk_indices: Expert indices selected (shape: batch x topk or tokens x topk)
            logits: Optional router logits (if available, used for probability mode)
            topk: Number of experts selected per token (if not inferrable from shape)
            selected_weights: Already-selected gate weights with exactly the
                same shape as ``topk_indices``. Prefer this when the router
                exposes top-k probabilities directly.
        
        Returns:
            RouterSignal with expert activations
        """
        # Convert to numpy
        if isinstance(topk_indices, torch.Tensor):
            topk_indices = topk_indices.cpu().numpy()
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        if isinstance(selected_weights, torch.Tensor):
            selected_weights = selected_weights.cpu().numpy()
        
        indices = np.asarray(topk_indices, dtype=np.int32)
        
        # Infer dimensions
        if indices.ndim == 2:
            num_tokens, k = indices.shape
        elif indices.ndim == 1:
            num_tokens = indices.shape[0]
            k = topk if topk is not None else 1
            indices = indices.reshape(-1, k)
        else:
            raise ValueError(f"Unexpected indices shape: {indices.shape}")
        
        # Extract weights if using probability mode
        weights = None
        if self.use_probabilities and selected_weights is not None:
            weights = np.asarray(selected_weights, dtype=np.float32)
            if weights.shape != indices.shape:
                raise ValueError(
                    "selected_weights shape must match topk_indices: "
                    f"{weights.shape} != {indices.shape}"
                )
        elif self.use_probabilities and logits is not None:
            logits_arr = np.asarray(logits, dtype=np.float32)
            if logits_arr.ndim == 2:
                # Shape: (tokens, num_experts)
                # Normalize to get probabilities
                exp_logits = np.exp(logits_arr - np.max(logits_arr, axis=1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                # Extract top-k probabilities
                weights = np.take_along_axis(probs, indices, axis=1)
            elif logits_arr.ndim == 3:
                # Shape: (batch, tokens, num_experts)
                batch_size = logits_arr.shape[0]
                weights_list = []
                for b in range(batch_size):
                    batch_logits = logits_arr[b]
                    exp_logits = np.exp(batch_logits - np.max(batch_logits, axis=-1, keepdims=True))
                    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                    batch_weights = np.take_along_axis(probs, indices[b], axis=-1)
                    weights_list.append(batch_weights)
                weights = np.stack(weights_list)
        
        return RouterSignal(
            layer=layer,
            expert_indices=indices,
            expert_weights=weights,
            num_tokens=num_tokens,
        )

    def compute_g_signal(self, signal: RouterSignal) -> dict[int, float]:
        """
        Compute g[l,e] values from a RouterSignal.
        
        Returns:
            Dictionary mapping expert_id -> g[l,e] value
        """
        g_values: dict[int, list[float]] = {}
        
        indices = signal.expert_indices
        num_tokens = signal.num_tokens
        
        if self.use_probabilities and signal.expert_weights is not None:
            # Probability mode: total routed probability mass normalized by
            # the number of tokens. A mean over only the selections of each
            # expert would discard invocation frequency.
            weights = signal.expert_weights
            flat_indices = indices.reshape(-1)
            flat_weights = weights.reshape(-1)
            
            for expert_id, weight in zip(flat_indices, flat_weights):
                if expert_id not in g_values:
                    g_values[expert_id] = []
                g_values[expert_id].append(float(weight))
            
            if num_tokens > 0:
                return {
                    eid: float(np.sum(vals) / num_tokens)
                    for eid, vals in g_values.items()
                }
            return {eid: 0.0 for eid in g_values}
        else:
            # Indicator mode: count selections, normalize by tokens * topk
            flat_indices = indices.reshape(-1)
            k = indices.shape[-1] if indices.ndim > 0 else 1
            total_selections = num_tokens * k
            
            for expert_id in flat_indices:
                if expert_id not in g_values:
                    g_values[expert_id] = []
                g_values[expert_id].append(1.0)
            
            # Normalize
            if total_selections > 0:
                return {eid: len(vals) / total_selections for eid, vals in g_values.items()}
            return {eid: 0.0 for eid in g_values.keys()}
