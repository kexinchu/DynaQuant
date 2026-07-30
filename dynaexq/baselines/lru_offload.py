"""
Blocking LRU expert-offload baseline.

This intentionally does *not* claim to implement ExpertFlow
(arXiv:2510.26730). ExpertFlow includes adaptive cross-layer prefetch,
two-tier cache coordination, and cache-aware token scheduling; a single LRU
cache with blocking copies is only a naive offload reference.

Design:
- All expert weights are kept in pinned host (CPU) memory at fp16.
- A GPU-side LRU cache holds up to ``gpu_expert_slots`` experts.
- On each token, the router selects top-k experts:
  - If the expert is in the GPU cache → hit (free)
  - If the expert is NOT in the GPU cache → miss → blocking H2D transfer
    + evict LRU if cache is full
- No prefetch, prediction, or cache-aware token scheduling.

The LRU cache wraps expert forward calls transparently so the model code
sees no difference between cached and non-cached experts.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_h2d_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(total, 1)


class LRUOffloadCache:
    """
    LRU cache of expert weight tensors on GPU.

    Args:
        gpu_expert_slots: Maximum number of experts held on GPU.
        device: GPU device to cache onto (default ``cuda:0``).
    """

    def __init__(
        self,
        gpu_expert_slots: int = 16,
        device: str = "cuda:0",
    ):
        self.gpu_expert_slots = gpu_expert_slots
        self.device = torch.device(device)

        # OrderedDict as LRU: most recently used at the END.
        # Key: (layer_idx, expert_idx), Value: dict[str, Tensor] on GPU.
        self._cache: OrderedDict[tuple[int, int], dict[str, torch.Tensor]] = OrderedDict()

        self.stats = CacheStats()

    def get_expert_weights(
        self,
        layer_idx: int,
        expert_idx: int,
        host_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Return GPU-resident copies of the expert's weight tensors.

        If cached, returns immediately (cache hit). Otherwise copies from
        pinned host memory → GPU (cache miss), evicting the LRU entry if
        the cache is full.

        Args:
            layer_idx: Decoder layer index.
            expert_idx: Expert index within the layer.
            host_weights: ``{"weight_name": cpu_tensor, ...}`` — the
                expert's weight tensors in pinned host memory.

        Returns:
            ``dict[str, Tensor]`` — same keys, tensors on ``self.device``.
        """
        key = (layer_idx, expert_idx)

        if key in self._cache:
            self._cache.move_to_end(key)
            self.stats.hits += 1
            return self._cache[key]

        # Cache miss: blocking H2D transfer.
        self.stats.misses += 1

        if len(self._cache) >= self.gpu_expert_slots:
            evict_key, evict_tensors = self._cache.popitem(last=False)
            for t in evict_tensors.values():
                del t
            self.stats.evictions += 1

        t0 = time.perf_counter()
        gpu_weights = {
            name: tensor.to(self.device, non_blocking=False)
            for name, tensor in host_weights.items()
        }
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.stats.total_h2d_ms += (time.perf_counter() - t0) * 1000

        self._cache[key] = gpu_weights
        return gpu_weights

    def clear(self) -> None:
        """Free all cached GPU tensors."""
        for tensors in self._cache.values():
            for t in tensors.values():
                del t
        self._cache.clear()

    def summary(self) -> dict:
        return {
            "capacity": self.gpu_expert_slots,
            "current_size": len(self._cache),
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate,
            "evictions": self.stats.evictions,
            "total_h2d_ms": round(self.stats.total_h2d_ms, 2),
        }


__all__ = ["LRUOffloadCache", "CacheStats"]
