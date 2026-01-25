from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

from .types import ExpertID, Residency, ResidencyLocation


@dataclass
class PoolConfig:
    hot_capacity_bytes: int
    cold_capacity_bytes: int
    transient_capacity_bytes: int


class MemoryManager:
    """Tiered pool allocator coordinating expert residency."""

    def __init__(self, config: PoolConfig) -> None:
        self._config = config
        self._residency: Dict[ExpertID, Residency] = {}

        self._hot_usage = 0
        self._hot_reserved = 0
        self._cold_usage = 0
        self._transient_usage = 0
        self._hot_lru: OrderedDict[ExpertID, float] = OrderedDict()
        self._pending_hot: Dict[ExpertID, int] = {}
        self._recent_evictions: List[ExpertID] = []
        self._lock = threading.Lock()

    # --------------------------------------------------------------------- #
    # Allocation helpers
    # --------------------------------------------------------------------- #
    def reserve_hot(self, expert: ExpertID, nbytes: int) -> bool:
        """Reserve space in the hot pool, evicting LRU experts if needed."""
        with self._lock:
            capacity = self._config.hot_capacity_bytes
            if nbytes > capacity:
                return False

            current_pending = self._pending_hot.pop(expert, 0)
            if current_pending:
                self._hot_reserved = max(0, self._hot_reserved - current_pending)

            self._ensure_hot_capacity(required_bytes=nbytes)
            if self._hot_usage + self._hot_reserved + nbytes > capacity:
                if current_pending:
                    self._pending_hot[expert] = current_pending
                    self._hot_reserved += current_pending
                return False

            self._pending_hot[expert] = nbytes
            self._hot_reserved += nbytes
            return True

    def place(self, expert: ExpertID, residency: Residency) -> None:
        """Record an expert residency update."""
        with self._lock:
            prev = self._residency.get(expert)
            if prev is not None:
                self._decrement_usage(prev, expert)

            if residency.location is ResidencyLocation.HBM:
                self._hot_lru.pop(expert, None)
                self._hot_lru[expert] = time.time()
            else:
                self._hot_lru.pop(expert, None)

            self._residency[expert] = residency
            self._increment_usage(residency)

            if expert in self._pending_hot:
                pending_bytes = self._pending_hot.pop(expert)
                self._hot_reserved = max(0, self._hot_reserved - pending_bytes)

    def evict_hot(self) -> List[ExpertID]:
        """Return experts evicted from the hot pool in the last reserve call."""
        with self._lock:
            evicted, self._recent_evictions = self._recent_evictions, []
            return evicted

    # ------------------------------------------------------------------ #
    # Query helpers
    # ------------------------------------------------------------------ #
    def residency(self, expert: ExpertID) -> Optional[Residency]:
        with self._lock:
            return self._residency.get(expert)

    def hot_occupancy(self) -> float:
        with self._lock:
            return self._hot_usage / max(1, self._config.hot_capacity_bytes)

    def cold_occupancy(self) -> float:
        with self._lock:
            return self._cold_usage / max(1, self._config.cold_capacity_bytes)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_hot_capacity(self, required_bytes: int) -> None:
        while (
            self._hot_usage + self._hot_reserved + required_bytes > self._config.hot_capacity_bytes
            and self._hot_lru
        ):
            expert, _ = self._hot_lru.popitem(last=False)
            residency = self._residency.get(expert)
            if residency is None:
                continue

            self._decrement_usage(residency, expert)
            self._residency.pop(expert, None)
            self._recent_evictions.append(expert)

    def _increment_usage(self, residency: Residency) -> None:
        if residency.location is ResidencyLocation.HBM:
            self._hot_usage += residency.bytes
        elif residency.location is ResidencyLocation.DRAM:
            self._cold_usage += residency.bytes
        else:
            self._transient_usage += residency.bytes

    def _decrement_usage(self, residency: Residency, expert: ExpertID) -> None:
        if residency.location is ResidencyLocation.HBM:
            self._hot_usage = max(0, self._hot_usage - residency.bytes)
            pending = self._pending_hot.pop(expert, None)
            if pending:
                self._hot_reserved = max(0, self._hot_reserved - pending)
        elif residency.location is ResidencyLocation.DRAM:
            self._cold_usage = max(0, self._cold_usage - residency.bytes)
        else:
            self._transient_usage = max(0, self._transient_usage - residency.bytes)


__all__ = ["MemoryManager", "PoolConfig"]

