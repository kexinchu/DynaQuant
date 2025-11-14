from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from .types import ExpertID, Residency, ResidencyLocation


@dataclass
class PoolConfig:
    hot_capacity_bytes: int
    cold_capacity_bytes: int
    transient_capacity_bytes: int
    hot_slots: int = 0
    hot_slot_bytes: int = 0
    cold_slots: int = 0
    cold_slot_bytes: int = 0


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

        self._have_hot_slots = config.hot_slots > 0 and config.hot_slot_bytes > 0
        self._have_cold_slots = config.cold_slots > 0 and config.cold_slot_bytes > 0

        self._hot_slot_size = config.hot_slot_bytes
        self._cold_slot_size = config.cold_slot_bytes

        self._hot_free_slots: Deque[int] = deque(
            range(config.hot_slots)) if self._have_hot_slots else deque()
        self._cold_free_slots: Deque[int] = deque(
            range(config.cold_slots)) if self._have_cold_slots else deque()

        self._hot_pending_slots: Dict[ExpertID, int] = {}
        self._cold_pending_slots: Dict[ExpertID, int] = {}
        self._hot_occupied_slots: Dict[ExpertID, int] = {}
        self._cold_occupied_slots: Dict[ExpertID, int] = {}

    # --------------------------------------------------------------------- #
    # Allocation helpers
    # --------------------------------------------------------------------- #
    def reserve_hot(self, expert: ExpertID, nbytes: int) -> Tuple[bool, Optional[int]]:
        """Reserve space in the hot pool, evicting LRU experts if needed."""
        with self._lock:
            if self._have_hot_slots:
                if nbytes > self._hot_slot_size:
                    raise ValueError(
                        f"Expert {expert} requires {nbytes} bytes, exceeds hot slot size {self._hot_slot_size}"
                    )

                pending = self._hot_pending_slots.get(expert)
                if pending is not None:
                    return True, pending

                if not self._hot_free_slots:
                    self._ensure_hot_slot()
                if not self._hot_free_slots:
                    return False, None

                slot_id = self._hot_free_slots.popleft()
                self._hot_pending_slots[expert] = slot_id
                self._hot_reserved += self._hot_slot_size
                return True, slot_id

            # Legacy byte-granular reservation
            capacity = self._config.hot_capacity_bytes
            if nbytes > capacity:
                return False, None

            current_pending = self._pending_hot.pop(expert, 0)
            if current_pending:
                self._hot_reserved = max(
                    0, self._hot_reserved - current_pending)

            self._ensure_hot_capacity(required_bytes=nbytes)
            if self._hot_usage + self._hot_reserved + nbytes > capacity:
                if current_pending:
                    self._pending_hot[expert] = current_pending
                    self._hot_reserved += current_pending
                return False, None

            self._pending_hot[expert] = nbytes
            self._hot_reserved += nbytes
            return True, None

    def reserve_cold(self, expert: ExpertID, nbytes: int) -> Tuple[bool, Optional[int]]:
        with self._lock:
            if self._have_cold_slots:
                if nbytes > self._cold_slot_size:
                    raise ValueError(
                        f"Expert {expert} requires {nbytes} bytes, exceeds cold slot size {self._cold_slot_size}"
                    )
                pending = self._cold_pending_slots.get(expert)
                if pending is not None:
                    return True, pending
                if not self._cold_free_slots:
                    return False, None
                slot_id = self._cold_free_slots.popleft()
                self._cold_pending_slots[expert] = slot_id
                self._cold_usage += self._cold_slot_size
                return True, slot_id

            # Legacy behaviour: allow unlimited cold storage
            return True, None

    def cancel_hot_reservation(self, expert: ExpertID) -> None:
        with self._lock:
            if self._have_hot_slots:
                slot = self._hot_pending_slots.pop(expert, None)
                if slot is not None:
                    self._hot_free_slots.appendleft(slot)
                    self._hot_reserved = max(
                        0, self._hot_reserved - self._hot_slot_size)
                return
            pending = self._pending_hot.pop(expert, None)
            if pending:
                self._hot_reserved = max(0, self._hot_reserved - pending)

    def cancel_cold_reservation(self, expert: ExpertID) -> None:
        with self._lock:
            if self._have_cold_slots:
                slot = self._cold_pending_slots.pop(expert, None)
                if slot is not None:
                    self._cold_free_slots.appendleft(slot)
                    self._cold_usage = max(
                        0, self._cold_usage - self._cold_slot_size)

    def place(self, expert: ExpertID, residency: Residency) -> None:
        """Record an expert residency update."""
        with self._lock:
            prev = self._residency.get(expert)
            if prev is not None:
                self._decrement_usage(prev, expert)

            if residency.location is ResidencyLocation.HBM:
                self._hot_lru.pop(expert, None)
                self._hot_lru[expert] = time.time()
                if self._have_hot_slots:
                    slot = residency.tags.get("slot_id")
                    pending_slot = self._hot_pending_slots.pop(expert, None)
                    if slot is None:
                        slot = pending_slot
                    if slot is None:
                        raise RuntimeError(
                            f"Missing hot slot assignment for {expert}")
                    self._hot_occupied_slots[expert] = int(slot)
                    residency.tags["slot_id"] = int(slot)
                    if pending_slot is not None:
                        self._hot_reserved = max(
                            0, self._hot_reserved - self._hot_slot_size)
                    self._hot_usage += self._hot_slot_size
            else:
                self._hot_lru.pop(expert, None)
                if residency.location is ResidencyLocation.DRAM and self._have_cold_slots:
                    slot = residency.tags.get("slot_id")
                    pending_slot = self._cold_pending_slots.pop(expert, None)
                    if slot is None:
                        slot = pending_slot
                    if slot is None:
                        raise RuntimeError(
                            f"Missing cold slot assignment for {expert}")
                    self._cold_occupied_slots[expert] = int(slot)
                    residency.tags["slot_id"] = int(slot)
                    # cold usage already incremented during reservation

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
            capacity = (
                self._config.hot_slots * self._hot_slot_size
                if self._have_hot_slots
                else self._config.hot_capacity_bytes
            )
            return self._hot_usage / max(1, capacity)

    def cold_occupancy(self) -> float:
        with self._lock:
            capacity = (
                self._config.cold_slots * self._cold_slot_size
                if self._have_cold_slots
                else self._config.cold_capacity_bytes
            )
            return self._cold_usage / max(1, capacity)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_hot_capacity(self, required_bytes: int) -> None:
        while (
            self._hot_usage + self._hot_reserved +
                required_bytes > self._config.hot_capacity_bytes
            and self._hot_lru
        ):
            expert, _ = self._hot_lru.popitem(last=False)
            residency = self._residency.get(expert)
            if residency is None:
                continue

            self._decrement_usage(residency, expert)
            self._residency.pop(expert, None)
            self._recent_evictions.append(expert)

    def _ensure_hot_slot(self) -> None:
        while not self._hot_free_slots and self._hot_lru:
            expert, _ = self._hot_lru.popitem(last=False)
            residency = self._residency.pop(expert, None)
            if residency is None:
                continue
            self._decrement_usage(residency, expert)
            self._recent_evictions.append(expert)

    def _increment_usage(self, residency: Residency) -> None:
        if residency.location is ResidencyLocation.HBM:
            if self._have_hot_slots:
                self._hot_usage = min(
                    self._config.hot_slots * self._hot_slot_size,
                    self._hot_usage + self._hot_slot_size,
                )
            else:
                self._hot_usage += residency.bytes
        elif residency.location is ResidencyLocation.DRAM:
            if not self._have_cold_slots:
                self._cold_usage += residency.bytes
        else:
            self._transient_usage += residency.bytes

    def _decrement_usage(self, residency: Residency, expert: ExpertID) -> None:
        if residency.location is ResidencyLocation.HBM:
            if self._have_hot_slots:
                slot = self._hot_occupied_slots.pop(expert, None)
                if slot is not None:
                    self._hot_free_slots.append(slot)
                    self._hot_usage = max(
                        0, self._hot_usage - self._hot_slot_size)
            else:
                self._hot_usage = max(0, self._hot_usage - residency.bytes)
                pending = self._pending_hot.pop(expert, None)
                if pending:
                    self._hot_reserved = max(0, self._hot_reserved - pending)
        elif residency.location is ResidencyLocation.DRAM:
            if self._have_cold_slots:
                slot = self._cold_occupied_slots.pop(expert, None)
                if slot is not None:
                    self._cold_free_slots.append(slot)
                    self._cold_usage = max(
                        0, self._cold_usage - self._cold_slot_size)
            else:
                self._cold_usage = max(0, self._cold_usage - residency.bytes)
        else:
            self._transient_usage = max(
                0, self._transient_usage - residency.bytes)


__all__ = ["MemoryManager", "PoolConfig"]
