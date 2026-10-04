"""One device-memory arena for every expert representation.

The arena has no layer, representation-tier, or lease-purpose partitions.
Allocations are byte extents carved from one or more preallocated segments.
Free extents are kept in address order, scanned for a best fit, and coalesced
by address when they are returned.
"""

from __future__ import annotations

import bisect
import itertools
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch


class ExtentState(str, Enum):
    FREE = "free"
    RESERVED = "reserved"
    PUBLISHED = "published"
    RECLAIM_PENDING = "reclaim_pending"
    RELEASED = "released"


@dataclass
class ArenaExtent:
    """A contiguous byte range in one arena segment."""

    segment_id: int
    offset: int
    capacity_bytes: int
    requested_bytes: int
    allocation_id: int
    state: ExtentState
    tensor: torch.Tensor

    @property
    def nbytes(self) -> int:
        """Bytes charged to the expert-memory budget."""
        return self.capacity_bytes


class SharedArenaAllocator:
    """Thread-safe, coalescing extent allocator over one logical arena.

    ``segment_bytes`` may contain more than one entry when the device cannot
    provide one allocation of ``capacity_bytes``. Segments share one capacity,
    but extents never coalesce across segment boundaries.
    """

    def __init__(
        self,
        capacity_bytes: int,
        device: torch.device,
        *,
        alignment: int = 256,
        segment_bytes: Iterable[int] | None = None,
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError("arena capacity must be positive")
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("alignment must be a positive power of two")
        sizes = list(segment_bytes) if segment_bytes is not None else [capacity_bytes]
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("arena segments must be positive")
        if sum(sizes) != capacity_bytes:
            raise ValueError("arena segment bytes must sum to capacity")

        self.capacity_bytes = capacity_bytes
        self.device = device
        self.alignment = alignment
        self._segments = [
            torch.empty(size, dtype=torch.uint8, device=device) for size in sizes
        ]
        self._lock = threading.RLock()
        self._allocation_ids = itertools.count(1)
        self._free_by_segment: dict[int, list[tuple[int, int]]] = {
            segment_id: [(0, size)] for segment_id, size in enumerate(sizes)
        }
        self._allocated: dict[int, ArenaExtent] = {}

    def rounded_bytes(self, requested_bytes: int) -> int:
        if requested_bytes <= 0:
            raise ValueError("allocation size must be positive")
        return (requested_bytes + self.alignment - 1) & ~(self.alignment - 1)

    def allocate(self, requested_bytes: int) -> ArenaExtent | None:
        """Bind the smallest free extent that can hold ``requested_bytes``."""
        charged = self.rounded_bytes(requested_bytes)
        with self._lock:
            best: tuple[int, int, int] | None = None
            for segment_id, ranges in self._free_by_segment.items():
                for index, (_offset, length) in enumerate(ranges):
                    if length < charged:
                        continue
                    candidate = (length, segment_id, index)
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                return None
            _length, segment_id, index = best
            offset, length = self._free_by_segment[segment_id].pop(index)
            if length > charged:
                self._insert_free(segment_id, offset + charged, length - charged)
            allocation_id = next(self._allocation_ids)
            extent = ArenaExtent(
                segment_id=segment_id,
                offset=offset,
                capacity_bytes=charged,
                requested_bytes=requested_bytes,
                allocation_id=allocation_id,
                state=ExtentState.RESERVED,
                tensor=self._segments[segment_id][offset : offset + charged],
            )
            self._allocated[allocation_id] = extent
            return extent

    def mark_published(self, extent: ArenaExtent) -> None:
        with self._lock:
            current = self._current(extent)
            if current.state != ExtentState.RESERVED:
                raise RuntimeError("only a reserved extent can be published")
            current.state = ExtentState.PUBLISHED

    def mark_reclaim_pending(self, extent: ArenaExtent) -> None:
        with self._lock:
            current = self._current(extent)
            if current.state != ExtentState.PUBLISHED:
                raise RuntimeError("only a published extent can await reclaim")
            current.state = ExtentState.RECLAIM_PENDING

    def free(self, extent: ArenaExtent) -> None:
        with self._lock:
            current = self._current(extent)
            del self._allocated[current.allocation_id]
            current.state = ExtentState.RELEASED
            self._insert_free(
                current.segment_id, current.offset, current.capacity_bytes
            )
            self._coalesce(current.segment_id)

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            state_bytes = {state.value: 0 for state in ExtentState}
            for extent in self._allocated.values():
                state_bytes[extent.state.value] += extent.capacity_bytes
            free_ranges = [
                length
                for ranges in self._free_by_segment.values()
                for _offset, length in ranges
            ]
            free_bytes = sum(free_ranges)
            largest = max(free_ranges, default=0)
            occupied = sum(state_bytes.values())
            if occupied + free_bytes != self.capacity_bytes:
                raise AssertionError("arena accounting does not sum to capacity")
            fragmentation = 0.0 if free_bytes == 0 else 1.0 - largest / free_bytes
            return {
                "capacity_bytes": self.capacity_bytes,
                "reserved_bytes": state_bytes[ExtentState.RESERVED.value],
                "published_bytes": state_bytes[ExtentState.PUBLISHED.value],
                "reclaim_pending_bytes": state_bytes[
                    ExtentState.RECLAIM_PENDING.value
                ],
                "free_bytes": free_bytes,
                "largest_free_extent_bytes": largest,
                "external_fragmentation": fragmentation,
                "allocation_count": len(self._allocated),
                "segment_count": len(self._segments),
            }

    def _current(self, extent: ArenaExtent) -> ArenaExtent:
        current = self._allocated.get(extent.allocation_id)
        if current is not extent:
            raise RuntimeError("stale or already released arena extent")
        return current

    def _insert_free(self, segment_id: int, offset: int, length: int) -> None:
        ranges = self._free_by_segment[segment_id]
        bisect.insort(ranges, (offset, length))

    def _coalesce(self, segment_id: int) -> None:
        ranges = self._free_by_segment[segment_id]
        if len(ranges) < 2:
            return
        merged: list[tuple[int, int]] = []
        for offset, length in ranges:
            if merged and merged[-1][0] + merged[-1][1] == offset:
                previous_offset, previous_length = merged[-1]
                merged[-1] = (previous_offset, previous_length + length)
            else:
                merged.append((offset, length))
        self._free_by_segment[segment_id] = merged


__all__ = ["ArenaExtent", "ExtentState", "SharedArenaAllocator"]
