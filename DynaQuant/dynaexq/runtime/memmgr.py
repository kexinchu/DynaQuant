"""
MemoryManager - Manage HBM/DRAM/SSD pools with LRU eviction
"""

import logging
import time
from typing import Dict, List, Optional, Set
from collections import OrderedDict
import threading

from .types import ExpertID, Residency

logger = logging.getLogger(__name__)


class MemoryPool:
    """
    Fixed-size memory pool with LRU eviction.
    """

    def __init__(self, name: str, capacity_bytes: int, slot_size_bytes: int):
        """
        Args:
            name: Pool name for logging
            capacity_bytes: Total pool capacity
            slot_size_bytes: Size of each slot (expert weights)
        """
        self.name = name
        self.capacity_bytes = capacity_bytes
        self.slot_size_bytes = slot_size_bytes
        self.max_slots = capacity_bytes // slot_size_bytes

        # OrderedDict for LRU: ExpertID -> Residency
        self.residents: OrderedDict[ExpertID, Residency] = OrderedDict()

        # Available slots
        self.free_slots = self.max_slots

        logger.info(
            f"MemoryPool '{name}': {capacity_bytes / 1e9:.2f} GB, "
            f"{self.max_slots} slots of {slot_size_bytes / 1e6:.2f} MB each"
        )

    def allocate(self, expert: ExpertID, residency: Residency) -> bool:
        """
        Allocate a slot for an expert.

        Returns:
            True if allocated, False if no space available
        """
        if expert in self.residents:
            # Already present, move to end (most recently used)
            self.residents.move_to_end(expert)
            self.residents[expert] = residency
            return True

        if self.free_slots > 0:
            self.residents[expert] = residency
            self.free_slots -= 1
            return True

        return False

    def evict_lru(self) -> Optional[ExpertID]:
        """
        Evict the least recently used expert.

        Returns:
            ExpertID of evicted expert, or None if pool is empty
        """
        if not self.residents:
            return None

        # Pop from front (least recently used)
        expert_id, _ = self.residents.popitem(last=False)
        self.free_slots += 1

        logger.debug(f"Evicted {expert_id} from {self.name}")
        return expert_id

    def remove(self, expert: ExpertID) -> bool:
        """
        Remove an expert from the pool.

        Returns:
            True if removed, False if not present
        """
        if expert in self.residents:
            del self.residents[expert]
            self.free_slots += 1
            return True
        return False

    def touch(self, expert: ExpertID) -> bool:
        """
        Mark expert as recently used (move to end of LRU).

        Returns:
            True if expert is in pool, False otherwise
        """
        if expert in self.residents:
            self.residents.move_to_end(expert)
            self.residents[expert].update_timestamp()
            return True
        return False

    def get(self, expert: ExpertID) -> Optional[Residency]:
        """Get residency info for an expert"""
        return self.residents.get(expert)

    def contains(self, expert: ExpertID) -> bool:
        """Check if expert is in pool"""
        return expert in self.residents

    def get_usage(self) -> float:
        """Get pool utilization (0.0 to 1.0)"""
        return (self.max_slots - self.free_slots) / self.max_slots if self.max_slots > 0 else 0.0

    def get_statistics(self) -> Dict:
        """Get pool statistics"""
        return {
            "name": self.name,
            "capacity_bytes": self.capacity_bytes,
            "slot_size_bytes": self.slot_size_bytes,
            "max_slots": self.max_slots,
            "used_slots": self.max_slots - self.free_slots,
            "free_slots": self.free_slots,
            "utilization": self.get_usage(),
        }


class MemoryManager:
    """
    Manages three memory pools: Hot (HBM W4), Cold (HBM W2), and Transient (staging).

    Also tracks experts in DRAM and SSD tiers.
    """

    def __init__(
        self,
        hot_pool_gb: float = 10.0,
        cold_pool_gb: float = 5.0,
        transient_pool_mb: float = 2048.0,
        w4_expert_size_mb: float = 256.0,
        w2_expert_size_mb: float = 64.0,
    ):
        """
        Args:
            hot_pool_gb: Size of hot pool (W4 experts) in GB
            cold_pool_gb: Size of cold pool (W2 experts) in GB
            transient_pool_mb: Size of transient staging pool in MB
            w4_expert_size_mb: Size of a single W4 expert in MB
            w2_expert_size_mb: Size of a single W2 expert in MB
        """
        self.w4_size = int(w4_expert_size_mb * 1e6)
        self.w2_size = int(w2_expert_size_mb * 1e6)

        # Create three pools
        self.hot_pool = MemoryPool(
            "HotPool",
            int(hot_pool_gb * 1e9),
            self.w4_size
        )

        self.cold_pool = MemoryPool(
            "ColdPool",
            int(cold_pool_gb * 1e9),
            self.w2_size
        )

        self.transient_pool = MemoryPool(
            "TransientPool",
            int(transient_pool_mb * 1e6),
            max(self.w4_size, self.w2_size)  # Must fit largest expert
        )

        # Track experts in lower tiers
        self.dram_experts: Dict[ExpertID, Residency] = {}
        self.ssd_experts: Dict[ExpertID, Residency] = {}

        # Global residency map
        self.residency_map: Dict[ExpertID, Residency] = {}

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.eviction_count = 0
        self.allocation_count = 0

        logger.info(
            f"MemoryManager initialized: "
            f"Hot={hot_pool_gb:.1f}GB, Cold={cold_pool_gb:.1f}GB, "
            f"Transient={transient_pool_mb:.0f}MB"
        )

    def reserve_hot(self, expert: ExpertID, nbytes: int) -> bool:
        """
        Reserve space in hot pool, evicting LRU if needed.

        Returns:
            True if successfully reserved
        """
        with self.lock:
            # If already in hot pool, just touch it
            if self.hot_pool.contains(expert):
                self.hot_pool.touch(expert)
                return True

            # Try to allocate
            residency = Residency(
                bitwidth="W4",
                location="HBM",
                bytes=nbytes
            )

            if self.hot_pool.allocate(expert, residency):
                self.allocation_count += 1
                self.residency_map[expert] = residency
                return True

            # Pool full, try to evict LRU
            evicted = self.hot_pool.evict_lru()
            if evicted:
                self.eviction_count += 1
                # Move evicted expert to DRAM or cold pool
                self._demote_to_dram(evicted)

                # Now allocate
                if self.hot_pool.allocate(expert, residency):
                    self.allocation_count += 1
                    self.residency_map[expert] = residency
                    return True

            return False

    def reserve_cold(self, expert: ExpertID, nbytes: int) -> bool:
        """Reserve space in cold pool, evicting LRU if needed"""
        with self.lock:
            if self.cold_pool.contains(expert):
                self.cold_pool.touch(expert)
                return True

            residency = Residency(
                bitwidth="W2",
                location="HBM",
                bytes=nbytes
            )

            if self.cold_pool.allocate(expert, residency):
                self.allocation_count += 1
                self.residency_map[expert] = residency
                return True

            # Evict and retry
            evicted = self.cold_pool.evict_lru()
            if evicted:
                self.eviction_count += 1
                self._demote_to_dram(evicted)

                if self.cold_pool.allocate(expert, residency):
                    self.allocation_count += 1
                    self.residency_map[expert] = residency
                    return True

            return False

    def place(self, expert: ExpertID, residency: Residency) -> None:
        """Place an expert with given residency info"""
        with self.lock:
            if residency.location == "HBM":
                if residency.bitwidth == "W4":
                    self.hot_pool.allocate(expert, residency)
                else:
                    self.cold_pool.allocate(expert, residency)
            elif residency.location == "DRAM":
                self.dram_experts[expert] = residency
            elif residency.location == "SSD":
                self.ssd_experts[expert] = residency

            self.residency_map[expert] = residency

    def get_residency(self, expert: ExpertID) -> Optional[Residency]:
        """Get current residency info for an expert"""
        with self.lock:
            return self.residency_map.get(expert)

    def evict_hot(self) -> List[ExpertID]:
        """Evict all hot pool experts (for emergency memory management)"""
        with self.lock:
            evicted = []
            while True:
                expert = self.hot_pool.evict_lru()
                if expert is None:
                    break
                evicted.append(expert)
                self._demote_to_dram(expert)
            return evicted

    def _demote_to_dram(self, expert: ExpertID) -> None:
        """Move expert from HBM to DRAM"""
        residency = self.residency_map.get(expert)
        if residency:
            residency.location = "DRAM"
            residency.hbm_ptr = None
            self.dram_experts[expert] = residency
            self.residency_map[expert] = residency

    def get_hbm_pressure(self) -> float:
        """
        Get HBM memory pressure (0.0 to 1.0).
        Returns max utilization across hot and cold pools.
        """
        with self.lock:
            return max(
                self.hot_pool.get_usage(),
                self.cold_pool.get_usage()
            )

    def get_statistics(self) -> Dict:
        """Get memory manager statistics"""
        with self.lock:
            return {
                "hot_pool": self.hot_pool.get_statistics(),
                "cold_pool": self.cold_pool.get_statistics(),
                "transient_pool": self.transient_pool.get_statistics(),
                "dram_experts": len(self.dram_experts),
                "ssd_experts": len(self.ssd_experts),
                "total_experts": len(self.residency_map),
                "eviction_count": self.eviction_count,
                "allocation_count": self.allocation_count,
                "hbm_pressure": self.get_hbm_pressure(),
            }
