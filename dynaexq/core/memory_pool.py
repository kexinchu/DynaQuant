"""
MemoryPool: deterministic pool-based allocation.

Provides fixed-size blocks from pre-allocated pools to avoid fragmentation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import torch

from .config import Tier


@dataclass
class PoolBlock:
    """A block in a memory pool."""
    block_id: int
    tensor: torch.Tensor
    in_use: bool = False
    pool_name: str = ""


class MemoryPool:
    """
    Fixed-size memory pool for a single tier and layer.
    
    Provides deterministic allocation without cudaMalloc.
    """
    
    def __init__(
        self,
        block_size_bytes: int,
        num_blocks: int,
        device: torch.device,
        name: str = "",
    ):
        """
        Args:
            block_size_bytes: Size of each block
            num_blocks: Number of blocks in pool
            device: Device to allocate on
        """
        self.block_size_bytes = block_size_bytes
        self.num_blocks = num_blocks
        self.device = device
        self.name = name
        
        # Pre-allocate blocks
        self._blocks: list[PoolBlock] = []
        for i in range(num_blocks):
            # Allocate as uint8 bytes for flexible storage
            tensor = torch.empty(
                (block_size_bytes,),
                dtype=torch.uint8,
                device=device,
            )
            self._blocks.append(
                PoolBlock(
                    block_id=i,
                    tensor=tensor,
                    in_use=False,
                    pool_name=name,
                )
            )
        
        self._free_queue: list[int] = list(range(num_blocks))
        self._lock = threading.Lock()
    
    def alloc(self) -> Optional[PoolBlock]:
        """Allocate a free block (returns None if pool exhausted)."""
        with self._lock:
            if not self._free_queue:
                return None
            block_id = self._free_queue.pop(0)
            block = self._blocks[block_id]
            block.in_use = True
            return block
    
    def free(self, block_id: int) -> None:
        """Free a block back to the pool."""
        with self._lock:
            if block_id < 0 or block_id >= self.num_blocks:
                return
            block = self._blocks[block_id]
            if block.in_use:
                block.in_use = False
                if block_id not in self._free_queue:
                    self._free_queue.append(block_id)
    
    def occupancy(self) -> float:
        """Get pool occupancy (0.0 to 1.0)."""
        with self._lock:
            return (self.num_blocks - len(self._free_queue)) / max(1, self.num_blocks)

    def snapshot(self) -> dict[str, int | float]:
        """Return allocation counters without exposing mutable blocks."""
        with self._lock:
            free_blocks = len(self._free_queue)
            used_blocks = self.num_blocks - free_blocks
            return {
                "block_size_bytes": self.block_size_bytes,
                "total_blocks": self.num_blocks,
                "used_blocks": used_blocks,
                "free_blocks": free_blocks,
                "allocated_bytes": self.block_size_bytes * self.num_blocks,
                "used_bytes": self.block_size_bytes * used_blocks,
                "occupancy": used_blocks / max(1, self.num_blocks),
            }


class PoolAllocator:
    """
    Multi-tier, multi-layer pool allocator.
    
    Manages separate pools for HI/LO tiers per layer.
    """
    
    def __init__(
        self,
        num_layers: int,
        hi_pool_sizes: list[int],  # bytes per layer
        lo_pool_sizes: list[int],  # bytes per layer
        device: torch.device,
        block_size_bytes: int = 1024 * 1024,  # 1MB blocks
        hi_block_sizes: Optional[list[int]] = None,
        lo_block_sizes: Optional[list[int]] = None,
        staging_pool_size_bytes: int = 0,
        staging_block_size_bytes: Optional[int] = None,
    ):
        """
        Args:
            num_layers: Number of layers
            hi_pool_sizes: HI pool size per layer (bytes)
            lo_pool_sizes: LO pool size per layer (bytes)
            device: Device to allocate on
            block_size_bytes: Backward-compatible common block size.
            hi_block_sizes: Exact HI expert block size per layer.
            lo_block_sizes: Exact LO expert block size per layer.
            staging_pool_size_bytes: Global transient-pool capacity.
            staging_block_size_bytes: Size of each global transient block.
        """
        self.num_layers = num_layers
        self.device = device
        self.block_size_bytes = block_size_bytes
        
        # Create pools per layer and tier
        self._hi_pools: list[MemoryPool] = []
        self._lo_pools: list[MemoryPool] = []
        
        for layer in range(num_layers):
            hi_size = hi_pool_sizes[layer] if layer < len(hi_pool_sizes) else 0
            lo_size = lo_pool_sizes[layer] if layer < len(lo_pool_sizes) else 0
            
            # A zero-capacity pool must contain zero blocks. Allocating one
            # block here would exceed the configured HBM budget before the
            # first request arrives.
            hi_block_size = (
                hi_block_sizes[layer]
                if hi_block_sizes is not None
                else block_size_bytes
            )
            lo_block_size = (
                lo_block_sizes[layer]
                if lo_block_sizes is not None
                else block_size_bytes
            )
            if hi_block_size <= 0 or lo_block_size <= 0:
                raise ValueError("pool block sizes must be positive")
            hi_blocks = max(0, hi_size // hi_block_size)
            lo_blocks = max(0, lo_size // lo_block_size)
            
            self._hi_pools.append(
                MemoryPool(hi_block_size, hi_blocks, device, f"hi:{layer}")
            )
            self._lo_pools.append(
                MemoryPool(lo_block_size, lo_blocks, device, f"lo:{layer}")
            )

        if staging_pool_size_bytes < 0:
            raise ValueError("staging_pool_size_bytes must be non-negative")
        if staging_pool_size_bytes:
            if staging_block_size_bytes is None or staging_block_size_bytes <= 0:
                raise ValueError(
                    "positive staging_block_size_bytes is required for a staging pool"
                )
            staging_blocks = (
                staging_pool_size_bytes // staging_block_size_bytes
            )
            if staging_blocks <= 0:
                raise ValueError("staging pool is smaller than one staging block")
            self._staging_pool: Optional[MemoryPool] = MemoryPool(
                staging_block_size_bytes,
                staging_blocks,
                device,
                "staging",
            )
        else:
            self._staging_pool = None
    
    def alloc(self, layer: int, tier: Tier) -> Optional[PoolBlock]:
        """Allocate a block from the appropriate pool."""
        block = self.alloc_resident(layer, tier)
        if block is None and self._staging_pool is not None:
            block = self._staging_pool.alloc()
        return block

    def alloc_resident(self, layer: int, tier: Tier) -> Optional[PoolBlock]:
        """Allocate only from the requested layer/tier resident pool."""
        if layer < 0 or layer >= self.num_layers:
            return None
        return (
            self._hi_pools[layer].alloc()
            if tier == Tier.HI
            else self._lo_pools[layer].alloc()
        )
    
    def free(self, layer: int, tier: Tier, block_id: int) -> None:
        """Free a block back to the appropriate pool."""
        if layer < 0 or layer >= self.num_layers:
            return
        
        if tier == Tier.HI:
            self._hi_pools[layer].free(block_id)
        else:
            self._lo_pools[layer].free(block_id)

    def free_block(self, block: PoolBlock) -> None:
        """Return a block to its originating resident or staging pool."""
        if block.pool_name == "staging":
            if self._staging_pool is None:
                raise RuntimeError("staging block has no owning pool")
            self._staging_pool.free(block.block_id)
            return
        try:
            tier_name, layer_text = block.pool_name.split(":", maxsplit=1)
            layer = int(layer_text)
            tier = Tier.HI if tier_name == "hi" else Tier.LO
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                f"block has invalid pool identity {block.pool_name!r}"
            ) from exc
        self.free(layer, tier, block.block_id)
    
    def occupancy(self, layer: int, tier: Tier) -> float:
        """Get pool occupancy for a layer/tier."""
        if layer < 0 or layer >= self.num_layers:
            return 0.0
        
        if tier == Tier.HI:
            return self._hi_pools[layer].occupancy()
        else:
            return self._lo_pools[layer].occupancy()

    def snapshot(self) -> dict:
        """Return per-layer pool capacity and utilization."""
        result = {
            "hi": [pool.snapshot() for pool in self._hi_pools],
            "lo": [pool.snapshot() for pool in self._lo_pools],
        }
        result["staging"] = (
            self._staging_pool.snapshot()
            if self._staging_pool is not None
            else None
        )
        return result
