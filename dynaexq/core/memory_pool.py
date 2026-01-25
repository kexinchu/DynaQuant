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


class MemoryPool:
    """
    Fixed-size memory pool for a single tier and layer.
    
    Provides deterministic allocation without cudaMalloc.
    """
    
    def __init__(self, block_size_bytes: int, num_blocks: int, device: torch.device):
        """
        Args:
            block_size_bytes: Size of each block
            num_blocks: Number of blocks in pool
            device: Device to allocate on
        """
        self.block_size_bytes = block_size_bytes
        self.num_blocks = num_blocks
        self.device = device
        
        # Pre-allocate blocks
        self._blocks: list[PoolBlock] = []
        for i in range(num_blocks):
            # Allocate as uint8 bytes for flexible storage
            tensor = torch.empty(
                (block_size_bytes,),
                dtype=torch.uint8,
                device=device,
            )
            self._blocks.append(PoolBlock(block_id=i, tensor=tensor, in_use=False))
        
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
    ):
        """
        Args:
            num_layers: Number of layers
            hi_pool_sizes: HI pool size per layer (bytes)
            lo_pool_sizes: LO pool size per layer (bytes)
            device: Device to allocate on
            block_size_bytes: Block size for pool allocation
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
            
            hi_blocks = max(1, hi_size // block_size_bytes)
            lo_blocks = max(1, lo_size // block_size_bytes)
            
            self._hi_pools.append(
                MemoryPool(block_size_bytes, hi_blocks, device)
            )
            self._lo_pools.append(
                MemoryPool(block_size_bytes, lo_blocks, device)
            )
    
    def alloc(self, layer: int, tier: Tier) -> Optional[PoolBlock]:
        """Allocate a block from the appropriate pool."""
        if layer < 0 or layer >= self.num_layers:
            return None
        
        if tier == Tier.HI:
            return self._hi_pools[layer].alloc()
        else:
            return self._lo_pools[layer].alloc()
    
    def free(self, layer: int, tier: Tier, block_id: int) -> None:
        """Free a block back to the appropriate pool."""
        if layer < 0 or layer >= self.num_layers:
            return
        
        if tier == Tier.HI:
            self._hi_pools[layer].free(block_id)
        else:
            self._lo_pools[layer].free(block_id)
    
    def occupancy(self, layer: int, tier: Tier) -> float:
        """Get pool occupancy for a layer/tier."""
        if layer < 0 or layer >= self.num_layers:
            return 0.0
        
        if tier == Tier.HI:
            return self._hi_pools[layer].occupancy()
        else:
            return self._lo_pools[layer].occupancy()

