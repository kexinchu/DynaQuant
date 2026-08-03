from __future__ import annotations

import torch

from dynaexq.core.config import Tier
from dynaexq.core.memory_pool import PoolAllocator


def test_zero_capacity_pool_allocates_no_hidden_block():
    allocator = PoolAllocator(
        num_layers=1,
        hi_pool_sizes=[0],
        lo_pool_sizes=[0],
        device=torch.device("cpu"),
        block_size_bytes=64,
    )
    assert allocator.alloc(0, Tier.HI) is None
    assert allocator.alloc(0, Tier.LO) is None


def test_per_tier_block_sizes_match_expert_footprints():
    allocator = PoolAllocator(
        num_layers=1,
        hi_pool_sizes=[300],
        lo_pool_sizes=[100],
        device=torch.device("cpu"),
        hi_block_sizes=[100],
        lo_block_sizes=[25],
    )
    hi = allocator.alloc(0, Tier.HI)
    lo = allocator.alloc(0, Tier.LO)
    assert hi.tensor.numel() == 100
    assert lo.tensor.numel() == 25


def test_global_staging_pool_backs_full_resident_pool_and_frees_by_origin():
    allocator = PoolAllocator(
        num_layers=1,
        hi_pool_sizes=[100],
        lo_pool_sizes=[25],
        device=torch.device("cpu"),
        hi_block_sizes=[100],
        lo_block_sizes=[25],
        staging_pool_size_bytes=200,
        staging_block_size_bytes=100,
    )
    resident = allocator.alloc(0, Tier.HI)
    staging = allocator.alloc(0, Tier.HI)
    assert resident is not None and resident.pool_name == "hi:0"
    assert staging is not None and staging.pool_name == "staging"
    assert allocator.snapshot()["staging"]["used_blocks"] == 1

    allocator.free_block(staging)
    assert allocator.snapshot()["staging"]["used_blocks"] == 0
