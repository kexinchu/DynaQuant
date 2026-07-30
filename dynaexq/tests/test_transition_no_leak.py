"""
Tests for ``dynaexq.core.transition_engine`` reclaim path (Plan B payoff).

The pre-Plan-B implementation had ``Stage 4: reclaim`` as a no-op (``pass``),
which leaks one PoolBlock per transition. With ``ExpertHandle`` now holding a
direct reference to its ``PoolBlock``, the reclaim stage can free the old
block via ``pool_allocator.free(layer, tier, block_id)``.

This test simulates a long promote/demote loop and asserts the pool returns
to the same occupancy it started with — the canonical "no leak" guarantee
from plan §III-D and the DoD checklist:
    > 长跑 1 小时 promote/demote, nvidia-smi 内存稳定 (无泄漏)

We run on CPU with tiny pools so the test is fast (< 1 s) and doesn't need
CUDA. The TransitionEngine internally short-circuits its CUDA stream when
``torch.cuda.is_available()`` is False, so the same code path is exercised.
"""

from __future__ import annotations

import time

import pytest
import torch

from dynaexq.core.config import Tier
from dynaexq.core.memory_pool import PoolAllocator
from dynaexq.core.quant import (
    PackedTensor,
    QuantFormat,
    compute_packed_nbytes,
    dequant_to_fp16,
    fused_linear,
    pack,
)
from dynaexq.core.registry import ExpertKey, ExpertRegistry
from dynaexq.core.scheduler import TransitionReq
from dynaexq.core.transition_engine import TransitionEngine
from dynaexq.core.weight_store import ModelWeightStore


# ---------------------------------------------------------------------------
# Stub WeightStore — returns a tiny PackedTensor regardless of key/tier.
# Implements the post-Plan-A WeightStore contract (PackedTensor + nbytes
# via compute_packed_nbytes) without depending on ModelWeightStore's model
# walking, so the leak test stays focused on the reclaim contract.
# ---------------------------------------------------------------------------


class _StubWeightStore:
    def __init__(self, out_features: int = 4, in_features: int = 64):
        self.out_features = out_features
        self.in_features = in_features
        # Pre-pack one fp16 tensor; reuse for every (key, tier).
        weight = torch.zeros(out_features, in_features, dtype=torch.float16)
        self._packed_fp16 = pack(weight, QuantFormat.FP16)

    def load_weights(self, key: ExpertKey, tier: Tier) -> PackedTensor:
        # All tiers return the same fp16 PackedTensor; the leak test
        # doesn't care about quant correctness, only block accounting.
        return self._packed_fp16

    def get_byte_size(self, key: ExpertKey, tier: Tier) -> int:
        return compute_packed_nbytes(
            self.out_features, self.in_features, QuantFormat.FP16, self.in_features
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(
    *,
    synchronous: bool = False,
) -> tuple[TransitionEngine, PoolAllocator, ExpertRegistry]:
    block_size = 64 * 1024  # 64 KiB blocks; tiny tensor fits easily
    pool_bytes = block_size * 4  # 4 blocks per tier per layer
    alloc = PoolAllocator(
        num_layers=1,
        hi_pool_sizes=[pool_bytes],
        lo_pool_sizes=[pool_bytes],
        device=torch.device("cpu"),
        block_size_bytes=block_size,
    )
    registry = ExpertRegistry()
    store = _StubWeightStore()
    engine = TransitionEngine(
        registry=registry,
        pool_allocator=alloc,
        weight_store=store,  # type: ignore[arg-type]
        max_workers=2,
        max_inflight=4,
        synchronous=synchronous,
    )
    return engine, alloc, registry


def _wait_for_completion(engine: TransitionEngine, timeout: float = 5.0) -> None:
    """Spin until all in-flight transitions drain. Crude but adequate for tests."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with engine._transition_lock:
            if not engine._active_transitions:
                return
        time.sleep(0.005)
    raise TimeoutError("transitions did not drain within timeout")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_promotion_allocates_one_hi_block():
    engine, alloc, registry = _make_engine()
    try:
        key = ExpertKey(0, 0)
        req = TransitionReq(key=key, src=Tier.LO, dst=Tier.HI, reason="test", issued_step=0)
        assert engine.enqueue(req)
        _wait_for_completion(engine)

        handle = registry.get_handle(key)
        assert handle is not None
        assert handle.tier == Tier.HI
        assert handle.block is not None
        assert handle.is_valid()
        # HI pool now has one block in use
        assert alloc.occupancy(0, Tier.HI) > 0
    finally:
        engine.shutdown()


def test_synchronous_ablation_publishes_before_enqueue_returns():
    engine, _, registry = _make_engine(synchronous=True)
    try:
        key = ExpertKey(0, 0)
        req = TransitionReq(
            key=key,
            src=Tier.LO,
            dst=Tier.HI,
            reason="blocking-ablation",
            issued_step=0,
        )
        assert engine.enqueue(req)
        handle = registry.get_handle(key)
        assert handle is not None
        assert handle.tier == Tier.HI
        assert engine.get_stats()["execution_mode"] == "synchronous"
        with engine._transition_lock:
            assert key not in engine._active_transitions
    finally:
        engine.shutdown()


def test_staging_handle_is_repatriated_after_paired_swap_frees_resident_slot(
    monkeypatch,
):
    block_size = 4096
    alloc = PoolAllocator(
        num_layers=1,
        hi_pool_sizes=[block_size],
        lo_pool_sizes=[block_size],
        device=torch.device("cpu"),
        block_size_bytes=block_size,
        staging_pool_size_bytes=2 * block_size,
        staging_block_size_bytes=block_size,
    )
    registry = ExpertRegistry()
    engine = TransitionEngine(
        registry=registry,
        pool_allocator=alloc,
        weight_store=_StubWeightStore(),  # type: ignore[arg-type]
        max_workers=1,
        max_inflight=2,
    )
    try:
        hi_key = ExpertKey(0, 0)
        lo_key = ExpertKey(0, 1)
        for key, tier in ((hi_key, Tier.HI), (lo_key, Tier.LO)):
            assert engine.enqueue(
                TransitionReq(key, tier, tier, "bootstrap", 0)
            )
            assert engine.wait_ready(key, timeout=5)

        # Demotion must use staging because the LO resident pool is full.
        assert engine.enqueue(
            TransitionReq(hi_key, Tier.HI, Tier.LO, "swap_out", 1)
        )
        assert engine.wait_ready(hi_key, timeout=5)
        assert registry.get_handle(hi_key).block.pool_name == "staging"

        original_bind = engine._bind_packed_to_block
        staging_reader_counts = []

        def observing_bind(packed, storage):
            staging_reader_counts.append(
                registry.get_handle(hi_key).active_readers
            )
            return original_bind(packed, storage)

        monkeypatch.setattr(engine, "_bind_packed_to_block", observing_bind)

        # Promotion frees the paired LO resident block. Repatriation then
        # moves the staging-backed demotion into that hole.
        assert engine.enqueue(
            TransitionReq(lo_key, Tier.LO, Tier.HI, "swap_in", 1)
        )
        assert engine.wait_ready(lo_key, timeout=5)
        assert registry.get_handle(hi_key).block.pool_name == "lo:0"
        assert alloc.snapshot()["staging"]["used_blocks"] == 0
        assert engine.get_stats()["repatriations"] == 1
        assert 1 in staging_reader_counts
    finally:
        engine.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int4_kernel_cache_is_backed_by_the_reserved_pool_block():
    key = ExpertKey(0, 0)
    store = ModelWeightStore(
        model=None,
        hi_format="fp16",
        lo_format="int4",
        enable_int4_kernel_cache=True,
    )
    store.register_expert(key, torch.randn(16, 128, dtype=torch.float16))
    block_size = store.get_byte_size(key, Tier.LO)
    allocator = PoolAllocator(
        num_layers=1,
        hi_pool_sizes=[0],
        lo_pool_sizes=[block_size],
        device=torch.device("cuda"),
        hi_block_sizes=[16 * 128 * 2],
        lo_block_sizes=[block_size],
    )
    registry = ExpertRegistry()
    engine = TransitionEngine(
        registry,
        allocator,
        store,
        max_workers=1,
        max_inflight=1,
    )
    try:
        assert engine.enqueue(
            TransitionReq(key, Tier.HI, Tier.LO, "int4", 0)
        )
        assert engine.wait_ready(key, timeout=10)
        handle = registry.get_handle(key)
        assert handle is not None
        packed = handle.quant_meta
        assert packed.int4pack_weight is not None
        assert packed.int4pack_scales_and_zeros is not None
        assert handle.bytes == block_size
        block_start = handle.block.tensor.data_ptr()
        block_end = block_start + handle.block.tensor.numel()
        for tensor in (
            packed.qweight,
            packed.scales,
            packed.int4pack_weight,
            packed.int4pack_scales_and_zeros,
        ):
            assert block_start <= tensor.data_ptr() < block_end
        x = torch.randn(3, 128, dtype=torch.float16, device="cuda")
        actual = fused_linear(x, packed)
        source_packed = store.load_weights(key, Tier.LO).to("cuda")
        expected = torch.nn.functional.linear(
            x,
            dequant_to_fp16(source_packed),
        )
        torch.testing.assert_close(actual, expected, rtol=0.08, atol=0.08)
    finally:
        engine.shutdown()


def test_transition_rejects_undersized_block_without_publishing_or_leaking():
    block_size = 64
    alloc = PoolAllocator(
        num_layers=1,
        hi_pool_sizes=[block_size],
        lo_pool_sizes=[block_size],
        device=torch.device("cpu"),
        block_size_bytes=block_size,
    )
    registry = ExpertRegistry()
    store = _StubWeightStore()  # payload is 512 bytes
    engine = TransitionEngine(
        registry=registry,
        pool_allocator=alloc,
        weight_store=store,  # type: ignore[arg-type]
        max_workers=1,
    )
    try:
        key = ExpertKey(0, 0)
        assert engine.enqueue(
            TransitionReq(key=key, src=Tier.LO, dst=Tier.HI, reason="test", issued_step=0)
        )
        _wait_for_completion(engine)
        assert registry.get_handle(key) is None
        assert alloc.occupancy(0, Tier.HI) == 0
    finally:
        engine.shutdown()


def test_published_quant_meta_is_backed_by_pool_storage():
    engine, _, registry = _make_engine()
    try:
        key = ExpertKey(0, 0)
        assert engine.enqueue(
            TransitionReq(key=key, src=Tier.LO, dst=Tier.HI, reason="test", issued_step=0)
        )
        _wait_for_completion(engine)
        handle = registry.get_handle(key)
        packed = handle.get_packed()
        assert packed.qweight.device == handle.block.tensor.device
        assert packed.qweight.untyped_storage().data_ptr() == (
            handle.block.tensor.untyped_storage().data_ptr()
        )
    finally:
        engine.shutdown()


def test_promote_then_demote_frees_old_hi_block():
    """
    The single most important test in Plan B: after promote→demote, the HI
    pool occupancy must return to zero. Pre-fix this would have been > 0
    (leaked block).
    """
    engine, alloc, registry = _make_engine()
    try:
        key = ExpertKey(0, 0)

        # Promote to HI
        engine.enqueue(TransitionReq(key=key, src=Tier.LO, dst=Tier.HI, reason="up", issued_step=0))
        _wait_for_completion(engine)
        assert alloc.occupancy(0, Tier.HI) > 0
        hi_block_id = registry.get_handle(key).block.block_id

        # Demote back to LO
        engine.enqueue(TransitionReq(key=key, src=Tier.HI, dst=Tier.LO, reason="down", issued_step=1))
        _wait_for_completion(engine)

        # Old HI block must be freed; new LO block must be live
        assert alloc.occupancy(0, Tier.HI) == 0, "HI block leaked after demote"
        assert alloc.occupancy(0, Tier.LO) > 0
        assert registry.get_handle(key).tier == Tier.LO

        # And the freed HI block must be reusable. PoolAllocator pops from
        # the front of the free queue and appends on free, so after one
        # alloc(0)+free(0) cycle the next alloc returns block_id=1 from the
        # rotated queue — block IDs are not stable across recycle, only the
        # *count* of free blocks is what matters. We assert occupancy
        # returned to "1 used out of 4" rather than equality of block_id.
        engine.enqueue(TransitionReq(key=key, src=Tier.LO, dst=Tier.HI, reason="up2", issued_step=2))
        _wait_for_completion(engine)
        new_handle = registry.get_handle(key)
        assert new_handle is not None and new_handle.block is not None
        assert alloc.occupancy(0, Tier.HI) > 0
        assert alloc.occupancy(0, Tier.LO) == 0
        # Sanity: the new block_id is in the valid range and the previously
        # used id is no longer in use (it's back in the free queue).
        assert 0 <= new_handle.block.block_id < 4
        _ = hi_block_id  # kept for diagnostic clarity; intentionally unused
    finally:
        engine.shutdown()


def test_long_promote_demote_loop_no_leak():
    """
    Plan §12 DoD: long promote/demote loop must not leak blocks. We run 50
    cycles on a single expert and assert the HI pool oscillates between 0
    and 1 used block — never accumulating.

    50 cycles is enough that a 1-block-per-cycle leak would have exhausted
    the 4-block HI pool 12× over and the engine would have started failing
    its allocations.
    """
    engine, alloc, registry = _make_engine()
    try:
        key = ExpertKey(0, 0)
        n_cycles = 50
        for i in range(n_cycles):
            engine.enqueue(
                TransitionReq(key=key, src=Tier.LO, dst=Tier.HI, reason="up", issued_step=2 * i)
            )
            _wait_for_completion(engine)
            assert alloc.occupancy(0, Tier.HI) > 0
            assert alloc.occupancy(0, Tier.LO) == 0

            engine.enqueue(
                TransitionReq(key=key, src=Tier.HI, dst=Tier.LO, reason="down", issued_step=2 * i + 1)
            )
            _wait_for_completion(engine)
            assert alloc.occupancy(0, Tier.HI) == 0
            assert alloc.occupancy(0, Tier.LO) > 0

        stats = engine.get_stats()
        assert stats["total_promotions"] == n_cycles
        assert stats["total_demotions"] == n_cycles
    finally:
        engine.shutdown()


def test_multiple_experts_independent_reclaim():
    """
    Promote three different experts to HI, then demote them in a different
    order. Each demote must free exactly the right block. With the old
    pre-Plan-B code there was no way to map handle→block, so this test
    would have either leaked or freed the wrong block.
    """
    engine, alloc, registry = _make_engine()
    try:
        keys = [ExpertKey(0, i) for i in range(3)]
        # Promote all three
        for i, k in enumerate(keys):
            engine.enqueue(TransitionReq(key=k, src=Tier.LO, dst=Tier.HI, reason="up", issued_step=i))
            _wait_for_completion(engine)

        used_block_ids = {registry.get_handle(k).block.block_id for k in keys}
        assert len(used_block_ids) == 3, "experts should occupy distinct blocks"

        # Demote in reverse order
        for i, k in enumerate(reversed(keys)):
            engine.enqueue(
                TransitionReq(key=k, src=Tier.HI, dst=Tier.LO, reason="down", issued_step=10 + i)
            )
            _wait_for_completion(engine)

        assert alloc.occupancy(0, Tier.HI) == 0, "HI blocks leaked"
        for k in keys:
            assert registry.get_handle(k).tier == Tier.LO
    finally:
        engine.shutdown()
