"""
Tests for ``dynaexq.core.budget_tracker`` (BudgetTracker, plan §III-D / §5.2).

Two layers of testing:

1. **Unit tests for the tracker itself**: try_reserve / commit / release
   lifecycle, per-tier independence, staging cap, double-release detection,
   thread safety. These run on plain Python objects with no torch.

2. **Integration tests with TransitionEngine**: a tight HI cap forces the
   engine to either reject ``enqueue`` (backpressure) or evict-then-promote
   in a steady-state cycle without leaking budget. These exercise the
   reservation handoff between TransitionEngine.enqueue, the worker
   thread's commit, and Stage 4 reclaim's release.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch

from dynaexq.core.budget_tracker import BudgetTracker, Reservation
from dynaexq.core.config import Tier
from dynaexq.core.memory_pool import PoolAllocator
from dynaexq.core.quant import QuantFormat, compute_packed_nbytes, pack
from dynaexq.core.registry import ExpertKey, ExpertRegistry
from dynaexq.core.scheduler import TransitionReq
from dynaexq.core.transition_engine import TransitionEngine


# ===========================================================================
# UNIT TESTS — BudgetTracker in isolation
# ===========================================================================


def test_constructor_rejects_negative_caps():
    with pytest.raises(ValueError):
        BudgetTracker(hi_cap=-1, lo_cap=10)
    with pytest.raises(ValueError):
        BudgetTracker(hi_cap=10, lo_cap=-1)
    with pytest.raises(ValueError):
        BudgetTracker(hi_cap=10, lo_cap=10, staging_cap=-1)
    with pytest.raises(ValueError):
        BudgetTracker(hi_cap=10, lo_cap=10, total_cap=-1)


def test_try_reserve_succeeds_below_cap():
    bt = BudgetTracker(hi_cap=1000, lo_cap=2000)
    r = bt.try_reserve(400, Tier.HI)
    assert r is not None
    assert r.tier == Tier.HI
    assert r.nbytes == 400
    assert r.is_pending()


def test_try_reserve_returns_none_when_cap_would_be_exceeded():
    bt = BudgetTracker(hi_cap=500, lo_cap=500)
    r1 = bt.try_reserve(300, Tier.HI)
    assert r1 is not None
    r2 = bt.try_reserve(300, Tier.HI)  # 300+300=600 > 500
    assert r2 is None


def test_try_reserve_rejects_zero_or_negative():
    bt = BudgetTracker(hi_cap=100, lo_cap=100)
    with pytest.raises(ValueError):
        bt.try_reserve(0, Tier.HI)
    with pytest.raises(ValueError):
        bt.try_reserve(-1, Tier.HI)


def test_per_tier_caps_are_independent():
    bt = BudgetTracker(hi_cap=100, lo_cap=200)
    # Saturate HI; LO should still have full headroom.
    assert bt.try_reserve(100, Tier.HI) is not None
    assert bt.try_reserve(1, Tier.HI) is None
    r_lo = bt.try_reserve(150, Tier.LO)
    assert r_lo is not None


def test_commit_moves_pending_to_committed():
    bt = BudgetTracker(hi_cap=1000, lo_cap=1000)
    r = bt.try_reserve(400, Tier.HI)
    snap = bt.snapshot()
    assert snap["hi_pending"] == 400
    assert snap["hi_committed"] == 0

    bt.commit(r)
    assert r.is_committed()
    snap = bt.snapshot()
    assert snap["hi_pending"] == 0
    assert snap["hi_committed"] == 400


def test_commit_is_idempotent_for_already_committed():
    bt = BudgetTracker(hi_cap=1000, lo_cap=1000)
    r = bt.try_reserve(400, Tier.HI)
    bt.commit(r)
    bt.commit(r)  # second call must not double-count
    assert bt.snapshot()["hi_committed"] == 400


def test_commit_after_release_raises():
    bt = BudgetTracker(hi_cap=1000, lo_cap=1000)
    r = bt.try_reserve(400, Tier.HI)
    bt.release(r)
    with pytest.raises(RuntimeError, match="released"):
        bt.commit(r)


def test_release_pending_returns_to_budget():
    bt = BudgetTracker(hi_cap=500, lo_cap=500)
    r = bt.try_reserve(400, Tier.HI)
    assert bt.try_reserve(200, Tier.HI) is None  # 400+200 > 500

    bt.release(r)
    assert r.is_released()
    # Budget recovered: a fresh reservation for the same bytes succeeds.
    r2 = bt.try_reserve(400, Tier.HI)
    assert r2 is not None


def test_release_committed_returns_to_budget():
    bt = BudgetTracker(hi_cap=500, lo_cap=500)
    r = bt.try_reserve(400, Tier.HI)
    bt.commit(r)
    assert bt.snapshot()["hi_committed"] == 400

    bt.release(r)
    assert bt.snapshot()["hi_committed"] == 0
    assert bt.try_reserve(500, Tier.HI) is not None


def test_double_release_raises():
    bt = BudgetTracker(hi_cap=500, lo_cap=500)
    r = bt.try_reserve(100, Tier.HI)
    bt.release(r)
    with pytest.raises(RuntimeError, match="double release"):
        bt.release(r)


def test_staging_cap_blocks_too_many_inflight():
    bt = BudgetTracker(hi_cap=10_000, lo_cap=10_000, staging_cap=300)
    r1 = bt.try_reserve(100, Tier.HI)
    r2 = bt.try_reserve(100, Tier.LO)
    r3 = bt.try_reserve(100, Tier.HI)  # 100+100+100 = 300 (== cap)
    assert r1 and r2 and r3
    # Next reservation pushes staging to 301 → reject
    assert bt.try_reserve(1, Tier.HI) is None


def test_staging_cap_recovers_on_commit():
    """Once a reservation is committed it leaves the staging pool, freeing
    headroom for new in-flight transitions."""
    bt = BudgetTracker(hi_cap=10_000, lo_cap=10_000, staging_cap=300)
    r1 = bt.try_reserve(150, Tier.HI)
    r2 = bt.try_reserve(150, Tier.HI)  # staging = 300, cap reached
    assert bt.try_reserve(1, Tier.HI) is None

    bt.commit(r1)  # frees 150 of staging
    r3 = bt.try_reserve(100, Tier.HI)
    assert r3 is not None


def test_staging_cap_zero_blocks_everything():
    bt = BudgetTracker(hi_cap=10_000, lo_cap=10_000, staging_cap=0)
    assert bt.try_reserve(1, Tier.HI) is None


def test_staging_cap_none_means_no_limit():
    bt = BudgetTracker(hi_cap=10_000, lo_cap=10_000, staging_cap=None)
    # Reserve a lot of bytes without ever committing.
    rs = [bt.try_reserve(1000, Tier.HI) for _ in range(10)]
    assert all(r is not None for r in rs)


def test_available_reflects_remaining_capacity():
    bt = BudgetTracker(hi_cap=1000, lo_cap=2000)
    assert bt.available(Tier.HI) == 1000
    assert bt.available(Tier.LO) == 2000

    bt.try_reserve(300, Tier.HI)
    assert bt.available(Tier.HI) == 700

    r = bt.try_reserve(500, Tier.HI)
    bt.commit(r)
    assert bt.available(Tier.HI) == 200  # 1000 - 300 (pending) - 500 (committed)


def test_available_clamped_by_staging_cap():
    bt = BudgetTracker(hi_cap=1000, lo_cap=1000, staging_cap=200)
    # Per-tier room is 1000 but staging only allows 200.
    assert bt.available(Tier.HI) == 200


def test_total_cap_counts_committed_and_pending_across_tiers():
    bt = BudgetTracker(
        hi_cap=1_000,
        lo_cap=1_000,
        staging_cap=1_000,
        total_cap=500,
    )
    hi = bt.try_reserve(300, Tier.HI)
    assert hi is not None
    bt.commit(hi)
    assert bt.available(Tier.LO) == 200
    assert bt.try_reserve(201, Tier.LO) is None
    lo = bt.try_reserve(200, Tier.LO)
    assert lo is not None
    assert bt.snapshot()["total_live"] == 500


def test_concurrent_reservations_never_exceed_cap():
    """Hammer try_reserve from many threads against a tight cap. The total
    bytes successfully reserved must not exceed the cap, and the
    BudgetTracker's internal counters must agree afterwards."""
    bt = BudgetTracker(hi_cap=1_000_000, lo_cap=0)
    n_threads = 16
    per_thread = 200
    chunk = 100  # 16 * 200 * 100 = 320_000 << 1_000_000, leaves headroom

    successes = [0] * n_threads

    def worker(tid: int) -> None:
        for _ in range(per_thread):
            r = bt.try_reserve(chunk, Tier.HI)
            if r is not None:
                successes[tid] += 1
                # Half commit, half release — exercises both paths.
                if successes[tid] % 2 == 0:
                    bt.commit(r)
                else:
                    bt.release(r)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = bt.snapshot()
    # Pending should now be 0 (everything either committed or released).
    assert snap["hi_pending"] == 0
    # Committed bytes must be ≤ cap and equal half(ish) of successes * chunk.
    committed_count = sum(s // 2 for s in successes)
    assert snap["hi_committed"] == committed_count * chunk
    assert snap["hi_committed"] <= bt.hi_cap


# ===========================================================================
# INTEGRATION TESTS — TransitionEngine + BudgetTracker
# ===========================================================================


class _StubWeightStore:
    """Plan-A-shaped stub: returns a tiny PackedTensor."""

    def __init__(self, out_features: int = 4, in_features: int = 64):
        self.out_features = out_features
        self.in_features = in_features
        weight = torch.zeros(out_features, in_features, dtype=torch.float16)
        self._packed = pack(weight, QuantFormat.FP16)

    def load_weights(self, key: ExpertKey, tier: Tier):
        return self._packed

    def get_byte_size(self, key: ExpertKey, tier: Tier) -> int:
        return compute_packed_nbytes(
            self.out_features, self.in_features, QuantFormat.FP16, self.in_features
        )


def _make_engine_with_budget(hi_cap: int, lo_cap: int, staging_cap=None):
    block_size = 64 * 1024
    pool_bytes = block_size * 4
    alloc = PoolAllocator(
        num_layers=1,
        hi_pool_sizes=[pool_bytes],
        lo_pool_sizes=[pool_bytes],
        device=torch.device("cpu"),
        block_size_bytes=block_size,
    )
    registry = ExpertRegistry()
    store = _StubWeightStore()
    bt = BudgetTracker(hi_cap=hi_cap, lo_cap=lo_cap, staging_cap=staging_cap)
    engine = TransitionEngine(
        registry=registry,
        pool_allocator=alloc,
        weight_store=store,  # type: ignore[arg-type]
        max_workers=2,
        max_inflight=4,
        budget_tracker=bt,
    )
    return engine, alloc, registry, bt, store


def _wait(engine, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with engine._transition_lock:
            if not engine._active_transitions:
                return
        time.sleep(0.005)
    raise TimeoutError("transitions did not drain")


def test_enqueue_charges_bytes_against_budget():
    engine, _, registry, bt, store = _make_engine_with_budget(
        hi_cap=10_000, lo_cap=10_000
    )
    try:
        nbytes = store.get_byte_size(ExpertKey(0, 0), Tier.HI)
        engine.enqueue(TransitionReq(key=ExpertKey(0, 0), src=Tier.LO, dst=Tier.HI, reason="up", issued_step=0))
        _wait(engine)

        snap = bt.snapshot()
        assert snap["hi_committed"] == nbytes
        assert snap["hi_pending"] == 0
        # The handle carries the reservation that owns these bytes.
        h = registry.get_handle(ExpertKey(0, 0))
        assert h.reservation is not None
        assert h.reservation.is_committed()
    finally:
        engine.shutdown()


def test_enqueue_returns_false_when_budget_full():
    """Cap so tight only one HI block fits. Second enqueue must be rejected."""
    engine, _, _, bt, store = _make_engine_with_budget(hi_cap=1, lo_cap=10_000)
    try:
        # Tight HI cap: only 1 byte allowed, but each reservation needs
        # store.get_byte_size() bytes >> 1 → first reserve fails.
        ok = engine.enqueue(
            TransitionReq(key=ExpertKey(0, 0), src=Tier.LO, dst=Tier.HI, reason="up", issued_step=0)
        )
        assert ok is False
        snap = bt.snapshot()
        assert snap["hi_pending"] == 0
        assert snap["hi_committed"] == 0
    finally:
        engine.shutdown()


def test_transition_unit_budget_rejection_rolls_back_first_reservation():
    one_size = compute_packed_nbytes(4, 64, QuantFormat.FP16, 64)
    engine, _, registry, bt, _ = _make_engine_with_budget(
        hi_cap=one_size,
        lo_cap=one_size * 4,
        staging_cap=one_size * 4,
    )
    try:
        unit = (
            TransitionReq(
                ExpertKey(0, 0), Tier.LO, Tier.HI, "first", 0
            ),
            TransitionReq(
                ExpertKey(0, 1), Tier.LO, Tier.HI, "second", 0
            ),
        )
        assert engine.enqueue_many(unit) is False
        assert registry.get_handle(unit[0].key) is None
        assert registry.get_handle(unit[1].key) is None
        snapshot = bt.snapshot()
        assert snapshot["hi_pending"] == 0
        assert snapshot["hi_committed"] == 0
        stats = engine.get_stats()
        assert stats["accepted_requests"] == 0
        assert stats["rejected_budget"] == 2
        assert stats["active_transitions"] == 0
    finally:
        engine.shutdown()


def test_enqueue_succeeds_after_eviction_releases_budget():
    """
    Cap is sized so exactly ONE HI expert fits at a time. Promote a, then
    promote b → b must succeed only after a is demoted (which releases a's
    bytes back to the budget). This is the canonical "evict-then-promote
    under tight cap" steady state from plan §III-D.
    """
    one_size = compute_packed_nbytes(4, 64, QuantFormat.FP16, 64)
    engine, _, registry, bt, _ = _make_engine_with_budget(
        hi_cap=one_size, lo_cap=one_size * 4
    )
    try:
        a = ExpertKey(0, 0)
        b = ExpertKey(0, 1)

        assert engine.enqueue(TransitionReq(key=a, src=Tier.LO, dst=Tier.HI, reason="up", issued_step=0))
        _wait(engine)
        assert bt.available(Tier.HI) == 0  # cap is full

        # Without an eviction, b cannot promote
        ok = engine.enqueue(TransitionReq(key=b, src=Tier.LO, dst=Tier.HI, reason="up", issued_step=1))
        assert ok is False

        # Demote a → releases bytes
        assert engine.enqueue(TransitionReq(key=a, src=Tier.HI, dst=Tier.LO, reason="down", issued_step=2))
        _wait(engine)
        assert bt.available(Tier.HI) == one_size

        # Now b's promotion is admitted
        assert engine.enqueue(TransitionReq(key=b, src=Tier.LO, dst=Tier.HI, reason="up", issued_step=3))
        _wait(engine)
        assert registry.get_handle(b).tier == Tier.HI
        assert bt.available(Tier.HI) == 0
    finally:
        engine.shutdown()


def test_long_loop_under_tight_budget_no_drift():
    """50 promote/demote cycles under a 1-block HI budget. After every
    cycle the budget must be back to the same numbers. Pre-Plan-A there
    would be no reservation accounting; pre-this-test no proof of steady
    state under enforcement."""
    one_size = compute_packed_nbytes(4, 64, QuantFormat.FP16, 64)
    engine, _, _, bt, _ = _make_engine_with_budget(
        hi_cap=one_size, lo_cap=one_size * 4
    )
    try:
        key = ExpertKey(0, 0)
        n_cycles = 50
        for i in range(n_cycles):
            assert engine.enqueue(TransitionReq(key=key, src=Tier.LO, dst=Tier.HI, reason="up", issued_step=2*i))
            _wait(engine)
            assert bt.snapshot()["hi_committed"] == one_size

            assert engine.enqueue(TransitionReq(key=key, src=Tier.HI, dst=Tier.LO, reason="down", issued_step=2*i+1))
            _wait(engine)
            snap = bt.snapshot()
            assert snap["hi_committed"] == 0
            assert snap["hi_pending"] == 0
    finally:
        engine.shutdown()
