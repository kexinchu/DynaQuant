"""
Tests for the Phase 4.3 event fence path.

The contract being guarded:

1. ``ExpertHandle.last_use_event`` is a duck-typed field; anything with
   a ``synchronize()`` method works. ``torch.cuda.Event`` satisfies it
   natively; these tests use a tiny fake event class because the CI
   machine may not have CUDA.

2. ``ExpertRegistry.mark_used(key, event)`` attaches the event to the
   currently-registered handle, atomically under the registry lock.

3. ``TransitionEngine._fence_before_reclaim(old_handle)``:
   - Calls ``old_handle.last_use_event.synchronize()`` when set and
     does NOT call the global fallback.
   - Reclaims a never-dispatched handle without a device sync.
   - Calls the global fallback when a dispatch lease existed but no
     per-handle event was recorded.

4. A full promote→demote cycle routes the old handle's fence correctly:
   if mark_used was called between promote and demote, the demote
   reclaim waits on THAT event and skips the global sync entirely.

These guarantees are the plan §5.3 "publish-after-complete" event
fence, replacing the conservative ``torch.cuda.synchronize()`` fallback
that Phase 4 shipped as a stopgap.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch

from dynaexq.core.budget_tracker import BudgetTracker
from dynaexq.core.config import Tier
from dynaexq.core.memory_pool import PoolAllocator
from dynaexq.core.quant import QuantFormat, compute_packed_nbytes, pack
from dynaexq.core.registry import ExpertHandle, ExpertKey, ExpertRegistry
from dynaexq.core.scheduler import TransitionReq
from dynaexq.core.transition_engine import TransitionEngine


# ---------------------------------------------------------------------------
# Fake event: the minimum surface TransitionEngine requires.
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Duck-typed fence satisfying ``_FenceLike``: one ``synchronize``
    method that increments a call counter. Used because
    ``torch.cuda.Event`` is not available on CPU-only CI."""

    def __init__(self):
        self.sync_calls = 0
        self._lock = threading.Lock()

    def synchronize(self) -> None:
        with self._lock:
            self.sync_calls += 1


# ---------------------------------------------------------------------------
# Stub WeightStore — Plan A shape: returns PackedTensor.
# ---------------------------------------------------------------------------


class _StubWeightStore:
    def __init__(self, out_features: int = 4, in_features: int = 64):
        self.out_features = out_features
        self.in_features = in_features
        self._packed = pack(
            torch.zeros(out_features, in_features, dtype=torch.float16),
            QuantFormat.FP16,
        )

    def load_weights(self, key: ExpertKey, tier: Tier):
        return self._packed

    def get_byte_size(self, key: ExpertKey, tier: Tier) -> int:
        return compute_packed_nbytes(
            self.out_features, self.in_features, QuantFormat.FP16, self.in_features
        )


def _make_engine():
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
    engine = TransitionEngine(
        registry=registry,
        pool_allocator=alloc,
        weight_store=_StubWeightStore(),  # type: ignore[arg-type]
        max_workers=2,
        max_inflight=4,
    )
    return engine, alloc, registry


def _wait(engine, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with engine._transition_lock:
            if not engine._active_transitions:
                return
        time.sleep(0.005)
    raise TimeoutError("transitions did not drain")


# ---------------------------------------------------------------------------
# Unit tests: ExpertRegistry.mark_used + ExpertHandle.last_use_event
# ---------------------------------------------------------------------------


def test_handle_default_last_use_event_is_none():
    h = ExpertHandle(tier=Tier.LO)
    assert h.last_use_event is None


def test_mark_used_attaches_event_to_registered_handle():
    reg = ExpertRegistry()
    key = ExpertKey(0, 0)
    h = ExpertHandle(tier=Tier.LO)
    reg.register(key, h)

    event = _FakeEvent()
    ok = reg.mark_used(key, event)
    assert ok is True
    assert reg.get_handle(key).last_use_event is event


def test_mark_used_returns_false_for_missing_key():
    reg = ExpertRegistry()
    event = _FakeEvent()
    assert reg.mark_used(ExpertKey(0, 0), event) is False


def test_mark_used_overwrites_previous_event():
    """Subsequent forwards of the same expert replace the stale fence
    with the fresh one — the reclaim path only needs the MOST RECENT
    use, not every historical one."""
    reg = ExpertRegistry()
    key = ExpertKey(0, 0)
    reg.register(key, ExpertHandle(tier=Tier.LO))

    e1 = _FakeEvent()
    e2 = _FakeEvent()
    reg.mark_used(key, e1)
    reg.mark_used(key, e2)
    assert reg.get_handle(key).last_use_event is e2


def test_mark_used_does_not_reattach_to_new_handle_after_register():
    """register() publishes a fresh handle, wiping any stale fence —
    it's the caller's responsibility to mark_used again after a new
    publish if they want the precise reclaim path."""
    reg = ExpertRegistry()
    key = ExpertKey(0, 0)
    reg.register(key, ExpertHandle(tier=Tier.LO))
    reg.mark_used(key, _FakeEvent())

    # A new handle replaces the old; its last_use_event defaults to None.
    reg.register(key, ExpertHandle(tier=Tier.HI))
    assert reg.get_handle(key).last_use_event is None


# ---------------------------------------------------------------------------
# Unit tests: TransitionEngine._fence_before_reclaim
# ---------------------------------------------------------------------------


def test_fence_before_reclaim_uses_event_when_present():
    """If last_use_event is set, the engine calls event.synchronize()
    and DOES NOT fall back to the global sync."""
    engine, _, _ = _make_engine()
    try:
        calls = []
        engine._fallback_global_sync = lambda: calls.append("global")  # type: ignore[method-assign]
        event = _FakeEvent()
        handle = ExpertHandle(tier=Tier.LO, last_use_event=event)

        engine._fence_before_reclaim(handle)

        assert event.sync_calls == 1
        assert calls == [], "global fallback must NOT be called when a per-handle fence is present"
    finally:
        engine.shutdown()


def test_fence_before_reclaim_waits_for_every_compute_stream():
    engine, _, _ = _make_engine()
    try:
        first = _FakeEvent()
        second = _FakeEvent()
        handle = ExpertHandle(tier=Tier.LO)
        handle.last_use_events = {1: first, 2: second}

        engine._fence_before_reclaim(handle)

        assert first.sync_calls == 1
        assert second.sync_calls == 1
    finally:
        engine.shutdown()


def test_fence_before_reclaim_skips_sync_for_never_dispatched_handle():
    engine, _, _ = _make_engine()
    try:
        calls = []
        engine._fallback_global_sync = lambda: calls.append("global")  # type: ignore[method-assign]
        handle = ExpertHandle(tier=Tier.LO, last_use_event=None)

        engine._fence_before_reclaim(handle)

        assert calls == []
        assert engine.get_stats()["unused_handle_reclaims"] == 1
    finally:
        engine.shutdown()


def test_fence_before_reclaim_falls_back_after_unfenced_dispatch():
    """A dispatch lease without its required event remains fail-safe."""
    engine, _, registry = _make_engine()
    try:
        calls = []
        engine._fallback_global_sync = lambda: calls.append("global")  # type: ignore[method-assign]
        key = ExpertKey(0, 0)
        handle = ExpertHandle(tier=Tier.LO, last_use_event=None)
        registry.register(key, handle)
        leased = registry.acquire_handle(key)
        assert leased is handle
        registry.release_handle(leased)

        engine._fence_before_reclaim(handle)

        assert calls == ["global"]
    finally:
        engine.shutdown()


def test_fence_before_reclaim_propagates_event_exception():
    """A broken fence (synchronize raises) should surface to the caller,
    not be silently swallowed — a corrupted fence is a serious bug that
    must halt the reclaim rather than risk freeing a live block."""
    engine, _, _ = _make_engine()
    try:
        class _BadEvent:
            def synchronize(self):
                raise RuntimeError("cuda event corrupted")

        handle = ExpertHandle(tier=Tier.LO, last_use_event=_BadEvent())
        with pytest.raises(RuntimeError, match="corrupted"):
            engine._fence_before_reclaim(handle)
    finally:
        engine.shutdown()


def test_fallback_global_sync_is_cpu_noop():
    """The CPU code path for the fallback must not touch torch.cuda at all
    beyond the ``is_available()`` check. This test just proves the call
    doesn't raise on a CPU-only machine."""
    TransitionEngine._fallback_global_sync()  # no assertion, just "doesn't raise"


# ---------------------------------------------------------------------------
# Integration: promote → mark_used → demote path
# ---------------------------------------------------------------------------


def test_demote_waits_on_fence_from_previous_forward():
    """
    Full cycle: promote expert 0 to HI, attach a fence event to the
    registered handle via mark_used (simulating a forward that just
    finished reading from block.tensor), then demote. Stage 4 reclaim
    must call the event's ``synchronize()`` exactly once and NOT fall
    back to the global sync.

    Pre-Phase-4.3 this path unconditionally called
    ``torch.cuda.synchronize()`` — the whole point of the refactor is
    that it now prefers the cheaper per-handle fence.
    """
    engine, alloc, registry = _make_engine()
    try:
        # Intercept the fallback so we can detect if it was called.
        global_calls = []
        engine._fallback_global_sync = lambda: global_calls.append("global")  # type: ignore[method-assign]

        key = ExpertKey(0, 0)

        # Promote
        assert engine.enqueue(
            TransitionReq(key=key, src=Tier.LO, dst=Tier.HI, reason="up", issued_step=0)
        )
        _wait(engine)

        # Simulate a forward kernel completing and recording a fence.
        event = _FakeEvent()
        assert registry.mark_used(key, event) is True

        # Demote — the OLD (HI) handle now carries the event, so Stage 4
        # should wait on it and skip the global fallback.
        assert engine.enqueue(
            TransitionReq(key=key, src=Tier.HI, dst=Tier.LO, reason="down", issued_step=1)
        )
        _wait(engine)

        assert event.sync_calls == 1, (
            "Stage 4 reclaim must wait on the per-handle fence exactly once"
        )
        assert global_calls == [], (
            "global fallback must not fire when a per-handle fence is available"
        )
        # Pool must have released the old HI block.
        assert alloc.occupancy(0, Tier.HI) == 0
        assert registry.get_handle(key).tier == Tier.LO
    finally:
        engine.shutdown()


def test_demote_without_mark_used_falls_back_to_global_sync():
    """Mirror case: no mark_used call → global fallback fires for demote."""
    engine, alloc, registry = _make_engine()
    try:
        global_calls = []
        engine._fallback_global_sync = lambda: global_calls.append("global")  # type: ignore[method-assign]

        key = ExpertKey(0, 0)

        assert engine.enqueue(
            TransitionReq(key=key, src=Tier.LO, dst=Tier.HI, reason="up", issued_step=0)
        )
        _wait(engine)

        # Simulate a faulty execution path that leased the handle but
        # deliberately released it without its required event.
        leased = registry.acquire_handle(key)
        assert leased is not None
        registry.release_handle(leased)

        assert engine.enqueue(
            TransitionReq(key=key, src=Tier.HI, dst=Tier.LO, reason="down", issued_step=1)
        )
        _wait(engine)

        # Exactly one fallback call — the demote's Stage 4. Promote's
        # Stage 4 has no old handle (first promote), so it should not
        # fence.
        assert global_calls == ["global"]
        assert alloc.occupancy(0, Tier.HI) == 0
    finally:
        engine.shutdown()


def test_long_loop_with_fences_never_calls_global_sync():
    """
    50 promote/demote cycles, mark_used on every freshly-registered
    handle. A real forward path marks use of an expert regardless of
    its tier — and in that realistic scenario the global fallback must
    fire zero times and every recorded fence must be synchronized
    exactly once.

    Note on the initial promote: there's no prior handle, so its
    Stage 4 has nothing to reclaim and therefore no fence is consulted.
    All subsequent 2N-1 Stage-4 calls (N demotes + (N-1) promotes)
    must route through a per-handle fence.

    If this regresses, the Phase 4.3 optimization is silently dead and
    the runtime is paying for a full device sync on every swap.
    """
    engine, alloc, registry = _make_engine()
    try:
        global_calls = []
        engine._fallback_global_sync = lambda: global_calls.append("global")  # type: ignore[method-assign]

        key = ExpertKey(0, 0)
        n_cycles = 50
        events: list[_FakeEvent] = []

        def _mark_fresh():
            """Simulate a forward pass reading the current handle and
            recording a compute-stream event afterward."""
            e = _FakeEvent()
            events.append(e)
            assert registry.mark_used(key, e) is True

        for i in range(n_cycles):
            assert engine.enqueue(
                TransitionReq(key=key, src=Tier.LO, dst=Tier.HI, reason="up", issued_step=2 * i)
            )
            _wait(engine)
            _mark_fresh()  # forward on the HI-tier handle

            assert engine.enqueue(
                TransitionReq(key=key, src=Tier.HI, dst=Tier.LO, reason="down", issued_step=2 * i + 1)
            )
            _wait(engine)
            _mark_fresh()  # forward on the LO-tier handle

        # Every recorded event should have been synchronized exactly
        # once — by the Stage 4 of the NEXT transition. The very last
        # event attached after the final demote is never synchronized
        # because no further transition follows; we expect exactly
        # ``2*n_cycles - 1`` events to be synced.
        synced = sum(e.sync_calls for e in events)
        assert synced == 2 * n_cycles - 1, (
            f"expected {2 * n_cycles - 1} fence synchronizations, got {synced}"
        )
        assert all(e.sync_calls in (0, 1) for e in events), (
            "no fence should be synchronized more than once"
        )
        assert global_calls == [], (
            f"global sync unexpectedly fired {len(global_calls)} times"
        )
        assert alloc.occupancy(0, Tier.HI) == 0
    finally:
        engine.shutdown()
