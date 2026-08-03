"""
Tests for ``dynaexq.core.registry`` after the Plan B refactor.

These guard the new contract:

* ``ExpertHandle`` holds a ``PoolBlock`` reference (not a bare tensor) and a
  ``PackedTensor`` quantization metadata. ``device_ptr`` is now a property
  that forwards to ``block.tensor``.
* ``ExpertHandle.format`` and ``ExpertHandle.bytes`` are kept in sync with
  ``quant_meta`` automatically by ``__post_init__``. Drift between the two
  must raise.
* ``ExpertRegistry.register`` increments the version monotonically per key.
* ``ExpertRegistry.get_old_handle`` returns the about-to-be-replaced handle
  (the snapshot used by ``TransitionEngine`` to free the old block).
"""

from __future__ import annotations

import threading

import pytest
import torch

from dynaexq.core.config import Tier
from dynaexq.core.memory_pool import PoolAllocator, PoolBlock
from dynaexq.core.quant import QuantFormat, pack
from dynaexq.core.registry import ExpertHandle, ExpertKey, ExpertRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_block(nbytes: int = 1024) -> PoolBlock:
    """A standalone PoolBlock that doesn't require a PoolAllocator."""
    return PoolBlock(
        block_id=0,
        tensor=torch.empty((nbytes,), dtype=torch.uint8),
        in_use=True,
    )


def _make_packed(out_features: int = 8, in_features: int = 128, fmt: QuantFormat = QuantFormat.INT4):
    w = torch.randn(out_features, in_features, dtype=torch.float16) * 0.1
    return pack(w, fmt)


# ---------------------------------------------------------------------------
# ExpertKey
# ---------------------------------------------------------------------------


def test_expert_key_hash_and_eq():
    a = ExpertKey(layer=3, expert=7)
    b = ExpertKey(layer=3, expert=7)
    c = ExpertKey(layer=3, expert=8)
    assert a == b and hash(a) == hash(b)
    assert a != c
    assert a != "not a key"
    d = {a: 1}
    assert d[b] == 1


# ---------------------------------------------------------------------------
# ExpertHandle field semantics
# ---------------------------------------------------------------------------


def test_handle_format_and_bytes_inferred_from_quant_meta():
    block = _make_block()
    packed = _make_packed(8, 128, QuantFormat.INT4)
    h = ExpertHandle(tier=Tier.LO, block=block, quant_meta=packed)
    assert h.format == "int4"
    assert h.bytes == packed.nbytes
    assert h.is_valid()


def test_handle_device_ptr_property_forwards_to_block_tensor():
    block = _make_block(nbytes=2048)
    packed = _make_packed()
    h = ExpertHandle(tier=Tier.HI, block=block, quant_meta=packed)
    # Identity, not equality — must be the *same* underlying storage
    assert h.device_ptr is block.tensor


def test_handle_without_block_is_invalid_and_device_ptr_none():
    h = ExpertHandle(tier=Tier.LO)
    assert not h.is_valid()
    assert h.device_ptr is None


def test_handle_format_drift_raises():
    """If a caller passes both quant_meta and an inconsistent format string,
    we must fail loudly rather than silently ignore one of them."""
    block = _make_block()
    packed = _make_packed(fmt=QuantFormat.INT4)
    with pytest.raises(ValueError, match="format"):
        ExpertHandle(tier=Tier.LO, block=block, quant_meta=packed, format="int2")


def test_handle_bytes_drift_raises():
    block = _make_block()
    packed = _make_packed(fmt=QuantFormat.INT4)
    with pytest.raises(ValueError, match="bytes"):
        ExpertHandle(
            tier=Tier.LO, block=block, quant_meta=packed, bytes=packed.nbytes + 1
        )


def test_handle_with_fp16_packed_meta_works():
    """FP16 tier still goes through PackedTensor (with fmt=FP16, scales=None).
    The runtime never special-cases on tier — only on quant_meta.fmt."""
    block = _make_block()
    packed = _make_packed(fmt=QuantFormat.FP16)
    h = ExpertHandle(tier=Tier.HI, block=block, quant_meta=packed)
    assert h.format == "fp16"
    assert h.bytes == packed.nbytes
    assert h.quant_meta.scales is None


# ---------------------------------------------------------------------------
# ExpertRegistry
# ---------------------------------------------------------------------------


def test_registry_register_then_get_returns_same_handle():
    reg = ExpertRegistry()
    key = ExpertKey(0, 0)
    block = _make_block()
    packed = _make_packed()
    h = ExpertHandle(tier=Tier.LO, block=block, quant_meta=packed)
    reg.register(key, h)
    got = reg.get_handle(key)
    assert got is h
    assert len(reg) == 1


def test_registry_get_handle_missing_returns_none():
    reg = ExpertRegistry()
    assert reg.get_handle(ExpertKey(0, 0)) is None


def test_registry_lease_prevents_reclaim_until_reader_releases():
    reg = ExpertRegistry()
    key = ExpertKey(0, 0)
    old = ExpertHandle(tier=Tier.LO)
    reg.register(key, old)
    assert reg.acquire_handle(key) is old

    replacement = ExpertHandle(tier=Tier.HI)
    reg.register(key, replacement)
    reclaimed = threading.Event()

    def wait_for_reader() -> None:
        reg.wait_until_unused(old)
        reclaimed.set()

    waiter = threading.Thread(target=wait_for_reader)
    waiter.start()
    assert not reclaimed.wait(timeout=0.05)

    fence = object()
    reg.release_handle(old, fence)
    assert reclaimed.wait(timeout=1.0)
    waiter.join()
    assert old.last_use_event is fence
    assert reg.get_handle(key) is replacement


def test_registry_rejects_unbalanced_lease_release():
    reg = ExpertRegistry()
    with pytest.raises(RuntimeError, match="without acquire"):
        reg.release_handle(ExpertHandle(tier=Tier.LO))


def test_registry_keeps_latest_fence_per_compute_stream():
    reg = ExpertRegistry()
    key = ExpertKey(0, 0)
    handle = ExpertHandle(tier=Tier.LO)
    reg.register(key, handle)

    first = object()
    replacement = object()
    other_stream = object()
    for event, stream_id in ((first, 7), (replacement, 7), (other_stream, 9)):
        assert reg.acquire_handle(key) is handle
        reg.release_handle(handle, event, stream_id)

    assert handle.last_use_events == {7: replacement, 9: other_stream}


def test_registry_register_increments_version():
    """
    Registry semantics: the first register call leaves the handle's
    version at whatever was constructed (0 by default); each subsequent
    register increments by exactly 1. This matches
    ``ExpertRegistry.register``'s ``handle.version = old.version + 1``.
    """
    reg = ExpertRegistry()
    key = ExpertKey(1, 4)

    h0 = ExpertHandle(tier=Tier.LO, block=_make_block(), quant_meta=_make_packed())
    reg.register(key, h0)
    assert reg.get_handle(key).version == 0  # no prior handle, untouched

    h1 = ExpertHandle(tier=Tier.HI, block=_make_block(), quant_meta=_make_packed(fmt=QuantFormat.FP16))
    reg.register(key, h1)
    assert reg.get_handle(key).version == 1  # 0 + 1

    h2 = ExpertHandle(tier=Tier.LO, block=_make_block(), quant_meta=_make_packed())
    reg.register(key, h2)
    assert reg.get_handle(key).version == 2  # 1 + 1


def test_registry_get_old_handle_snapshot_for_reclaim():
    """
    TransitionEngine relies on get_old_handle returning the *current*
    registered handle BEFORE register(new) is called, so it can free the
    old block in stage 4. Verify the snapshot semantics.
    """
    reg = ExpertRegistry()
    key = ExpertKey(0, 0)
    block_a = _make_block()
    h_a = ExpertHandle(tier=Tier.LO, block=block_a, quant_meta=_make_packed())
    reg.register(key, h_a)

    snapshot = reg.get_old_handle(key)
    assert snapshot is h_a
    assert snapshot.block is block_a

    block_b = _make_block()
    h_b = ExpertHandle(tier=Tier.HI, block=block_b, quant_meta=_make_packed(fmt=QuantFormat.FP16))
    reg.register(key, h_b)

    # snapshot still points at the old object — that's the whole point
    assert snapshot is h_a
    assert snapshot.block is block_a
    # but the registry now sees the new one
    assert reg.get_handle(key) is h_b


def test_registry_concurrent_register_is_thread_safe():
    """
    Hammer registry.register from many threads. After the storm settles, the
    final version must equal the total number of registrations and the
    handle must be one of the registered objects.
    """
    reg = ExpertRegistry()
    key = ExpertKey(0, 0)

    n_threads = 8
    n_per_thread = 50
    seen_handles = [None] * (n_threads * n_per_thread)

    def worker(tid: int):
        for i in range(n_per_thread):
            h = ExpertHandle(
                tier=Tier.LO,
                block=_make_block(),
                quant_meta=_make_packed(),
            )
            reg.register(key, h)
            seen_handles[tid * n_per_thread + i] = h

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = reg.get_handle(key)
    assert final is not None
    # First register leaves version=0; each subsequent register adds 1.
    # So after N total registrations the final version is N - 1.
    assert final.version == n_threads * n_per_thread - 1
    # ``final in seen_handles`` would fall through to dataclass __eq__,
    # which compares fields including torch tensors → ambiguous bool. Use
    # identity instead.
    assert any(final is h for h in seen_handles)


# ---------------------------------------------------------------------------
# Integration with PoolAllocator: handle.block survives an alloc/free cycle
# ---------------------------------------------------------------------------


def test_handle_block_id_round_trips_through_pool_allocator():
    """
    The whole point of Plan B: TransitionEngine must be able to take
    ``old_handle.block.block_id`` and pass it to
    ``pool_allocator.free(layer, tier, block_id)`` to actually release HBM.
    """
    alloc = PoolAllocator(
        num_layers=1,
        hi_pool_sizes=[4 * 1024 * 1024],
        lo_pool_sizes=[4 * 1024 * 1024],
        device=torch.device("cpu"),
        block_size_bytes=1024 * 1024,
    )

    # Allocate one HI block, wrap it in a handle, then free it via the
    # block.block_id stored on the handle.
    block = alloc.alloc(layer=0, tier=Tier.HI)
    assert block is not None
    initial_occ = alloc.occupancy(0, Tier.HI)
    assert initial_occ > 0

    h = ExpertHandle(tier=Tier.HI, block=block, quant_meta=_make_packed(fmt=QuantFormat.FP16))
    # Simulate what TransitionEngine does in stage 4:
    alloc.free(0, h.tier, h.block.block_id)

    # Block must be back in the free pool
    final_occ = alloc.occupancy(0, Tier.HI)
    assert final_occ < initial_occ
