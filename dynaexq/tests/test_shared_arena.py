from __future__ import annotations

import threading

import pytest
import torch

from dynaexq.core.config import Tier
from dynaexq.core.expert_memory import ExpertMemoryManager, MemoryRequest
from dynaexq.core.registry import ExpertKey
from dynaexq.core.shared_arena import ExtentState, SharedArenaAllocator


def _arena(capacity: int = 1024, *, alignment: int = 16) -> SharedArenaAllocator:
    return SharedArenaAllocator(capacity, torch.device("cpu"), alignment=alignment)


def test_split_best_fit_and_coalesce() -> None:
    arena = _arena()
    first = arena.allocate(100)
    second = arena.allocate(200)
    assert first is not None and first.capacity_bytes == 112
    assert second is not None and second.offset == 112
    arena.free(first)
    replacement = arena.allocate(80)
    assert replacement is not None and replacement.offset == 0
    arena.free(replacement)
    arena.free(second)
    snapshot = arena.snapshot()
    assert snapshot["free_bytes"] == 1024
    assert snapshot["largest_free_extent_bytes"] == 1024
    assert snapshot["external_fragmentation"] == 0.0


def test_no_coalescing_across_segments() -> None:
    arena = SharedArenaAllocator(
        1024,
        torch.device("cpu"),
        alignment=16,
        segment_bytes=[512, 512],
    )
    assert arena.allocate(600) is None
    snapshot = arena.snapshot()
    assert snapshot["free_bytes"] == 1024
    assert snapshot["largest_free_extent_bytes"] == 512


def test_stale_and_double_free_are_rejected() -> None:
    arena = _arena()
    extent = arena.allocate(64)
    assert extent is not None
    arena.free(extent)
    with pytest.raises(RuntimeError):
        arena.free(extent)


def test_reservation_is_all_or_nothing_and_cross_tier() -> None:
    manager = ExpertMemoryManager(_arena(256, alignment=16))
    first = manager.try_reserve_exchange(
        "x1",
        [MemoryRequest(ExpertKey(0, 0), Tier.LO, "res", 96)],
    )
    assert first is not None
    manager.publish(first.claims[0])
    failed = manager.try_reserve_exchange(
        "x2",
        [
            MemoryRequest(ExpertKey(1, 0), Tier.HI, "fid", 80),
            MemoryRequest(ExpertKey(2, 0), Tier.LO, "look", 96),
        ],
    )
    assert failed is None
    snapshot = manager.snapshot()
    assert snapshot["published_bytes"] == 96
    assert snapshot["reserved_bytes"] == 0
    assert snapshot["free_bytes"] == 160


def test_abort_keeps_published_claim_and_drops_reserved_claim() -> None:
    manager = ExpertMemoryManager(_arena(256, alignment=16))
    reservation = manager.try_reserve_exchange(
        "x",
        [
            MemoryRequest(ExpertKey(0, 0), Tier.LO, "res", 64),
            MemoryRequest(ExpertKey(1, 0), Tier.HI, "look", 64),
        ],
    )
    assert reservation is not None
    manager.publish(reservation.claims[0])
    manager.abort(reservation)
    snapshot = manager.snapshot()
    assert snapshot["published_bytes"] == 64
    assert snapshot["reserved_bytes"] == 0
    assert reservation.claims[0].extent.state == ExtentState.PUBLISHED


def test_concurrent_allocations_never_overlap_or_exceed_capacity() -> None:
    arena = _arena(4096, alignment=16)
    extents = []
    lock = threading.Lock()

    def allocate() -> None:
        extent = arena.allocate(32)
        if extent is not None:
            with lock:
                extents.append(extent)

    threads = [threading.Thread(target=allocate) for _ in range(160)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    intervals = sorted((item.offset, item.offset + item.capacity_bytes) for item in extents)
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
    assert arena.snapshot()["reserved_bytes"] <= 4096


def test_one_donor_cannot_fund_two_concurrent_exchanges() -> None:
    manager = ExpertMemoryManager(_arena(256, alignment=16))
    owner = manager.try_reserve_exchange(
        "owner", [MemoryRequest(ExpertKey(0, 0), Tier.LO, "res", 64)]
    )
    assert owner is not None
    manager.publish(owner.claims[0])
    assert manager.hold_donors("x1", owner.claims)
    assert not manager.hold_donors("x2", owner.claims)
    manager.release_donor_holds("x1")
    assert manager.hold_donors("x2", owner.claims)


def test_donor_hold_and_destinations_are_one_transaction() -> None:
    manager = ExpertMemoryManager(_arena(128, alignment=16))
    owner = manager.try_reserve_exchange(
        "owner", [MemoryRequest(ExpertKey(0, 0), Tier.LO, "res", 64)]
    )
    assert owner is not None
    manager.publish(owner.claims[0])
    failed = manager.try_reserve_exchange(
        "too-large",
        [MemoryRequest(ExpertKey(1, 0), Tier.HI, "look", 80)],
        donors=owner.claims,
    )
    assert failed is None
    assert manager.snapshot()["held_donor_count"] == 0
    accepted = manager.try_reserve_exchange(
        "fits",
        [MemoryRequest(ExpertKey(1, 0), Tier.LO, "look", 64)],
        donors=owner.claims,
    )
    assert accepted is not None
    assert manager.snapshot()["held_donor_count"] == 1
    manager.abort(accepted)
    assert manager.snapshot()["held_donor_count"] == 0


def test_rollback_can_release_published_destination_before_visibility() -> None:
    manager = ExpertMemoryManager(_arena(128, alignment=16))
    reservation = manager.try_reserve_exchange(
        "x", [MemoryRequest(ExpertKey(0, 0), Tier.LO, "res", 64)]
    )
    assert reservation is not None
    manager.publish(reservation.claims[0])
    manager.rollback_before_registry_publish(reservation)
    snapshot = manager.snapshot()
    assert snapshot["published_bytes"] == 0
    assert snapshot["free_bytes"] == 128
    assert snapshot["claim_count"] == 0


def test_exchange_cannot_finish_before_held_donor_is_reclaimed() -> None:
    manager = ExpertMemoryManager(_arena(192, alignment=16))
    owner = manager.try_reserve_exchange(
        "owner", [MemoryRequest(ExpertKey(0, 0), Tier.LO, "res", 64)]
    )
    assert owner is not None
    manager.publish(owner.claims[0])
    reservation = manager.try_reserve_exchange(
        "x",
        [MemoryRequest(ExpertKey(1, 0), Tier.LO, "look", 64)],
        donors=owner.claims,
    )
    assert reservation is not None
    with pytest.raises(RuntimeError, match="before reclaiming held donors"):
        manager.finish(reservation)
    manager.release_held_donor("x", owner.claims[0])
    manager.finish(reservation)
    assert reservation.closed


def test_one_extent_reports_residency_and_fidelity_components() -> None:
    manager = ExpertMemoryManager(_arena(256, alignment=16))
    reservation = manager.try_reserve_exchange(
        "high",
        [
            MemoryRequest(
                ExpertKey(0, 0),
                Tier.HI,
                "res",
                80,
                accounting=(48, 32, 0),
            )
        ],
    )
    assert reservation is not None
    manager.publish(reservation.claims[0])
    snapshot = manager.snapshot()
    assert snapshot["fid_bytes"] == 48
    assert snapshot["res_bytes"] == 32
    assert snapshot["look_bytes"] == 0
    assert snapshot["fid_bytes"] + snapshot["res_bytes"] == snapshot["published_bytes"]


def test_lookahead_reclassification_does_not_move_or_resize_extent() -> None:
    manager = ExpertMemoryManager(_arena(256, alignment=16))
    reservation = manager.try_reserve_exchange(
        "prefetch",
        [MemoryRequest(ExpertKey(1, 2), Tier.HI, "look", 80)],
    )
    assert reservation is not None
    claim = reservation.claims[0]
    extent = claim.extent
    assert manager.snapshot()["look_bytes"] == 80
    manager.reclassify(claim, purpose="res", accounting=(48, 32, 0))
    snapshot = manager.snapshot()
    assert claim.extent is extent
    assert snapshot["look_bytes"] == 0
    assert snapshot["fid_bytes"] == 48
    assert snapshot["res_bytes"] == 32


def test_accounting_components_must_sum_to_requested_bytes() -> None:
    with pytest.raises(ValueError, match="sum"):
        MemoryRequest(
            ExpertKey(0, 0),
            Tier.HI,
            "res",
            80,
            accounting=(40, 30, 0),
        )
