from __future__ import annotations

import torch

from dynaexq.core.config import Tier
from dynaexq.core.expert_memory import ExpertMemoryManager
from dynaexq.core.quant import QuantFormat, compute_packed_nbytes, pack
from dynaexq.core.registry import ExpertKey, ExpertRegistry
from dynaexq.core.scheduler import TransitionReq
from dynaexq.core.shared_arena import SharedArenaAllocator
from dynaexq.core.transition_engine import TransitionEngine
from dynaexq.lease.plan import DonorSpec, ExchangePlan, LeaseSpec, TransitionStep


class _Store:
    def __init__(self) -> None:
        self.weight = pack(torch.zeros(4, 64, dtype=torch.float16), QuantFormat.FP16)

    def load_weights(self, key: ExpertKey, tier: Tier):
        return self.weight

    def get_byte_size(self, key: ExpertKey, tier: Tier) -> int:
        return compute_packed_nbytes(4, 64, QuantFormat.FP16, 64)


class _TieredFootprintStore(_Store):
    def get_byte_size(self, key: ExpertKey, tier: Tier) -> int:
        return 1024 if tier == Tier.HI else 512


def _engine(capacity: int) -> tuple[TransitionEngine, ExpertMemoryManager, ExpertRegistry]:
    arena = SharedArenaAllocator(capacity, torch.device("cpu"), alignment=16)
    manager = ExpertMemoryManager(arena)
    registry = ExpertRegistry()
    engine = TransitionEngine(
        registry=registry,
        pool_allocator=None,
        weight_store=_Store(),  # type: ignore[arg-type]
        max_workers=1,
        max_inflight=2,
        memory_manager=manager,
        synchronous=True,
    )
    return engine, manager, registry


def _tiered_engine(capacity: int) -> tuple[TransitionEngine, ExpertMemoryManager, ExpertRegistry]:
    arena = SharedArenaAllocator(capacity, torch.device("cpu"), alignment=16)
    manager = ExpertMemoryManager(arena)
    registry = ExpertRegistry()
    engine = TransitionEngine(
        registry=registry,
        pool_allocator=None,
        weight_store=_TieredFootprintStore(),  # type: ignore[arg-type]
        max_workers=1,
        max_inflight=2,
        memory_manager=manager,
        synchronous=True,
    )
    return engine, manager, registry


def _request(key: ExpertKey, src: Tier, dst: Tier, step: int) -> TransitionReq:
    return TransitionReq(key, src, dst, "test", step)


def test_transition_uses_one_arena_across_tiers_and_reclaims_old_extent() -> None:
    engine, manager, registry = _engine(2048)
    key = ExpertKey(0, 0)
    try:
        assert engine.enqueue(_request(key, Tier.LO, Tier.LO, 0))
        first = registry.get_handle(key)
        assert first is not None and first.memory_claim is not None
        assert manager.snapshot()["published_bytes"] == 512

        assert engine.enqueue(_request(key, Tier.LO, Tier.HI, 1))
        second = registry.get_handle(key)
        assert second is not None and second is not first
        assert second.memory_claim is not None
        snapshot = manager.snapshot()
        assert snapshot["published_bytes"] == 512
        assert snapshot["reserved_bytes"] == 0
        assert snapshot["free_bytes"] == 1536
        assert "pool" not in engine.get_stats()
    finally:
        engine.shutdown()


def test_rejected_destination_keeps_published_handle_and_arena_state() -> None:
    engine, manager, registry = _engine(512)
    key = ExpertKey(0, 0)
    try:
        assert engine.enqueue(_request(key, Tier.LO, Tier.LO, 0))
        published = registry.get_handle(key)
        before = manager.snapshot()
        assert not engine.enqueue(_request(key, Tier.LO, Tier.HI, 1))
        assert registry.get_handle(key) is published
        after = manager.snapshot()
        for field in ("published_bytes", "reserved_bytes", "free_bytes", "claim_count"):
            assert after[field] == before[field]
        assert after["reservation_failures_budget"] == 1
    finally:
        engine.shutdown()


def test_registry_publish_failure_rolls_back_destination(monkeypatch) -> None:
    engine, manager, registry = _engine(1024)
    key = ExpertKey(0, 0)
    before = manager.snapshot()

    def fail_register(_key, _handle) -> None:
        raise RuntimeError("injected registry failure")

    monkeypatch.setattr(registry, "register", fail_register)
    try:
        assert engine.enqueue(_request(key, Tier.LO, Tier.LO, 0))
        after = manager.snapshot()
        assert registry.get_handle(key) is None
        for field in ("published_bytes", "reserved_bytes", "free_bytes", "claim_count"):
            assert after[field] == before[field]
        assert engine.get_stats()["failed_transitions"] == 1
    finally:
        engine.shutdown()


def test_transition_engine_attributes_high_increment_to_fidelity() -> None:
    engine, manager, _registry = _tiered_engine(2048)
    try:
        key = ExpertKey(0, 0)
        assert engine.enqueue(_request(key, Tier.HI, Tier.HI, 0))
        snapshot = manager.snapshot()
        assert snapshot["published_bytes"] == 1024
        assert snapshot["res_bytes"] == 512
        assert snapshot["fid_bytes"] == 512
        assert snapshot["look_bytes"] == 0
    finally:
        engine.shutdown()


def test_exchange_plan_atomically_replaces_donor_with_lookahead() -> None:
    engine, manager, registry = _tiered_engine(1024)
    donor_key = ExpertKey(0, 0)
    request_key = ExpertKey(1, 1)
    try:
        assert engine.enqueue(_request(donor_key, Tier.LO, Tier.LO, 0))
        donor = registry.get_handle(donor_key)
        assert donor is not None and donor.memory_claim is not None
        plan = ExchangePlan(
            "exchange-1",
            (LeaseSpec(request_key, Tier.LO, "look", 512, 0.0, 1.0),),
            (
                DonorSpec(
                    donor.memory_claim.claim_id,
                    donor_key,
                    "res",
                    0.0,
                ),
            ),
            (
                TransitionStep("copy", "copy"),
                TransitionStep("publish", "publish", ("copy",)),
                TransitionStep("reclaim", "reclaim", ("publish",)),
            ),
            0.01,
        )
        assert engine.enqueue_exchange(plan)
        assert registry.get_handle(donor_key) is None
        requested = registry.get_handle(request_key)
        assert requested is not None and requested.tier == Tier.LO
        snapshot = manager.snapshot()
        assert snapshot["published_bytes"] == 512
        assert snapshot["look_bytes"] == 512
        assert snapshot["res_bytes"] == 0
        assert snapshot["held_donor_count"] == 0
        acquired = registry.acquire_handle(request_key)
        assert acquired is requested
        consumed = manager.snapshot()
        assert consumed["look_bytes"] == 0
        assert consumed["res_bytes"] == 512
        registry.release_handle(acquired)
    finally:
        engine.shutdown()


def test_exchange_plan_rejects_stale_donor_without_mutation() -> None:
    engine, manager, registry = _tiered_engine(1024)
    donor_key = ExpertKey(0, 0)
    request_key = ExpertKey(1, 1)
    try:
        assert engine.enqueue(_request(donor_key, Tier.LO, Tier.LO, 0))
        donor = registry.get_handle(donor_key)
        assert donor is not None and donor.memory_claim is not None
        before = manager.snapshot()
        plan = ExchangePlan(
            "stale",
            (LeaseSpec(request_key, Tier.LO, "look", 512, 0.0, 1.0),),
            (DonorSpec(donor.memory_claim.claim_id + 1, donor_key, "res", 0.0),),
            (),
            0.01,
        )
        assert not engine.enqueue_exchange(plan)
        assert registry.get_handle(donor_key) is donor
        assert registry.get_handle(request_key) is None
        after = manager.snapshot()
        for field in ("published_bytes", "reserved_bytes", "claim_count"):
            assert after[field] == before[field]
    finally:
        engine.shutdown()
