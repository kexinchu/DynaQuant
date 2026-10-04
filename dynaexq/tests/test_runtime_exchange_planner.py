from __future__ import annotations

import torch

from dynaexq.core.config import Tier
from dynaexq.core.expert_memory import ExpertMemoryManager
from dynaexq.core.quant import QuantFormat, pack
from dynaexq.core.registry import ExpertKey, ExpertRegistry
from dynaexq.core.scheduler import TransitionReq
from dynaexq.core.hotness_tracker import HotnessTracker
from dynaexq.core.shared_arena import SharedArenaAllocator
from dynaexq.core.transition_engine import TransitionEngine
from dynaexq.lease.exchange import Exchange
from dynaexq.policy.runtime import RuntimeExchangePlanner, RuntimeResidencyController
from dynaexq.policy.risk import QualityRiskAccount


class _Store:
    def __init__(self) -> None:
        self.weight = pack(
            torch.zeros(4, 64, dtype=torch.float16),
            QuantFormat.FP16,
        )

    def load_weights(self, key: ExpertKey, tier: Tier):
        return self.weight

    def get_byte_size(self, key: ExpertKey, tier: Tier) -> int:
        return 1024 if tier == Tier.HI else 512


def _runtime(capacity: int = 4096):
    manager = ExpertMemoryManager(
        SharedArenaAllocator(capacity, torch.device("cpu"), alignment=16)
    )
    registry = ExpertRegistry()
    engine = TransitionEngine(
        registry=registry,
        pool_allocator=None,
        weight_store=_Store(),  # type: ignore[arg-type]
        max_workers=1,
        max_inflight=4,
        memory_manager=manager,
        synchronous=True,
    )
    return engine, manager, registry, RuntimeExchangePlanner(registry, engine)


def _transition(key: ExpertKey, src: Tier, dst: Tier, step: int = 1):
    return TransitionReq(key, src, dst, "test", step)


def test_precision_pair_is_one_quota_preserving_exchange() -> None:
    engine, manager, registry, planner = _runtime()
    hot = ExpertKey(0, 0)
    cold = ExpertKey(0, 1)
    try:
        assert engine.enqueue(_transition(hot, Tier.LO, Tier.LO, 0))
        assert engine.enqueue(_transition(cold, Tier.HI, Tier.HI, 0))
        assert planner.submit_precision_unit(
            [
                _transition(cold, Tier.HI, Tier.LO),
                _transition(hot, Tier.LO, Tier.HI),
            ]
        )
        assert registry.get_handle(hot).tier == Tier.HI
        assert registry.get_handle(cold).tier == Tier.LO
        snapshot = manager.snapshot()
        assert snapshot["published_bytes"] == 1536
        assert snapshot["fid_bytes"] == 512
        assert snapshot["res_bytes"] == 1024
        assert planner.snapshot()["accepted"] == 1
    finally:
        engine.shutdown()


def test_lookahead_exchange_keeps_transient_purpose_until_dispatch() -> None:
    engine, manager, registry, planner = _runtime(1536)
    donor_key = ExpertKey(0, 0)
    target_key = ExpertKey(1, 1)
    try:
        assert engine.enqueue(_transition(donor_key, Tier.LO, Tier.LO, 0))
        exchange = Exchange(
            "lookahead",
            layer=target_key.layer,
            expert=target_key.expert,
            tier="lo",
            nbytes=512,
            donor_layer=donor_key.layer,
            donor_expert=donor_key.expert,
            donor_tier="lo",
            gain_s=0.01,
        )
        assert planner.submit_exchange(exchange, issued_step=3)
        assert registry.get_handle(donor_key) is None
        assert manager.snapshot()["look_bytes"] == 512
        handle = registry.acquire_handle(target_key)
        assert handle is not None
        assert manager.snapshot()["look_bytes"] == 0
        assert manager.snapshot()["res_bytes"] == 512
        registry.release_handle(handle)
    finally:
        engine.shutdown()


def test_runtime_plan_rejects_declared_size_drift_at_executor_boundary() -> None:
    engine, _manager, _registry, planner = _runtime()
    try:
        exchange = Exchange(
            "lookahead",
            layer=0,
            expert=1,
            tier="lo",
            nbytes=1,
            gain_s=0.01,
        )
        plan = planner.plan_exchange(exchange, issued_step=0)
        assert plan is not None
        # The planner resolves the authoritative footprint instead of trusting
        # replay metadata carried by Exchange.nbytes.
        assert plan.requests[0].nbytes == 512
    finally:
        engine.shutdown()


def test_residency_controller_atomically_replaces_a_cold_same_layer_donor() -> None:
    engine, manager, registry, planner = _runtime(1536)
    first = ExpertKey(0, 0)
    donor = ExpertKey(0, 1)
    target = ExpertKey(0, 2)
    try:
        assert engine.enqueue(_transition(first, Tier.LO, Tier.LO, 0))
        assert engine.enqueue(_transition(donor, Tier.LO, Tier.LO, 0))
        tracker = HotnessTracker(1, 3, alpha=0.5)
        tracker.update(0, {first.expert: 1.0, target.expert: 0.9})
        controller = RuntimeResidencyController(
            planner,
            tracker,
            num_layers=1,
            transition_reserve_bytes=1024,
        )
        controller.ensure_selected(0, {first.expert, target.expert}, issued_step=7)
        assert registry.get_handle(first) is not None
        assert registry.get_handle(target) is not None
        assert registry.get_handle(donor) is None
        assert manager.snapshot()["published_bytes"] == 1024
        snapshot = controller.snapshot()
        assert snapshot["resident_hits"] == 1
        assert snapshot["demand_misses"] == 1
        assert snapshot["demand_bytes"] == 512
    finally:
        engine.shutdown()


def test_residency_controller_admits_positive_counterfactual_lookahead() -> None:
    engine, manager, registry, planner = _runtime(2048)
    layer0 = ExpertKey(0, 0)
    donor = ExpertKey(1, 0)
    target = ExpertKey(1, 1)
    try:
        assert engine.enqueue(_transition(layer0, Tier.LO, Tier.LO, 0))
        assert engine.enqueue(_transition(donor, Tier.LO, Tier.LO, 0))
        tracker = HotnessTracker(2, 2, alpha=0.5)
        tracker.update(1, {target.expert: 1.0})
        controller = RuntimeResidencyController(
            planner,
            tracker,
            num_layers=2,
            lookahead_depth=1,
            lookahead_per_layer=1,
            transition_reserve_bytes=1024,
        )
        controller.on_route(
            0,
            torch.tensor([[layer0.expert]]),
            issued_step=4,
        )
        assert registry.get_handle(target) is not None
        assert registry.get_handle(donor) is None
        assert manager.snapshot()["look_bytes"] == 512
        assert planner.snapshot()["records"][-1]["predicted_gain_s"] > 0
        controller.ensure_selected(1, {target.expert}, issued_step=4)
        handle = registry.acquire_handle(target)
        assert handle is not None
        registry.release_handle(handle)
        assert manager.snapshot()["look_bytes"] == 0
        assert manager.snapshot()["res_bytes"] == 1024
        assert controller.snapshot()["prefetched_hits"] == 1
        record = planner.snapshot()["records"][-1]
        assert record["outcome"] == "hit"
        assert record["realized_gain_s"] > 0.0

        controller.ensure_selected(1, {donor.expert}, issued_step=5)
        record = planner.snapshot()["records"][0]
        assert record["reload_bytes"] == 512
        assert record["reload_cost_s"] >= 0.0
        assert controller.snapshot()["eviction_reload_bytes"] == 512
    finally:
        engine.shutdown()


def test_lookahead_reuses_one_counterfactual_snapshot_for_rejections() -> None:
    engine, _manager, registry, planner = _runtime(4096)
    current = ExpertKey(0, 0)
    donor = ExpertKey(1, 0)
    try:
        assert engine.enqueue(_transition(current, Tier.LO, Tier.LO, 0))
        assert engine.enqueue(_transition(donor, Tier.LO, Tier.LO, 0))
        tracker = HotnessTracker(2, 4, alpha=0.5)
        tracker.update(1, {1: 1.0, 2: 0.8, 3: 0.6})
        controller = RuntimeResidencyController(
            planner,
            tracker,
            num_layers=2,
            lookahead_depth=1,
            lookahead_per_layer=1,
        )
        snapshots = 0
        original = controller._counterfactual_state

        def counted_state():
            nonlocal snapshots
            snapshots += 1
            return original()

        controller._counterfactual_state = counted_state  # type: ignore[method-assign]
        controller._counterfactual_gain = (  # type: ignore[method-assign]
            lambda exchange, *, state=None: -1.0
        )

        controller.on_route(0, torch.tensor([[0]]), issued_step=1)

        assert snapshots == 1
        assert controller.snapshot()["counterfactual_candidates"] == 3
        assert controller.snapshot()["counterfactual_rejected"] == 3
        assert registry.get_handle(donor) is not None
    finally:
        engine.shutdown()


def test_independent_value_ablation_scores_request_and_donor_separately() -> None:
    engine, _manager, registry, planner = _runtime(2048)
    current = ExpertKey(0, 0)
    donor = ExpertKey(1, 0)
    target = ExpertKey(1, 1)
    try:
        assert engine.enqueue(_transition(current, Tier.LO, Tier.LO, 0))
        assert engine.enqueue(_transition(donor, Tier.LO, Tier.LO, 0))
        tracker = HotnessTracker(2, 2, alpha=0.5)
        tracker.update(1, {target.expert: 1.0})
        controller = RuntimeResidencyController(
            planner,
            tracker,
            num_layers=2,
            lookahead_depth=1,
            lookahead_per_layer=1,
            valuation_mode="independent",
            transition_reserve_bytes=1024,
        )

        controller.on_route(0, torch.tensor([[0]]), issued_step=1)

        assert registry.get_handle(target) is not None
        assert registry.get_handle(donor) is None
        assert planner.snapshot()["records"][-1]["predicted_gain_s"] > 0
        assert controller.snapshot()["valuation_mode"] == "independent"
    finally:
        engine.shutdown()


def test_residency_controller_uses_only_slack_above_transition_reserve() -> None:
    engine, manager, registry, planner = _runtime(3072)
    current = ExpertKey(0, 0)
    resident = ExpertKey(1, 0)
    target = ExpertKey(1, 1)
    try:
        assert engine.enqueue(_transition(current, Tier.LO, Tier.LO, 0))
        assert engine.enqueue(_transition(resident, Tier.LO, Tier.LO, 0))
        tracker = HotnessTracker(2, 2, alpha=0.5)
        tracker.update(1, {target.expert: 1.0})
        controller = RuntimeResidencyController(
            planner,
            tracker,
            num_layers=2,
            lookahead_depth=1,
            lookahead_per_layer=1,
            transition_reserve_bytes=1024,
        )

        controller.on_route(0, torch.tensor([[0]]), issued_step=1)

        assert registry.get_handle(target) is not None
        assert registry.get_handle(resident) is not None
        assert manager.snapshot()["free_bytes"] == 1536
        assert planner.snapshot()["records"][-1]["donor_uses"] == []
    finally:
        engine.shutdown()


def test_quality_risk_rejects_a_precision_swap_that_drops_required_hi() -> None:
    engine, _manager, registry, planner = _runtime()
    required = ExpertKey(0, 0)
    replacement = ExpertKey(0, 1)
    try:
        assert engine.enqueue(_transition(required, Tier.HI, Tier.HI, 0))
        assert engine.enqueue(_transition(replacement, Tier.LO, Tier.LO, 0))
        tracker = HotnessTracker(1, 2, alpha=0.5)
        tracker.seed_scores([[0.8, 0.2]])
        risk = QualityRiskAccount(
            {required: 1.0, replacement: 0.1},
            limit=0.2,
        )
        controller = RuntimeResidencyController(
            planner,
            tracker,
            num_layers=1,
            quality_risk=risk,
        )

        feasible = controller.precision_unit_feasible(
            [
                _transition(required, Tier.HI, Tier.LO),
                _transition(replacement, Tier.LO, Tier.HI),
            ]
        )

        assert not feasible
        assert controller.snapshot()["quality_risk_rejected"] == 1
    finally:
        engine.shutdown()


def test_quality_risk_protects_required_hi_from_residency_eviction() -> None:
    engine, _manager, registry, planner = _runtime(2048)
    required = ExpertKey(0, 0)
    cold_low = ExpertKey(0, 1)
    target = ExpertKey(0, 2)
    try:
        assert engine.enqueue(_transition(required, Tier.HI, Tier.HI, 0))
        assert engine.enqueue(_transition(cold_low, Tier.LO, Tier.LO, 0))
        tracker = HotnessTracker(1, 3, alpha=0.5)
        tracker.seed_scores([[0.8, 0.0, 0.2]])
        risk = QualityRiskAccount(
            {required: 1.0, cold_low: 0.1, target: 0.1},
            limit=0.1,
        )
        controller = RuntimeResidencyController(
            planner,
            tracker,
            num_layers=1,
            quality_risk=risk,
            transition_reserve_bytes=1024,
        )

        controller.ensure_selected(0, {target.expert}, issued_step=2)

        assert registry.get_handle(required) is not None
        assert registry.get_handle(cold_low) is None
        assert registry.get_handle(target) is not None
    finally:
        engine.shutdown()


def test_fidelity_growth_reclaims_residency_to_restore_transition_reserve() -> None:
    engine, manager, registry, planner = _runtime(3072)
    target = ExpertKey(0, 0)
    cold_resident = ExpertKey(0, 1)
    try:
        assert engine.enqueue(_transition(target, Tier.LO, Tier.LO, 0))
        assert engine.enqueue(_transition(cold_resident, Tier.LO, Tier.LO, 0))
        tracker = HotnessTracker(1, 2, alpha=0.5)
        tracker.seed_scores([[1.0, 0.0]])
        controller = RuntimeResidencyController(
            planner,
            tracker,
            num_layers=1,
            transition_reserve_bytes=2048,
        )
        promotion = [_transition(target, Tier.LO, Tier.HI, 1)]

        donors = controller.precision_extra_donors(promotion)
        assert donors == [cold_resident]
        assert planner.submit_precision_unit(
            promotion,
            additional_donors=donors,
        )

        assert registry.get_handle(target).tier == Tier.HI
        assert registry.get_handle(cold_resident) is None
        assert manager.snapshot()["free_bytes"] == 2048
        assert planner.snapshot()["records"][-1]["donor_uses"] == ["res", "res"]
    finally:
        engine.shutdown()


def test_slack_only_ablation_prefetches_without_naming_a_donor() -> None:
    engine, manager, registry, planner = _runtime(2048)
    current = ExpertKey(0, 0)
    resident = ExpertKey(1, 0)
    target = ExpertKey(1, 1)
    try:
        assert engine.enqueue(_transition(current, Tier.LO, Tier.LO, 0))
        assert engine.enqueue(_transition(resident, Tier.LO, Tier.LO, 0))
        tracker = HotnessTracker(2, 2, alpha=0.5)
        tracker.update(1, {target.expert: 1.0})
        controller = RuntimeResidencyController(
            planner,
            tracker,
            num_layers=2,
            lookahead_depth=1,
            lookahead_per_layer=1,
            donor_mode="slack_only",
        )

        controller.on_route(0, torch.tensor([[0]]), issued_step=1)

        assert registry.get_handle(target) is not None
        assert registry.get_handle(resident) is not None
        assert manager.snapshot()["look_bytes"] == 512
        assert planner.snapshot()["records"][-1]["donor_uses"] == []
        assert controller.snapshot()["donor_mode"] == "slack_only"
    finally:
        engine.shutdown()
