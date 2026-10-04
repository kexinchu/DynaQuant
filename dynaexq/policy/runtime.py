"""Translate controller decisions into executable shared-arena exchanges."""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field, replace

import torch

from dynaexq.core.config import Tier
from dynaexq.core.registry import ExpertHandle, ExpertKey, ExpertRegistry
from dynaexq.core.scheduler import TransitionReq
from dynaexq.core.transition_engine import TransitionEngine
from dynaexq.core.hotness_tracker import HotnessTracker
from dynaexq.policy.risk import QualityRiskAccount
from dynaexq.lease.exchange import (
    Exchange,
    Placement,
    counterfactual_gain,
    independent_gain,
)
from dynaexq.lease.plan import DonorSpec, ExchangePlan, LeaseSpec, TransitionStep


@dataclass
class RuntimePlannerStats:
    submitted: int = 0
    accepted: int = 0
    rejected: int = 0
    noops: int = 0
    records: list[dict] = field(default_factory=list)


class RuntimeExchangePlanner:
    """Build byte-accurate plans from slow- or fast-loop decisions.

    This class contains no policy search. It is the typed boundary between a
    controller and the device executor, and records whether every proposed
    exchange reached admission.
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        transition_engine: TransitionEngine,
    ) -> None:
        self.registry = registry
        self.transition_engine = transition_engine
        self.weight_store = transition_engine.weight_store
        self._ids = itertools.count(1)
        self.stats = RuntimePlannerStats()

    def submit_precision_unit(
        self,
        requests: list[TransitionReq],
        *,
        additional_donors: list[ExpertKey] | None = None,
    ) -> bool:
        """Execute one quota-preserving precision swap as one transaction."""
        plan = self.plan_precision_unit(
            requests,
            additional_donors=additional_donors,
        )
        if plan is None:
            self.stats.noops += 1
            return True
        return self._submit(plan, source="precision")

    def plan_precision_unit(
        self,
        requests: list[TransitionReq],
        *,
        additional_donors: list[ExpertKey] | None = None,
    ) -> ExchangePlan | None:
        if not requests:
            return None
        keys = [request.key for request in requests]
        if len(keys) != len(set(keys)):
            raise ValueError("precision unit repeats an expert")
        layers = {key.layer for key in keys}
        if len(layers) != 1:
            raise ValueError("one precision unit must stay within one layer")
        if len(requests) > 2:
            raise ValueError("a precision unit contains at most one swap pair")

        leases: list[LeaseSpec] = []
        donors: dict[int, DonorSpec] = {}
        changed = False
        for request in requests:
            current = self.registry.get_handle(request.key)
            if current is not None and current.tier == request.dst:
                continue
            if current is None or current.memory_claim is None:
                raise RuntimeError(
                    f"precision transition lacks a published donor: {request.key}"
                )
            changed = True
            claim = current.memory_claim
            donors[claim.claim_id] = self._donor(request.key, current)
            leases.append(
                LeaseSpec(
                    request.key,
                    request.dst,
                    "fid" if request.dst == Tier.HI else "res",
                    self.weight_store.get_byte_size(request.key, request.dst),
                    float(request.issued_step),
                    float(request.issued_step + 1),
                )
            )
        for donor_key in additional_donors or ():
            if donor_key in keys:
                raise ValueError("an additional donor cannot be a precision target")
            handle = self.registry.get_handle(donor_key)
            if handle is None or handle.memory_claim is None:
                raise RuntimeError(
                    f"additional precision donor is not published: {donor_key}"
                )
            donors[handle.memory_claim.claim_id] = self._donor(donor_key, handle)
        if not changed:
            return None
        step = min(request.issued_step for request in requests)
        return ExchangePlan(
            exchange_id=f"precision:{step}:{next(self._ids)}",
            requests=tuple(leases),
            donors=tuple(donors.values()),
            schedule=self._schedule(),
            predicted_gain_s=0.0,
        )

    def submit_exchange(self, exchange: Exchange, *, issued_step: int) -> bool:
        plan = self.plan_exchange(exchange, issued_step=issued_step)
        if plan is None:
            self.stats.noops += 1
            return True
        return self._submit(plan, source=exchange.kind)

    def plan_exchange(
        self,
        exchange: Exchange,
        *,
        issued_step: int,
    ) -> ExchangePlan | None:
        """Translate one admitted replay exchange into an executor plan."""
        if exchange.kind == "cancel":
            raise ValueError("cancellation is a release-only operation")
        if exchange.kind not in {"lookahead", "promote", "demote"}:
            raise ValueError(f"unsupported runtime exchange {exchange.kind}")
        tier = {"lo": Tier.LO, "hi": Tier.HI}[exchange.tier]
        key = ExpertKey(exchange.layer, exchange.expert)
        current = self.registry.get_handle(key)
        if exchange.kind == "lookahead" and current is not None:
            return None

        donors: dict[int, DonorSpec] = {}
        if current is not None:
            if current.memory_claim is None:
                raise RuntimeError(f"target handle has no arena claim: {key}")
            donors[current.memory_claim.claim_id] = self._donor(key, current)
        if exchange.donor_expert is not None:
            donor_key = ExpertKey(
                exchange.donor_layer
                if exchange.donor_layer is not None
                else exchange.layer,
                exchange.donor_expert,
            )
            donor_handle = self.registry.get_handle(donor_key)
            if donor_handle is None or donor_handle.memory_claim is None:
                raise RuntimeError(f"explicit donor is not published: {donor_key}")
            donors[donor_handle.memory_claim.claim_id] = self._donor(
                donor_key,
                donor_handle,
            )

        purpose = {
            "lookahead": "look",
            "promote": "fid",
            "demote": "res",
        }[exchange.kind]
        nbytes = self.weight_store.get_byte_size(key, tier)
        return ExchangePlan(
            exchange_id=f"{exchange.kind}:{issued_step}:{next(self._ids)}",
            requests=(
                LeaseSpec(
                    key,
                    tier,
                    purpose,
                    nbytes,
                    float(issued_step),
                    float(issued_step + 1),
                ),
            ),
            donors=tuple(donors.values()),
            schedule=self._schedule(),
            predicted_gain_s=exchange.gain_s,
        )

    def snapshot(self) -> dict:
        return {
            "submitted": self.stats.submitted,
            "accepted": self.stats.accepted,
            "rejected": self.stats.rejected,
            "noops": self.stats.noops,
            "records": list(self.stats.records),
        }

    def record_lookahead_outcome(
        self,
        key: ExpertKey,
        *,
        outcome: str,
        realized_gain_s: float,
    ) -> None:
        """Attach a first-use outcome to the accepted lookahead exchange."""
        encoded = f"{key.layer}:{key.expert}"
        for record in reversed(self.stats.records):
            if (
                record["source"] == "lookahead"
                and record["accepted"]
                and encoded in record["request_keys"]
                and record["outcome"] == "pending"
            ):
                record["outcome"] = outcome
                record["realized_gain_s"] = max(0.0, realized_gain_s)
                return

    def record_donor_reload(
        self,
        key: ExpertKey,
        *,
        reload_bytes: int,
        reload_cost_s: float,
    ) -> None:
        """Charge a later demand reload to the exchange that evicted donor."""
        encoded = f"{key.layer}:{key.expert}"
        for record in reversed(self.stats.records):
            if (
                record["source"] == "lookahead"
                and record["accepted"]
                and encoded in record["donor_keys"]
            ):
                record["reload_bytes"] += int(reload_bytes)
                record["reload_cost_s"] += max(0.0, reload_cost_s)
                return

    def _submit(self, plan: ExchangePlan, *, source: str) -> bool:
        self.stats.submitted += 1
        accepted = self.transition_engine.enqueue_exchange(plan)
        self.stats.accepted += int(accepted)
        self.stats.rejected += int(not accepted)
        self.stats.records.append(
            {
                "exchange_id": plan.exchange_id,
                "source": source,
                "accepted": accepted,
                "predicted_gain_s": plan.predicted_gain_s,
                "request_bytes": sum(item.nbytes for item in plan.requests),
                "request_uses": [item.purpose for item in plan.requests],
                "donor_uses": [item.purpose for item in plan.donors],
                "request_keys": [
                    f"{item.key.layer}:{item.key.expert}"
                    for item in plan.requests
                ],
                "donor_keys": [
                    f"{item.key.layer}:{item.key.expert}"
                    for item in plan.donors
                ],
                "outcome": (
                    "pending" if accepted and source == "lookahead" else None
                ),
                "realized_gain_s": 0.0,
                "reload_bytes": 0,
                "reload_cost_s": 0.0,
            }
        )
        return accepted

    @staticmethod
    def _donor(key: ExpertKey, handle: ExpertHandle) -> DonorSpec:
        assert handle.memory_claim is not None
        return DonorSpec(
            handle.memory_claim.claim_id,
            key,
            handle.memory_claim.request.purpose,
            0.0,
        )

    @staticmethod
    def _schedule() -> tuple[TransitionStep, ...]:
        return (
            TransitionStep("copy", "copy"),
            TransitionStep("publish", "publish", ("copy",)),
            TransitionStep("reclaim", "reclaim", ("publish",)),
        )


@dataclass
class RuntimeResidencyStats:
    route_events: int = 0
    selected_experts: int = 0
    resident_hits: int = 0
    demand_misses: int = 0
    prefetched_hits: int = 0
    demand_bytes: int = 0
    demand_wait_s: float = 0.0
    prefetch_submitted: int = 0
    prefetch_rejected: int = 0
    depth_increases: int = 0
    depth_decreases: int = 0
    counterfactual_candidates: int = 0
    counterfactual_rejected: int = 0
    quality_risk_rejected: int = 0
    realized_lookahead_gain_s: float = 0.0
    eviction_reload_bytes: int = 0
    eviction_reload_cost_s: float = 0.0


class RuntimeResidencyController:
    """Resolve routed misses and issue bounded cross-layer lookahead copies."""

    def __init__(
        self,
        planner: RuntimeExchangePlanner,
        tracker: HotnessTracker,
        *,
        num_layers: int,
        lookahead_depth: int = 0,
        lookahead_per_layer: int = 1,
        wait_timeout_s: float = 300.0,
        bandwidth_bytes_s: float = 20e9,
        layer_time_s: float = 1e-3,
        valuation_mode: str = "counterfactual",
        donor_mode: str = "explicit",
        quality_risk: QualityRiskAccount | None = None,
        transition_reserve_bytes: int = 0,
    ) -> None:
        if lookahead_depth < 0:
            raise ValueError("lookahead_depth must be non-negative")
        if lookahead_per_layer < 0:
            raise ValueError("lookahead_per_layer must be non-negative")
        if bandwidth_bytes_s <= 0 or layer_time_s <= 0:
            raise ValueError("bandwidth and layer time must be positive")
        if valuation_mode not in {"counterfactual", "independent"}:
            raise ValueError(f"unknown valuation mode {valuation_mode!r}")
        if donor_mode not in {"explicit", "slack_only"}:
            raise ValueError(f"unknown donor mode {donor_mode!r}")
        if transition_reserve_bytes < 0:
            raise ValueError("transition reserve must be non-negative")
        self.planner = planner
        self.registry = planner.registry
        self.transition_engine = planner.transition_engine
        self.weight_store = planner.weight_store
        self.tracker = tracker
        self.num_layers = num_layers
        self.lookahead_depth = lookahead_depth
        self._effective_depth = min(1, lookahead_depth)
        self.lookahead_per_layer = lookahead_per_layer
        self.wait_timeout_s = wait_timeout_s
        self.bandwidth_bytes_s = bandwidth_bytes_s
        self.layer_time_s = layer_time_s
        self.valuation_mode = valuation_mode
        self.donor_mode = donor_mode
        self.quality_risk = quality_risk
        self.transition_reserve_bytes = transition_reserve_bytes
        self.stats = RuntimeResidencyStats()
        self._prefetching: set[ExpertKey] = set()
        self._prefetch_donor_by_target: dict[ExpertKey, ExpertKey] = {}
        self._evicted_by_lookahead: set[ExpertKey] = set()
        self._latest_step = 0
        self._lock = threading.RLock()

    def on_route(
        self,
        layer: int,
        topk_indices: torch.Tensor,
        *,
        issued_step: int,
    ) -> None:
        selected = {
            int(expert)
            for expert in topk_indices.detach().reshape(-1).cpu().tolist()
        }
        with self._lock:
            self._latest_step = issued_step
            self.stats.route_events += 1
            self.stats.selected_experts += len(selected)
            self._adapt_lookahead_depth()
            self._issue_lookahead(layer, issued_step=issued_step)

    def ensure_selected(
        self,
        layer: int,
        selected: set[int],
        *,
        issued_step: int | None,
    ) -> None:
        """Synchronously make every routed expert visible in the registry."""
        with self._lock:
            self._ensure_selected(
                layer,
                selected,
                issued_step=(
                    self._latest_step if issued_step is None else issued_step
                ),
            )

    def precision_unit_feasible(
        self,
        requests: list[TransitionReq],
    ) -> bool:
        """Check a proposed fidelity swap without converting risk to time."""
        if self.quality_risk is None:
            return True
        high = self._high_experts()
        for request in requests:
            if request.dst == Tier.HI:
                high.add(request.key)
            else:
                high.discard(request.key)
        weights = self._routing_weights()
        current_risk = self.quality_risk.risk(
            weights, self._high_experts()
        )
        resulting_risk = self.quality_risk.risk(weights, high)
        feasible = (
            resulting_risk <= self.quality_risk.limit + self.quality_risk.tolerance
            or resulting_risk < current_risk - self.quality_risk.tolerance
        )
        if not feasible:
            self.stats.quality_risk_rejected += 1
        return feasible

    def precision_extra_donors(
        self,
        requests: list[TransitionReq],
    ) -> list[ExpertKey]:
        """Reclaim cold LO placements when fidelity growth consumes slack.

        Destination copies use the transition reserve while in flight. Extra
        donors ensure that the same reserve is available again after publish,
        turning a fidelity increase into an explicit residency-byte trade.
        """
        manager = self.transition_engine.memory_manager
        if manager is None or not requests:
            return []
        handles = self.registry.handle_snapshot()
        target_keys = {request.key for request in requests}
        request_bytes = sum(
            manager.allocator.rounded_bytes(
                self.weight_store.get_byte_size(request.key, request.dst)
            )
            for request in requests
        )
        replaced_bytes = sum(
            handle.memory_claim.extent.nbytes
            for key, handle in handles.items()
            if key in target_keys and handle.memory_claim is not None
        )
        free_bytes = int(manager.snapshot()["free_bytes"])
        post_publish_free = free_bytes - request_bytes + replaced_bytes
        needed = self.transition_reserve_bytes - post_publish_free
        if needed <= 0:
            return []
        candidates = [
            (key, handle)
            for key, handle in handles.items()
            if key not in target_keys
            and handle.tier == Tier.LO
            and handle.active_readers == 0
            and handle.memory_claim is not None
            and key not in self._prefetching
            and key not in self._prefetch_donor_by_target.values()
        ]
        candidates.sort(
            key=lambda item: (
                item[1].memory_claim.request.purpose != "look",
                self.tracker.get_score(item[0].layer, item[0].expert),
                item[0].layer,
                item[0].expert,
            )
        )
        donors = []
        reclaimed = 0
        for key, handle in candidates:
            donors.append(key)
            assert handle.memory_claim is not None
            reclaimed += handle.memory_claim.extent.nbytes
            if reclaimed >= needed:
                return donors
        raise RuntimeError(
            "fidelity repair lacks reclaimable LO residency/lookahead bytes"
        )

    def _ensure_selected(
        self,
        layer: int,
        selected: set[int],
        *,
        issued_step: int,
    ) -> None:
        protected = {ExpertKey(layer, expert) for expert in selected}
        for key in sorted(protected, key=lambda item: item.expert):
            if self.registry.get_handle(key) is not None:
                if key in self._prefetching:
                    self._prefetching.discard(key)
                    self._prefetch_donor_by_target.pop(key, None)
                    self.stats.prefetched_hits += 1
                    self._record_prefetch_hit(key, observed_wait_s=0.0)
                else:
                    self.stats.resident_hits += 1
                continue

            started = time.perf_counter()
            if key in self._prefetching:
                if not self.transition_engine.wait_ready(
                    key, timeout=self.wait_timeout_s
                ):
                    raise TimeoutError(f"prefetch timed out for routed expert {key}")
                self._prefetching.discard(key)
                self._prefetch_donor_by_target.pop(key, None)
                if self.registry.get_handle(key) is not None:
                    self.stats.prefetched_hits += 1
                    observed_wait = time.perf_counter() - started
                    self.stats.demand_wait_s += observed_wait
                    self._record_prefetch_hit(
                        key,
                        observed_wait_s=observed_wait,
                    )
                    continue

            was_eviction_reload = key in self._evicted_by_lookahead
            nbytes = self.weight_store.get_byte_size(key, Tier.LO)
            donor = self._select_donor(
                layer,
                protected,
                request_bytes=nbytes,
            )
            donor_key, donor_tier = (
                donor if donor is not None else (None, None)
            )
            exchange = Exchange(
                "lookahead",
                layer=layer,
                expert=key.expert,
                tier="lo",
                nbytes=nbytes,
                donor_layer=(donor_key.layer if donor_key is not None else None),
                donor_expert=(
                    donor_key.expert if donor_key is not None else None
                ),
                donor_tier=(
                    "hi"
                    if donor_tier == Tier.HI
                    else "lo" if donor_tier == Tier.LO else None
                ),
            )
            if not self.planner.submit_exchange(exchange, issued_step=issued_step):
                raise RuntimeError(f"demand exchange rejected for routed expert {key}")
            if not self.transition_engine.wait_ready(
                key, timeout=self.wait_timeout_s
            ):
                raise TimeoutError(f"demand exchange timed out for routed expert {key}")
            if self.registry.get_handle(key) is None:
                raise RuntimeError(f"demand exchange did not publish routed expert {key}")
            self.stats.demand_misses += 1
            self.stats.demand_bytes += nbytes
            observed_wait = time.perf_counter() - started
            self.stats.demand_wait_s += observed_wait
            if was_eviction_reload:
                self._evicted_by_lookahead.discard(key)
                self.stats.eviction_reload_bytes += nbytes
                self.stats.eviction_reload_cost_s += observed_wait
                self.planner.record_donor_reload(
                    key,
                    reload_bytes=nbytes,
                    reload_cost_s=observed_wait,
                )

    def _issue_lookahead(self, layer: int, *, issued_step: int) -> None:
        if self._effective_depth <= 0 or self.lookahead_per_layer <= 0:
            return
        upper = min(self.num_layers, layer + self._effective_depth + 1)
        # Registry placement and route-score snapshots are invariant while we
        # reject candidates.  Building them once avoids an O(candidates x
        # resident-experts) scan on every routed layer.  After an admission we
        # refresh because publication changes the exact donor state.
        counterfactual_state = None
        for future in range(layer + 1, upper):
            scores = self.tracker.get_layer_scores(future)
            if scores.size == 0 or not bool((scores > 0).any()):
                continue
            ranking = sorted(
                range(len(scores)),
                key=lambda expert: (-float(scores[expert]), expert),
            )
            issued = 0
            for expert in ranking:
                if issued >= self.lookahead_per_layer:
                    break
                key = ExpertKey(future, expert)
                if self.registry.get_handle(key) is not None or key in self._prefetching:
                    continue
                nbytes = self.weight_store.get_byte_size(key, Tier.LO)
                donor_key = None
                donor_tier = None
                if self.donor_mode == "explicit":
                    try:
                        donor = self._select_donor(
                            future,
                            {key},
                            request_bytes=nbytes,
                        )
                        donor_key, donor_tier = (
                            donor if donor is not None else (None, None)
                        )
                    except RuntimeError:
                        break
                exchange = Exchange(
                    "lookahead",
                    layer=future,
                    expert=expert,
                    tier="lo",
                    nbytes=nbytes,
                    donor_layer=(donor_key.layer if donor_key is not None else None),
                    donor_expert=(donor_key.expert if donor_key is not None else None),
                    donor_tier=(
                        "hi"
                        if donor_tier == Tier.HI
                        else "lo" if donor_tier == Tier.LO else None
                    ),
                )
                self.stats.counterfactual_candidates += 1
                if (
                    self.valuation_mode == "counterfactual"
                    and counterfactual_state is None
                ):
                    counterfactual_state = self._counterfactual_state()
                gain = self._exchange_gain(exchange, state=counterfactual_state)
                if gain is None or gain <= 0:
                    self.stats.counterfactual_rejected += 1
                    continue
                exchange = replace(exchange, gain_s=gain)
                if self.planner.submit_exchange(exchange, issued_step=issued_step):
                    self._prefetching.add(key)
                    if donor_key is not None:
                        self._prefetch_donor_by_target[key] = donor_key
                        self._evicted_by_lookahead.add(donor_key)
                    self.stats.prefetch_submitted += 1
                    issued += 1
                    counterfactual_state = None
                else:
                    self.stats.prefetch_rejected += 1
                    break

    def _record_prefetch_hit(
        self,
        key: ExpertKey,
        *,
        observed_wait_s: float,
    ) -> None:
        nbytes = self.weight_store.get_byte_size(key, Tier.LO)
        blocking_copy_s = nbytes / self.bandwidth_bytes_s
        realized = max(0.0, blocking_copy_s - observed_wait_s)
        self.stats.realized_lookahead_gain_s += realized
        self.planner.record_lookahead_outcome(
            key,
            outcome="hit",
            realized_gain_s=realized,
        )

    def _adapt_lookahead_depth(self) -> None:
        """Adjust the overlap envelope from realized prefetch usefulness."""
        if self.lookahead_depth <= 0 or self.stats.route_events % 32 != 0:
            return
        submitted = self.stats.prefetch_submitted
        if submitted < 8:
            return
        hit_rate = self.stats.prefetched_hits / submitted
        if hit_rate < 0.25 and self._effective_depth > 0:
            self._effective_depth -= 1
            self.stats.depth_decreases += 1
        elif (
            hit_rate > 0.60
            and self.stats.demand_misses > 0
            and self._effective_depth < self.lookahead_depth
        ):
            self._effective_depth += 1
            self.stats.depth_increases += 1

    def _counterfactual_state(
        self,
    ) -> tuple[list[set[int]], list[Placement]]:
        """Snapshot replay inputs shared by candidates at one route event."""
        handles = self.registry.handle_snapshot()
        placements = [Placement() for _ in range(self.num_layers)]
        for key, handle in handles.items():
            target = (
                placements[key.layer].high
                if handle.tier == Tier.HI
                else placements[key.layer].low
            )
            target.add(key.expert)
        active: list[set[int]] = []
        for layer in range(self.num_layers):
            scores = self.tracker.get_layer_scores(layer)
            positive = [
                expert for expert in range(len(scores)) if scores[expert] > 0
            ]
            positive.sort(key=lambda expert: (-float(scores[expert]), expert))
            active.append(set(positive[: min(8, len(positive))]))
        return active, placements

    def _counterfactual_gain(
        self,
        exchange: Exchange,
        *,
        state: tuple[list[set[int]], list[Placement]] | None = None,
    ) -> float | None:
        """Value one request together with the exact donor it consumes."""
        base_active, placements = (
            self._counterfactual_state() if state is None else state
        )
        active = list(base_active)
        active[exchange.layer] = set(base_active[exchange.layer])
        active[exchange.layer].add(exchange.expert)
        key = ExpertKey(exchange.layer, exchange.expert)
        lo_bytes = self.weight_store.get_byte_size(key, Tier.LO)
        hi_bytes = self.weight_store.get_byte_size(key, Tier.HI)
        manager = self.transition_engine.memory_manager
        if manager is None:
            raise RuntimeError("counterfactual admission requires the shared arena")
        return counterfactual_gain(
            active,
            placements,
            exchange,
            layer_time_s=self.layer_time_s,
            bandwidth_bytes_s=self.bandwidth_bytes_s,
            lo_bytes=lo_bytes,
            hi_bytes=hi_bytes,
            horizon=max(1, self._effective_depth),
            budget=manager.budget,
            scope="global",
        )

    def _exchange_gain(
        self,
        exchange: Exchange,
        *,
        state: tuple[list[set[int]], list[Placement]] | None = None,
    ) -> float | None:
        if self.valuation_mode == "counterfactual":
            return self._counterfactual_gain(exchange, state=state)
        request_probability = max(
            0.0,
            min(1.0, self.tracker.get_score(exchange.layer, exchange.expert)),
        )
        donor_probability = 0.0
        if exchange.donor_expert is not None:
            donor_probability = max(
                0.0,
                min(
                    1.0,
                    self.tracker.get_score(
                        exchange.donor_layer
                        if exchange.donor_layer is not None
                        else exchange.layer,
                        exchange.donor_expert,
                    ),
                ),
            )
        return independent_gain(
            exchange,
            request_probability=request_probability,
            donor_probability=donor_probability,
            bandwidth_bytes_s=self.bandwidth_bytes_s,
        )

    def _select_donor(
        self,
        layer: int,
        protected: set[ExpertKey],
        *,
        request_bytes: int,
    ) -> tuple[ExpertKey, Tier] | None:
        manager = self.transition_engine.memory_manager
        if manager is None:
            raise RuntimeError("donor selection requires the shared arena")
        arena = manager.snapshot()
        charged = manager.allocator.rounded_bytes(request_bytes)
        if (
            int(arena["free_bytes"]) - self.transition_reserve_bytes
            >= charged
        ):
            return None
        candidates = []
        for key, handle in self.registry.handle_snapshot().items():
            if (
                key.layer != layer
                or key in protected
                or key in self._prefetching
                or key in self._prefetch_donor_by_target.values()
            ):
                continue
            if handle.active_readers != 0 or handle.memory_claim is None:
                continue
            if (
                handle.tier == Tier.HI
                and self.quality_risk is not None
                and not self._risk_feasible_without(key)
            ):
                continue
            candidates.append((key, handle))
        if not candidates:
            raise RuntimeError(
                f"no reclaimable resident donor for layer {layer}; "
                "resident quota must exceed the routed working set"
            )
        key, handle = min(
            candidates,
            key=lambda item: (
                item[1].tier == Tier.HI,
                self.tracker.get_score(item[0].layer, item[0].expert),
                item[0].expert,
            ),
        )
        return key, handle.tier

    def _routing_weights(self) -> dict[ExpertKey, float]:
        return {
            ExpertKey(layer, expert): float(score)
            for layer in range(self.num_layers)
            for expert, score in enumerate(
                self.tracker.get_layer_scores(layer)
            )
        }

    def _high_experts(self) -> set[ExpertKey]:
        return {
            key
            for key, handle in self.registry.handle_snapshot().items()
            if handle.tier == Tier.HI
        }

    def _risk_feasible_without(self, key: ExpertKey) -> bool:
        assert self.quality_risk is not None
        high = self._high_experts()
        high.discard(key)
        return self.quality_risk.feasible(self._routing_weights(), high)

    def snapshot(self) -> dict:
        snapshot = {
            **vars(self.stats),
            "lookahead_depth": self.lookahead_depth,
            "effective_lookahead_depth": self._effective_depth,
            "lookahead_per_layer": self.lookahead_per_layer,
            "prefetch_inflight": len(self._prefetching),
            "bandwidth_bytes_s": self.bandwidth_bytes_s,
            "layer_time_s": self.layer_time_s,
            "valuation_mode": self.valuation_mode,
            "donor_mode": self.donor_mode,
            "transition_reserve_bytes": self.transition_reserve_bytes,
        }
        if self.quality_risk is not None:
            weights = self._routing_weights()
            high = self._high_experts()
            snapshot["quality_risk"] = {
                **self.quality_risk.snapshot(),
                "current": self.quality_risk.risk(weights, high),
                "feasible": self.quality_risk.feasible(weights, high),
            }
        return snapshot


__all__ = [
    "RuntimeExchangePlanner",
    "RuntimePlannerStats",
    "RuntimeResidencyController",
    "RuntimeResidencyStats",
]
