"""Envelope controller on one expert-memory arena.

This is the design's fast/slow loop: ``S_t`` changes by at most one layer from
stall and pollution, lookahead tiers of one expert are mutually exclusive, and
a fidelity lease cannot be spent before its tenure ends. The per-layer
controller in ``online.py`` remains the fixed-horizon baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dynaexq.lease.envelope import EnvelopeState, initial_depth, pollution_ratio, update_envelope
from dynaexq.lease.exchange import Exchange, Placement, admit_exchanges, placement_bytes
from dynaexq.lease.replay import simulate_forward
from dynaexq.policy.online import _rank


@dataclass
class AdaptiveStats:
    exposed_s: list[float] = field(default_factory=list)
    exchanges: list[Exchange] = field(default_factory=list)
    depths: list[int] = field(default_factory=list)
    pollution: list[float] = field(default_factory=list)
    expired_prefetches: int = 0
    converted_prefetches: int = 0


def _tick_holds(placements: list[Placement]) -> None:
    for placement in placements:
        placement.hold = {
            expert: remaining - 1
            for expert, remaining in placement.hold.items()
            if remaining > 1
        }


def _predicted(counts: list[int], width: int) -> set[int]:
    return set(_rank(counts)[:width])


def _donor_for(
    placement: Placement,
    counts: list[int],
    protected: set[int],
    *,
    nbytes: int,
    lo_bytes: int,
    hi_bytes: int,
    budget: int,
    placements: list[Placement],
) -> tuple[int, int, str] | None:
    """Return ``(layer is filled by caller, expert, tier)`` or None when slack suffices."""
    if placement_bytes(placements, lo_bytes, hi_bytes) + nbytes <= budget:
        return None
    for _issue, expert, tier in placement.prefetches:
        if expert not in protected:
            return expert, tier, "prefetch"
    outside = [expert for expert in placement.low if expert not in protected and placement.hold.get(expert, 0) <= 0]
    if outside:
        expert = min(outside, key=lambda item: (counts[item], item))
        return expert, "lo", "resident"
    protected_low = [expert for expert in placement.low if placement.hold.get(expert, 0) <= 0]
    if protected_low:
        expert = min(protected_low, key=lambda item: (counts[item], item))
        return expert, "lo", "resident"
    return None


def run_adaptive(
    trials: list[list[set[int]]],
    counts: list[list[int]],
    placements: list[Placement],
    *,
    layer_time_s: float,
    bandwidth_bytes_s: float,
    lo_bytes: int,
    hi_bytes: int,
    budget: int,
    predict_width: int,
    absent_per_layer: float,
    s_min: int = 1,
    s_max: int = 4,
    beta: float = 0.0,
    lam: float = 1.0,
    theta: float = 0.2,
    hold_for: int = 2,
    slow_every: int = 2,
    min_high: int = 0,
    admissions_per_layer: int = 4,
    candidate_cap: int = 8,
) -> AdaptiveStats:
    stats = AdaptiveStats()
    envelope = EnvelopeState(
        initial_depth(
            absent_per_layer,
            lo_bytes,
            bandwidth_bytes_s,
            layer_time_s,
            s_min=s_min,
            s_max=s_max,
        )
    )
    for trial_index, trial in enumerate(trials):
        for placement in placements:
            placement.prefetches.clear()
        if trial_index % slow_every == 0:
            _repair_quality(
                placements,
                counts,
                min_high=min_high,
                budget=budget,
                lo_bytes=lo_bytes,
                hi_bytes=hi_bytes,
                hold_for=hold_for,
            )
        admitted_bytes = 0
        period_exchanges: list[Exchange] = []
        for index in range(len(trial)):
            depth = envelope.depth
            candidates: list[Exchange] = []
            forecast = [_predicted(counts[layer], predict_width) for layer in range(len(trial))]
            for past in range(index + 1):
                forecast[past] = set(trial[past])
            last = min(len(trial), index + depth + 1)
            for future in range(index + 1, last):
                predicted = forecast[future]
                resident = placements[future].high | placements[future].low
                for _issue, expert, _tier in list(placements[future].prefetches):
                    if expert not in predicted:
                        candidates.append(
                            Exchange("cancel", layer=future, expert=expert, tier=_tier, nbytes=lo_bytes)
                        )
                missing = [expert for expert in _rank(counts[future]) if expert in predicted - resident]
                for expert in missing[:candidate_cap]:
                    for tier, nbytes in (("lo", lo_bytes), ("hi", hi_bytes)):
                        donor = _donor_for(
                            placements[future],
                            counts[future],
                            predicted,
                            nbytes=nbytes,
                            lo_bytes=lo_bytes,
                            hi_bytes=hi_bytes,
                            budget=budget,
                            placements=placements,
                        )
                        donor_expert = None if donor is None else donor[0]
                        donor_tier = None if donor is None else donor[1]
                        candidates.append(
                            Exchange(
                                kind="lookahead",
                                layer=future,
                                expert=expert,
                                tier=tier,
                                nbytes=nbytes,
                                donor_layer=future if donor_expert is not None else None,
                                donor_expert=donor_expert,
                                donor_tier="lo" if donor_tier in {None, "prefetch"} else donor_tier,
                            )
                        )
            admitted = admit_exchanges(
                forecast,
                placements,
                candidates,
                layer_time_s=layer_time_s,
                bandwidth_bytes_s=bandwidth_bytes_s,
                lo_bytes=lo_bytes,
                hi_bytes=hi_bytes,
                horizon=depth,
                budget=budget,
                limit=admissions_per_layer,
                scope="global",
            )
            stats.exchanges.extend(admitted)
            period_exchanges.extend(admitted)
            admitted_bytes += sum(item.nbytes for item in admitted if item.kind == "lookahead")
        plan = [[] for _ in trial]
        for destination, placement in enumerate(placements):
            for issue, expert, tier in placement.prefetches:
                plan[issue].append((destination, expert, tier))
        result = simulate_forward(
            trial,
            [item.as_resident() for item in placements],
            plan,
            layer_time_s=layer_time_s,
            bandwidth_bytes_s=bandwidth_bytes_s,
            lo_bytes=lo_bytes,
            hi_bytes=hi_bytes,
        )
        expired_bytes = 0
        for layer, placement in enumerate(placements):
            for _issue, expert, tier in placement.prefetches:
                size = hi_bytes if tier == "hi" else lo_bytes
                if expert in trial[layer]:
                    target = placement.high if tier == "hi" else placement.low
                    target.add(expert)
                    stats.converted_prefetches += 1
                else:
                    expired_bytes += size
                    stats.expired_prefetches += 1
            placement.prefetches.clear()
        reload_bytes = 0
        for exchange in period_exchanges:
            if (
                exchange.donor_expert is not None
                and exchange.donor_layer is not None
                and exchange.donor_expert in trial[exchange.donor_layer]
            ):
                reload_bytes += lo_bytes
        ratio = pollution_ratio(expired_bytes, reload_bytes, admitted_bytes, lo_bytes)
        routed_time = max(layer_time_s, len(trial) * layer_time_s)
        stall = result.exposed_s / routed_time
        envelope = update_envelope(
            envelope,
            stall=stall,
            pollution=ratio,
            beta=beta,
            lam=lam,
            theta=theta,
            s_min=s_min,
            s_max=s_max,
        )
        stats.exposed_s.append(result.exposed_s)
        stats.depths.append(envelope.depth)
        stats.pollution.append(ratio)
        _tick_holds(placements)
        for layer, needed in enumerate(trial):
            for expert in needed:
                counts[layer][expert] += 1
    return stats


def _repair_quality(
    placements: list[Placement],
    counts: list[list[int]],
    *,
    min_high: int,
    budget: int,
    lo_bytes: int,
    hi_bytes: int,
    hold_for: int,
) -> None:
    """Promote the cheapest low residents until every layer meets the floor."""
    increment = hi_bytes - lo_bytes
    for placement, layer_counts in zip(placements, counts):
        while len(placement.high) < min_high:
            lows = [expert for expert in _rank(layer_counts) if expert in placement.low]
            if not lows:
                break
            promote = lows[0]
            if placement_bytes(placements, lo_bytes, hi_bytes) + increment <= budget:
                placement.low.remove(promote)
                placement.high.add(promote)
                placement.hold[promote] = hold_for
                continue
            victims = [expert for expert in reversed(_rank(layer_counts)) if expert in placement.low and expert != promote]
            need = increment
            chosen: list[int] = []
            freed = max(0, budget - placement_bytes(placements, lo_bytes, hi_bytes))
            for expert in victims:
                if freed >= need:
                    break
                chosen.append(expert)
                freed += lo_bytes
            if freed < need:
                break
            for expert in chosen:
                placement.low.remove(expert)
            placement.low.remove(promote)
            placement.high.add(promote)
            placement.hold[promote] = hold_for
