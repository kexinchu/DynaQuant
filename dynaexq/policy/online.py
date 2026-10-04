"""Online byte-exchange controller driven by a frequency predictor.

Residents persist across requests. Lookahead leases live for one request and
convert to residency only when the prefetched expert is actually used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dynaexq.lease.exchange import (
    Exchange,
    Placement,
    admit_exchanges,
    independent_gain,
)
from dynaexq.lease.replay import simulate_forward


@dataclass
class OnlineStats:
    exposed_s: list[float] = field(default_factory=list)
    exchanges: list[Exchange] = field(default_factory=list)
    expired_prefetches: int = 0
    converted_prefetches: int = 0


def _rank(counts: list[int]) -> list[int]:
    return sorted(range(len(counts)), key=lambda expert: (-counts[expert], expert))


def initial_placement(
    counts: list[list[int]],
    *,
    n_high: int,
    n_low: int,
    n_experts: int,
) -> list[Placement]:
    placements = []
    for layer_counts in counts:
        ranking = [expert for expert in _rank(layer_counts) if expert < n_experts]
        high = set(ranking[:n_high])
        low = set(ranking[n_high : n_high + n_low])
        placements.append(Placement(high=high, low=low))
    return placements


def _enforce_quality(
    placements: list[Placement],
    counts: list[list[int]],
    *,
    min_high: int,
    budget: int,
    lo_bytes: int,
    hi_bytes: int,
) -> None:
    """Promote the hottest low residents until the fidelity floor fits.

    A promotion needs ``hi - lo`` extra bytes. When slack is short, the coldest
    other low residents are the donors. One donor is not enough when the high
    tier is more than twice the low tier.
    """
    increment = hi_bytes - lo_bytes
    for layer, layer_counts in zip(placements, counts):
        while len(layer.high) < min_high and layer.low:
            ranking = _rank(layer_counts)
            promote = next(expert for expert in ranking if expert in layer.low)
            if layer.total_bytes(lo_bytes, hi_bytes) + increment <= budget:
                layer.low.remove(promote)
                layer.high.add(promote)
                continue
            victims = [
                expert
                for expert in reversed(ranking)
                if expert in layer.low and expert != promote
            ]
            need = increment - max(0, budget - layer.total_bytes(lo_bytes, hi_bytes))
            chosen: list[int] = []
            freed = 0
            for expert in victims:
                chosen.append(expert)
                freed += lo_bytes
                if freed >= need:
                    break
            if freed < need:
                break
            for expert in chosen:
                layer.low.remove(expert)
            layer.low.remove(promote)
            layer.high.add(promote)
            if layer.total_bytes(lo_bytes, hi_bytes) > budget:
                raise RuntimeError("quality repair exceeded the layer budget")


def _predicted_active(
    counts: list[list[int]],
    width: int,
    start: int,
    horizon: int,
    observed: list[set[int]],
) -> list[set[int]]:
    predicted = [set(layer) for layer in observed]
    last = min(len(counts), start + horizon + 1)
    for layer in range(start + 1, last):
        predicted[layer] = set(_rank(counts[layer])[:width])
    return predicted


def run_online(
    trials: list[list[set[int]]],
    counts: list[list[int]],
    placements: list[Placement],
    *,
    layer_time_s: float,
    bandwidth_bytes_s: float,
    lo_bytes: int,
    hi_bytes: int,
    budget: int,
    horizon: int,
    min_high: int,
    valuation: str,
    predict_width: int,
    admissions_per_layer: int = 8,
    candidate_cap: int = 16,
) -> OnlineStats:
    """Run the measured stream. ``valuation`` is ``counterfactual`` or ``independent``."""
    if valuation not in {"counterfactual", "independent"}:
        raise ValueError(f"unknown valuation {valuation}")
    stats = OnlineStats()
    width = max(1, predict_width)
    for trial in trials:
        for placement in placements:
            placement.prefetches.clear()
        _enforce_quality(
            placements,
            counts,
            min_high=min_high,
            budget=budget,
            lo_bytes=lo_bytes,
            hi_bytes=hi_bytes,
        )
        for index in range(len(trial)):
            predicted = _predicted_active(counts, width, index, horizon, trial)
            candidates: list[Exchange] = []
            last = min(len(trial), index + horizon + 1)
            for future in range(index + 1, last):
                resident = placements[future].high | placements[future].low
                missing = [
                    expert
                    for expert in _rank(counts[future])
                    if expert in predicted[future] - resident
                ][:candidate_cap]
                for expert in missing:
                    donor = _donor(placements[future], counts[future], predicted[future])
                    if donor is None and placements[future].total_bytes(lo_bytes, hi_bytes) + lo_bytes > budget:
                        continue
                    donor_expert, donor_tier = donor if donor is not None else (None, None)
                    candidates.append(
                        Exchange(
                            kind="lookahead",
                            layer=future,
                            expert=expert,
                            tier="lo",
                            nbytes=lo_bytes,
                            donor_layer=future if donor is not None else None,
                            donor_expert=donor_expert,
                            donor_tier=donor_tier,
                        )
                    )
            if valuation == "counterfactual":
                admitted = admit_exchanges(
                    predicted,
                    placements,
                    candidates,
                    layer_time_s=layer_time_s,
                    bandwidth_bytes_s=bandwidth_bytes_s,
                    lo_bytes=lo_bytes,
                    hi_bytes=hi_bytes,
                    horizon=horizon,
                    budget=budget,
                    limit=admissions_per_layer,
                )
                stats.exchanges.extend(admitted)
            else:
                stats.exchanges.extend(
                    _admit_independent(
                        placements,
                        candidates,
                        counts,
                        bandwidth_bytes_s=bandwidth_bytes_s,
                        lo_bytes=lo_bytes,
                        hi_bytes=hi_bytes,
                        budget=budget,
                        limit=admissions_per_layer,
                    )
                )
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
        stats.exposed_s.append(result.exposed_s)
        _settle_prefetches(placements, trial, stats)
        for layer, needed in enumerate(trial):
            for expert in needed:
                counts[layer][expert] += 1
        _fill_slack(placements, counts, budget, lo_bytes, hi_bytes)
    return stats


def _donor(
    placement: Placement,
    counts: list[int],
    protected: set[int],
) -> tuple[int, str] | None:
    pool = [expert for expert in placement.low if expert not in protected] or list(placement.low)
    if not pool:
        return None
    expert = min(pool, key=lambda item: (counts[item], item))
    return expert, "lo"


def _admit_independent(
    placements: list[Placement],
    candidates: list[Exchange],
    counts: list[list[int]],
    *,
    bandwidth_bytes_s: float,
    lo_bytes: int,
    hi_bytes: int,
    budget: int,
    limit: int,
) -> list[Exchange]:
    from dynaexq.lease.exchange import _apply

    admitted = []
    remaining = list(candidates)
    while remaining and len(admitted) < limit:
        def score(exchange: Exchange) -> float:
            request_p = counts[exchange.layer][exchange.expert]
            donor_p = 0
            if exchange.donor_expert is not None and exchange.donor_layer is not None:
                donor_p = counts[exchange.donor_layer][exchange.donor_expert]
            scale = max(1, max(counts[exchange.layer]))
            return independent_gain(
                exchange,
                request_probability=request_p / scale,
                donor_probability=donor_p / scale,
                bandwidth_bytes_s=bandwidth_bytes_s,
            )

        best = max(remaining, key=score)
        if score(best) <= 0:
            break
        if not _apply(placements, best, lo_bytes, hi_bytes, budget):
            remaining.remove(best)
            continue
        admitted.append(best)
        remaining.remove(best)
    return admitted


def _settle_prefetches(
    placements: list[Placement],
    trial: list[set[int]],
    stats: OnlineStats,
) -> None:
    for layer, placement in enumerate(placements):
        for _issue, expert, tier in placement.prefetches:
            if expert in trial[layer]:
                target = placement.high if tier == "hi" else placement.low
                target.add(expert)
                stats.converted_prefetches += 1
            else:
                stats.expired_prefetches += 1
        placement.prefetches.clear()


def _fill_slack(
    placements: list[Placement],
    counts: list[list[int]],
    budget: int,
    lo_bytes: int,
    hi_bytes: int,
) -> None:
    for placement, layer_counts in zip(placements, counts):
        for expert in _rank(layer_counts):
            if expert in placement.high or expert in placement.low:
                continue
            if placement.total_bytes(lo_bytes, hi_bytes) + lo_bytes > budget:
                break
            placement.low.add(expert)
