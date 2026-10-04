"""Feasible byte exchanges and counterfactual latency value.

A request is scored only together with the donor leases that fund it. The
latency value is the change in replayed exposed wait. Quality risk stays a
feasibility check: this module never converts it into milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dynaexq.lease.replay import simulate_forward


@dataclass(frozen=True, slots=True)
class Exchange:
    kind: str
    layer: int
    expert: int
    tier: str
    nbytes: int
    donor_layer: int | None = None
    donor_expert: int | None = None
    donor_tier: str | None = None
    gain_s: float = 0.0
    hold_for: int = 0

    @property
    def density(self) -> float:
        return self.gain_s / self.nbytes if self.nbytes else 0.0


@dataclass
class Placement:
    """Published experts plus lookahead copies issued from this layer.

    ``prefetches`` holds ``(future_layer, expert, tier)`` copies that occupy a
    destination until that future layer consumes them. They are not published
    residents, so the replay charges their transfer time.
    """

    high: set[int] = field(default_factory=set)
    low: set[int] = field(default_factory=set)
    staging_slots: int = 0
    # (issue_layer, expert, tier) destinations charged to the shared budget.
    prefetches: list[tuple[int, int, str]] = field(default_factory=list)
    # Trials remaining before a high-precision expert may be demoted.
    hold: dict[int, int] = field(default_factory=dict)

    def copy(self) -> Placement:
        return Placement(
            set(self.high),
            set(self.low),
            self.staging_slots,
            list(self.prefetches),
            dict(self.hold),
        )

    def resident_bytes(self, lo_bytes: int, hi_bytes: int) -> int:
        return len(self.high) * hi_bytes + len(self.low) * lo_bytes

    def prefetch_bytes(self, lo_bytes: int, hi_bytes: int) -> int:
        total = 0
        for _issue, _expert, tier in self.prefetches:
            total += hi_bytes if tier == "hi" else lo_bytes
        return total

    def total_bytes(self, lo_bytes: int, hi_bytes: int) -> int:
        return (
            self.resident_bytes(lo_bytes, hi_bytes)
            + self.staging_slots * lo_bytes
            + self.prefetch_bytes(lo_bytes, hi_bytes)
        )

    def as_resident(self) -> dict[int, str]:
        state = {expert: "lo" for expert in self.low}
        state.update({expert: "hi" for expert in self.high})
        return state


def placement_bytes(placements: list[Placement], lo_bytes: int, hi_bytes: int) -> int:
    return sum(item.total_bytes(lo_bytes, hi_bytes) for item in placements)


def within_budget(
    placements: list[Placement],
    lo_bytes: int,
    hi_bytes: int,
    budget: int,
    *,
    scope: str,
) -> bool:
    """``layer`` gives every layer its own budget. ``global`` is one arena."""
    if scope == "global":
        return placement_bytes(placements, lo_bytes, hi_bytes) <= budget
    if scope == "layer":
        return all(item.total_bytes(lo_bytes, hi_bytes) <= budget for item in placements)
    raise ValueError(f"unknown budget scope {scope}")


def migration_cost(
    exchange: Exchange,
    *,
    bandwidth_bytes_s: float,
    layer_time_s: float,
    horizon: int,
) -> float:
    """Exposed tail of a copy this exchange itself introduces."""
    if exchange.kind not in {"lookahead", "promote"} or exchange.nbytes <= 0:
        return 0.0
    overlap = max(1, horizon) * layer_time_s
    return max(0.0, exchange.nbytes / bandwidth_bytes_s - overlap)


def donor_rank(exchange: Exchange, *, protected: set[int]) -> int:
    """Lower rank is consumed first: cancel, then unprotected, then protected."""
    if exchange.kind == "cancel" or exchange.donor_expert is None:
        return 0
    if exchange.donor_expert in protected:
        return 2
    return 1


def _open_loop_prefetch(
    active: list[set[int]],
    placements: list[Placement],
    horizon: int,
) -> list[list[tuple[int, int, str]]]:
    """Issue explicit prefetches, then fill each layer's staging slots.

    Slot bytes live in the destination layer. The copy is issued ``horizon``
    layers earlier so it can overlap that many compute windows. Horizon 1
    issues the copy during the immediately preceding layer.
    """
    plan: list[list[tuple[int, int, str]]] = [[] for _ in active]
    for destination, placement in enumerate(placements):
        for issue, expert, tier in placement.prefetches:
            if 0 <= issue < len(plan):
                plan[issue].append((destination, expert, tier))
        if placement.staging_slots <= 0 or destination == 0:
            continue
        issue = max(0, destination - max(1, horizon))
        resident = placement.high | placement.low
        already = {expert for future, expert, _tier in plan[issue] if future == destination}
        issued = 0
        for expert in sorted(active[destination] - resident - already):
            if issued >= placement.staging_slots:
                break
            plan[issue].append((destination, expert, "lo"))
            issued += 1
    return plan


def exposed_seconds(
    active: list[set[int]],
    placements: list[Placement],
    *,
    layer_time_s: float,
    bandwidth_bytes_s: float,
    lo_bytes: int,
    hi_bytes: int,
    horizon: int,
) -> float:
    plan = _open_loop_prefetch(active, placements, horizon)
    result = simulate_forward(
        active,
        [item.as_resident() for item in placements],
        plan,
        layer_time_s=layer_time_s,
        bandwidth_bytes_s=bandwidth_bytes_s,
        lo_bytes=lo_bytes,
        hi_bytes=hi_bytes,
    )
    return result.exposed_s


def counterfactual_gain(
    active: list[set[int]],
    placements: list[Placement],
    exchange: Exchange,
    *,
    layer_time_s: float,
    bandwidth_bytes_s: float,
    lo_bytes: int,
    hi_bytes: int,
    horizon: int,
    budget: int,
    scope: str = "layer",
) -> float | None:
    """Return saved exposed seconds minus migration cost, or None if infeasible.

    Applying a lookahead lease to an expert that is already resident saves
    nothing: the replay sees the same published state. A copy that misses its
    overlap window is charged again through ``C_move``, so the net gain is
    smaller than the raw difference and can be negative.
    """
    trial = [item.copy() for item in placements]
    if not _apply(trial, exchange, lo_bytes, hi_bytes, budget, scope=scope):
        return None
    before = exposed_seconds(
        active,
        placements,
        layer_time_s=layer_time_s,
        bandwidth_bytes_s=bandwidth_bytes_s,
        lo_bytes=lo_bytes,
        hi_bytes=hi_bytes,
        horizon=horizon,
    )
    after = exposed_seconds(
        active,
        trial,
        layer_time_s=layer_time_s,
        bandwidth_bytes_s=bandwidth_bytes_s,
        lo_bytes=lo_bytes,
        hi_bytes=hi_bytes,
        horizon=horizon,
    )
    introduced = sum(len(item.prefetches) for item in trial) - sum(
        len(item.prefetches) for item in placements
    )
    move = (
        migration_cost(
            exchange,
            bandwidth_bytes_s=bandwidth_bytes_s,
            layer_time_s=layer_time_s,
            horizon=horizon,
        )
        if introduced > 0 or exchange.kind == "promote"
        else 0.0
    )
    return before - after - move


def independent_gain(
    exchange: Exchange,
    *,
    request_probability: float,
    donor_probability: float,
    bandwidth_bytes_s: float,
) -> float:
    """Score the request and the donor as if they did not share a queue.

    A prefetch of an already resident expert still receives a full transfer
    credit. That double count is the behavior the counterfactual replay removes.
    """
    gain = request_probability * exchange.nbytes / bandwidth_bytes_s
    if exchange.donor_expert is not None and exchange.donor_tier is not None:
        gain -= donor_probability * exchange.nbytes / bandwidth_bytes_s
    return gain


def apply_exchange(
    placements: list[Placement],
    exchange: Exchange,
    lo_bytes: int,
    hi_bytes: int,
    budget: int,
    *,
    scope: str = "layer",
) -> bool:
    """Apply one exchange. A rejected exchange leaves ``placements`` unchanged."""
    return _apply(placements, exchange, lo_bytes, hi_bytes, budget, scope=scope)


def _release_donor(trial: list[Placement], exchange: Exchange) -> bool:
    if exchange.donor_expert is None:
        return True
    donor_layer = exchange.donor_layer if exchange.donor_layer is not None else exchange.layer
    donor = trial[donor_layer]
    if exchange.donor_expert in donor.low:
        donor.low.remove(exchange.donor_expert)
        donor.hold.pop(exchange.donor_expert, None)
        return True
    if exchange.donor_expert in donor.high:
        if donor.hold.get(exchange.donor_expert, 0) > 0:
            return False
        donor.high.remove(exchange.donor_expert)
        donor.hold.pop(exchange.donor_expert, None)
        return True
    kept = [item for item in donor.prefetches if item[1] != exchange.donor_expert]
    if len(kept) != len(donor.prefetches):
        donor.prefetches = kept
        return True
    return False


def _apply(
    placements: list[Placement],
    exchange: Exchange,
    lo_bytes: int,
    hi_bytes: int,
    budget: int,
    *,
    scope: str = "layer",
) -> bool:
    trial = [item.copy() for item in placements]
    layer = trial[exchange.layer]
    if exchange.kind == "lookahead":
        future = exchange.layer
        if exchange.expert in layer.high or exchange.expert in layer.low:
            return True
        existing = [tier for _issue, expert, tier in layer.prefetches if expert == exchange.expert]
        if existing:
            return existing[0] == exchange.tier
        if not _release_donor(trial, exchange):
            return False
        issue_layer = max(0, future - 1)
        layer.prefetches.append((issue_layer, exchange.expert, exchange.tier))
    elif exchange.kind == "cancel":
        kept = [item for item in layer.prefetches if item[1] != exchange.expert]
        if len(kept) == len(layer.prefetches):
            return False
        layer.prefetches = kept
    elif exchange.kind == "demote":
        if exchange.expert not in layer.high or layer.hold.get(exchange.expert, 0) > 0:
            return False
        layer.high.remove(exchange.expert)
        layer.hold.pop(exchange.expert, None)
        layer.low.add(exchange.expert)
    elif exchange.kind == "promote":
        if exchange.expert in layer.high:
            return True
        if exchange.expert in layer.low:
            layer.low.remove(exchange.expert)
        elif not _release_donor(trial, exchange):
            return False
        layer.high.add(exchange.expert)
        if exchange.hold_for > 0:
            layer.hold[exchange.expert] = exchange.hold_for
    else:
        raise ValueError(f"unknown exchange kind {exchange.kind}")
    if not within_budget(trial, lo_bytes, hi_bytes, budget, scope=scope):
        return False
    for index, item in enumerate(trial):
        placements[index] = item
    return True


def admit_exchanges(
    active: list[set[int]],
    placements: list[Placement],
    candidates: list[Exchange],
    *,
    layer_time_s: float,
    bandwidth_bytes_s: float,
    lo_bytes: int,
    hi_bytes: int,
    horizon: int,
    budget: int,
    limit: int,
    scope: str = "layer",
) -> list[Exchange]:
    """Greedy admission by saved wait per requested byte, recomputed after each pick."""
    admitted: list[Exchange] = []
    remaining = list(candidates)
    while remaining and len(admitted) < limit:
        scored: list[tuple[float, Exchange]] = []
        for candidate in remaining:
            gain = counterfactual_gain(
                active,
                placements,
                candidate,
                layer_time_s=layer_time_s,
                bandwidth_bytes_s=bandwidth_bytes_s,
                lo_bytes=lo_bytes,
                hi_bytes=hi_bytes,
                horizon=horizon,
                budget=budget,
                scope=scope,
            )
            if gain is not None and gain > 0:
                scored.append((gain / candidate.nbytes, candidate))
        if not scored:
            break
        _density, best = max(scored, key=lambda item: (item[0], -item[1].nbytes))
        gain = counterfactual_gain(
            active,
            placements,
            best,
            layer_time_s=layer_time_s,
            bandwidth_bytes_s=bandwidth_bytes_s,
            lo_bytes=lo_bytes,
            hi_bytes=hi_bytes,
            horizon=horizon,
            budget=budget,
            scope=scope,
        )
        if gain is None or not _apply(placements, best, lo_bytes, hi_bytes, budget, scope=scope):
            remaining.remove(best)
            continue
        admitted.append(
            Exchange(
                kind=best.kind,
                layer=best.layer,
                expert=best.expert,
                tier=best.tier,
                nbytes=best.nbytes,
                donor_layer=best.donor_layer,
                donor_expert=best.donor_expert,
                donor_tier=best.donor_tier,
                gain_s=gain,
            )
        )
        remaining.remove(best)
    return admitted
