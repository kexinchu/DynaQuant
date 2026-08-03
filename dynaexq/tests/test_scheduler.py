"""
Tests for ``dynaexq.core.scheduler.PrecisionScheduler`` (Phase 5).

Coverage:

* Plain top-N (δ=0) reproduces the pre-Phase-5 behavior.
* Cold start: when no residents exist yet, the scheduler fills the quota
  unconditionally (no δ check) — otherwise the system never warms up.
* δ blocks marginal swaps that would oscillate the system.
* δ allows swaps when the gap clearly exceeds the margin.
* Multiple swaps in one tick.
* Quota covers all experts → no transitions.
* Symmetric eviction: a resident is evicted iff some outsider beats it
  by > δ.
* Rate limit caps the number of requests per tick, prioritized by gap.
* δ-margin must reject negative values.
"""

from __future__ import annotations

import numpy as np
import pytest

from dynaexq.core.config import Tier
from dynaexq.core.hotness_tracker import HotnessTracker
from dynaexq.core.registry import ExpertKey
from dynaexq.core.scheduler import PrecisionScheduler, TransitionReq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tracker_with_scores(scores_per_layer: list[list[float]]) -> HotnessTracker:
    """Build a HotnessTracker pre-loaded with deterministic scores.

    The HotnessTracker stores EMA scores in a flat ndarray; we directly
    set those rather than feeding it through ``update()`` so test
    expectations don't depend on EMA decay.
    """
    n_layers = len(scores_per_layer)
    experts_per_layer = [len(s) for s in scores_per_layer]
    tracker = HotnessTracker(n_layers, experts_per_layer, alpha=0.9)
    # HotnessTracker stores scores in the private ``_scores`` ndarray;
    # we write straight to it so test expectations don't depend on EMA decay.
    for l, scores in enumerate(scores_per_layer):
        tracker._scores[l, : len(scores)] = np.array(scores, dtype=np.float32)
    return tracker


def _make_scheduler(
    scores_per_layer: list[list[float]],
    n_hi: list[int],
    delta: float = 0.0,
    rate_limit=None,
) -> tuple[PrecisionScheduler, HotnessTracker]:
    n_layers = len(scores_per_layer)
    experts_per_layer = [len(s) for s in scores_per_layer]
    sched = PrecisionScheduler(
        num_layers=n_layers,
        experts_per_layer=experts_per_layer,
        n_hi=n_hi,
        update_period_steps=1,
        rate_limit=rate_limit,
        delta_score_margin=delta,
    )
    tracker = _tracker_with_scores(scores_per_layer)
    return sched, tracker


def _residents_after(reqs: list[TransitionReq], current: dict[ExpertKey, Tier]) -> set[int]:
    """Apply transition reqs to a single-layer current map and return the
    resulting HI expert id set."""
    state = dict(current)
    for r in reqs:
        state[r.key] = r.dst
    return {k.expert for k, t in state.items() if t == Tier.HI}


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_negative_delta_raises():
    with pytest.raises(ValueError, match="non-negative"):
        PrecisionScheduler(
            num_layers=1,
            experts_per_layer=4,
            n_hi=[2],
            delta_score_margin=-0.01,
        )


def test_invalid_period_and_rate_limit_raise():
    with pytest.raises(ValueError, match="update_period_steps"):
        PrecisionScheduler(
            num_layers=1,
            experts_per_layer=4,
            n_hi=[2],
            update_period_steps=0,
        )
    with pytest.raises(ValueError, match="rate_limit"):
        PrecisionScheduler(
            num_layers=1,
            experts_per_layer=4,
            n_hi=[2],
            rate_limit=0,
        )


# ---------------------------------------------------------------------------
# Cold start: quota empty → fill, regardless of δ
# ---------------------------------------------------------------------------


def test_cold_start_fills_quota_with_top_n_no_delta_check():
    """First tick: nothing is HI yet. The scheduler must fill the quota
    even though δ would normally block borderline swaps. Without this
    behavior the system would never leave the all-LO cold state."""
    sched, tracker = _make_scheduler(
        scores_per_layer=[[0.10, 0.11, 0.05, 0.30]],
        n_hi=[2],
        delta=0.50,  # huge δ — would block any normal swap
    )
    reqs = sched.plan(step=1, tracker=tracker, current_tiers={})
    promotes = [r for r in reqs if r.dst == Tier.HI]
    # Top 2 by score: expert 3 (0.30) and expert 1 (0.11)
    assert {r.key.expert for r in promotes} == {3, 1}
    assert all(r.src == Tier.LO for r in promotes)
    assert not any(r.dst == Tier.LO for r in reqs)


def test_cold_start_partial_residents_fills_remaining():
    """A second cold-start scenario: some residents already exist but
    quota is not yet full. Fill the slack from top of outsiders, no δ."""
    sched, tracker = _make_scheduler(
        scores_per_layer=[[0.50, 0.10, 0.05, 0.30]],
        n_hi=[3],
        delta=1.0,
    )
    # Pretend expert 0 is already a resident (1/3 of quota).
    current = {ExpertKey(0, 0): Tier.HI}
    reqs = sched.plan(step=1, tracker=tracker, current_tiers=current)
    residents = _residents_after(reqs, current)
    # Cold start path picks top-3 by score: 0 (0.50), 3 (0.30), 1 (0.10)
    assert residents == {0, 3, 1}


# ---------------------------------------------------------------------------
# δ=0 reproduces plain top-N (backward compatibility)
# ---------------------------------------------------------------------------


def test_delta_zero_matches_plain_top_n():
    """With δ=0 the scheduler must match plain top-N projection (the
    pre-Phase-5 behavior). This is the backward-compat guard for every
    existing user that constructs a scheduler without specifying δ."""
    sched, tracker = _make_scheduler(
        scores_per_layer=[[0.30, 0.20, 0.40, 0.10]],
        n_hi=[2],
        delta=0.0,
    )
    # Cold start picks top-2: 2 (0.40), 0 (0.30)
    reqs = sched.plan(step=1, tracker=tracker, current_tiers={})
    residents = _residents_after(reqs, {})
    assert residents == {2, 0}


# ---------------------------------------------------------------------------
# δ blocks small swaps (the whole point of hysteresis)
# ---------------------------------------------------------------------------


def test_delta_blocks_borderline_swap():
    """
    Setup: residents are {0, 1} (scores 0.50, 0.40). Outsider 2 has score
    0.42 — barely above resident 1 by 0.02. With δ=0.05 this swap should
    be **rejected** (the gap of 0.02 is below the margin), so no
    transitions are emitted on this tick.
    """
    sched, tracker = _make_scheduler(
        scores_per_layer=[[0.50, 0.40, 0.42, 0.10]],
        n_hi=[2],
        delta=0.05,
    )
    current = {ExpertKey(0, 0): Tier.HI, ExpertKey(0, 1): Tier.HI}
    reqs = sched.plan(step=1, tracker=tracker, current_tiers=current)
    assert reqs == [], (
        "δ=0.05 should block the 0.40 → 0.42 swap (gap 0.02 < δ)"
    )


def test_delta_allows_clear_swap():
    """
    Same setup but the outsider's score is now 0.60 — clearly above
    resident 1 (0.40) by 0.20. With δ=0.05 the swap MUST happen.
    """
    sched, tracker = _make_scheduler(
        scores_per_layer=[[0.50, 0.40, 0.60, 0.10]],
        n_hi=[2],
        delta=0.05,
    )
    current = {ExpertKey(0, 0): Tier.HI, ExpertKey(0, 1): Tier.HI}
    reqs = sched.plan(step=1, tracker=tracker, current_tiers=current)
    residents = _residents_after(reqs, current)
    assert residents == {0, 2}
    # Exactly one promote (expert 2) and one demote (expert 1).
    promotes = [r for r in reqs if r.dst == Tier.HI]
    demotes = [r for r in reqs if r.dst == Tier.LO]
    assert len(promotes) == 1 and promotes[0].key.expert == 2
    assert len(demotes) == 1 and demotes[0].key.expert == 1


def test_delta_at_exact_boundary_does_not_swap():
    """
    Swap condition is **strict**: ``S[outsider] > S[resident] + δ``.
    A gap of exactly δ should NOT swap (otherwise the system thrashes
    when scores happen to land on the boundary).
    """
    sched, tracker = _make_scheduler(
        scores_per_layer=[[0.50, 0.40, 0.45, 0.10]],
        n_hi=[2],
        delta=0.05,
    )
    current = {ExpertKey(0, 0): Tier.HI, ExpertKey(0, 1): Tier.HI}
    reqs = sched.plan(step=1, tracker=tracker, current_tiers=current)
    assert reqs == [], "gap of exactly δ is not strictly greater than δ"


# ---------------------------------------------------------------------------
# Multiple swaps in a single tick
# ---------------------------------------------------------------------------


def test_multiple_swaps_in_one_tick():
    """
    n_hi=2, residents {0,1} with scores {0.30, 0.20}. Two outsiders 2,3
    have scores {0.80, 0.70} — both clearly above any resident by > δ.
    Both should swap in one tick.
    """
    sched, tracker = _make_scheduler(
        scores_per_layer=[[0.30, 0.20, 0.80, 0.70]],
        n_hi=[2],
        delta=0.05,
    )
    current = {ExpertKey(0, 0): Tier.HI, ExpertKey(0, 1): Tier.HI}
    reqs = sched.plan(step=1, tracker=tracker, current_tiers=current)
    residents = _residents_after(reqs, current)
    assert residents == {2, 3}


def test_partial_swap_only_strong_outsider_succeeds():
    """
    Residents {0, 1} with scores {0.50, 0.40}. Outsiders {2, 3} with
    scores {0.70, 0.43}. With δ=0.05:
    - Outsider 2 (0.70) vs lowest resident 1 (0.40): gap 0.30 > 0.05 → swap
    - Outsider 3 (0.43) vs new lowest resident 0 (0.50): gap is negative → stop
    Result: residents become {0, 2}, only expert 1 demoted, only expert 2 promoted.
    """
    sched, tracker = _make_scheduler(
        scores_per_layer=[[0.50, 0.40, 0.70, 0.43]],
        n_hi=[2],
        delta=0.05,
    )
    current = {ExpertKey(0, 0): Tier.HI, ExpertKey(0, 1): Tier.HI}
    reqs = sched.plan(step=1, tracker=tracker, current_tiers=current)
    residents = _residents_after(reqs, current)
    assert residents == {0, 2}


# ---------------------------------------------------------------------------
# Degenerate cases
# ---------------------------------------------------------------------------


def test_quota_covers_all_experts_no_transitions():
    """If n_hi[l] >= n_experts[l] every expert is HI and the scheduler
    on subsequent ticks emits nothing."""
    sched, tracker = _make_scheduler(
        scores_per_layer=[[0.1, 0.2, 0.3]],
        n_hi=[3],
    )
    current = {ExpertKey(0, e): Tier.HI for e in range(3)}
    reqs = sched.plan(step=1, tracker=tracker, current_tiers=current)
    assert reqs == []


def test_zero_quota_emits_nothing():
    sched, tracker = _make_scheduler(
        scores_per_layer=[[0.1, 0.2, 0.3, 0.4]],
        n_hi=[0],
    )
    reqs = sched.plan(step=1, tracker=tracker, current_tiers={})
    assert reqs == []


def test_should_update_only_at_period_boundary():
    sched, tracker = _make_scheduler(
        scores_per_layer=[[0.1, 0.2, 0.3, 0.4]],
        n_hi=[2],
    )
    sched.update_period_steps = 5
    # step 0 → True (0 % 5 == 0); step 3 → False; step 5 → True
    assert sched.should_update(0) is True
    assert sched.should_update(3) is False
    assert sched.should_update(5) is True


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------


def test_rate_limit_caps_requests_and_prefers_largest_gap():
    """
    n_hi=2, residents {0,1} with scores {0.30, 0.20}, outsiders {2,3,4}
    with scores {0.90, 0.80, 0.70}. All three would normally swap, but
    rate_limit=2 should cap to 2 requests, prioritized by score_gap.
    The expected promotes are experts 2 (gap 0.70) and 3 (gap 0.50);
    paired with demotes of 1 then 0.

    A replacement is an indivisible pair, so the cap must retain one
    demotion and its matching promotion rather than two promotions.
    """
    sched, tracker = _make_scheduler(
        scores_per_layer=[[0.30, 0.20, 0.90, 0.80, 0.70]],
        n_hi=[2],
        delta=0.05,
        rate_limit=2,
    )
    current = {ExpertKey(0, 0): Tier.HI, ExpertKey(0, 1): Tier.HI}
    reqs = sched.plan(step=1, tracker=tracker, current_tiers=current)
    assert len(reqs) == 2
    assert [r.dst for r in reqs] == [Tier.LO, Tier.HI]
    assert len({r.key.layer for r in reqs}) == 1


def test_odd_rate_limit_does_not_split_steady_state_swap():
    sched, tracker = _make_scheduler(
        scores_per_layer=[[0.30, 0.20, 0.90, 0.80]],
        n_hi=[2],
        delta=0.05,
        rate_limit=1,
    )
    current = {ExpertKey(0, 0): Tier.HI, ExpertKey(0, 1): Tier.HI}
    assert sched.plan(step=1, tracker=tracker, current_tiers=current) == []
