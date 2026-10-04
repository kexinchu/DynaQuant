"""Ledger and deadline-replay invariants. No GPU and no checkpoint required."""

from __future__ import annotations

import pytest

from dynaexq.lease.envelope import EnvelopeState, pollution_ratio, update_envelope
from dynaexq.lease.exchange import (
    Exchange,
    Placement,
    apply_exchange,
    counterfactual_gain,
    donor_rank,
    independent_gain,
)
from dynaexq.lease.ledger import BudgetViolation, ByteLease, LeaseLedger
from dynaexq.lease.replay import simulate_forward
from dynaexq.policy.adaptive import run_adaptive
from dynaexq.policy.online import initial_placement, run_online


def test_reserve_all_is_atomic():
    ledger = LeaseLedger(100)
    first = ByteLease("res", 0, 1, 60, 0.0, 1.0)
    assert ledger.try_reserve_all([first])
    second = ByteLease("look", 0, 2, 50, 0.0, 1.0)
    assert not ledger.try_reserve_all([second])
    assert ledger.reserved_bytes() == 60
    assert ledger.slack() == 40


def test_over_reserve_raises_and_keeps_the_invariant():
    ledger = LeaseLedger(100)
    ledger.reserve(ByteLease("res", 0, 1, 80, 0.0, 1.0))
    with pytest.raises(BudgetViolation):
        ledger.reserve(ByteLease("fid", 0, 1, 30, 0.0, 1.0))
    ledger.assert_invariant()
    assert ledger.slack() == 20


def test_peak_counts_source_and_replacement_together():
    ledger = LeaseLedger(100)
    replacement = ByteLease("res", 0, 2, 40, 1.0, 2.0)
    assert ledger.peak_of([[replacement]]) == 40
    ledger.reserve(ByteLease("res", 0, 1, 50, 0.0, 1.0))
    assert ledger.peak_of([[replacement]]) == 90


def test_resident_expert_does_not_stall():
    result = simulate_forward(
        [{0}, {0}],
        [{0: "lo"}, {0: "lo"}],
        [[], []],
        layer_time_s=2.0,
        bandwidth_bytes_s=1000.0,
        lo_bytes=1000,
        hi_bytes=4000,
    )
    assert result.exposed_s == 0.0
    assert result.demand_bytes == 0


def test_demand_miss_costs_one_transfer():
    result = simulate_forward(
        [{0}],
        [{}],
        [[]],
        layer_time_s=2.0,
        bandwidth_bytes_s=1000.0,
        lo_bytes=1000,
        hi_bytes=4000,
    )
    assert result.exposed_s == pytest.approx(1.0)
    assert result.demand_bytes == 1000


def test_prefetch_that_finishes_before_the_deadline_hides_the_transfer():
    result = simulate_forward(
        [set(), {0}],
        [{}, {}],
        [[(1, 0, "lo")], []],
        layer_time_s=2.0,
        bandwidth_bytes_s=1000.0,
        lo_bytes=1000,
        hi_bytes=4000,
    )
    assert result.exposed_s == 0.0
    assert result.layers[1].prefetched_hits == 1


def test_late_prefetch_still_exposes_the_remaining_wait():
    result = simulate_forward(
        [set(), {0}],
        [{}, {}],
        [[(1, 0, "lo")], []],
        layer_time_s=0.25,
        bandwidth_bytes_s=1000.0,
        lo_bytes=1000,
        hi_bytes=4000,
    )
    assert result.exposed_s == pytest.approx(0.75)
    assert result.layers[1].late_prefetches == 1


def test_counterfactual_gives_no_credit_for_prefetching_a_resident():
    placements = [Placement(low={0}), Placement(low={0})]
    exchange = Exchange("lookahead", layer=1, expert=0, tier="lo", nbytes=1000)
    gain = counterfactual_gain(
        [set(), {0}],
        placements,
        exchange,
        layer_time_s=2.0,
        bandwidth_bytes_s=1000.0,
        lo_bytes=1000,
        hi_bytes=4000,
        horizon=1,
        budget=10_000,
    )
    assert gain == pytest.approx(0.0)
    assert independent_gain(
        exchange,
        request_probability=1.0,
        donor_probability=0.0,
        bandwidth_bytes_s=1000.0,
    ) == pytest.approx(1.0)


def test_quality_repair_evicts_enough_low_residents_to_fund_one_promotion():
    counts = [[0, 0, 0, 5, 1, 1, 1]]
    placements = initial_placement(counts, n_high=0, n_low=4, n_experts=7)
    # Fill the budget with four low residents. One high tier costs three low tiers,
    # so the repair must evict more than one donor.
    run_online(
        [[{3}]],
        counts,
        placements,
        layer_time_s=1.0,
        bandwidth_bytes_s=1000.0,
        lo_bytes=10,
        hi_bytes=30,
        budget=40,
        horizon=1,
        min_high=1,
        valuation="counterfactual",
        predict_width=1,
    )
    assert len(placements[0].high) == 1
    assert placements[0].total_bytes(10, 30) <= 40


def test_online_prefetch_hides_a_predicted_miss():
    counts = [[0, 0, 0] for _ in range(3)]
    counts[1][2] = 5
    counts[2][2] = 5
    placements = initial_placement(counts, n_high=0, n_low=0, n_experts=3)
    trials = [[set(), {2}, {2}]]
    stats = run_online(
        trials,
        counts,
        placements,
        layer_time_s=2.0,
        bandwidth_bytes_s=1000.0,
        lo_bytes=1000,
        hi_bytes=4000,
        budget=4000,
        horizon=1,
        min_high=0,
        valuation="counterfactual",
        predict_width=1,
        admissions_per_layer=2,
        candidate_cap=4,
    )
    cold = simulate_forward(
        trials[0],
        [{}, {}, {}],
        [[], [], []],
        layer_time_s=2.0,
        bandwidth_bytes_s=1000.0,
        lo_bytes=1000,
        hi_bytes=4000,
    )
    assert stats.exposed_s[0] < cold.exposed_s
    assert stats.exchanges


def test_rejected_exchange_does_not_mutate_placement():
    placements = [Placement(low={1}, staging_slots=0), Placement(low={1})]
    before = placements[1].copy()
    exchange = Exchange(
        "lookahead",
        layer=1,
        expert=2,
        tier="lo",
        nbytes=1000,
        donor_layer=1,
        donor_expert=9,
        donor_tier="lo",
    )
    gain = counterfactual_gain(
        [{1}, {2}],
        placements,
        exchange,
        layer_time_s=2.0,
        bandwidth_bytes_s=1000.0,
        lo_bytes=1000,
        hi_bytes=4000,
        horizon=1,
        budget=1000,
    )
    assert gain is None
    assert placements[1].low == before.low
    assert placements[1].prefetches == before.prefetches


def _replay_kwargs() -> dict:
    return {
        "layer_time_s": 0.1,
        "bandwidth_bytes_s": 1000.0,
        "lo_bytes": 1000,
        "hi_bytes": 4000,
        "horizon": 1,
    }


def test_global_budget_funds_a_lookahead_from_another_layer():
    placements = [Placement(low={0}), Placement(low={1})]
    exchange = Exchange(
        "lookahead",
        layer=1,
        expert=2,
        tier="lo",
        nbytes=1000,
        donor_layer=0,
        donor_expert=0,
        donor_tier="lo",
    )
    assert not apply_exchange(placements, exchange, 1000, 4000, 1000, scope="layer")
    assert placements[0].low == {0}
    funded = [item.copy() for item in placements]
    assert apply_exchange(funded, exchange, 1000, 4000, 2000, scope="global")
    assert funded[0].low == set()
    assert any(expert == 2 for _issue, expert, _tier in funded[1].prefetches)


def test_low_and_high_lookahead_of_one_expert_are_mutually_exclusive():
    placements = [Placement(), Placement()]
    low = Exchange("lookahead", layer=1, expert=2, tier="lo", nbytes=1000)
    high = Exchange("lookahead", layer=1, expert=2, tier="hi", nbytes=4000)
    assert apply_exchange(placements, low, 1000, 4000, 10_000)
    assert not apply_exchange(placements, high, 1000, 4000, 10_000)
    assert [tier for _issue, _expert, tier in placements[1].prefetches] == ["lo"]


def test_envelope_hysteresis_moves_by_at_most_one_layer():
    state = EnvelopeState(depth=2)
    held = update_envelope(
        state, stall=0.3, pollution=0.3, beta=0.0, lam=1.0, theta=0.2, s_min=1, s_max=4
    )
    assert held.depth == 2
    longer = update_envelope(
        state, stall=1.0, pollution=0.0, beta=0.0, lam=1.0, theta=0.2, s_min=1, s_max=4
    )
    assert longer.depth == 3
    shorter = update_envelope(
        longer, stall=0.0, pollution=1.0, beta=0.0, lam=1.0, theta=0.2, s_min=1, s_max=4
    )
    assert shorter.depth == 2


def test_pollution_from_a_useless_prefetch_does_not_lengthen_the_envelope():
    ratio = pollution_ratio(
        unused_bytes=1000,
        reload_bytes=0,
        admitted_lookahead_bytes=1000,
        block_bytes=100,
    )
    assert ratio == pytest.approx(1.0)
    state = EnvelopeState(depth=2)
    updated = update_envelope(
        state, stall=0.1, pollution=ratio, beta=0.0, lam=1.0, theta=0.2, s_min=1, s_max=4
    )
    assert updated.depth < state.depth


def test_evicting_a_protected_resident_is_charged_its_reload():
    placements = [Placement(), Placement(low={0})]
    exchange = Exchange(
        "lookahead",
        layer=1,
        expert=1,
        tier="lo",
        nbytes=1000,
        donor_layer=1,
        donor_expert=0,
        donor_tier="lo",
    )
    gain = counterfactual_gain(
        [set(), {0, 1}],
        placements,
        exchange,
        budget=5000,
        scope="layer",
        **_replay_kwargs(),
    )
    assert gain is not None and gain <= 0
    assert 0 in placements[1].low


def test_cancelling_an_unused_lookahead_outranks_evicting_a_resident():
    cancel = Exchange("cancel", layer=1, expert=9, tier="lo", nbytes=1000)
    evict = Exchange(
        "lookahead",
        layer=1,
        expert=1,
        tier="lo",
        nbytes=1000,
        donor_layer=1,
        donor_expert=0,
        donor_tier="lo",
    )
    assert donor_rank(cancel, protected={0}) < donor_rank(evict, protected={0})
    placements = [Placement(), Placement(low={0}, prefetches=[(0, 9, "lo")])]
    assert apply_exchange(placements, cancel, 1000, 4000, 2000)
    assert placements[1].prefetches == []
    assert 0 in placements[1].low


def test_migration_cost_can_make_a_late_copy_negative():
    placements = [Placement(), Placement()]
    exchange = Exchange("lookahead", layer=1, expert=0, tier="lo", nbytes=1000)
    gain = counterfactual_gain(
        [set(), {0}],
        placements,
        exchange,
        budget=10_000,
        **_replay_kwargs(),
    )
    assert gain is not None and gain < 0


def test_fidelity_inside_its_tenure_cannot_be_demoted():
    placements = [Placement(high={1}, hold={1: 2})]
    demote = Exchange("demote", layer=0, expert=1, tier="lo", nbytes=3000)
    assert not apply_exchange(placements, demote, 1000, 4000, 10_000)
    assert placements[0].high == {1}
    placements[0].hold[1] = 0
    assert apply_exchange(placements, demote, 1000, 4000, 10_000)
    assert placements[0].high == set() and placements[0].low == {1}


def test_stall_lengthens_the_envelope_and_pollution_does_not():
    missed = run_adaptive(
        [[{}, {1}]],
        [[0, 0], [0, 0]],
        [Placement(), Placement()],
        layer_time_s=0.1,
        bandwidth_bytes_s=1000.0,
        lo_bytes=1000,
        hi_bytes=4000,
        budget=100,
        predict_width=1,
        absent_per_layer=0,
        min_high=0,
    )
    assert missed.depths == [2]
    assert missed.exposed_s[0] > 0

    unused = run_adaptive(
        [[set(), set()]],
        [[0, 0], [5, 0]],
        [Placement(), Placement()],
        layer_time_s=2.0,
        bandwidth_bytes_s=1000.0,
        lo_bytes=1000,
        hi_bytes=4000,
        budget=10_000,
        predict_width=1,
        absent_per_layer=0,
        min_high=0,
    )
    assert unused.expired_prefetches >= 1
    assert unused.pollution[0] > 0
    assert unused.depths[0] <= 1


def test_ready_experts_overlap_a_miss_in_the_same_layer():
    result = simulate_forward(
        [{0, 1}],
        [{0: "lo"}],
        [[]],
        layer_time_s=1.0,
        bandwidth_bytes_s=1000.0,
        lo_bytes=1000,
        hi_bytes=4000,
    )
    assert result.exposed_s == pytest.approx(0.5)
    assert result.exposed_s < 1.0
