from __future__ import annotations

import pytest

from dynaexq.core.hotness_tracker import HotnessTracker


def test_unselected_experts_decay_after_workload_shift():
    tracker = HotnessTracker(num_layers=1, experts_per_layer=3, alpha=0.5)
    tracker.update(0, {0: 1.0})
    assert tracker.get_layer_scores(0).tolist() == pytest.approx([0.5, 0.0, 0.0])
    tracker.update(0, {1: 1.0})
    assert tracker.get_layer_scores(0).tolist() == pytest.approx([0.25, 0.5, 0.0])


def test_alpha_zero_tracks_current_window_exactly():
    tracker = HotnessTracker(num_layers=1, experts_per_layer=2, alpha=0.0)
    tracker.update(0, {0: 0.8})
    tracker.update(0, {1: 0.6})
    assert tracker.get_layer_scores(0).tolist() == pytest.approx([0.0, 0.6])


def test_cumulative_calibration_scores_are_order_invariant():
    forward = HotnessTracker(num_layers=1, experts_per_layer=3, alpha=0.9)
    reverse = HotnessTracker(num_layers=1, experts_per_layer=3, alpha=0.9)
    observations = ({0: 0.8, 2: 0.2}, {1: 0.6, 2: 0.4})
    for observation in observations:
        forward.update(0, observation)
    for observation in reversed(observations):
        reverse.update(0, observation)
    assert forward.get_cumulative_layer_scores(0).tolist() == pytest.approx(
        [0.4, 0.3, 0.3]
    )
    assert reverse.get_cumulative_layer_scores(0) == pytest.approx(
        forward.get_cumulative_layer_scores(0)
    )
    # Online EMA intentionally remains order-sensitive.
    assert reverse.get_layer_scores(0) != pytest.approx(
        forward.get_layer_scores(0)
    )


def test_reset_clears_online_and_calibration_state():
    tracker = HotnessTracker(num_layers=1, experts_per_layer=2, alpha=0.5)
    tracker.update(0, {0: 1.0})
    tracker.reset()
    assert tracker.get_layer_scores(0).tolist() == [0.0, 0.0]
    assert tracker.get_cumulative_layer_scores(0).tolist() == [0.0, 0.0]
    assert tracker.get_state().update_counts.tolist() == [0]


@pytest.mark.parametrize("alpha", [-0.1, 1.0, 1.1])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError):
        HotnessTracker(num_layers=1, experts_per_layer=2, alpha=alpha)
