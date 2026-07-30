from __future__ import annotations

import numpy as np
import pytest

from dynaexq.core.router_observer import RouterObserver


def test_selected_weights_are_not_reindexed_as_full_logits():
    observer = RouterObserver(use_probabilities=True)
    indices = np.array([[3, 1], [2, 3]])
    weights = np.array([[0.7, 0.3], [0.6, 0.4]], dtype=np.float32)
    signal = observer.extract_signal(
        layer=0,
        topk_indices=indices,
        selected_weights=weights,
    )
    values = observer.compute_g_signal(signal)
    assert values[3] == pytest.approx(0.55)
    assert values[1] == pytest.approx(0.15)
    assert values[2] == pytest.approx(0.3)


def test_selected_weight_shape_must_match_indices():
    observer = RouterObserver(use_probabilities=True)
    with pytest.raises(ValueError, match="shape"):
        observer.extract_signal(
            layer=0,
            topk_indices=np.array([[0, 1]]),
            selected_weights=np.array([[1.0]]),
        )


def test_probability_mode_preserves_frequency():
    observer = RouterObserver(use_probabilities=True)
    signal = observer.extract_signal(
        layer=0,
        topk_indices=np.array([[0], [0], [1], [0]]),
        selected_weights=np.ones((4, 1), dtype=np.float32),
    )
    values = observer.compute_g_signal(signal)
    assert values == pytest.approx({0: 0.75, 1: 0.25})
