import numpy as np

from dynaexq.runtime.monitor import ExpertMonitor
from dynaexq.runtime.types import ExpertID


def test_monitor_updates_scores_with_ewma():
    monitor = ExpertMonitor(ewma_alpha=0.5, epoch_decay=0.5)
    layer = 0
    expert = ExpertID(layer=layer, idx=3)

    topk_idx = np.array([[3, 1], [3, 2]])
    logits = np.array([[0.8, 0.1], [0.7, 0.2]])
    monitor.update_batch(layer, topk_idx, logits)

    first_score = monitor.score(expert)
    assert 0.0 < first_score <= 1.0

    topk_idx2 = np.array([[3, 4], [3, 5]])
    logits2 = np.array([[0.2, 0.1], [0.3, 0.2]])
    monitor.update_batch(layer, topk_idx2, logits2)

    second_score = monitor.score(expert)
    assert second_score < first_score


def test_epoch_tick_decays_scores():
    monitor = ExpertMonitor(ewma_alpha=0.5, epoch_decay=0.1)
    layer = 1
    expert = ExpertID(layer=layer, idx=7)

    monitor.update_batch(
        layer,
        np.array([[7]]),
        np.array([[0.9]]),
    )
    before = monitor.score(expert)
    monitor.epoch_tick()
    after = monitor.score(expert)
    assert after < before

