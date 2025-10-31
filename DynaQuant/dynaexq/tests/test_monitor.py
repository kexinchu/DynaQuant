"""
Unit tests for ExpertMonitor
"""

import unittest
import numpy as np
import time

from dynaexq.runtime.monitor import ExpertMonitor
from dynaexq.runtime.types import ExpertID


class TestExpertMonitor(unittest.TestCase):
    """Test ExpertMonitor functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.monitor = ExpertMonitor(
            ewma_alpha=0.5,
            epoch_duration=1.0,  # Short epoch for testing
            num_layers=4,
            num_experts_per_layer=8,
        )

    def test_initialization(self):
        """Test monitor initialization"""
        self.assertEqual(self.monitor.ewma_alpha, 0.5)
        self.assertEqual(self.monitor.epoch_duration, 1.0)
        self.assertEqual(self.monitor.current_epoch, 0)

    def test_update_batch(self):
        """Test batch update"""
        topk_idx = np.array([[0, 1], [2, 3]])  # 2 tokens, top-2
        logits = np.array([[0.6, 0.4], [0.7, 0.3]])

        self.monitor.update_batch(layer=0, topk_idx=topk_idx, logits=logits)

        # Check that statistics are recorded
        expert0 = ExpertID(layer=0, idx=0)
        self.assertGreater(self.monitor.total_logit_mass[expert0], 0)
        self.assertEqual(self.monitor.batch_count[expert0], 1)

    def test_score_calculation(self):
        """Test hotness score calculation"""
        topk_idx = np.array([[0, 1]])
        logits = np.array([[0.8, 0.2]])

        # Update and tick epoch
        self.monitor.update_batch(layer=0, topk_idx=topk_idx, logits=logits)
        self.monitor.epoch_tick()

        expert0 = ExpertID(layer=0, idx=0)
        expert1 = ExpertID(layer=0, idx=1)

        score0 = self.monitor.score(expert0)
        score1 = self.monitor.score(expert1)

        # Expert 0 should have higher score
        self.assertGreater(score0, score1)

    def test_epoch_tick(self):
        """Test epoch ticking"""
        initial_epoch = self.monitor.current_epoch

        topk_idx = np.array([[0, 1]])
        self.monitor.update_batch(layer=0, topk_idx=topk_idx)

        self.monitor.epoch_tick()

        self.assertEqual(self.monitor.current_epoch, initial_epoch + 1)
        self.assertEqual(len(self.monitor.batch_count), 0)  # Should be reset

    def test_automatic_epoch_tick(self):
        """Test automatic epoch tick after duration"""
        topk_idx = np.array([[0, 1]])
        self.monitor.update_batch(layer=0, topk_idx=topk_idx)

        # Wait for epoch duration
        time.sleep(1.1)

        # Next update should trigger epoch tick
        initial_epoch = self.monitor.current_epoch
        self.monitor.update_batch(layer=0, topk_idx=topk_idx)

        self.assertEqual(self.monitor.current_epoch, initial_epoch + 1)

    def test_multiple_layers(self):
        """Test tracking multiple layers"""
        for layer in range(4):
            topk_idx = np.array([[0, 1]])
            self.monitor.update_batch(layer=layer, topk_idx=topk_idx)

        self.monitor.epoch_tick()

        # Check that all layers have scores
        for layer in range(4):
            expert = ExpertID(layer=layer, idx=0)
            score = self.monitor.score(expert)
            self.assertGreater(score, 0)

    def test_ewma_decay(self):
        """Test EWMA decay for inactive experts"""
        topk_idx = np.array([[0, 1]])

        # Make expert 0 hot
        self.monitor.update_batch(layer=0, topk_idx=topk_idx)
        self.monitor.epoch_tick()

        expert0 = ExpertID(layer=0, idx=0)
        initial_score = self.monitor.score(expert0)

        # Several epochs without activity
        for _ in range(3):
            self.monitor.epoch_tick()

        final_score = self.monitor.score(expert0)

        # Score should decay
        self.assertLess(final_score, initial_score)

    def test_statistics(self):
        """Test statistics reporting"""
        topk_idx = np.array([[0, 1], [2, 3]])
        self.monitor.update_batch(layer=0, topk_idx=topk_idx)

        stats = self.monitor.get_statistics()

        self.assertIn("current_epoch", stats)
        self.assertIn("total_experts_tracked", stats)
        self.assertIn("active_experts_this_epoch", stats)
        self.assertGreater(stats["active_experts_this_epoch"], 0)


if __name__ == '__main__':
    unittest.main()
