"""
Unit tests for PrecisionController
"""

import unittest
import numpy as np

from dynaexq.runtime.controller import PrecisionController
from dynaexq.runtime.monitor import ExpertMonitor
from dynaexq.runtime.types import ExpertID


class TestPrecisionController(unittest.TestCase):
    """Test PrecisionController functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.controller = PrecisionController(
            tau_h=0.7,
            tau_c=0.3,
            max_w4_slots=4,
            num_layers=4,
            num_experts_per_layer=8,
        )

        self.monitor = ExpertMonitor(
            ewma_alpha=0.5,
            epoch_duration=300.0,
            num_layers=4,
            num_experts_per_layer=8,
        )

    def test_initialization(self):
        """Test controller initialization"""
        self.assertEqual(self.controller.tau_h, 0.7)
        self.assertEqual(self.controller.tau_c, 0.3)
        self.assertEqual(self.controller.max_w4_slots, 4)

    def test_promote_hot_experts(self):
        """Test promotion of hot experts to W4"""
        # Make some experts hot
        for expert_idx in [0, 1]:
            expert = ExpertID(layer=0, idx=expert_idx)
            topk_idx = np.array([[expert_idx]])
            logits = np.array([[1.0]])
            self.monitor.update_batch(
                layer=0, topk_idx=topk_idx, logits=logits)

        self.monitor.epoch_tick()

        # Plan precision
        active = [ExpertID(layer=0, idx=i) for i in range(8)]
        targets = self.controller.plan(active, self.monitor)

        # Hot experts should be W4
        self.assertEqual(targets[ExpertID(layer=0, idx=0)], "W4")
        self.assertEqual(targets[ExpertID(layer=0, idx=1)], "W4")

    def test_demote_cold_experts(self):
        """Test demotion of cold experts to W2"""
        # Make expert hot first
        expert = ExpertID(layer=0, idx=0)
        self.controller.current_precision[expert] = "W4"

        # Don't use it (score will be 0, below tau_c)
        active = [expert]
        targets = self.controller.plan(active, self.monitor)

        # Should be demoted to W2
        self.assertEqual(targets[expert], "W2")

    def test_hysteresis(self):
        """Test hysteresis prevents oscillation"""
        expert = ExpertID(layer=0, idx=0)

        # Set score in hysteresis band (between tau_c and tau_h)
        self.monitor.hotness[expert] = 0.5  # Between 0.3 and 0.7

        # If currently W4, should stay W4
        self.controller.current_precision[expert] = "W4"
        targets = self.controller.plan([expert], self.monitor)
        self.assertEqual(targets[expert], "W4")

        # If currently W2, should stay W2
        self.controller.current_precision[expert] = "W2"
        targets = self.controller.plan([expert], self.monitor)
        self.assertEqual(targets[expert], "W2")

    def test_pool_capacity_limit(self):
        """Test that max_w4_slots is enforced"""
        # Try to promote more than max_w4_slots experts
        for expert_idx in range(8):
            expert = ExpertID(layer=0, idx=expert_idx)
            self.monitor.hotness[expert] = 0.9  # All very hot

        active = [ExpertID(layer=0, idx=i) for i in range(8)]
        targets = self.controller.plan(active, self.monitor)

        # Count W4 assignments
        w4_count = sum(1 for p in targets.values() if p == "W4")

        # Should not exceed max_w4_slots
        self.assertLessEqual(w4_count, self.controller.max_w4_slots)

    def test_prioritize_hottest_experts(self):
        """Test that hottest experts get W4 slots when limited"""
        # Set different scores
        for expert_idx in range(8):
            expert = ExpertID(layer=0, idx=expert_idx)
            # Score decreases with index
            self.monitor.hotness[expert] = 0.9 - expert_idx * 0.1

        active = [ExpertID(layer=0, idx=i) for i in range(8)]
        targets = self.controller.plan(active, self.monitor)

        # First 4 (hottest) should be W4
        for i in range(4):
            self.assertEqual(targets[ExpertID(layer=0, idx=i)], "W4")

        # Rest should be W2
        for i in range(4, 8):
            self.assertEqual(targets[ExpertID(layer=0, idx=i)], "W2")

    def test_get_diff(self):
        """Test diff computation"""
        target = {
            ExpertID(layer=0, idx=0): "W4",
            ExpertID(layer=0, idx=1): "W2",
        }

        current = {
            ExpertID(layer=0, idx=0): "W2",
            ExpertID(layer=0, idx=1): "W4",
        }

        diff = self.controller.get_diff(target, current)

        self.assertEqual(len(diff["upgrades"]), 1)
        self.assertEqual(len(diff["downgrades"]), 1)
        self.assertIn(ExpertID(layer=0, idx=0), diff["upgrades"])
        self.assertIn(ExpertID(layer=0, idx=1), diff["downgrades"])

    def test_adapt_thresholds(self):
        """Test adaptive threshold adjustment"""
        initial_tau_h = self.controller.tau_h
        initial_tau_c = self.controller.tau_c

        # Low ready ratio should widen hysteresis
        self.controller.adapt_thresholds(ready_ratio=0.95, hbm_pressure=0.5)

        self.assertGreater(self.controller.tau_h, initial_tau_h)
        self.assertLess(self.controller.tau_c, initial_tau_c)

    def test_statistics(self):
        """Test statistics reporting"""
        stats = self.controller.get_statistics()

        self.assertIn("tau_h", stats)
        self.assertIn("tau_c", stats)
        self.assertIn("max_w4_slots", stats)
        self.assertIn("current_w4_experts", stats)
        self.assertIn("current_w2_experts", stats)


if __name__ == '__main__':
    unittest.main()
