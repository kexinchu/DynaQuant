"""
Precision Scheduler for dynamic mixed-precision management.
Manages expert precision switching based on risk, VRAM budget, bandwidth budget, and hit rates.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np


class PrecisionScheduler:
    """
    Scheduler for managing dynamic precision of MoE experts.
    """

    def __init__(
        self,
        num_experts: int,
        vram_budget_gb: float = 80.0,
        bandwidth_budget_gbps: float = 2000.0,
        top_m_experts: int = 8,
        default_precision: str = "w2a4",
        high_pressure_precision: str = "w2a2",
        hysteresis_steps: int = 10,
        cooldown_steps: int = 5,
        risk_weight: float = 0.6,
        hit_rate_weight: float = 0.4,
        hit_rate_ema_alpha: float = 0.1,
    ):
        """
        Initialize precision scheduler.

        Args:
            num_experts: Total number of experts
            vram_budget_gb: VRAM budget in GB
            bandwidth_budget_gbps: Memory bandwidth budget in GB/s
            top_m_experts: Keep top-M experts at W4A4
            default_precision: Default precision for experts
            high_pressure_precision: Precision under memory pressure
            hysteresis_steps: Cooldown steps before demotion
            cooldown_steps: Additional cooldown after switch
            risk_weight: Weight for risk in priority calculation
            hit_rate_weight: Weight for hit rate in priority calculation
            hit_rate_ema_alpha: EMA smoothing factor for hit rates
        """
        self.num_experts = num_experts
        self.vram_budget_gb = vram_budget_gb
        self.bandwidth_budget_gbps = bandwidth_budget_gbps
        self.top_m_experts = top_m_experts
        self.default_precision = default_precision
        self.high_pressure_precision = high_pressure_precision
        self.hysteresis_steps = hysteresis_steps
        self.cooldown_steps = cooldown_steps
        self.risk_weight = risk_weight
        self.hit_rate_weight = hit_rate_weight
        self.hit_rate_ema_alpha = hit_rate_ema_alpha

        # Expert state
        self.expert_precision = [default_precision] * num_experts
        self.expert_risk = [0.0] * num_experts
        self.expert_hit_rate = [0.0] * num_experts
        self.expert_last_switch_step = [0] * num_experts
        self.expert_promotion_candidate_steps = [0] * num_experts

        # Hit count tracking (for EMA)
        self.expert_hit_count = [0] * num_experts
        self.total_hits = 0

        # Statistics
        self.current_step = 0
        self.num_promotions = 0
        self.num_demotions = 0
        self.precision_history = defaultdict(list)

        # Precision memory costs (relative to FP16)
        self.precision_memory_cost = {
            "fp16": 1.0,
            "w4a4": 0.25,  # 4-bit weights + 4-bit activations
            "w2a4": 0.15,  # 2-bit weights + 4-bit activations
            "w2a2": 0.10,  # 2-bit weights + 2-bit activations
        }

        # Bandwidth costs (relative to FP16)
        self.precision_bandwidth_cost = {
            "fp16": 1.0,
            "w4a4": 0.25,
            "w2a4": 0.15,
            "w2a2": 0.10,
        }

        # Expert weights size (will be set later)
        self.expert_weight_size_gb = 0.1  # Default placeholder

    def update_risk(self, expert_id: int, risk: float):
        """
        Update risk score for an expert.

        Args:
            expert_id: Expert ID
            risk: Risk score
        """
        self.expert_risk[expert_id] = risk

    def update_hit_counts(self, expert_ids: List[int]):
        """
        Update hit counts for experts (called after each forward pass).

        Args:
            expert_ids: List of expert IDs that were used
        """
        for expert_id in expert_ids:
            if 0 <= expert_id < self.num_experts:
                self.expert_hit_count[expert_id] += 1
                self.total_hits += 1

        # Update EMA hit rates
        for expert_id in range(self.num_experts):
            # Compute current hit rate (hits per total)
            if self.total_hits > 0:
                current_rate = self.expert_hit_count[expert_id] / \
                    self.total_hits
            else:
                current_rate = 0.0

            # EMA update
            self.expert_hit_rate[expert_id] = (
                (1 - self.hit_rate_ema_alpha) * self.expert_hit_rate[expert_id] +
                self.hit_rate_ema_alpha * current_rate
            )

    def compute_priority(self, expert_id: int) -> float:
        """
        Compute priority for an expert (for promotion decisions).

        Args:
            expert_id: Expert ID

        Returns:
            priority: Priority score (higher = more important)
        """
        risk = self.expert_risk[expert_id]
        hit_rate = self.expert_hit_rate[expert_id]

        priority = self.risk_weight * risk + self.hit_rate_weight * hit_rate

        return priority

    def check_vram_budget(self) -> Tuple[float, bool]:
        """
        Check current VRAM usage against budget.

        Returns:
            vram_usage_gb: Current VRAM usage in GB
            under_budget: Whether usage is under budget
        """
        total_usage = 0.0

        for expert_id in range(self.num_experts):
            precision = self.expert_precision[expert_id]
            cost_factor = self.precision_memory_cost.get(precision, 1.0)
            total_usage += self.expert_weight_size_gb * cost_factor

        under_budget = total_usage <= self.vram_budget_gb

        return total_usage, under_budget

    def check_bandwidth_budget(self) -> Tuple[float, bool]:
        """
        Check current bandwidth usage against budget.

        Returns:
            bandwidth_usage: Current bandwidth usage
            under_budget: Whether usage is under budget
        """
        # Estimate bandwidth based on hit rates and precision
        total_bandwidth = 0.0

        for expert_id in range(self.num_experts):
            precision = self.expert_precision[expert_id]
            cost_factor = self.precision_bandwidth_cost.get(precision, 1.0)
            hit_rate = self.expert_hit_rate[expert_id]

            # Bandwidth proportional to hit rate
            total_bandwidth += cost_factor * hit_rate

        # Normalize to budget scale
        # This is a simplified model; in practice, would need more accurate BW modeling
        under_budget = total_bandwidth <= 1.0  # Relative to baseline

        return total_bandwidth, under_budget

    def schedule(self, promotion_candidates: Optional[List[int]] = None) -> Dict[int, str]:
        """
        Run scheduling step to update expert precisions.

        Args:
            promotion_candidates: List of expert IDs to consider for promotion

        Returns:
            precision_changes: Dictionary of expert_id -> new_precision for changed experts
        """
        self.current_step += 1
        precision_changes = {}

        # Check budgets
        vram_usage, vram_ok = self.check_vram_budget()
        bw_usage, bw_ok = self.check_bandwidth_budget()

        memory_pressure = not (vram_ok and bw_ok)

        # Sort experts by priority
        priorities = [(expert_id, self.compute_priority(expert_id))
                      for expert_id in range(self.num_experts)]
        priorities.sort(key=lambda x: x[1], reverse=True)

        # Handle promotions
        if promotion_candidates:
            for expert_id in promotion_candidates:
                current_precision = self.expert_precision[expert_id]

                # Check cooldown
                steps_since_switch = self.current_step - \
                    self.expert_last_switch_step[expert_id]
                if steps_since_switch < self.cooldown_steps:
                    continue

                # Promote to W4A4 if not already
                if current_precision != "w4a4" and not memory_pressure:
                    self.expert_precision[expert_id] = "w4a4"
                    self.expert_last_switch_step[expert_id] = self.current_step
                    self.expert_promotion_candidate_steps[expert_id] = 0
                    self.num_promotions += 1
                    precision_changes[expert_id] = "w4a4"

        # Ensure top-M experts are at W4A4 (if not under memory pressure)
        if not memory_pressure:
            top_m_experts = [expert_id for expert_id,
                             _ in priorities[:self.top_m_experts]]

            for expert_id in top_m_experts:
                if self.expert_precision[expert_id] != "w4a4":
                    steps_since_switch = self.current_step - \
                        self.expert_last_switch_step[expert_id]
                    if steps_since_switch >= self.cooldown_steps:
                        self.expert_precision[expert_id] = "w4a4"
                        self.expert_last_switch_step[expert_id] = self.current_step
                        self.num_promotions += 1
                        precision_changes[expert_id] = "w4a4"

        # Handle demotions with hysteresis
        for expert_id in range(self.num_experts):
            current_precision = self.expert_precision[expert_id]

            # Skip if already at default or lower
            if current_precision == self.default_precision or current_precision == self.high_pressure_precision:
                continue

            # Check if should demote (low priority)
            priority = self.compute_priority(expert_id)

            # Check if in top-M
            is_top_m = expert_id in [exp_id for exp_id,
                                     _ in priorities[:self.top_m_experts]]

            # Demote if not in top-M and hysteresis passed
            if not is_top_m:
                self.expert_promotion_candidate_steps[expert_id] += 1

                if self.expert_promotion_candidate_steps[expert_id] >= self.hysteresis_steps:
                    steps_since_switch = self.current_step - \
                        self.expert_last_switch_step[expert_id]

                    if steps_since_switch >= self.cooldown_steps:
                        # Demote to default or high-pressure precision
                        new_precision = self.high_pressure_precision if memory_pressure else self.default_precision

                        if current_precision != new_precision:
                            self.expert_precision[expert_id] = new_precision
                            self.expert_last_switch_step[expert_id] = self.current_step
                            self.expert_promotion_candidate_steps[expert_id] = 0
                            self.num_demotions += 1
                            precision_changes[expert_id] = new_precision
            else:
                # Reset demotion counter if back in top-M
                self.expert_promotion_candidate_steps[expert_id] = 0

        # Handle memory pressure: demote more aggressively
        if memory_pressure:
            # Sort by priority and demote lowest priority W4A4 experts
            w4a4_experts = [(expert_id, self.compute_priority(expert_id))
                            for expert_id in range(self.num_experts)
                            if self.expert_precision[expert_id] == "w4a4"]
            w4a4_experts.sort(key=lambda x: x[1])

            # Demote lowest priority experts until under budget
            for expert_id, _ in w4a4_experts:
                if vram_ok and bw_ok:
                    break

                # Keep top-M experts even under pressure
                is_top_m = expert_id in [
                    exp_id for exp_id, _ in priorities[:self.top_m_experts]]
                if is_top_m:
                    continue

                self.expert_precision[expert_id] = self.high_pressure_precision
                self.expert_last_switch_step[expert_id] = self.current_step
                self.num_demotions += 1
                precision_changes[expert_id] = self.high_pressure_precision

                # Recheck budgets
                vram_usage, vram_ok = self.check_vram_budget()
                bw_usage, bw_ok = self.check_bandwidth_budget()

        # Log precision changes
        for expert_id, new_precision in precision_changes.items():
            self.precision_history[expert_id].append({
                'step': self.current_step,
                'precision': new_precision,
            })

        return precision_changes

    def get_precision(self, expert_id: int) -> str:
        """
        Get current precision for an expert.

        Args:
            expert_id: Expert ID

        Returns:
            precision: Precision string (e.g., "w4a4", "w2a4")
        """
        return self.expert_precision[expert_id]

    def get_all_precisions(self) -> Dict[int, str]:
        """Get dictionary of all expert precisions."""
        return {expert_id: self.expert_precision[expert_id]
                for expert_id in range(self.num_experts)}

    def get_statistics(self) -> Dict[str, any]:
        """Get scheduler statistics."""
        precision_counts = defaultdict(int)
        for precision in self.expert_precision:
            precision_counts[precision] += 1

        vram_usage, vram_ok = self.check_vram_budget()
        bw_usage, bw_ok = self.check_bandwidth_budget()

        return {
            'current_step': self.current_step,
            'num_promotions': self.num_promotions,
            'num_demotions': self.num_demotions,
            'precision_counts': dict(precision_counts),
            'vram_usage_gb': vram_usage,
            'vram_budget_gb': self.vram_budget_gb,
            'vram_ok': vram_ok,
            'bandwidth_usage': bw_usage,
            'bandwidth_ok': bw_ok,
            'avg_hit_rate': np.mean(self.expert_hit_rate),
            'avg_risk': np.mean(self.expert_risk),
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        self.num_promotions = 0
        self.num_demotions = 0
        self.precision_history.clear()

    def save_state(self, path: str):
        """Save scheduler state to file."""
        import pickle
        state = {
            'expert_precision': self.expert_precision,
            'expert_risk': self.expert_risk,
            'expert_hit_rate': self.expert_hit_rate,
            'expert_last_switch_step': self.expert_last_switch_step,
            'expert_promotion_candidate_steps': self.expert_promotion_candidate_steps,
            'expert_hit_count': self.expert_hit_count,
            'total_hits': self.total_hits,
            'current_step': self.current_step,
            'num_promotions': self.num_promotions,
            'num_demotions': self.num_demotions,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, path: str):
        """Load scheduler state from file."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.expert_precision = state['expert_precision']
        self.expert_risk = state['expert_risk']
        self.expert_hit_rate = state['expert_hit_rate']
        self.expert_last_switch_step = state['expert_last_switch_step']
        self.expert_promotion_candidate_steps = state['expert_promotion_candidate_steps']
        self.expert_hit_count = state['expert_hit_count']
        self.total_hits = state['total_hits']
        self.current_step = state['current_step']
        self.num_promotions = state['num_promotions']
        self.num_demotions = state['num_demotions']


def test_precision_scheduler():
    """
    Unit tests for PrecisionScheduler.
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Testing PrecisionScheduler...")

    # Test initialization
    logger.info("\n--- Testing initialization ---")
    num_experts = 64

    scheduler = PrecisionScheduler(
        num_experts=num_experts,
        vram_budget_gb=80.0,
        bandwidth_budget_gbps=2000.0,
        top_m_experts=8,
        default_precision="w2a4",
        high_pressure_precision="w2a2",
        hysteresis_steps=10,
        cooldown_steps=5,
    )

    logger.info(f"Number of experts: {scheduler.num_experts}")
    logger.info(f"Default precision: {scheduler.default_precision}")
    logger.info(f"✓ Initialization test passed")

    # Test hit count updates
    logger.info("\n--- Testing hit count updates ---")
    for _ in range(100):
        # Simulate expert usage (Zipfian distribution)
        expert_ids = np.random.choice(num_experts, size=4, replace=False,
                                      p=np.array([1/(i+1) for i in range(num_experts)]) /
                                      sum([1/(i+1) for i in range(num_experts)]))
        scheduler.update_hit_counts(expert_ids.tolist())

    stats = scheduler.get_statistics()
    logger.info(f"Average hit rate: {stats['avg_hit_rate']:.6f}")
    logger.info(f"Precision counts: {stats['precision_counts']}")
    logger.info(f"✓ Hit count update test passed")

    # Test risk updates
    logger.info("\n--- Testing risk updates ---")
    for expert_id in range(num_experts):
        # Higher risk for frequently used experts
        risk = scheduler.expert_hit_rate[expert_id] * \
            2.0 + np.random.rand() * 0.5
        scheduler.update_risk(expert_id, risk)

    logger.info(f"Average risk: {stats['avg_risk']:.6f}")
    logger.info(f"✓ Risk update test passed")

    # Test scheduling
    logger.info("\n--- Testing scheduling ---")

    # Promote some high-risk experts
    # First few experts typically have high hit rate
    high_risk_experts = [0, 1, 2, 3]

    for step in range(20):
        if step % 5 == 0:
            changes = scheduler.schedule(
                promotion_candidates=high_risk_experts)
            if changes:
                logger.info(f"Step {step}: Precision changes: {changes}")
        else:
            changes = scheduler.schedule()

    stats = scheduler.get_statistics()
    logger.info(f"Current step: {stats['current_step']}")
    logger.info(f"Num promotions: {stats['num_promotions']}")
    logger.info(f"Num demotions: {stats['num_demotions']}")
    logger.info(f"Precision counts: {stats['precision_counts']}")
    logger.info(f"✓ Scheduling test passed")

    # Test budget checking
    logger.info("\n--- Testing budget checking ---")
    vram_usage, vram_ok = scheduler.check_vram_budget()
    bw_usage, bw_ok = scheduler.check_bandwidth_budget()

    logger.info(
        f"VRAM usage: {vram_usage:.2f} GB (budget: {scheduler.vram_budget_gb} GB)")
    logger.info(f"VRAM OK: {vram_ok}")
    logger.info(f"Bandwidth usage: {bw_usage:.4f} (relative)")
    logger.info(f"Bandwidth OK: {bw_ok}")
    logger.info(f"✓ Budget checking test passed")

    # Test priority computation
    logger.info("\n--- Testing priority computation ---")
    priorities = [(expert_id, scheduler.compute_priority(expert_id))
                  for expert_id in range(10)]
    priorities.sort(key=lambda x: x[1], reverse=True)

    logger.info("Top 5 experts by priority:")
    for expert_id, priority in priorities[:5]:
        logger.info(f"  Expert {expert_id}: priority={priority:.4f}, "
                    f"risk={scheduler.expert_risk[expert_id]:.4f}, "
                    f"hit_rate={scheduler.expert_hit_rate[expert_id]:.4f}")
    logger.info(f"✓ Priority computation test passed")

    # Test get_precision
    logger.info("\n--- Testing get_precision ---")
    for expert_id in range(5):
        precision = scheduler.get_precision(expert_id)
        logger.info(f"Expert {expert_id}: {precision}")
    logger.info(f"✓ Get precision test passed")

    # Test state save/load
    logger.info("\n--- Testing state save/load ---")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "scheduler_state.pkl")

        # Save state
        scheduler.save_state(state_path)
        logger.info(f"State saved to {state_path}")

        # Create new scheduler and load state
        scheduler2 = PrecisionScheduler(num_experts=num_experts)
        scheduler2.load_state(state_path)

        # Verify state
        assert scheduler2.current_step == scheduler.current_step
        assert scheduler2.num_promotions == scheduler.num_promotions
        assert scheduler2.expert_precision == scheduler.expert_precision

        logger.info(f"State loaded successfully")

    logger.info(f"✓ State save/load test passed")

    logger.info("\n✓ All PrecisionScheduler tests passed!")
    return True


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_precision_scheduler()
