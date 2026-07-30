from __future__ import annotations

import pytest

from dynaexq.core.budget_init import BudgetInitializer
from dynaexq.core.config import Tier


def _footprint(layer: int, tier: Tier) -> int:
    return 40 if tier == Tier.HI else 10


def test_budget_includes_transient_headroom():
    result = BudgetInitializer(
        num_layers=2,
        experts_per_layer=4,
        memory_footprint_fn=_footprint,
        device_mem_bytes=400,
        max_inflight=2,
    ).compute()
    assert result.transient_bytes == 80
    assert result.resident_budget == 320
    assert result.total_reserved_bytes <= result.available_memory


def test_budget_subtracts_runtime_workspace_before_expert_allocation():
    result = BudgetInitializer(
        num_layers=1,
        experts_per_layer=4,
        memory_footprint_fn=_footprint,
        device_mem_bytes=240,
        reserve_runtime_bytes=50,
        max_inflight=1,
    ).compute()
    assert result.available_memory == 190
    assert result.runtime_workspace_bytes == 50
    assert result.total_reserved_bytes <= 190


def test_all_low_infeasible_raises_instead_of_returning_invalid_result():
    initializer = BudgetInitializer(
        num_layers=2,
        experts_per_layer=8,
        memory_footprint_fn=_footprint,
        device_mem_bytes=150,
        max_inflight=1,
    )
    with pytest.raises(ValueError, match="All-LO"):
        initializer.compute()


def test_greedy_starts_from_all_low_footprint():
    result = BudgetInitializer(
        num_layers=1,
        experts_per_layer=4,
        memory_footprint_fn=_footprint,
        device_mem_bytes=180,
        max_inflight=1,
    ).compute(strategy="greedy")
    # 40 transient leaves 140 resident: all-LO costs 40 and each upgrade 30,
    # so exactly three experts can be high precision.
    assert result.n_hi == [3]
    assert result.total_expert_bytes == 130


def test_invalid_precision_order_raises():
    def bad(layer: int, tier: Tier) -> int:
        return 10 if tier == Tier.HI else 20

    with pytest.raises(ValueError, match="cannot be smaller"):
        BudgetInitializer(
            num_layers=1,
            experts_per_layer=1,
            memory_footprint_fn=bad,
            device_mem_bytes=100,
        ).compute()


def test_exact_high_precision_ratio_uses_floor_per_layer():
    result = BudgetInitializer(
        num_layers=2,
        experts_per_layer=7,
        memory_footprint_fn=_footprint,
        device_mem_bytes=1_000,
        max_inflight=1,
    ).compute(high_precision_ratio=0.30)
    assert result.n_hi == [2, 2]


def test_infeasible_exact_ratio_fails_instead_of_silently_reducing():
    initializer = BudgetInitializer(
        num_layers=1,
        experts_per_layer=4,
        memory_footprint_fn=_footprint,
        device_mem_bytes=150,
        max_inflight=1,
    )
    with pytest.raises(ValueError, match="requested high-precision ratio"):
        initializer.compute(high_precision_ratio=1.0)


@pytest.mark.parametrize("ratio", (-0.01, 1.01))
def test_exact_ratio_must_be_a_probability(ratio):
    initializer = BudgetInitializer(
        num_layers=1,
        experts_per_layer=1,
        memory_footprint_fn=_footprint,
        device_mem_bytes=100,
        max_inflight=1,
    )
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        initializer.compute(high_precision_ratio=ratio)
