from types import SimpleNamespace

import pytest

from dynaexq.core import DynaExqConfig
from dynaexq.experiments.eval_dynamic import (
    QWEN30B_LOW_POOL_BYTES,
    _build_allocation_protocol,
)


def _args(**overrides):
    values = {
        "phi_override": None,
        "fidelity_share_pct": None,
        "lookahead_share_pct": None,
        "residency_share_pct": None,
        "allocation_policy": None,
        "workload_switch_trace": None,
        "predictor_noise_prob": None,
        "copy_rate_gbps": None,
        "byte_inflation": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _config():
    return DynaExqConfig.from_yaml("dynaexq/configs/qwen30b_joint_a6000.yaml")


def test_p1_share_to_bytes_is_deterministic_and_sums_to_budget():
    config = _config()
    protocol = _build_allocation_protocol(
        _args(
            phi_override=0.85,
            allocation_policy="dynabyte",
            fidelity_share_pct=20,
            lookahead_share_pct=10,
            residency_share_pct=70,
        ),
        config,
    )
    assert protocol is not None
    assigned = protocol["assigned_bytes"]
    assert sum(assigned.values()) == protocol["expert_memory_budget_bytes"]
    assert protocol["expert_memory_budget_bytes"] == int(
        round(QWEN30B_LOW_POOL_BYTES * 0.85)
    )
    assert protocol["phi_budget_basis"] == {
        "name": "qwen30b_measured_low_pool",
        "low_pool_bytes": QWEN30B_LOW_POOL_BYTES,
    }
    assert protocol["shares_pct"] == {
        "fidelity": 20,
        "residency": 70,
        "lookahead": 10,
    }
    assert config.memory.device_mem_bytes == (
        protocol["expert_memory_budget_bytes"]
        + config.memory.reserve_kv_bytes
        + config.memory.reserve_act_bytes
        + config.memory.reserve_dense_bytes
        + config.memory.reserve_kernel_workspace_bytes
        + config.memory.safety_margin_bytes
    )
    assert config.memory.initial_high_precision_ratio == 0.20
    assert config.memory.resident_ratio == 0.70


def test_zero_lookahead_share_disables_lookahead_admission():
    config = _config()
    protocol = _build_allocation_protocol(
        _args(
            allocation_policy="dynabyte",
            fidelity_share_pct=30,
            lookahead_share_pct=0,
            residency_share_pct=70,
        ),
        config,
    )
    assert protocol is not None
    assert protocol["assigned_bytes"]["lookahead"] == 0
    assert config.scheduler.lookahead_depth == 0
    assert config.scheduler.lookahead_per_layer == 0


def test_p1_shares_must_sum_to_one_hundred():
    with pytest.raises(ValueError, match="sum to 100"):
        _build_allocation_protocol(
            _args(
                allocation_policy="dynabyte",
                fidelity_share_pct=40,
                lookahead_share_pct=40,
                residency_share_pct=40,
            ),
            _config(),
        )
