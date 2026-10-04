"""Keep the paper policy configurations component-isolated."""

from pathlib import Path

from dynaexq.core.config import DynaExqConfig


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load(policy: str) -> DynaExqConfig:
    return DynaExqConfig.from_yaml(
        CONFIG_DIR / f"qwen30b_{policy}_a6000.yaml"
    )


def test_qwen30b_component_policy_configs_are_isolated() -> None:
    joint = _load("joint")
    assert joint.scheduler.enabled
    assert joint.scheduler.lookahead_depth > 0
    assert joint.memory.resident_ratio < 1.0
    assert joint.memory.initial_high_precision_ratio > 0.0

    residency = _load("residency")
    assert not residency.scheduler.enabled
    assert residency.scheduler.lookahead_depth == 0
    assert residency.memory.resident_ratio < 1.0
    assert residency.memory.initial_high_precision_ratio == 0.0

    fidelity = _load("fidelity")
    assert fidelity.scheduler.enabled
    assert fidelity.scheduler.lookahead_depth == 0
    assert fidelity.memory.resident_ratio == 1.0
    assert fidelity.memory.initial_high_precision_ratio > 0.0

    uniform_low = _load("uniform_low")
    assert not uniform_low.scheduler.enabled
    assert uniform_low.scheduler.lookahead_depth == 0
    assert uniform_low.memory.resident_ratio == 1.0
    assert uniform_low.memory.initial_high_precision_ratio == 0.0

    static_mixed = _load("static_mixed")
    assert not static_mixed.scheduler.enabled
    assert static_mixed.scheduler.lookahead_depth == 0
    assert static_mixed.memory.resident_ratio == 1.0
    assert static_mixed.memory.initial_high_precision_ratio > 0.0

    lookahead = _load("lookahead")
    assert not lookahead.scheduler.enabled
    assert lookahead.scheduler.lookahead_depth > 0
    assert lookahead.memory.resident_ratio < 1.0
    assert lookahead.memory.initial_high_precision_ratio == 0.0

    all_high = _load("all_high_reference")
    assert not all_high.scheduler.enabled
    assert all_high.scheduler.lookahead_depth == 0
    assert all_high.memory.resident_ratio < 1.0
    assert all_high.memory.initial_high_precision_ratio == 0.0
    assert all_high.precision.hi == "fp16"
    assert all_high.precision.lo == "fp16"
