"""Artifact-level component contracts for the Qwen3-30B policy lane."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


POLICY_CONTRACTS = {
    "all_high_reference": {
        "scheduler_enabled": False,
        "controller": True,
        "high_precision_ratio": 0.0,
        "resident_ratio": 0.5,
        "lookahead_depth": 0,
        "hi_format": "fp16",
        "lo_format": "fp16",
    },
    "joint": {
        "scheduler_enabled": True,
        "controller": True,
        "high_precision_ratio": 0.25,
        "resident_ratio": 0.5,
        "lookahead_depth": 3,
    },
    "residency": {
        "scheduler_enabled": False,
        "controller": True,
        "high_precision_ratio": 0.0,
        "resident_ratio": 0.5,
        "lookahead_depth": 0,
    },
    "fidelity": {
        "scheduler_enabled": True,
        "controller": False,
        "high_precision_ratio": 0.15,
        "resident_ratio": 1.0,
        "lookahead_depth": 0,
    },
    "uniform_low": {
        "scheduler_enabled": False,
        "controller": False,
        "high_precision_ratio": 0.0,
        "resident_ratio": 1.0,
        "lookahead_depth": 0,
    },
    "static_mixed": {
        "scheduler_enabled": False,
        "controller": False,
        "high_precision_ratio": 0.15,
        "resident_ratio": 1.0,
        "lookahead_depth": 0,
    },
    "lookahead": {
        "scheduler_enabled": False,
        "controller": True,
        "high_precision_ratio": 0.0,
        "resident_ratio": 0.5,
        "lookahead_depth": 3,
    },
}


def calibration_policy_problems(artifact: dict[str, Any]) -> list[str]:
    """Validate the calibrated routing, sensitivity, and fidelity state."""
    if artifact.get("artifact_type") != "dynaexq_initial_expert_ranking":
        return []
    model = artifact.get("model_config", {})
    layers = model.get("layers")
    experts = model.get("experts_per_layer")
    if not isinstance(layers, int) or not isinstance(experts, int):
        return ["calibration model shape is absent"]
    fields = {
        "routing_weights": artifact.get("routing_weights"),
        "normalized_sensitivity": artifact.get("normalized_sensitivity"),
        "fidelity_ranking": artifact.get("fidelity_ranking"),
    }
    problems = []
    expected_layers = {str(layer) for layer in range(layers)}
    for name, value in fields.items():
        if not isinstance(value, dict) or set(value) != expected_layers:
            problems.append(f"{name} does not cover every layer")
            continue
        for layer in expected_layers:
            row = value.get(layer)
            if not isinstance(row, list) or len(row) != experts:
                problems.append(f"{name}[{layer}] has the wrong width")
                break
            if name == "fidelity_ranking":
                if set(row) != set(range(experts)):
                    problems.append(f"{name}[{layer}] is not a permutation")
                    break
            elif any(
                not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or float(item) < 0.0
                for item in row
            ):
                problems.append(f"{name}[{layer}] contains an invalid value")
                break
    if problems:
        return problems
    encoded = json.dumps(
        fields,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_hash = hashlib.sha256(encoded).hexdigest()
    if artifact.get("policy_state_sha256") != expected_hash:
        problems.append("calibration policy-state hash mismatch")
    return problems


def policy_from_filename(filename: str) -> str | None:
    for policy in POLICY_CONTRACTS:
        if filename.startswith(f"{policy}_"):
            return policy
    return None


def policy_contract_problems(
    filename: str,
    artifact: dict[str, Any],
) -> list[str]:
    """Return component-boundary violations encoded in one raw artifact."""
    policy = policy_from_filename(filename)
    if policy is None:
        return []
    expected = POLICY_CONTRACTS[policy]
    problems = []
    if artifact.get("policy") != policy:
        problems.append(
            f"artifact policy={artifact.get('policy')!r}, expected={policy!r}"
        )
    config_path = artifact.get("config_path")
    expected_config = f"qwen30b_{policy}_a6000.yaml"
    if not isinstance(config_path, str) or not config_path.endswith(
        expected_config
    ):
        problems.append(
            f"{policy} config_path={config_path!r}, "
            f"expected suffix={expected_config!r}"
        )
    wrapper = artifact.get("wrapper_stats", {})
    scheduler_enabled = wrapper.get("scheduler_enabled")
    if scheduler_enabled is not expected["scheduler_enabled"]:
        problems.append(
            f"{policy} scheduler_enabled={scheduler_enabled!r}, "
            f"expected={expected['scheduler_enabled']!r}"
        )

    controller = wrapper.get("residency_controller")
    if expected["controller"]:
        if not isinstance(controller, dict):
            problems.append(f"{policy} residency controller is absent")
        elif controller.get("lookahead_depth") != expected["lookahead_depth"]:
            problems.append(
                f"{policy} lookahead_depth={controller.get('lookahead_depth')!r}, "
                f"expected={expected['lookahead_depth']!r}"
            )
    elif controller is not None:
        problems.append(f"{policy} unexpectedly has a residency controller")

    initialization = artifact.get("runtime_initialization", {})
    for field, expected_value in (
        ("requested_high_precision_ratio", expected["high_precision_ratio"]),
        ("requested_resident_ratio", expected["resident_ratio"]),
    ):
        value = initialization.get(field)
        if not isinstance(value, (int, float)) or abs(value - expected_value) > 1e-9:
            problems.append(
                f"{policy} {field}={value!r}, expected={expected_value!r}"
            )
    precision = artifact.get("config", {}).get("precision", {})
    for field, expected_value in (
        ("hi", expected.get("hi_format", "fp16")),
        ("lo", expected.get("lo_format", "int4")),
    ):
        value = precision.get(field)
        if value != expected_value:
            problems.append(
                f"{policy} precision.{field}={value!r}, "
                f"expected={expected_value!r}"
            )
    return problems
