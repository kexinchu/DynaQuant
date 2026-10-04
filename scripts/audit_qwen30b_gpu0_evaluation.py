#!/usr/bin/env python3
"""Audit the resumable Qwen3-30B/GPU0 evaluation lane."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

from qwen30b_policy_contract import (
    calibration_policy_problems,
    policy_contract_problems,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "evaluation" / "qwen30b"
DERIVED = ROOT / "results" / "evaluation" / "derived" / "qwen30b"
FIGURES = ROOT / "SIGMETRICS_2027" / "figures"
SEEDS = (42, 43, 44, 45, 46)
REGIMES = ("prefill", "decode", "mixed")
POLICIES = (
    "uniform_low",
    "static_mixed",
    "residency",
    "lookahead",
    "fidelity",
)


def expected() -> list[Path]:
    paths = [
        RESULTS / "calibration_wikitext103_256x2048.json",
        RESULTS / "joint_runtime_smoke.json",
        RESULTS / "residency_runtime_smoke.json",
        RESULTS / "fidelity_runtime_smoke.json",
        RESULTS / "uniform_low_runtime_smoke.json",
        RESULTS / "static_mixed_runtime_smoke.json",
        RESULTS / "lookahead_runtime_smoke.json",
        RESULTS / "all_high_reference_runtime_smoke.json",
        RESULTS / "all_high_reference_quality_seed42.json",
        RESULTS / "joint_quality_seed42.json",
        RESULTS / "routing_hotset.json",
        RESULTS / "overhead_seed42.json",
    ]
    paths.extend(
        RESULTS / f"ablation_{name}_seed42.json"
        for name in (
            "full",
            "static",
            "blocking",
            "no_hysteresis",
            "no_tenure",
            "ind_value",
            "slack_only",
            "single_timescale",
        )
    )
    paths.extend(
        RESULTS / f"sensitivity_hi{ratio}_seed42.json"
        for ratio in (0, 5, 10, 15, 20, 25, 30)
    )
    for policy in POLICIES:
        paths.append(RESULTS / f"{policy}_quality_seed42.json")
        for seed in SEEDS:
            paths.extend(
                RESULTS / f"{policy}_trace_{regime}_seed{seed}.json"
                for regime in REGIMES
            )
    for seed in SEEDS:
        paths.extend(
            RESULTS / f"joint_trace_{regime}_seed{seed}.json"
            for regime in REGIMES
        )
        paths.extend(
            RESULTS / f"joint_perf_seed{seed}_bs{batch}.json"
            for batch in (1, 2, 4, 8, 16, 32)
        )
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(
    path: Path,
    *,
    calibration_ranking_sha256: str | None,
    calibration_source_sha256: str | None,
) -> list[str]:
    problems: list[str] = []
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot read JSON: {error}"]
    if int(artifact.get("schema_version", 0)) < 2:
        problems.append("schema_version < 2")
    checkpoint = artifact.get("checkpoint", {})
    if not checkpoint.get("weight_hashes_included"):
        problems.append("checkpoint weight hashes absent")
    environment = artifact.get("environment", {})
    gpus = environment.get("gpus", [])
    if gpus != ["NVIDIA RTX A6000"]:
        problems.append(f"unexpected GPU inventory: {gpus}")
    artifact_type = artifact.get("artifact_type")
    is_calibration = artifact_type == "dynaexq_initial_expert_ranking"
    source_sha256 = environment.get("git", {}).get("source_tree_sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        problems.append("runtime source hash absent")
    if is_calibration:
        problems.extend(calibration_policy_problems(artifact))
    initialization = artifact.get("runtime_initialization", {})
    if not is_calibration:
        arena = initialization.get("arena_after_bootstrap")
        if not isinstance(arena, dict):
            problems.append("bootstrap arena snapshot absent")
        initial_map = artifact.get("initial_map", {})
        if not isinstance(initial_map.get("policy_state_sha256"), str):
            problems.append("initial-map policy-state hash absent")
        if (
            calibration_ranking_sha256 is not None
            and initial_map.get("ranking_sha256")
            != calibration_ranking_sha256
        ):
            problems.append("initial-map ranking differs from calibration")
        if (
            calibration_source_sha256 is not None
            and source_sha256 != calibration_source_sha256
        ):
            problems.append("runtime source hash differs from calibration")
    transitions = artifact.get("transition_stats", {})
    if int(transitions.get("failed_transitions", 0)) != 0:
        problems.append("failed transitions are nonzero")
    if int(transitions.get("active_transitions", 0)) != 0:
        problems.append("active transitions remain")
    final_arena = transitions.get("arena")
    if isinstance(final_arena, dict):
        published = int(final_arena.get("published_bytes", -1))
        reserved = int(final_arena.get("reserved_bytes", -1))
        capacity = int(final_arena.get("capacity_bytes", -1))
        purposes = sum(
            int(final_arena.get(name, -1))
            for name in ("fid_bytes", "res_bytes", "look_bytes")
        )
        if min(published, reserved, capacity) < 0:
            problems.append("final arena byte counters absent")
        elif published + reserved > capacity:
            problems.append("final arena exceeds capacity")
        if purposes != published:
            problems.append("final purpose bytes do not sum to published bytes")
        if int(final_arena.get("reclaim_pending_bytes", -1)) != 0:
            problems.append("reclaim-pending arena bytes remain")
        if int(final_arena.get("held_donor_count", -1)) != 0:
            problems.append("held donors remain")
    elif not is_calibration:
        problems.append("final arena snapshot absent")
    problems.extend(policy_contract_problems(path.name, artifact))
    return problems


def main() -> None:
    calibration_path = RESULTS / "calibration_wikitext103_256x2048.json"
    calibration_ranking_sha256 = None
    calibration_source_sha256 = None
    calibration_policy_sha256 = None
    if calibration_path.is_file():
        try:
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibration_ranking_sha256 = calibration.get("ranking_sha256")
            calibration_policy_sha256 = calibration.get(
                "policy_state_sha256"
            )
            calibration_source_sha256 = (
                calibration.get("environment", {})
                .get("git", {})
                .get("source_tree_sha256")
            )
        except (OSError, json.JSONDecodeError):
            pass
    complete = []
    invalid = {}
    missing = []
    failed = {}
    for path in expected():
        if not path.is_file():
            failure_path = Path(f"{path}.failed")
            if failure_path.is_file():
                try:
                    failed[path.relative_to(ROOT).as_posix()] = (
                        failure_path.read_text(encoding="utf-8").strip()
                    )
                except OSError as error:
                    failed[path.relative_to(ROOT).as_posix()] = str(error)
                continue
            missing.append(path.relative_to(ROOT).as_posix())
            continue
        problems = validate(
            path,
            calibration_ranking_sha256=calibration_ranking_sha256,
            calibration_source_sha256=calibration_source_sha256,
        )
        if path != calibration_path and path.is_file():
            try:
                policy_hash = json.loads(
                    path.read_text(encoding="utf-8")
                ).get("initial_map", {}).get("policy_state_sha256")
            except (OSError, json.JSONDecodeError):
                policy_hash = None
            if (
                calibration_policy_sha256 is not None
                and policy_hash != calibration_policy_sha256
            ):
                problems.append(
                    "initial-map policy state differs from calibration"
                )
        if problems:
            invalid[path.relative_to(ROOT).as_posix()] = problems
        else:
            complete.append(path.relative_to(ROOT).as_posix())
    report = {
        "schema_version": 1,
        "expected": len(expected()),
        "complete": len(complete),
        "invalid": invalid,
        "failed": failed,
        "missing": missing,
    }
    if not invalid and not failed and not missing:
        derived_problems = []
        manifest_path = DERIVED / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            derived_problems.append(f"cannot read derived manifest: {error}")
            manifest = {}
        expected_rows = {
            "latency": 120,
            "quality": 82,
            "sensitivity": 7,
            "runtime": 129,
        }
        counts = manifest.get("row_counts", {})
        for name, expected_count in expected_rows.items():
            if counts.get(name) != expected_count:
                derived_problems.append(
                    f"{name} rows={counts.get(name)!r}, expected={expected_count}"
                )
        if int(counts.get("mechanisms", 0)) <= 0:
            derived_problems.append("mechanism rows are empty")
        hashes = manifest.get("source_tree_sha256", [])
        if hashes != [calibration_source_sha256]:
            derived_problems.append(
                "derived source hash does not equal calibration source hash"
            )
        for name in (
            "eval_qwen30b_latency.pdf",
            "eval_qwen30b_mechanisms.pdf",
            "eval_qwen30b_sensitivity.pdf",
        ):
            path = FIGURES / name
            if not path.is_file() or path.stat().st_size == 0:
                derived_problems.append(f"missing derived figure: {name}")
                continue
            registered = manifest.get("rendered_figures", {}).get(name, {})
            if registered.get("sha256") != sha256_file(path):
                derived_problems.append(
                    f"derived figure hash is absent or stale: {name}"
                )
        statistics_path = DERIVED / "latency_statistics.json"
        registered_statistics = manifest.get("statistical_outputs", {}).get(
            "latency_statistics.json", {}
        )
        if not statistics_path.is_file() or statistics_path.stat().st_size == 0:
            derived_problems.append("missing paired latency statistics")
        elif registered_statistics.get("sha256") != sha256_file(
            statistics_path
        ):
            derived_problems.append(
                "paired latency-statistics hash is absent or stale"
            )
        else:
            try:
                statistics_payload = json.loads(
                    statistics_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                derived_problems.append(
                    f"cannot read paired latency statistics: {error}"
                )
            else:
                if statistics_payload.get("bootstrap_replicates") != 2000:
                    derived_problems.append(
                        "unexpected paired-bootstrap replicate count"
                    )
                comparisons = statistics_payload.get(
                    "paired_joint_minus_baseline", []
                )
                if len(comparisons) != 30:
                    derived_problems.append(
                        f"paired latency comparisons={len(comparisons)}, "
                        "expected=30"
                    )
        quality_path = DERIVED / "quality_statistics.json"
        registered_quality = manifest.get("statistical_outputs", {}).get(
            "quality_statistics.json", {}
        )
        if not quality_path.is_file() or quality_path.stat().st_size == 0:
            derived_problems.append("missing all-high quality comparisons")
        elif registered_quality.get("sha256") != sha256_file(quality_path):
            derived_problems.append(
                "quality-statistics hash is absent or stale"
            )
        else:
            try:
                quality_payload = json.loads(
                    quality_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                derived_problems.append(
                    f"cannot read quality statistics: {error}"
                )
            else:
                quality_comparisons = quality_payload.get(
                    "comparisons", []
                )
                if len(quality_comparisons) != 76:
                    derived_problems.append(
                        f"quality comparisons={len(quality_comparisons)}, "
                        "expected=76"
                    )
        report["derived_problems"] = derived_problems
    else:
        report["derived_problems"] = [
            "raw matrix is incomplete; derived outputs not audited"
        ]
    print(json.dumps(report, indent=2))
    raise SystemExit(
        1
        if invalid or failed or missing or report["derived_problems"]
        else 0
    )


if __name__ == "__main__":
    main()
