#!/usr/bin/env python3
"""Fast structural gate for one resumable Qwen3-30B artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynaexq.experiments.eval_quality import environment_metadata
from dynaexq.experiments.eval_quality import PAPER_PROTOCOL
from qwen30b_policy_contract import (
    calibration_policy_problems,
    policy_contract_problems,
)


def _source_hash_is_allowed(
    source_hash: str,
    current_source_hash: str | None | set[str],
) -> bool:
    if current_source_hash is None:
        return True
    if isinstance(current_source_hash, set):
        return source_hash in current_source_hash
    return source_hash == current_source_hash


def _quality_reference_problems(artifact: dict) -> list[str]:
    found = []
    if artifact.get("artifact_type") != "quality_evaluation":
        found.append("all-high reference is not a quality_evaluation artifact")
    if artifact.get("paper_model") != "qwen30b":
        found.append("all-high reference paper_model is not qwen30b")
    if artifact.get("paper_method") != "reference_fp16":
        found.append("all-high reference paper_method is not reference_fp16")
    if artifact.get("method") != "reference_fp16":
        found.append("all-high reference method is not reference_fp16")
    if artifact.get("seed") != PAPER_PROTOCOL["seed"]:
        found.append("all-high reference seed is not the paper seed")
    if artifact.get("evaluation_protocol", {}).get("name") != PAPER_PROTOCOL["name"]:
        found.append("all-high reference protocol is not the paper protocol")
    benchmark_names = set(artifact.get("benchmarks", {}))
    expected = {"wikitext", "mmlu_pro", "gpqa", "aime25", "gsm8k", "humaneval"}
    if benchmark_names != expected:
        found.append(
            "all-high reference benchmark set mismatch: "
            f"{sorted(benchmark_names)}"
        )
    failed = {
        name: result.get("failed")
        for name, result in artifact.get("benchmarks", {}).items()
        if int(result.get("failed", 0)) != 0
    }
    if failed:
        found.append(f"all-high reference infrastructure failures: {failed}")
    placement = artifact.get("model_placement")
    if not isinstance(placement, dict):
        found.append("all-high reference model_placement absent")
        return found
    if placement.get("requested_device_map") != "auto":
        found.append("all-high reference did not request device_map=auto")
    if placement.get("required_cuda_device_count") != 2:
        found.append("all-high reference did not require two CUDA devices")
    if placement.get("resolved_cuda_devices") != [0, 1]:
        found.append("all-high reference did not resolve to CUDA [0, 1]")
    resolved = placement.get("resolved_device_map")
    if not isinstance(resolved, dict) or not resolved:
        found.append("all-high reference resolved device map absent")
    return found


def _p1_allocation_problems(path: Path, artifact: dict) -> list[str]:
    if not (
        path.name.startswith("rq2_grid_")
        or path.name.startswith("rq2_switch_")
    ):
        return []
    found = []
    protocol = artifact.get("allocation_protocol")
    if not isinstance(protocol, dict):
        return ["P1 allocation_protocol absent"]
    if protocol.get("name") != "qwen30b_p1_joint_allocation_v1":
        found.append("P1 allocation protocol name mismatch")
    infeasible = protocol.get("memory_feasible") is False
    assigned = protocol.get("assigned_bytes")
    if not isinstance(assigned, dict):
        found.append("P1 assigned byte pools absent")
    else:
        total = 0
        for name in ("fidelity", "residency", "lookahead"):
            value = assigned.get(name)
            if not isinstance(value, int) or value < 0:
                found.append(f"P1 assigned {name} bytes absent")
            else:
                total += value
        budget = protocol.get("expert_memory_budget_bytes")
        if not isinstance(budget, int) or budget < 0:
            found.append("P1 expert memory budget absent")
        elif total > budget:
            found.append("P1 assigned byte pools exceed B")
    if infeasible:
        reason = protocol.get("infeasible_reason")
        if not isinstance(reason, str) or not reason:
            found.append("P1 infeasible artifact lacks reason")
        if artifact.get("artifact_type") != "dynaexq_p1_allocation_infeasible":
            found.append("P1 infeasible artifact type mismatch")
        return found
    realized = protocol.get("realized")
    if not isinstance(realized, dict):
        found.append("P1 realized allocation absent")
    else:
        final_arena = realized.get("final_arena")
        if not isinstance(final_arena, dict):
            found.append("P1 final arena absent")
        else:
            purpose_sum = sum(
                int(final_arena.get(name, -1))
                for name in ("fid_bytes", "res_bytes", "look_bytes")
            )
            if purpose_sum != int(final_arena.get("published_bytes", -2)):
                found.append("P1 arena purpose bytes are inconsistent")
    if path.name.startswith("rq2_switch_"):
        if not protocol.get("workload_switch_trace"):
            found.append("P1 switch trace name absent")
        benchmark = artifact.get("benchmark", {})
        if not isinstance(benchmark, dict) or "metrics" not in benchmark:
            found.append("P1 switch benchmark metrics absent")
    return found


def problems(path: Path, *, current_source_hash: str | None | set[str]) -> list[str]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot read JSON: {error}"]
    found = []
    if int(artifact.get("schema_version", 0)) < 2:
        found.append("schema_version < 2")
    if path.name == "all_high_reference_quality_seed42.json":
        source_hash = (
            artifact.get("environment", {}).get("git", {}).get(
                "source_tree_sha256"
            )
        )
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            found.append("runtime source hash absent")
        elif not _source_hash_is_allowed(source_hash, current_source_hash):
            found.append("runtime source hash differs from current tree")
        if not artifact.get("checkpoint", {}).get("weight_hashes_included"):
            found.append("checkpoint weight hashes absent")
        found.extend(_quality_reference_problems(artifact))
        return found
    if artifact.get("environment", {}).get("gpus") != ["NVIDIA RTX A6000"]:
        found.append("unexpected GPU inventory")
    source_hash = (
        artifact.get("environment", {}).get("git", {}).get(
            "source_tree_sha256"
        )
    )
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        found.append("runtime source hash absent")
    elif not _source_hash_is_allowed(source_hash, current_source_hash):
        found.append("runtime source hash differs from current tree")
    if not artifact.get("checkpoint", {}).get("weight_hashes_included"):
        found.append("checkpoint weight hashes absent")
    is_calibration = (
        artifact.get("artifact_type") == "dynaexq_initial_expert_ranking"
    )
    if is_calibration:
        found.extend(calibration_policy_problems(artifact))
    else:
        policy_hash = artifact.get("initial_map", {}).get(
            "policy_state_sha256"
        )
        if not isinstance(policy_hash, str) or len(policy_hash) != 64:
            found.append("initial-map policy-state hash absent")
    transitions = artifact.get("transition_stats", {})
    if int(transitions.get("failed_transitions", 0)) != 0:
        found.append("failed transitions are nonzero")
    if int(transitions.get("active_transitions", 0)) != 0:
        found.append("active transitions remain")
    found.extend(_p1_allocation_problems(path, artifact))
    protocol = artifact.get("allocation_protocol")
    if (
        isinstance(protocol, dict)
        and protocol.get("memory_feasible") is False
        and path.name.startswith(("rq2_grid_", "rq2_switch_"))
    ):
        return found
    is_p1_artifact = path.name.startswith(("rq2_grid_", "rq2_switch_"))
    if artifact.get("policy") == "joint":
        quality_risk = (
            (artifact.get("wrapper_stats") or {})
            .get("residency_controller") or {}
        ).get("quality_risk")
        if not isinstance(quality_risk, dict):
            if not is_p1_artifact:
                found.append("joint policy quality-risk snapshot absent")
        else:
            try:
                current = float(quality_risk["current"])
                limit = float(quality_risk["limit"])
                tolerance = float(quality_risk["tolerance"])
            except (KeyError, TypeError, ValueError):
                found.append("joint policy quality-risk values absent")
            else:
                if (
                    quality_risk.get("feasible") is not True
                    or current > limit + tolerance
                ):
                    found.append("joint policy ended outside quality-risk limit")
    arena = transitions.get("arena")
    if is_calibration and not isinstance(arena, dict):
        found.extend(policy_contract_problems(path.name, artifact))
        return found
    if not isinstance(arena, dict):
        found.append("final arena snapshot absent")
        return found
    published = int(arena.get("published_bytes", -1))
    reserved = int(arena.get("reserved_bytes", -1))
    capacity = int(arena.get("capacity_bytes", -1))
    if min(published, reserved, capacity) < 0:
        found.append("arena byte counters absent")
    elif published + reserved > capacity:
        found.append("arena exceeds capacity")
    purposes = sum(
        int(arena.get(name, -1))
        for name in ("fid_bytes", "res_bytes", "look_bytes")
    )
    if purposes != published:
        found.append("purpose bytes do not sum to published bytes")
    if int(arena.get("reclaim_pending_bytes", -1)) != 0:
        found.append("reclaim-pending bytes remain")
    if int(arena.get("held_donor_count", -1)) != 0:
        found.append("held donors remain")
    found.extend(policy_contract_problems(path.name, artifact))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--allow-stale-source", action="store_true")
    args = parser.parse_args()
    current_source_hash = None
    if not args.allow_stale_source:
        current_source_hash = environment_metadata().get("git", {}).get(
            "source_tree_sha256"
        )
        if not current_source_hash:
            raise SystemExit("cannot compute current runtime source hash")
    found = problems(
        args.artifact,
        current_source_hash=current_source_hash,
    )
    if found:
        print(json.dumps({"artifact": str(args.artifact), "problems": found}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
