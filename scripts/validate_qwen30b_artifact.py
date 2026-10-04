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
from qwen30b_policy_contract import (
    calibration_policy_problems,
    policy_contract_problems,
)


def problems(path: Path, *, current_source_hash: str | None) -> list[str]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot read JSON: {error}"]
    found = []
    if int(artifact.get("schema_version", 0)) < 2:
        found.append("schema_version < 2")
    if artifact.get("environment", {}).get("gpus") != ["NVIDIA RTX A6000"]:
        found.append("unexpected GPU inventory")
    source_hash = (
        artifact.get("environment", {}).get("git", {}).get(
            "source_tree_sha256"
        )
    )
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        found.append("runtime source hash absent")
    elif current_source_hash is not None and source_hash != current_source_hash:
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
