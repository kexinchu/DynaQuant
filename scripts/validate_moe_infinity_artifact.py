#!/usr/bin/env python3
"""Validate one formal MoE-Infinity artifact before committing it."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument(
        "--batch-size", required=True, type=int, choices=(1, 2, 4, 8, 16, 32)
    )
    args = parser.parse_args()

    formal_root = args.formal_root.resolve()
    artifact = args.artifact.resolve()
    sys.path.insert(0, str(formal_root))
    from scripts.audit_paper_results import validate_manifest_artifact

    data = json.loads(artifact.read_text(encoding="utf-8"))
    claim_id = f"performance:qwen30b:moe_infinity:bs{args.batch_size}"
    problems = validate_manifest_artifact(
        "performance",
        str(artifact),
        data,
        claim_id,
    )
    head = subprocess.check_output(
        ["git", "-C", str(formal_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    artifact_git = data.get("environment", {}).get("git", {})
    if artifact_git.get("commit") != head:
        problems.append("experiment commit does not match worktree HEAD")

    benchmark = data.get("benchmark", {})
    samples = benchmark.get("samples", [])
    e2e = [float(sample["model_e2e_ms"]) for sample in samples]
    mean = statistics.fmean(e2e) if e2e else None
    stdev = statistics.stdev(e2e) if len(e2e) > 1 else 0.0
    summary = {
        "artifact": str(artifact),
        "claim_id": claim_id,
        "validation_problems": problems,
        "sample_count": len(samples),
        "e2e_coefficient_of_variation": (
            stdev / mean if mean not in (None, 0.0) else None
        ),
        "metrics": benchmark.get("metrics", {}),
        "baseline_runtime_stats": data.get("baseline_runtime_stats", {}),
        "git": artifact_git,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
