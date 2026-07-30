#!/usr/bin/env python3
"""Aggregate eight audited DynaExQ WikiText points into one curve artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynaexq.experiments.eval_dynamic import (
    PERPLEXITY_LOW_RATIOS_PCT,
)
from dynaexq.experiments.eval_quality import (
    PAPER_PROTOCOL,
    SCHEMA_VERSION,
    environment_metadata,
)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_git(data: dict) -> bool:
    environment = data.get("environment")
    git = environment.get("git", {}) if isinstance(environment, dict) else {}
    return bool(git.get("commit")) and git.get("dirty") is False


def build_curve(
    point_paths: list[Path],
    *,
    paper_model: str,
) -> dict:
    """Validate source points and return a provenance-rich curve artifact."""
    if len(point_paths) != len(PERPLEXITY_LOW_RATIOS_PCT):
        raise ValueError("exactly eight --point artifacts are required")
    sources = []
    by_ratio = {}
    checkpoint = None
    config = None
    initial_map = None
    for path in point_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("schema_version", 0)) < 2:
            raise ValueError(f"legacy point artifact: {path}")
        if data.get("artifact_type") != "dynaexq_perplexity_point":
            raise ValueError(f"perplexity point type mismatch: {path}")
        if data.get("paper_model") != paper_model:
            raise ValueError(f"paper model mismatch: {path}")
        if data.get("evaluation_protocol", {}).get("name") != PAPER_PROTOCOL["name"]:
            raise ValueError(f"paper protocol mismatch: {path}")
        if data.get("selection_policy") != "calibrated_coldest_prefix":
            raise ValueError(f"selection policy mismatch: {path}")
        if data.get("wrapper_stats", {}).get("scheduler_enabled") is not False:
            raise ValueError(f"perplexity point was not frozen: {path}")
        if not _clean_git(data):
            raise ValueError(f"point was not produced by a clean commit: {path}")
        ratio = int(data.get("low_ratio_pct", -1))
        if ratio in by_ratio:
            raise ValueError(f"duplicate low-precision ratio: {ratio}")
        if ratio not in PERPLEXITY_LOW_RATIOS_PCT:
            raise ValueError(f"unexpected low-precision ratio: {ratio}")
        if checkpoint is None:
            checkpoint = data["checkpoint"]
            config = data["config"]
            initial_map = data["initial_map"]
        elif (
            data.get("checkpoint") != checkpoint
            or data.get("config") != config
            or data.get("initial_map") != initial_map
        ):
            raise ValueError("curve points do not share checkpoint/config/map")
        result = data.get("benchmarks", {}).get("wikitext")
        if not isinstance(result, dict):
            raise ValueError(f"point has no WikiText result: {path}")
        if data.get("low_experts") is None:
            raise ValueError(f"point has no explicit low-expert sets: {path}")
        by_ratio[ratio] = {
            "low_ratio_pct": ratio,
            "low_experts_per_layer": data["low_experts_per_layer"],
            "selection_policy": data["selection_policy"],
            "low_experts_sha256": data["low_experts_sha256"],
            "perplexity": result["score"],
            "total_nll": result["total_nll"],
            "total_tokens": result["total_tokens"],
            "windows": result["windows"],
            "window_tokens": result["window_tokens"],
            "stride_tokens": result["stride_tokens"],
            "window_details": result["window_details"],
            "dataset": result["dataset"],
        }
        sources.append(
            {
                "path": path.resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "low_ratio_pct": ratio,
            }
        )
    if tuple(sorted(by_ratio)) != PERPLEXITY_LOW_RATIOS_PCT:
        raise ValueError("curve ratio grid is incomplete")
    assert checkpoint is not None and config is not None and initial_map is not None
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "perplexity_curve",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_model": paper_model,
        "checkpoint": checkpoint,
        "config": config,
        "seed": PAPER_PROTOCOL["seed"],
        "evaluation_protocol": PAPER_PROTOCOL,
        "ranking_sha256": initial_map["ranking_sha256"],
        "expert_ranking": initial_map["expert_ranking"],
        "points": [by_ratio[ratio] for ratio in PERPLEXITY_LOW_RATIOS_PCT],
        "source_points": sorted(
            sources,
            key=lambda source: source["low_ratio_pct"],
        ),
        "environment": environment_metadata(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-model",
        required=True,
        choices=("qwen30b", "qwen80b"),
    )
    parser.add_argument("--point", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_curve(args.point, paper_model=args.paper_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paper_model": args.paper_model,
                "points": len(artifact["points"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
