#!/usr/bin/env python3
"""Paired significance comparison for two paper-quality artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynaexq.experiments.eval_quality import (
    PAPER_PROTOCOL,
    SCHEMA_VERSION,
    environment_metadata,
)


BENCHMARKS = ("mmlu_pro", "gpqa", "aime25", "gsm8k", "humaneval")
QUALITY_COMPARISONS = {
    "qwen30b": ("static_int4", "dynaexq"),
    "qwen80b": ("static_int2", "dynaexq"),
    "phi35": ("static_int4", "dynaexq"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _benchmarks(data: dict[str, Any], result_key: str | None) -> dict[str, Any]:
    if result_key is None:
        value = data.get("benchmarks")
    else:
        value = data.get("results", {}).get(result_key, {}).get("benchmarks")
    if not isinstance(value, dict):
        raise ValueError("artifact does not contain the requested benchmarks")
    return value


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    """Two-sided exact McNemar p-value using the binomial null."""
    if left_only < 0 or right_only < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail_count = 1
    term = 1
    for index in range(1, min(left_only, right_only) + 1):
        term = term * (discordant - index + 1) // index
        tail_count += term
    tail = tail_count / (2**discordant)
    return min(1.0, 2.0 * tail)


def _correctness(result: dict[str, Any]) -> dict[str, bool]:
    details = result.get("details")
    if not isinstance(details, list):
        raise ValueError("benchmark result lacks per-example details")
    correctness: dict[str, bool] = {}
    for item in details:
        if not isinstance(item, dict):
            raise ValueError("benchmark detail is not an object")
        sample_id = item.get("sample_id")
        correct = item.get("correct")
        if not isinstance(sample_id, str) or not isinstance(correct, bool):
            raise ValueError("benchmark detail lacks sample_id/correct")
        if sample_id in correctness:
            raise ValueError(f"duplicate sample_id {sample_id!r}")
        correctness[sample_id] = correct
    if not correctness:
        raise ValueError("benchmark result has no per-example details")
    return correctness


def compare_benchmark(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    """Compare matched predictions after checking dataset and sample identity."""
    left_dataset = left.get("dataset")
    right_dataset = right.get("dataset")
    identity_fields = (
        "repository",
        "revision",
        "config",
        "split",
        "source_rows",
        "fingerprint",
        "evaluated_rows",
    )
    if not isinstance(left_dataset, dict) or not isinstance(right_dataset, dict):
        raise ValueError("both benchmark results require dataset provenance")
    if any(left_dataset.get(key) != right_dataset.get(key) for key in identity_fields):
        raise ValueError("dataset provenance differs between artifacts")

    left_correct = _correctness(left)
    right_correct = _correctness(right)
    if left_correct.keys() != right_correct.keys():
        raise ValueError("sample IDs differ between artifacts")
    for name, result, correctness in (
        ("left", left, left_correct),
        ("right", right, right_correct),
    ):
        if (
            int(result.get("total", -1)) != len(correctness)
            or int(result.get("evaluated", -1)) != len(correctness)
            or int(result.get("failed", result.get("skipped", -1))) != 0
            or int(result.get("skipped", result.get("failed", -1))) != 0
            or int(result["dataset"].get("evaluated_rows", -1))
            != len(correctness)
        ):
            raise ValueError(f"{name} benchmark is incomplete")
        expected_score = sum(correctness.values()) / len(correctness)
        if abs(float(result.get("score", -1.0)) - expected_score) > 1e-12:
            raise ValueError(f"{name} benchmark score does not match details")

    left_only = sum(
        left_correct[key] and not right_correct[key] for key in left_correct
    )
    right_only = sum(
        right_correct[key] and not left_correct[key] for key in left_correct
    )
    both_correct = sum(
        left_correct[key] and right_correct[key] for key in left_correct
    )
    both_wrong = len(left_correct) - left_only - right_only - both_correct
    left_accuracy = (both_correct + left_only) / len(left_correct)
    right_accuracy = (both_correct + right_only) / len(left_correct)
    return {
        "total": len(left_correct),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "left_accuracy": left_accuracy,
        "right_accuracy": right_accuracy,
        "delta_percentage_points": (right_accuracy - left_accuracy) * 100.0,
        "mcnemar_exact_p": exact_mcnemar_p(left_only, right_only),
    }


def _holm_adjust(results: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(
        results,
        key=lambda benchmark: results[benchmark]["mcnemar_exact_p"],
    )
    running = 0.0
    count = len(ordered)
    for rank, benchmark in enumerate(ordered):
        raw = results[benchmark]["mcnemar_exact_p"]
        running = max(running, min(1.0, (count - rank) * raw))
        results[benchmark]["holm_adjusted_p"] = running
        results[benchmark]["significant_at_0_05"] = running < 0.05


def compare_artifacts(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_result_key: str | None = None,
    right_result_key: str | None = None,
) -> dict[str, Any]:
    left_benchmarks = _benchmarks(left, left_result_key)
    right_benchmarks = _benchmarks(right, right_result_key)
    results = {}
    for benchmark in BENCHMARKS:
        if benchmark not in left_benchmarks or benchmark not in right_benchmarks:
            raise ValueError(f"missing benchmark {benchmark!r}")
        results[benchmark] = compare_benchmark(
            left_benchmarks[benchmark],
            right_benchmarks[benchmark],
        )
    _holm_adjust(results)
    return results


def _relative_source(path: Path) -> tuple[Path, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("quality sources must be stored inside the repository")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved, resolved.relative_to(ROOT).as_posix()


def _validate_quality_source(
    data: dict[str, Any],
    *,
    paper_model: str,
    paper_method: str,
) -> None:
    expected_type = (
        "dynaexq_quality"
        if paper_method == "dynaexq"
        else "quality_evaluation"
    )
    git = data.get("environment", {}).get("git", {})
    if (
        int(data.get("schema_version", 0)) < 2
        or data.get("artifact_type") != expected_type
        or not data.get("created_at")
        or data.get("paper_model") != paper_model
        or data.get("paper_method") != paper_method
        or data.get("evaluation_protocol", {}).get("name") != "tc_main_v2"
        or data.get("seed") != PAPER_PROTOCOL["seed"]
        or not git.get("commit")
        or git.get("dirty") is not False
        or int(data.get("environment", {}).get("process_max_rss_bytes", 0))
        <= 0
    ):
        raise ValueError(
            f"invalid {paper_model}/{paper_method} quality provenance"
        )
    checkpoint = data.get("checkpoint", {})
    if checkpoint.get("local") is True:
        if not checkpoint.get("weight_hashes_included"):
            raise ValueError("local quality checkpoint lacks weight hashes")
    elif not checkpoint.get("revision"):
        raise ValueError("remote quality checkpoint is not pinned")


def build_significance_artifact(
    left_path: Path,
    right_path: Path,
    *,
    paper_model: str,
) -> dict[str, Any]:
    """Build a paper-grade paired comparison from two immutable sources."""
    if paper_model not in QUALITY_COMPARISONS:
        raise ValueError(f"unknown paper model {paper_model!r}")
    left_resolved, left_relative = _relative_source(left_path)
    right_resolved, right_relative = _relative_source(right_path)
    if left_resolved == right_resolved:
        raise ValueError("paired comparison requires two different artifacts")
    left = json.loads(left_resolved.read_text(encoding="utf-8"))
    right = json.loads(right_resolved.read_text(encoding="utf-8"))
    left_method, right_method = QUALITY_COMPARISONS[paper_model]
    _validate_quality_source(
        left,
        paper_model=paper_model,
        paper_method=left_method,
    )
    _validate_quality_source(
        right,
        paper_model=paper_model,
        paper_method=right_method,
    )
    results = compare_artifacts(left, right)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "quality_significance",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_model": paper_model,
        "seed": PAPER_PROTOCOL["seed"],
        "evaluation_protocol": {
            "name": "tc_paired_quality_v1",
            "test": "paired_exact_mcnemar_two_sided",
            "multiple_testing": "holm",
            "family": list(BENCHMARKS),
            "alpha": 0.05,
        },
        "comparison": {
            "left_paper_method": left_method,
            "right_paper_method": right_method,
        },
        "sources": {
            "left": {
                "path": left_relative,
                "sha256": _sha256(left_resolved),
                "checkpoint": left["checkpoint"],
            },
            "right": {
                "path": right_relative,
                "sha256": _sha256(right_resolved),
                "checkpoint": right["checkpoint"],
            },
        },
        "benchmarks": results,
        "environment": environment_metadata(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-model",
        required=True,
        choices=tuple(QUALITY_COMPARISONS),
    )
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        output = build_significance_artifact(
            args.left,
            args.right,
            paper_model=args.paper_model,
        )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paper_model": args.paper_model,
                "benchmarks": len(output["benchmarks"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
