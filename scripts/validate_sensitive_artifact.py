#!/usr/bin/env python3
"""Validate raw ablation, sensitivity, or runtime-overhead provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


KINDS = {
    "ablation": "dynaexq_ablation",
    "sensitivity": "dynaexq_sensitivity",
    "overhead": "dynaexq_overhead",
}
BENCHMARKS = {"mmlu_pro", "gpqa", "aime25", "gsm8k", "humaneval"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--kind", choices=tuple(KINDS), required=True)
    parser.add_argument(
        "--paper-model", choices=("qwen30b", "qwen80b"), required=True
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--ablation-config")
    parser.add_argument("--ratio", type=int)
    args = parser.parse_args()

    data = json.loads(args.artifact.read_text(encoding="utf-8"))
    problems: list[str] = []
    if int(data.get("schema_version", 0)) < 2:
        problems.append("legacy schema")
    if data.get("artifact_type") != KINDS[args.kind]:
        problems.append("artifact type mismatch")
    if data.get("paper_model") != args.paper_model or data.get("seed") != 42:
        problems.append("model or seed mismatch")
    checkpoint = data.get("checkpoint", {})
    if checkpoint.get("local") is True:
        if checkpoint.get("weight_hashes_included") is not True:
            problems.append("local checkpoint is not hashed")
    elif not checkpoint.get("revision"):
        problems.append("remote checkpoint is not pinned")
    environment = data.get("environment", {})
    git = environment.get("git", {}) if isinstance(environment, dict) else {}
    if git.get("commit") != args.expected_commit or git.get("dirty") is not False:
        problems.append("git provenance mismatch")
    if int(environment.get("process_max_rss_bytes", 0)) <= 0:
        problems.append("missing process peak RSS")
    if set(data.get("benchmarks", {})) != BENCHMARKS:
        problems.append("quality benchmark set mismatch")
    elif any(
        int(result.get("failed", 0)) != 0
        for result in data["benchmarks"].values()
    ):
        problems.append("quality benchmark infrastructure failure")
    benchmark = data.get("benchmark", {})
    protocol = data.get("evaluation_protocol", {})
    if (
        benchmark.get("batch_size") != 32
        or benchmark.get("input_tokens") != 2048
        or benchmark.get("output_tokens_per_sequence") != 256
        or benchmark.get("warmup_iterations") != 5
        or benchmark.get("measured_iterations") != 100
        or len(benchmark.get("samples", [])) != 100
        or protocol.get("name") != "tc_main_v2"
    ):
        problems.append("formal performance protocol mismatch")
    stats = data.get("wrapper_stats", {})
    transitions = data.get("transition_stats", {})
    if (
        int(stats.get("scheduler_update_count", -1))
        != len(stats.get("scheduler_update_samples_ms", []))
        or int(transitions.get("failed_transitions", -1)) != 0
    ):
        problems.append("runtime telemetry mismatch")

    if args.kind == "ablation":
        if args.ablation_config is None:
            parser.error("--ablation-config is required for ablation")
        if data.get("ablation_config") != args.ablation_config:
            problems.append("ablation configuration mismatch")
    elif args.kind == "sensitivity":
        if args.ratio is None:
            parser.error("--ratio is required for sensitivity")
        if int(data.get("hi_ratio_pct", -1)) != args.ratio:
            problems.append("sensitivity ratio mismatch")

    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
