#!/usr/bin/env python3
"""Validate common provenance and identity of a raw mechanism artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARTIFACT_TYPES = {
    "activation": "activation_density",
    "routing_trace": "routing_active_set_trace",
    "routing_hotset": "routing_hotset_bundle",
    "perplexity_point": "dynaexq_perplexity_point",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--kind", choices=tuple(ARTIFACT_TYPES), required=True)
    parser.add_argument(
        "--paper-model",
        choices=("qwen30b", "qwen80b", "phi35"),
        required=True,
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--ratio", type=int)
    args = parser.parse_args()

    data = json.loads(args.artifact.read_text(encoding="utf-8"))
    problems: list[str] = []
    if int(data.get("schema_version", 0)) < 2:
        problems.append("legacy schema")
    if data.get("artifact_type") != ARTIFACT_TYPES[args.kind]:
        problems.append("artifact type mismatch")
    if data.get("paper_model") != args.paper_model:
        problems.append("paper model mismatch")
    if data.get("seed") != 42:
        problems.append("seed mismatch")
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

    if args.kind == "activation":
        protocol = data.get("protocol", {})
        if (
            protocol.get("name") != "tc_activation_density_v1"
            or protocol.get("batch_sizes") != [1, 2, 4, 8, 16, 32]
            or protocol.get("repeats") != 5
            or set(data.get("stages", {})) != {"prefill", "decode"}
        ):
            problems.append("activation protocol mismatch")
    elif args.kind == "routing_trace":
        protocol = data.get("protocol", {})
        if (
            protocol.get("name") != "tc_routing_active_set_v1"
            or protocol.get("warmup_trials") != 2
            or protocol.get("measured_trials") != 10
            or len(data.get("points", [])) != 17
        ):
            problems.append("routing trace protocol mismatch")
    elif args.kind == "routing_hotset":
        protocol = data.get("profile_protocol", {})
        if (
            protocol.get("name") != "tc_routing_hotset_v1"
            or set(data.get("workloads", {}))
            != {"wikitext", "gsm8k", "humaneval"}
        ):
            problems.append("routing hotset protocol mismatch")
    elif args.kind == "perplexity_point":
        if args.ratio is None:
            parser.error("--ratio is required for a perplexity point")
        result = data.get("benchmarks", {}).get("wikitext")
        if (
            int(data.get("low_ratio_pct", -1)) != args.ratio
            or data.get("selection_policy") != "calibrated_coldest_prefix"
            or not isinstance(result, dict)
            or int(result.get("total_tokens", 0)) <= 0
        ):
            problems.append("perplexity point mismatch")

    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
