#!/usr/bin/env python3
"""Validate and summarize one formal native-performance artifact."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


FORMAL_ROOT = Path("/home/kec23008/DynaQuant-formal")
EXPECTED_COMMIT = "ee5283bfacf12428b5a6fcff284ddb4eb28a9cb9"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(FORMAL_ROOT))
    from scripts.audit_paper_results import (  # noqa: PLC0415
        _validate_performance_benchmark,
        validate_dynamic_runtime,
    )

    data = json.loads(args.artifact.read_text(encoding="utf-8"))
    benchmark = data.get("benchmark", {})
    problems = _validate_performance_benchmark(str(args.artifact), benchmark)
    if data.get("paper_method") == "dynaexq":
        problems.extend(validate_dynamic_runtime(str(args.artifact), data))
    git = data.get("environment", {}).get("git", {})
    if git.get("commit") != EXPECTED_COMMIT:
        problems.append("experiment commit mismatch")
    if git.get("dirty") is not False:
        problems.append("dirty git provenance")

    samples = benchmark.get("samples", [])
    e2e = [float(sample["model_e2e_ms"]) for sample in samples]
    mean = statistics.fmean(e2e) if e2e else None
    stdev = statistics.stdev(e2e) if len(e2e) > 1 else 0.0
    transition_stats = data.get("transition_stats", {})
    summary = {
        "artifact": str(args.artifact),
        "validation_problems": problems,
        "paper_model": data.get("paper_model"),
        "paper_method": data.get("paper_method"),
        "batch_size": benchmark.get("batch_size"),
        "input_tokens": benchmark.get("input_tokens"),
        "output_tokens_per_sequence": benchmark.get(
            "output_tokens_per_sequence"
        ),
        "warmup_iterations": benchmark.get("warmup_iterations"),
        "measured_iterations": benchmark.get("measured_iterations"),
        "sample_count": len(samples),
        "metrics": benchmark.get("metrics", {}),
        "e2e_coefficient_of_variation": (
            stdev / mean if mean not in (None, 0.0) else None
        ),
        "git": git,
        "transition_totals": {
            key: transition_stats.get(key)
            for key in (
                "total_promotions",
                "total_demotions",
                "failed_transitions",
                "active_transitions",
            )
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
