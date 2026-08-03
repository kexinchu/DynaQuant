#!/usr/bin/env python3
"""Validate one formal static or DynaExQ quality artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def validate_required_multi_gpu_placement(
    data: dict,
    *,
    paper_model: str,
    paper_method: str,
) -> None:
    """Require auditable two-GPU placement for oversized FP16 references."""
    if paper_method != "reference_fp16" or paper_model not in {
        "qwen30b",
        "phi35",
    }:
        return
    placement = data.get("model_placement")
    if not isinstance(placement, dict):
        raise ValueError("oversized FP16 reference lacks model_placement")
    if placement.get("requested_device_map") != "auto":
        raise ValueError("oversized FP16 reference did not request device_map=auto")
    if placement.get("required_cuda_device_count") != 2:
        raise ValueError("oversized FP16 reference did not require two CUDA devices")
    if placement.get("resolved_cuda_devices") != [0, 1]:
        raise ValueError("oversized FP16 reference did not resolve to CUDA [0, 1]")
    resolved = placement.get("resolved_device_map")
    if not isinstance(resolved, dict) or not resolved:
        raise ValueError("oversized FP16 reference lacks resolved hf_device_map")
    targets = set(resolved.values())
    normalized = set()
    for target in targets:
        text = str(target)
        if text.isdigit():
            normalized.add(int(text))
        elif text.startswith("cuda:") and text[5:].isdigit():
            normalized.add(int(text[5:]))
        else:
            raise ValueError(
                "oversized FP16 reference contains non-CUDA placement "
                f"target {text!r}"
            )
    if normalized != {0, 1}:
        raise ValueError(
            "oversized FP16 reference resolved map does not use both CUDA devices"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--paper-model",
        choices=("qwen30b", "qwen80b", "phi35"),
        required=True,
    )
    parser.add_argument(
        "--paper-method",
        choices=(
            "reference_fp16",
            "static_int4",
            "static_int2",
            "dynaexq",
        ),
        required=True,
    )
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()

    root = args.formal_root.resolve()
    artifact_path = args.artifact.resolve()
    sys.path.insert(0, str(root))
    from scripts.compare_quality_artifacts import (  # noqa: PLC0415
        _validate_quality_source,
    )

    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    _validate_quality_source(
        data,
        paper_model=args.paper_model,
        paper_method=args.paper_method,
    )
    try:
        validate_required_multi_gpu_placement(
            data,
            paper_model=args.paper_model,
            paper_method=args.paper_method,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    git = data["environment"]["git"]
    if git.get("commit") != args.expected_commit:
        raise SystemExit(
            "artifact commit mismatch: "
            f"{git.get('commit')} != {args.expected_commit}"
        )
    benchmark_names = set(data.get("benchmarks", {}))
    expected = {"mmlu_pro", "gpqa", "aime25", "gsm8k", "humaneval"}
    if benchmark_names != expected:
        raise SystemExit(
            f"quality benchmark set mismatch: {sorted(benchmark_names)}"
        )
    failed = {
        name: result.get("failed")
        for name, result in data["benchmarks"].items()
        if int(result.get("failed", 0)) != 0
    }
    if failed:
        raise SystemExit(f"quality benchmark infrastructure failures: {failed}")


if __name__ == "__main__":
    main()
