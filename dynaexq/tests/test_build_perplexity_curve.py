from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import build_perplexity_curve as builder


def _point(ratio: int) -> dict:
    return {
        "schema_version": 2,
        "artifact_type": "dynaexq_perplexity_point",
        "paper_model": "qwen30b",
        "checkpoint": {"local": False, "revision": "immutable"},
        "config": {"model": {"name": "test"}},
        "initial_map": {
            "ranking_sha256": "ranking-sha",
            "expert_ranking": {"0": [0, 1]},
        },
        "evaluation_protocol": {"name": "tc_main_v2"},
        "selection_policy": "calibrated_coldest_prefix",
        "low_ratio_pct": ratio,
        "low_experts_per_layer": ratio,
        "low_experts_sha256": f"low-{ratio}",
        "low_experts": {"0": []},
        "wrapper_stats": {"scheduler_enabled": False},
        "benchmarks": {
            "wikitext": {
                "score": 5.0 + ratio / 100.0,
                "total_nll": 10.0,
                "total_tokens": 5,
                "windows": 1,
                "window_tokens": 2048,
                "stride_tokens": 2048,
                "window_details": [
                    {
                        "window_index": 0,
                        "begin_token": 0,
                        "end_token": 6,
                        "target_tokens": 5,
                        "mean_loss": 2.0,
                        "nll": 10.0,
                    }
                ],
                "dataset": {
                    "revision": "dataset-sha",
                    "fingerprint": "fingerprint",
                },
            }
        },
        "environment": {
            "git": {"commit": "clean-commit", "dirty": False}
        },
    }


def test_curve_builder_requires_one_clean_consistent_point_per_ratio(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    paths = []
    for ratio in builder.PERPLEXITY_LOW_RATIOS_PCT:
        path = tmp_path / f"ratio-{ratio}.json"
        path.write_text(json.dumps(_point(ratio)), encoding="utf-8")
        paths.append(path)
    artifact = builder.build_curve(paths, paper_model="qwen30b")
    assert [point["low_ratio_pct"] for point in artifact["points"]] == list(
        builder.PERPLEXITY_LOW_RATIOS_PCT
    )
    assert len(artifact["source_points"]) == 8
    assert all(len(source["sha256"]) == 64 for source in artifact["source_points"])

    inconsistent = _point(100)
    inconsistent["checkpoint"] = {"local": False, "revision": "different"}
    paths[-1].write_text(json.dumps(inconsistent), encoding="utf-8")
    with pytest.raises(ValueError, match="share checkpoint"):
        builder.build_curve(paths, paper_model="qwen30b")


def test_curve_builder_script_is_directly_executable():
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "build_perplexity_curve.py"),
            "--help",
        ],
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
