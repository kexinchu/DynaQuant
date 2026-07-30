from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import compare_quality_artifacts as comparison_module
from scripts.compare_quality_artifacts import (
    BENCHMARKS,
    build_significance_artifact,
    compare_artifacts,
    compare_benchmark,
    exact_mcnemar_p,
)


def _result(correctness):
    dataset = {
        "repository": "test/data",
        "revision": "sha",
        "config": None,
        "split": "test",
        "source_rows": len(correctness),
        "fingerprint": "fingerprint",
        "evaluated_rows": len(correctness),
    }
    return {
        "dataset": dataset,
        "total": len(correctness),
        "evaluated": len(correctness),
        "failed": 0,
        "skipped": 0,
        "score": sum(correctness) / len(correctness),
        "details": [
            {"sample_id": f"sample/{index}", "correct": correct}
            for index, correct in enumerate(correctness)
        ],
    }


def test_exact_mcnemar_and_paired_delta():
    comparison = compare_benchmark(
        _result([True, False, False, False]),
        _result([False, True, True, True]),
    )
    assert comparison["left_only_correct"] == 1
    assert comparison["right_only_correct"] == 3
    assert comparison["delta_percentage_points"] == 50.0
    assert comparison["mcnemar_exact_p"] == pytest.approx(0.625)
    assert exact_mcnemar_p(0, 0) == 1.0


def test_artifact_comparison_adds_holm_adjustment():
    left = {"benchmarks": {name: _result([True, False]) for name in BENCHMARKS}}
    right = {"benchmarks": {name: _result([True, True]) for name in BENCHMARKS}}
    results = compare_artifacts(left, right)
    assert set(results) == set(BENCHMARKS)
    assert all(0.0 <= item["holm_adjusted_p"] <= 1.0 for item in results.values())


def test_comparison_rejects_different_sample_identity():
    left = _result([True])
    right = _result([False])
    right["details"][0]["sample_id"] = "different"
    with pytest.raises(ValueError, match="sample IDs differ"):
        compare_benchmark(left, right)


def test_comparison_rejects_different_dataset_revision():
    left = _result([True])
    right = _result([False])
    right["dataset"]["revision"] = "other"
    with pytest.raises(ValueError, match="dataset provenance differs"):
        compare_benchmark(left, right)


def _quality_artifact(paper_method, correctness):
    artifact_type = (
        "dynaexq_quality"
        if paper_method == "dynaexq"
        else "quality_evaluation"
    )
    return {
        "schema_version": 2,
        "artifact_type": artifact_type,
        "created_at": "2026-01-01T00:00:00+00:00",
        "paper_model": "qwen30b",
        "paper_method": paper_method,
        "checkpoint": {
            "local": False,
            "revision": f"{paper_method}-checkpoint",
        },
        "seed": 42,
        "evaluation_protocol": {"name": "tc_main_v2"},
        "environment": {
            "git": {"commit": f"{paper_method}-code", "dirty": False},
            "process_max_rss_bytes": 1024,
        },
        "benchmarks": {
            name: _result(correctness)
            for name in BENCHMARKS
        },
    }


def test_builder_binds_clean_sources_and_canonical_methods(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(comparison_module, "ROOT", tmp_path)
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(
        json.dumps(_quality_artifact("static_int4", [True, False])),
        encoding="utf-8",
    )
    right_path.write_text(
        json.dumps(_quality_artifact("dynaexq", [True, True])),
        encoding="utf-8",
    )
    artifact = build_significance_artifact(
        left_path,
        right_path,
        paper_model="qwen30b",
    )
    assert artifact["artifact_type"] == "quality_significance"
    assert artifact["comparison"] == {
        "left_paper_method": "static_int4",
        "right_paper_method": "dynaexq",
    }
    assert artifact["sources"]["left"]["path"] == "left.json"
    assert artifact["benchmarks"]["mmlu_pro"][
        "delta_percentage_points"
    ] == 50.0

    dirty = _quality_artifact("static_int4", [True, False])
    dirty["environment"]["git"]["dirty"] = True
    left_path.write_text(json.dumps(dirty), encoding="utf-8")
    with pytest.raises(ValueError, match="quality provenance"):
        build_significance_artifact(
            left_path,
            right_path,
            paper_model="qwen30b",
        )


def test_comparison_script_is_directly_executable():
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "compare_quality_artifacts.py"),
            "--help",
        ],
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
