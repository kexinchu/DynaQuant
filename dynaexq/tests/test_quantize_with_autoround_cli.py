from __future__ import annotations

import json

import pytest

from scripts.quantize_with_autoround import (
    _load_calibration_prompts,
    _load_source_manifest,
    _validate_calibration_token_coverage,
    quantize_model,
)


def test_autoround_calibration_loader_rejects_test_split(tmp_path):
    path = tmp_path / "calibration.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "split": "test",
                "prompt": "not allowed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="test-split"):
        _load_calibration_prompts(str(path), nsamples=1)


def test_autoround_calibration_loader_is_stable_and_hashed(tmp_path):
    path = tmp_path / "calibration.jsonl"
    records = [
        {
            "id": f"sample-{index}",
            "split": "train",
            "prompt": f"prompt {index}",
        }
        for index in range(3)
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    prompts, digest = _load_calibration_prompts(str(path), nsamples=2)
    assert prompts == ["prompt 0", "prompt 1"]
    assert digest is not None and len(digest) == 64


def test_autoround_refuses_existing_output_before_loading_model(tmp_path):
    output = tmp_path / "already-exists"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        quantize_model(
            str(tmp_path / "source"),
            str(output),
        )


def test_autoround_rejects_invalid_group_size_before_loading_model(tmp_path):
    with pytest.raises(ValueError, match="group_size"):
        quantize_model(
            str(tmp_path / "source"),
            str(tmp_path / "output"),
            group_size=0,
        )


def test_source_manifest_must_match_model_path(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_type": "model_snapshot_manifest",
                "local_path": str(tmp_path / "different"),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        _load_source_manifest(str(manifest), model_path=source.resolve())


class _FakeTokenizer:
    def __init__(self, lengths):
        self.lengths = lengths

    def __call__(self, prompts, **kwargs):
        assert kwargs["truncation"] is False
        return {"length": self.lengths[: len(prompts)]}


def test_calibration_token_coverage_rejects_short_prompts():
    with pytest.raises(ValueError, match="shorter than seqlen"):
        _validate_calibration_token_coverage(
            ["long", "short"],
            _FakeTokenizer([2048, 1024]),
            seqlen=2048,
        )


def test_calibration_token_coverage_records_full_batch():
    result = _validate_calibration_token_coverage(
        ["a", "b"],
        _FakeTokenizer([2200, 2400]),
        seqlen=2048,
    )
    assert result == {
        "validated_prompt_count": 2,
        "minimum_tokens_before_truncation": 2200,
        "maximum_tokens_before_truncation": 2400,
        "required_tokens": 2048,
    }
