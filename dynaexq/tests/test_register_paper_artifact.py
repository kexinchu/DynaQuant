from __future__ import annotations

import json

import pytest

from scripts.register_paper_artifact import build_manifest, write_manifest_atomic


def _artifact(path):
    path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")


def test_registers_relative_artifact_with_hash_and_command(tmp_path):
    artifact = tmp_path / "results" / "paper" / "perf.json"
    artifact.parent.mkdir(parents=True)
    _artifact(artifact)
    manifest_path = artifact.parent / "manifest.json"

    manifest = build_manifest(
        manifest_path=manifest_path,
        artifact_path=artifact,
        group="performance",
        claim_id="performance:qwen30b:static_ptq:bs32",
        command="python -m benchmark --seed 42",
        root=tmp_path,
    )
    record = manifest["groups"]["performance"][0]
    assert record["claim_id"] == "performance:qwen30b:static_ptq:bs32"
    assert record["path"] == "results/paper/perf.json"
    assert len(record["sha256"]) == 64
    assert record["command"] == "python -m benchmark --seed 42"

    write_manifest_atomic(manifest_path, manifest)
    assert json.loads(manifest_path.read_text()) == manifest


def test_duplicate_registration_requires_replace(tmp_path):
    artifact = tmp_path / "artifact.json"
    _artifact(artifact)
    manifest_path = tmp_path / "manifest.json"
    first = build_manifest(
        manifest_path=manifest_path,
        artifact_path=artifact,
        group="ablation",
        claim_id="ablation:qwen30b:full",
        command="first",
        root=tmp_path,
    )
    write_manifest_atomic(manifest_path, first)

    with pytest.raises(ValueError, match="already registered"):
        build_manifest(
            manifest_path=manifest_path,
            artifact_path=artifact,
            group="ablation",
            claim_id="ablation:qwen30b:full",
            command="second",
            root=tmp_path,
        )

    replaced = build_manifest(
        manifest_path=manifest_path,
        artifact_path=artifact,
        group="ablation",
        claim_id="ablation:qwen30b:full",
        command="second",
        root=tmp_path,
        replace=True,
    )
    assert replaced["groups"]["ablation"][0]["command"] == "second"


def test_rejects_legacy_or_external_artifact(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text('{"schema_version": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        build_manifest(
            manifest_path=tmp_path / "manifest.json",
            artifact_path=legacy,
            group="runtime_overhead",
            claim_id="runtime_overhead:qwen30b",
            command="run",
            root=tmp_path,
        )

    external = tmp_path.parent / "external-artifact.json"
    external.write_text('{"schema_version": 2}', encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="inside repository"):
            build_manifest(
                manifest_path=tmp_path / "manifest.json",
                artifact_path=external,
                group="runtime_overhead",
                claim_id="runtime_overhead:qwen30b",
                command="run",
                root=tmp_path,
            )
    finally:
        external.unlink()


def test_claim_prefix_must_match_group(tmp_path):
    artifact = tmp_path / "artifact.json"
    _artifact(artifact)
    with pytest.raises(ValueError, match="must start"):
        build_manifest(
            manifest_path=tmp_path / "manifest.json",
            artifact_path=artifact,
            group="performance",
            claim_id="ablation:qwen30b:full",
            command="run",
            root=tmp_path,
        )
