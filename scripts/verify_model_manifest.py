#!/usr/bin/env python3
"""Verify a local model snapshot against a committed DynaExQ manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.build_model_manifest import validate_local_snapshot


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(
    manifest_path: Path,
    *,
    model_dir: Path | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid model manifest") from error
    if manifest.get("artifact_type") not in {
        "model_snapshot_manifest",
        "quantized_model_manifest",
    }:
        raise ValueError("unsupported model manifest artifact type")
    declared_path = Path(str(manifest.get("local_path", ""))).resolve()
    snapshot = (
        declared_path
        if model_dir is None
        else model_dir.expanduser().resolve()
    )
    config, actual_files, structure = validate_local_snapshot(snapshot)
    actual = {
        path.relative_to(snapshot).as_posix(): path for path in actual_files
    }
    expected_records = manifest.get("files")
    if not isinstance(expected_records, list) or not expected_records:
        raise ValueError("model manifest contains no file records")
    expected: dict[str, dict[str, Any]] = {}
    for record in expected_records:
        if not isinstance(record, dict):
            raise ValueError("model manifest has an invalid file record")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative or relative in expected:
            raise ValueError("model manifest has an invalid file path")
        expected[relative] = record
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    if missing or unexpected:
        raise ValueError(
            "snapshot file set differs from manifest: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )

    verified_bytes = 0
    for relative, record in expected.items():
        path = actual[relative]
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError(f"invalid manifest record for {relative}")
        if path.stat().st_size != size:
            raise ValueError(f"size mismatch for {relative}")
        if _sha256(path) != digest.lower():
            raise ValueError(f"SHA-256 mismatch for {relative}")
        verified_bytes += size

    for key in (
        "tensor_count",
        "weight_shard_count",
        "indexed_tensor_bytes",
        "declared_hidden_layer_count",
        "verified_hidden_layer_count",
        "verified_safetensors_tensor_count",
        "verified_tensor_bytes",
    ):
        if key in manifest and structure.get(key) != manifest[key]:
            raise ValueError(f"structural manifest mismatch for {key}")
    if manifest.get("model_type") != config.get("model_type"):
        raise ValueError("model_type differs from manifest")
    if manifest.get("architectures") != config.get("architectures"):
        raise ValueError("architectures differ from manifest")
    if manifest.get("snapshot_size_bytes") != verified_bytes:
        raise ValueError("snapshot byte total differs from manifest")
    return {
        "artifact_type": "model_manifest_verification",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "declared_local_path": str(declared_path),
        "verified_local_path": str(snapshot),
        "relocated": snapshot != declared_path,
        "file_count": len(expected),
        "verified_bytes": verified_bytes,
        "model_type": config.get("model_type"),
        **structure,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Optional explicit path for a relocated byte-identical snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = verify_manifest(args.manifest, model_dir=args.model_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
