#!/usr/bin/env python3
"""Build a content-addressed manifest for a locally quantized checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
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


def build_quantized_manifest(
    model_dir: Path,
    *,
    allow_relocated: bool = False,
) -> dict[str, Any]:
    """Validate and hash a derived checkpoint and its parent provenance."""
    model_dir = model_dir.expanduser().resolve()
    provenance_path = model_dir / "quantization_provenance.json"
    if not provenance_path.is_file():
        raise ValueError("quantized checkpoint has no provenance")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid quantization provenance") from error
    if provenance.get("artifact_type") != "local_quantization_provenance":
        raise ValueError("quantization provenance has the wrong artifact type")
    export_path = Path(
        str(provenance.get("output", {}).get("path", ""))
    ).resolve()
    relocated = export_path != model_dir
    if relocated and not allow_relocated:
        raise ValueError("quantization provenance output path does not match")

    config, files, structure = validate_local_snapshot(model_dir)
    file_records = [
        {
            "path": path.relative_to(model_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    source_manifest = provenance.get("source_manifest", {})
    return {
        "schema_version": "1.0",
        "artifact_type": "quantized_model_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "local_path": str(model_dir),
        "export_path": str(export_path),
        "relocated_copy": relocated,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "source_manifest": {
            "path": source_manifest.get("path"),
            "sha256": source_manifest.get("sha256"),
        },
        "quantization_provenance": {
            "path": str(provenance_path),
            "sha256": _sha256(provenance_path),
        },
        **structure,
        "file_count": len(file_records),
        "snapshot_size_bytes": sum(
            int(record["size_bytes"]) for record in file_records
        ),
        "files": file_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-relocated",
        action="store_true",
        help=(
            "Accept a byte-for-byte copied checkpoint whose provenance names "
            "its original export directory."
        ),
    )
    args = parser.parse_args()
    manifest = build_quantized_manifest(
        args.model_dir,
        allow_relocated=args.allow_relocated,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
