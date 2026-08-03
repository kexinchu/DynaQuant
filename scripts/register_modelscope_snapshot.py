#!/usr/bin/env python3
"""Register and verify a downloaded ModelScope snapshot by content set.

ModelScope repositories may expose only a moving ``master`` ref.  This tool
queries the remote file catalog, derives a stable SHA-256 over every path,
size, and content digest, then verifies the complete local snapshot against
that catalog before writing the ordinary DynaExQ model manifest.
"""

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

from scripts.build_model_manifest import ROOT, build_manifest


def normalize_catalog(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for item in files:
        if item.get("Type") != "blob":
            continue
        path = item.get("Path")
        size = item.get("Size")
        sha256 = item.get("Sha256")
        if (
            not isinstance(path, str)
            or not path
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
        ):
            raise ValueError("ModelScope returned an invalid file record")
        records.append(
            {
                "path": path,
                "size_bytes": size,
                "sha256": sha256.lower(),
                "provider_file_revision": item.get("Revision"),
                "committed_date": item.get("CommittedDate"),
            }
        )
    records.sort(key=lambda record: record["path"])
    if not records:
        raise ValueError("ModelScope returned an empty file catalog")
    paths = [record["path"] for record in records]
    if len(paths) != len(set(paths)):
        raise ValueError("ModelScope returned duplicate file paths")
    return records


def content_set_sha256(catalog: list[dict[str, Any]]) -> str:
    identity = [
        {
            "path": record["path"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for record in catalog
    ]
    payload = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_manifest_against_catalog(
    manifest: dict[str, Any],
    catalog: list[dict[str, Any]],
) -> None:
    local = {
        record["path"]: (record["size_bytes"], record["sha256"])
        for record in manifest["files"]
    }
    remote = {
        record["path"]: (record["size_bytes"], record["sha256"])
        for record in catalog
    }
    missing = sorted(remote.keys() - local.keys())
    unexpected = sorted(local.keys() - remote.keys())
    mismatched = sorted(
        path
        for path in remote.keys() & local.keys()
        if remote[path] != local[path]
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "local snapshot does not match ModelScope catalog: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}, "
            f"mismatched={mismatched[:3]}"
        )


def register_snapshot(
    model_dir: Path,
    *,
    repository: str,
    requested_revision: str,
    output: Path,
    catalog_output: Path,
) -> dict[str, Any]:
    from modelscope.hub.api import HubApi

    raw_files = HubApi().get_model_files(
        repository,
        revision=requested_revision,
        recursive=True,
    )
    catalog = normalize_catalog(raw_files)
    content_revision = content_set_sha256(catalog)
    manifest = build_manifest(
        model_dir,
        provider="modelscope",
        repository=repository,
        revision=content_revision,
        revision_kind="content_set_sha256",
        requested_revision=requested_revision,
    )
    verify_manifest_against_catalog(manifest, catalog)
    catalog_artifact = {
        "schema_version": "1.0",
        "artifact_type": "modelscope_remote_file_catalog",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository,
        "requested_revision": requested_revision,
        "content_set_sha256": content_revision,
        "file_count": len(catalog),
        "files": catalog,
    }
    catalog_output.parent.mkdir(parents=True, exist_ok=True)
    catalog_output.write_text(
        json.dumps(catalog_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["source"]["catalog_path"] = str(catalog_output.resolve())
    manifest["source"]["catalog_sha256"] = hashlib.sha256(
        catalog_output.read_bytes()
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--requested-revision", default="master")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--catalog-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    catalog_output = args.catalog_output.expanduser().resolve()
    if not output.is_relative_to(ROOT) or not catalog_output.is_relative_to(ROOT):
        raise SystemExit("outputs must be inside the repository")
    if output.exists() or catalog_output.exists():
        raise SystemExit("refusing to overwrite an existing artifact")
    manifest = register_snapshot(
        args.model_dir.expanduser().resolve(),
        repository=args.repository,
        requested_revision=args.requested_revision,
        output=output,
        catalog_output=catalog_output,
    )
    print(manifest["source"]["revision"])


if __name__ == "__main__":
    main()
