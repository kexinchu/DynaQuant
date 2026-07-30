#!/usr/bin/env python3
"""Register one immutable paper artifact in the strict result manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results" / "paper" / "manifest.json"
GROUPS = (
    "quality_significance",
    "performance",
    "ablation",
    "runtime_overhead",
    "budget_sensitivity",
    "activation_density",
    "offload_waiting",
    "routing_hotset",
    "perplexity_curve",
    "figure_bundle",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_artifact_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"artifact must be inside repository: {resolved}") from error
    if relative == Path("results/paper/manifest.json"):
        raise ValueError("the manifest cannot register itself")
    return relative.as_posix()


def build_manifest(
    *,
    manifest_path: Path,
    artifact_path: Path,
    group: str,
    claim_id: str,
    command: str,
    root: Path = ROOT,
    replace: bool = False,
) -> dict[str, Any]:
    """Return an updated manifest after validating one artifact."""
    if group not in GROUPS:
        raise ValueError(f"unknown result group {group!r}")
    if not claim_id.startswith(f"{group}:"):
        raise ValueError(
            f"claim_id must start with {group!r} followed by ':'"
        )
    if not command.strip():
        raise ValueError("reproduction command must not be empty")
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"artifact is not valid JSON: {artifact_path}") from error
    if not isinstance(artifact, dict):
        raise ValueError("artifact root must be a JSON object")
    if int(artifact.get("schema_version", 0)) < 2:
        raise ValueError("only schema_version >= 2 artifacts can be registered")

    relative = _relative_artifact_path(artifact_path, root)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", 0)) != 2:
            raise ValueError("existing manifest has an unsupported schema")
    else:
        manifest = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "groups": {name: [] for name in GROUPS},
        }

    groups = manifest.setdefault("groups", {})
    for name in GROUPS:
        groups.setdefault(name, [])
    existing = [
        index
        for index, record in enumerate(groups[group])
        if record.get("claim_id") == claim_id
    ]
    if existing and not replace:
        raise ValueError(
            f"{claim_id} is already registered; use --replace"
        )

    record = {
        "claim_id": claim_id,
        "path": relative,
        "sha256": sha256(artifact_path),
        "command": command.strip(),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    if existing:
        groups[group][existing[0]] = record
        for index in reversed(existing[1:]):
            del groups[group][index]
    else:
        groups[group].append(record)
    groups[group].sort(key=lambda item: item["claim_id"])
    return manifest


def write_manifest_atomic(path: Path, manifest: dict[str, Any]) -> None:
    """Replace the manifest atomically without leaving a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=GROUPS)
    parser.add_argument(
        "--claim-id",
        required=True,
        help="Exact manuscript claim covered by this artifact",
    )
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--command", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(
        manifest_path=args.manifest,
        artifact_path=args.artifact,
        group=args.group,
        claim_id=args.claim_id,
        command=args.command,
        replace=args.replace,
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    write_manifest_atomic(args.manifest, manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "artifact": str(args.artifact),
                "group": args.group,
                "claim_id": args.claim_id,
                "sha256": sha256(args.artifact),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
