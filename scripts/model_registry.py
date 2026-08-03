#!/usr/bin/env python3
"""List, resolve, verify, or restore the immutable paper model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "release" / "model_registry.json"


def load_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported model-registry schema")
    artifacts = payload.get("artifacts")
    paper_models = payload.get("paper_models")
    if not isinstance(artifacts, dict) or not isinstance(paper_models, dict):
        raise ValueError("model registry is missing artifacts or paper_models")
    return payload


def list_registry(registry: dict[str, Any]) -> None:
    for artifact_id, artifact in registry["artifacts"].items():
        print(
            f"{artifact_id}\t{artifact['precision']}\t"
            f"{artifact['repository']}@{artifact['revision']}"
        )


def resolve_method(
    registry: dict[str, Any],
    paper_model: str,
    method: str,
) -> None:
    model = registry["paper_models"].get(paper_model)
    if model is None:
        raise ValueError(f"unknown paper model: {paper_model}")
    binding = model["methods"].get(method)
    if binding is None:
        available = ", ".join(model["methods"])
        raise ValueError(f"unknown method {method!r}; choose one of: {available}")
    result = {
        "paper_model": paper_model,
        "paper_label": model["paper_label"],
        "method": method,
        "binding": binding,
    }
    artifact_id = binding.get("artifact") or binding.get("source_artifact")
    if artifact_id is not None:
        result["artifact"] = registry["artifacts"][artifact_id]
    print(json.dumps(result, indent=2, sort_keys=True))


def verify_remote(registry: dict[str, Any], artifact_ids: list[str]) -> None:
    api = HfApi()
    failures: list[str] = []
    for artifact_id in artifact_ids:
        artifact = registry["artifacts"][artifact_id]
        info = api.model_info(
            artifact["repository"],
            revision=artifact["revision"],
            files_metadata=True,
        )
        remote_safetensors_bytes = sum(
            sibling.size or 0
            for sibling in info.siblings
            if sibling.rfilename.endswith(".safetensors")
        )
        expected = artifact.get("expected_safetensors_bytes")
        status = "ok"
        if info.sha != artifact["revision"]:
            status = "revision-mismatch"
        elif expected is not None and remote_safetensors_bytes != expected:
            status = "size-mismatch"
        if status != "ok":
            failures.append(artifact_id)
        print(
            f"{artifact_id}\t{status}\t{info.sha}\t"
            f"{remote_safetensors_bytes} safetensors bytes"
        )
    if failures:
        raise RuntimeError(
            "remote model verification failed: " + ", ".join(failures)
        )


def download_artifact(
    registry: dict[str, Any],
    artifact_id: str,
    output_root: Path,
) -> None:
    artifact = registry["artifacts"].get(artifact_id)
    if artifact is None:
        raise ValueError(f"unknown artifact: {artifact_id}")
    destination = output_root / artifact["default_local_directory"]
    snapshot_download(
        repo_id=artifact["repository"],
        revision=artifact["revision"],
        local_dir=destination,
    )
    print(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List every immutable artifact")

    resolve = subparsers.add_parser(
        "resolve", help="Resolve a paper model/method to its artifact"
    )
    resolve.add_argument("paper_model")
    resolve.add_argument("method")

    verify = subparsers.add_parser(
        "verify-remote", help="Verify immutable revisions and weight sizes"
    )
    verify.add_argument("artifacts", nargs="*", help="Artifact IDs; default all")

    download = subparsers.add_parser(
        "download", help="Restore one artifact at its immutable revision"
    )
    download.add_argument("artifact")
    download.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_registry(args.registry)
    if args.command == "list":
        list_registry(registry)
    elif args.command == "resolve":
        resolve_method(registry, args.paper_model, args.method)
    elif args.command == "verify-remote":
        artifact_ids = args.artifacts or list(registry["artifacts"])
        unknown = sorted(set(artifact_ids) - set(registry["artifacts"]))
        if unknown:
            raise ValueError("unknown artifacts: " + ", ".join(unknown))
        verify_remote(registry, artifact_ids)
    elif args.command == "download":
        download_artifact(registry, args.artifact, args.output_root)


if __name__ == "__main__":
    main()
