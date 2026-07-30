#!/usr/bin/env python3
"""Build a content-addressed manifest for a local model snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HEX_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
LAYER_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.")
NUMBERED_SAFETENSORS_SHARD_RE = re.compile(
    r"^.+-(\d+)-of-(\d+)\.safetensors$"
)
SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_files(model_dir: Path) -> list[Path]:
    files: list[Path] = []
    for candidate in sorted(model_dir.rglob("*")):
        relative = candidate.relative_to(model_dir)
        if ".cache" in relative.parts:
            continue
        if candidate.name in {".msc", ".mv"}:
            # Provider-client bookkeeping is not part of the model snapshot
            # and can contain host-specific serialized metadata.
            continue
        if candidate.is_symlink():
            target = candidate.resolve()
            if not target.is_relative_to(model_dir):
                raise ValueError(
                    f"model snapshot contains an external symlink: {relative}"
                )
        if candidate.is_file():
            files.append(candidate)
    if not files:
        raise ValueError("model snapshot contains no files")
    return files


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise ValueError("truncated safetensors header length")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 0 or header_length > path.stat().st_size - 8:
                raise ValueError("invalid safetensors header length")
            header = json.loads(stream.read(header_length))
    except (OSError, json.JSONDecodeError, struct.error) as error:
        raise ValueError(f"invalid safetensors shard: {path.name}") from error
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header: {path.name}")
    header.pop("__metadata__", None)
    return header


def _validate_safetensors_index(
    model_dir: Path,
    weight_map: dict[str, Any],
    shards: list[str],
    *,
    indexed_tensor_bytes: int,
    declared_primary_shards: int | None,
) -> dict[str, int]:
    if not all(name.endswith(".safetensors") for name in shards):
        return {}
    indexed_by_shard: dict[str, set[str]] = {
        shard: set() for shard in shards
    }
    for tensor_name, shard_value in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("weight index contains an invalid tensor name")
        shard = str(shard_value)
        if shard not in indexed_by_shard:
            raise ValueError("weight index contains an invalid shard mapping")
        indexed_by_shard[shard].add(tensor_name)

    bytes_by_shard: dict[str, int] = {}
    for shard in shards:
        header = _read_safetensors_header(model_dir / shard)
        actual_names = set(header)
        expected_names = indexed_by_shard[shard]
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        if missing or extra:
            raise ValueError(
                f"safetensors index/header mismatch in {shard}: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        shard_bytes = 0
        for tensor_name, descriptor in header.items():
            if not isinstance(descriptor, dict):
                raise ValueError(
                    f"invalid tensor descriptor for {tensor_name}"
                )
            dtype = descriptor.get("dtype")
            shape = descriptor.get("shape")
            offsets = descriptor.get("data_offsets")
            if (
                dtype not in SAFETENSORS_DTYPE_BYTES
                or not isinstance(shape, list)
                or not all(
                    isinstance(size, int) and not isinstance(size, bool)
                    and size >= 0
                    for size in shape
                )
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(
                    isinstance(offset, int) and not isinstance(offset, bool)
                    and offset >= 0
                    for offset in offsets
                )
                or offsets[1] < offsets[0]
            ):
                raise ValueError(
                    f"invalid tensor descriptor for {tensor_name}"
                )
            expected_bytes = (
                math.prod(shape) * SAFETENSORS_DTYPE_BYTES[dtype]
            )
            if offsets[1] - offsets[0] != expected_bytes:
                raise ValueError(
                    f"invalid tensor byte range for {tensor_name}"
                )
            shard_bytes += expected_bytes
        bytes_by_shard[shard] = shard_bytes

    verified_bytes = sum(bytes_by_shard.values())
    primary_shards = list(shards)
    if declared_primary_shards is not None:
        numbered_parts: dict[int, str] = {}
        for shard in shards:
            match = NUMBERED_SAFETENSORS_SHARD_RE.match(shard)
            if match is None or int(match.group(2)) != declared_primary_shards:
                continue
            part = int(match.group(1))
            if part in numbered_parts:
                raise ValueError("weight index has duplicate numbered shards")
            numbered_parts[part] = shard
        expected_parts = set(range(1, declared_primary_shards + 1))
        if set(numbered_parts) != expected_parts:
            raise ValueError(
                "weight index does not contain every declared primary shard"
            )
        primary_shards = [
            numbered_parts[part] for part in sorted(numbered_parts)
        ]

    verified_indexed_bytes = sum(
        bytes_by_shard[shard] for shard in primary_shards
    )
    if verified_indexed_bytes != indexed_tensor_bytes:
        raise ValueError(
            "weight-index total_size does not match safetensors headers: "
            f"{indexed_tensor_bytes} != {verified_indexed_bytes}"
        )
    return {
        "verified_safetensors_tensor_count": len(weight_map),
        "verified_tensor_bytes": verified_bytes,
        "verified_indexed_tensor_bytes": verified_indexed_bytes,
        "verified_primary_shard_count": len(primary_shards),
        "verified_auxiliary_tensor_bytes": (
            verified_bytes - verified_indexed_bytes
        ),
        "verified_auxiliary_shard_count": len(shards) - len(primary_shards),
    }


def _validate_model_structure(
    config: dict[str, Any],
    weight_map: dict[str, Any],
) -> dict[str, Any]:
    layer_count = config.get("num_hidden_layers", config.get("n_layer"))
    if layer_count is None:
        return {}
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count <= 0
    ):
        raise ValueError("config has an invalid hidden-layer count")
    layer_indices = {
        int(match.group(1))
        for key in weight_map
        if (match := LAYER_KEY_RE.match(str(key))) is not None
    }
    if not layer_indices:
        raise ValueError(
            "weight index has no model.layers tensors despite a declared "
            "hidden-layer count"
        )
    expected = set(range(layer_count))
    if layer_indices != expected:
        missing = sorted(expected - layer_indices)
        unexpected = sorted(layer_indices - expected)
        raise ValueError(
            "weight index does not cover every declared hidden layer: "
            f"missing={missing}, unexpected={unexpected}"
        )

    names = {str(key) for key in weight_map}
    required = {"model.embed_tokens.weight", "model.norm.weight"}
    if not bool(config.get("tie_word_embeddings", False)):
        required.add("lm_head.weight")
    missing_required = sorted(required - names)
    if missing_required:
        raise ValueError(
            "weight index is missing required terminal tensors: "
            f"{missing_required}"
        )
    return {
        "declared_hidden_layer_count": layer_count,
        "verified_hidden_layer_count": len(layer_indices),
        "required_terminal_tensors_verified": sorted(required),
    }


def _validate_weight_index(
    model_dir: Path,
    files: list[Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    indexes = [
        path
        for path in files
        if path.name
        in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    ]
    if len(indexes) != 1:
        raise ValueError("expected exactly one supported weight index")
    try:
        index = json.loads(indexes[0].read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("invalid model weight index") from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model weight index has no tensors")
    if not all(isinstance(value, str) and value for value in weight_map.values()):
        raise ValueError("weight index contains an invalid shard name")
    shards = sorted({str(value) for value in weight_map.values()})
    file_relatives = {
        path.relative_to(model_dir).as_posix() for path in files
    }
    missing = [name for name in shards if name not in file_relatives]
    if missing:
        raise ValueError(f"weight index references missing shards: {missing}")
    total_size = index.get("metadata", {}).get("total_size")
    if (
        isinstance(total_size, bool)
        or not isinstance(total_size, int)
        or total_size <= 0
    ):
        raise ValueError("weight index has no valid total tensor size")
    declared_primary_shards = index.get("metadata", {}).get("total_shards")
    if declared_primary_shards is not None and (
        isinstance(declared_primary_shards, bool)
        or not isinstance(declared_primary_shards, int)
        or declared_primary_shards <= 0
    ):
        raise ValueError("weight index has an invalid primary shard count")
    return {
        "index_file": indexes[0].relative_to(model_dir).as_posix(),
        "tensor_count": len(weight_map),
        "weight_shard_count": len(shards),
        "indexed_tensor_bytes": total_size,
        **_validate_model_structure(config, weight_map),
        **_validate_safetensors_index(
            model_dir,
            weight_map,
            shards,
            indexed_tensor_bytes=total_size,
            declared_primary_shards=declared_primary_shards,
        ),
    }


def validate_local_snapshot(
    model_dir: Path,
) -> tuple[dict[str, Any], list[Path], dict[str, Any]]:
    """Validate local config, shard index, headers, and declared structure."""
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise ValueError(f"model directory does not exist: {model_dir}")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ValueError("model snapshot has no config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid config.json") from error
    if not isinstance(config, dict):
        raise ValueError("config.json must contain an object")
    files = _model_files(model_dir)
    return config, files, _validate_weight_index(model_dir, files, config)


def build_manifest(
    model_dir: Path,
    *,
    provider: str,
    repository: str,
    revision: str,
    revision_kind: str = "provider_commit",
    requested_revision: str | None = None,
) -> dict[str, Any]:
    model_dir = model_dir.expanduser().resolve()
    revision = revision.strip().lower()
    if not HEX_REVISION_RE.fullmatch(revision):
        raise ValueError("revision must be a 40--64 character hexadecimal ID")
    config, files, weight_index = validate_local_snapshot(model_dir)
    file_records = [
        {
            "path": path.relative_to(model_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    source = {
        "provider": provider,
        "repository": repository,
        "revision": revision,
        "revision_kind": revision_kind,
    }
    if requested_revision is not None:
        source["requested_revision"] = requested_revision
    return {
        "schema_version": "1.0",
        "artifact_type": "model_snapshot_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "local_path": str(model_dir),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        **weight_index,
        "file_count": len(file_records),
        "snapshot_size_bytes": sum(
            int(record["size_bytes"]) for record in file_records
        ),
        "files": file_records,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument(
        "--provider",
        required=True,
        choices=("huggingface", "modelscope"),
    )
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(ROOT):
        raise SystemExit("output must be inside the repository")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        args.model_dir,
        provider=args.provider,
        repository=args.repository,
        revision=args.revision,
    )
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
