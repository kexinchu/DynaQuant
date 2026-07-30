#!/usr/bin/env python3
"""Derive a mixed-W2 AutoRound checkpoint from a mixed-W4 parent.

The conversion is intentionally calibration-free.  It reconstructs each
root-bit W4 matrix from the parent's AutoGPTQ tensors, applies deterministic
group-wise symmetric W2 RTN, and repacks the result in the same AutoGPTQ
layout.  Modules listed in ``extra_config`` (for example W8 dense layers and
FP16 gates) are copied byte-for-byte at the tensor level.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.build_model_manifest import validate_local_snapshot


ALGORITHM = "dynaexq_autogptq_w4_to_w2_rtn_integer_domain_v2"
IGNORED_PROVIDER_FILES = {".msc", ".mv"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_parent_manifest(path: Path, parent: Path) -> tuple[dict[str, Any], str]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid parent manifest") from error
    if manifest.get("artifact_type") not in {
        "model_snapshot_manifest",
        "quantized_model_manifest",
    }:
        raise ValueError("parent manifest has the wrong artifact type")
    if Path(str(manifest.get("local_path", ""))).resolve() != parent:
        raise ValueError("parent manifest local_path does not match --parent")
    return manifest, _sha256(path)


def _unpack_axis0(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack AutoGPTQ qweight to unsigned values shaped [in, out]."""
    pack_factor = 32 // bits
    shifts = torch.arange(pack_factor, dtype=torch.int64) * bits
    values = (
        packed.to(torch.int64).unsqueeze(1)
        >> shifts.view(1, pack_factor, 1)
    ) & ((1 << bits) - 1)
    return values.reshape(packed.shape[0] * pack_factor, packed.shape[1])


def _unpack_axis1(
    packed: torch.Tensor,
    bits: int,
    *,
    out_features: int,
) -> torch.Tensor:
    """Unpack AutoGPTQ qzeros to stored values shaped [groups, out]."""
    pack_factor = 32 // bits
    shifts = torch.arange(pack_factor, dtype=torch.int64) * bits
    values = (
        packed.to(torch.int64).unsqueeze(-1)
        >> shifts.view(1, 1, pack_factor)
    ) & ((1 << bits) - 1)
    return values.reshape(packed.shape[0], -1)[:, :out_features]


def _pack_axis0(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned values shaped [in, out] as AutoGPTQ qweight."""
    pack_factor = 32 // bits
    if values.shape[0] % pack_factor:
        raise ValueError("input features are not divisible by the pack factor")
    grouped = values.to(torch.int64).reshape(
        values.shape[0] // pack_factor,
        pack_factor,
        values.shape[1],
    )
    shifts = torch.arange(pack_factor, dtype=torch.int64) * bits
    packed = torch.sum(grouped << shifts.view(1, pack_factor, 1), dim=1)
    return packed.to(torch.int32)


def _pack_axis1(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack stored zero points shaped [groups, out] as AutoGPTQ qzeros."""
    pack_factor = 32 // bits
    if values.shape[1] % pack_factor:
        raise ValueError("output features are not divisible by the pack factor")
    grouped = values.to(torch.int64).reshape(
        values.shape[0],
        values.shape[1] // pack_factor,
        pack_factor,
    )
    shifts = torch.arange(pack_factor, dtype=torch.int64) * bits
    packed = torch.sum(grouped << shifts.view(1, 1, pack_factor), dim=2)
    return packed.to(torch.int32)


def requantize_autogptq_tensor(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    source_bits: int = 4,
    source_group_size: int = 128,
    target_bits: int = 2,
    target_group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Requantize one AutoGPTQ matrix without materializing an FP16 model."""
    if source_bits != 4 or target_bits != 2:
        raise ValueError("this audited converter only supports W4-to-W2")
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise ValueError("AutoGPTQ qweight and qzeros must be int32")
    if scales.ndim != 2 or qweight.ndim != 2 or qzeros.ndim != 2:
        raise ValueError("invalid AutoGPTQ tensor rank")

    unsigned = _unpack_axis0(qweight, source_bits)
    in_features, out_features = unsigned.shape
    if source_group_size % target_group_size:
        raise ValueError("target groups must evenly partition source groups")
    if in_features % source_group_size or in_features % target_group_size:
        raise ValueError("input features do not satisfy source/target groups")
    if out_features % (32 // target_bits):
        raise ValueError("output features do not satisfy W2 packing")
    source_groups = in_features // source_group_size
    if tuple(scales.shape) != (source_groups, out_features):
        raise ValueError("source scale shape does not match qweight")
    if qzeros.shape[0] != source_groups:
        raise ValueError("source zero-point groups do not match qweight")

    # AutoGPTQ stores (zero_point - 1); inference adds one after unpacking.
    zero_points = _unpack_axis1(
        qzeros,
        source_bits,
        out_features=out_features,
    ) + 1
    signed_source = (
        unsigned.reshape(source_groups, source_group_size, out_features)
        - zero_points.unsqueeze(1)
    ).to(torch.int8)

    target_groups = in_features // target_group_size
    grouped_codes = signed_source.reshape(
        target_groups,
        target_group_size,
        out_features,
    )
    # For signed W2, AutoRound uses representable values [-2, -1, 0, 1]
    # and divides absmax by qmax=1.  The W4 scale is constant across each
    # 128-value source group.  Since target groups (64) evenly partition
    # source groups, q2 = round((q4*s4)/(max(abs(q4))*s4)); s4 cancels.
    # Working in the integer domain avoids a full reconstructed FP32 matrix.
    code_absmax = grouped_codes.abs().amax(dim=1)
    safe_code_absmax = code_absmax.clamp(min=1)
    source_scales_for_target = scales.to(torch.float32).repeat_interleave(
        source_group_size // target_group_size,
        dim=0,
    )
    target_scales = (
        source_scales_for_target.abs() * code_absmax.to(torch.float32)
    ).clamp(min=1e-10)
    effective_codes = torch.where(
        source_scales_for_target.unsqueeze(1) < 0,
        -grouped_codes,
        grouped_codes,
    )
    signed = torch.round(
        effective_codes.to(torch.float32)
        / safe_code_absmax.unsqueeze(1).to(torch.float32)
    ).clamp(-2, 1)
    unsigned_target = (signed.to(torch.int64) + 2).reshape(
        in_features,
        out_features,
    )
    target_qweight = _pack_axis0(unsigned_target, target_bits)

    # AutoGPTQ stores zero_point - 1, hence W2 symmetric zp=2 is stored as 1.
    stored_zeros = torch.ones(
        (target_groups, out_features),
        dtype=torch.int64,
    )
    target_qzeros = _pack_axis1(stored_zeros, target_bits)
    return target_qweight, target_qzeros, target_scales.to(torch.float16)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _copy_support_files(parent: Path, temporary: Path, shard_names: set[str]) -> None:
    for source in sorted(parent.rglob("*")):
        relative = source.relative_to(parent)
        if source.is_dir() or ".cache" in relative.parts:
            continue
        if relative.as_posix() in shard_names:
            continue
        if source.name in {
            "config.json",
            "model.safetensors.index.json",
            "quantization_provenance.json",
            *IGNORED_PROVIDER_FILES,
        }:
            continue
        destination = temporary / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def derive_checkpoint(
    parent: Path,
    output: Path,
    *,
    parent_manifest_path: Path,
    target_group_size: int = 64,
) -> dict[str, Any]:
    parent = parent.expanduser().resolve()
    output = output.expanduser().resolve()
    parent_manifest_path = parent_manifest_path.expanduser().resolve()
    if not parent.is_dir():
        raise ValueError("parent checkpoint does not exist")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if target_group_size <= 0:
        raise ValueError("target group size must be positive")
    parent_manifest, parent_manifest_sha256 = _load_parent_manifest(
        parent_manifest_path,
        parent,
    )

    config = _read_json(parent / "config.json", "parent config")
    index = _read_json(
        parent / "model.safetensors.index.json",
        "parent weight index",
    )
    qconfig = config.get("quantization_config")
    if not isinstance(qconfig, dict):
        raise ValueError("parent has no quantization_config")
    required_parent = {
        "quant_method": "auto-round",
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "packing_format": "auto_round:auto_gptq",
    }
    for key, expected in required_parent.items():
        if qconfig.get(key) != expected:
            raise ValueError(
                f"unsupported parent quantization_config[{key!r}]"
            )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("parent weight index is empty")
    shard_names = {str(value) for value in weight_map.values()}
    for tensor_name, shard_name in weight_map.items():
        if tensor_name.endswith(".qweight"):
            base = tensor_name[: -len(".qweight")]
            for suffix in (".qzeros", ".scales"):
                peer = base + suffix
                if weight_map.get(peer) != shard_name:
                    raise ValueError(
                        f"AutoGPTQ tensor trio crosses shards: {base}"
                    )

    temporary = output.with_name(f".{output.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)

    extra_config = qconfig.get("extra_config", {})
    if not isinstance(extra_config, dict):
        raise ValueError("parent extra_config must be an object")
    converted_modules = 0
    copied_quantized_modules = 0
    output_tensor_bytes = 0
    try:
        _copy_support_files(parent, temporary, shard_names)
        for shard_name in sorted(shard_names):
            source_tensors = load_file(str(parent / shard_name), device="cpu")
            output_tensors: dict[str, torch.Tensor] = {}
            consumed: set[str] = set()
            for tensor_name in sorted(source_tensors):
                if tensor_name in consumed:
                    continue
                if not tensor_name.endswith(".qweight"):
                    output_tensors[tensor_name] = source_tensors[tensor_name]
                    consumed.add(tensor_name)
                    continue
                base = tensor_name[: -len(".qweight")]
                qzeros_name = base + ".qzeros"
                scales_name = base + ".scales"
                module_bits = int(
                    extra_config.get(base, {}).get("bits", qconfig["bits"])
                )
                if module_bits == 4:
                    converted = requantize_autogptq_tensor(
                        source_tensors[tensor_name],
                        source_tensors[qzeros_name],
                        source_tensors[scales_name],
                        source_bits=4,
                        source_group_size=128,
                        target_bits=2,
                        target_group_size=target_group_size,
                    )
                    for name, tensor in zip(
                        (tensor_name, qzeros_name, scales_name),
                        converted,
                        strict=True,
                    ):
                        output_tensors[name] = tensor
                    converted_modules += 1
                else:
                    for name in (tensor_name, qzeros_name, scales_name):
                        output_tensors[name] = source_tensors[name]
                    copied_quantized_modules += 1
                consumed.update((tensor_name, qzeros_name, scales_name))
            save_file(output_tensors, str(temporary / shard_name))
            output_tensor_bytes += sum(
                tensor.numel() * tensor.element_size()
                for tensor in output_tensors.values()
            )

        target_qconfig = dict(qconfig)
        target_qconfig.update(
            {
                "bits": 2,
                "group_size": target_group_size,
                "iters": 0,
            }
        )
        config["quantization_config"] = target_qconfig
        config["dynaexq_derivation"] = {
            "algorithm": ALGORITHM,
            "parent_manifest_sha256": parent_manifest_sha256,
            "source_bits": 4,
            "source_group_size": 128,
            "target_bits": 2,
            "target_group_size": target_group_size,
        }
        (temporary / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        target_index = dict(index)
        target_metadata = dict(target_index.get("metadata", {}))
        target_metadata["total_size"] = output_tensor_bytes
        target_index["metadata"] = target_metadata
        (temporary / "model.safetensors.index.json").write_text(
            json.dumps(target_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        provenance = {
            "schema_version": "1.0",
            "artifact_type": "local_quantization_provenance",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_manifest": {
                "path": str(parent_manifest_path),
                "sha256": parent_manifest_sha256,
                "source": parent_manifest.get("source"),
            },
            "calibration": {
                "path": None,
                "sha256": None,
                "sample_count": 0,
                "reason": "deterministic_requantization_from_quantized_parent",
            },
            "parameters": {
                "algorithm": ALGORITHM,
                "source_bits": 4,
                "source_group_size": 128,
                "target_bits": 2,
                "target_group_size": target_group_size,
                "sym": True,
                "packing_format": "auto_round:auto_gptq",
                "rounding": "torch_round_nearest_even",
                "source_zero_point_convention": "stored_zp_plus_one",
                "target_stored_zero_point": 1,
                "negative_source_scale_handling": (
                    "fold_sign_into_target_codes_and_store_positive_scale"
                ),
            },
            "dependencies": {
                package: importlib.metadata.version(package)
                for package in ("torch", "safetensors", "transformers")
            },
            "output": {
                "path": str(output),
                "model_type": config.get("model_type"),
                "architectures": config.get("architectures"),
                "converted_w4_modules": converted_modules,
                "copied_override_quantized_modules": copied_quantized_modules,
                "tensor_bytes_before_provenance": output_tensor_bytes,
            },
            "limitations": [
                "W2 is requantized from reconstructed W4 values, not BF16.",
                "Quantization error can compound relative to direct BF16-to-W2 PTQ.",
                "Parent extra_config W8/FP16 overrides are retained unchanged.",
            ],
        }
        (temporary / "quantization_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_local_snapshot(temporary)
        os.replace(temporary, output)
    except BaseException:
        # Preserve partial output for forensic diagnosis; a subsequent run
        # refuses to reuse it.
        raise
    return provenance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--target-group-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    provenance = derive_checkpoint(
        args.parent,
        args.output,
        parent_manifest_path=args.parent_manifest,
        target_group_size=args.target_group_size,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "converted_w4_modules": provenance["output"][
                    "converted_w4_modules"
                ],
                "copied_override_quantized_modules": provenance["output"][
                    "copied_override_quantized_modules"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
