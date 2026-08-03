#!/usr/bin/env python3
"""
Quantize models using Intel AutoRound to generate Int4 versions.
Uses GPU by default when available (device_map=0); pass --device-map for multi-GPU or CPU.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _resolve_device_map(device: str, device_map: str | None):
    """Resolve device_map for AutoRound: prefer device_map; else map --device to device_map."""
    if device_map is not None:
        # Allow int as string, e.g. "0", "1", or "cuda:1", "auto"
        try:
            return int(device_map)
        except ValueError:
            return device_map.strip()
    if device is None or device.lower() == "auto":
        return 0  # GPU 0 when available (AutoRound default)
    if device.lower() == "cpu":
        return "cpu"
    # "cuda:0", "cuda:1" -> use as-is; "0", "1" -> int for single GPU
    s = device.strip()
    if s.isdigit():
        return int(s)
    return s


def _load_calibration_prompts(
    path_text: str | None,
    *,
    nsamples: int,
) -> tuple[list[str] | None, str | None]:
    if path_text is None:
        return None, None
    path = Path(path_text).expanduser().resolve()
    prompts: list[str] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid calibration JSONL at line {line_number}"
                ) from error
            split = str(record.get("split", "")).lower()
            if split == "test":
                raise ValueError(
                    "test-split prompts cannot be used for quantization "
                    "calibration"
                )
            sample_id = str(record.get("id", "")).strip()
            prompt = next(
                (
                    record.get(field)
                    for field in ("prompt", "text", "question")
                    if isinstance(record.get(field), str)
                    and record[field].strip()
                ),
                None,
            )
            if not sample_id or prompt is None:
                raise ValueError(
                    "each calibration row requires a stable id and prompt/text"
                )
            if sample_id in seen_ids:
                raise ValueError(f"duplicate calibration id: {sample_id}")
            seen_ids.add(sample_id)
            prompts.append(prompt)
    if len(prompts) < nsamples:
        raise ValueError(
            f"calibration has {len(prompts)} prompts; {nsamples} required"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return prompts[:nsamples], digest


def _load_source_manifest(
    path_text: str | None,
    *,
    model_path: Path,
) -> tuple[dict | None, str | None, str | None]:
    if path_text is None:
        return None, None, None
    manifest_path = Path(path_text).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid source model manifest") from error
    if manifest.get("artifact_type") != "model_snapshot_manifest":
        raise ValueError("source manifest has the wrong artifact type")
    manifest_model_path = Path(str(manifest.get("local_path", ""))).resolve()
    if manifest_model_path != model_path:
        raise ValueError(
            "source manifest local_path does not match --model-path"
        )
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return manifest, str(manifest_path), digest


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in (
        "auto-round",
        "transformers",
        "torch",
        "safetensors",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


def _execution_settings(
    *,
    quantizer_script_sha256: str | None = None,
) -> dict:
    import torch

    cuda_devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        cuda_devices.append(
            {
                "logical_index": index,
                "name": properties.name,
                "uuid": str(getattr(properties, "uuid", "")),
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    return {
        "torch_deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_deterministic_warn_only_enabled": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pytorch_cuda_alloc_conf": (
            os.environ.get("PYTORCH_ALLOC_CONF")
            or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        ),
        "visible_cuda_devices": cuda_devices,
        "quantizer_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        if quantizer_script_sha256 is None
        else quantizer_script_sha256,
    }


def _write_provenance(
    *,
    output_dir: Path,
    manifest: dict | None,
    manifest_path: str | None,
    manifest_sha256: str | None,
    calibration_jsonl: str | None,
    calibration_sha256: str | None,
    calibration_prompts: list[str] | None,
    calibration_token_validation: dict[str, int] | None,
    scheme: str,
    iters: int,
    nsamples: int,
    seqlen: int,
    seed: int,
    batch_size: int | None,
    low_gpu_mem_usage: bool,
    enable_torch_compile: bool,
    resolved_device_map,
    platform: str,
    output_format: str,
    execution_settings: dict,
    recovery: dict | None = None,
) -> Path:
    from scripts.build_model_manifest import validate_local_snapshot

    config, files, structure = validate_local_snapshot(output_dir)
    provenance = {
        "schema_version": "1.0",
        "artifact_type": "local_quantization_provenance",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": {
            "path": manifest_path,
            "sha256": manifest_sha256,
            "source": None if manifest is None else manifest.get("source"),
        },
        "calibration": {
            "path": (
                None
                if calibration_jsonl is None
                else str(Path(calibration_jsonl).expanduser().resolve())
            ),
            "sha256": calibration_sha256,
            "sample_count": (
                None
                if calibration_prompts is None
                else len(calibration_prompts)
            ),
            "token_validation": calibration_token_validation,
        },
        "parameters": {
            "scheme": scheme,
            "iters": iters,
            "nsamples": nsamples,
            "seqlen": seqlen,
            "seed": seed,
            "batch_size": batch_size,
            "low_gpu_mem_usage": low_gpu_mem_usage,
            "enable_torch_compile": enable_torch_compile,
            "device_map": str(resolved_device_map),
            "platform": platform,
            "output_format": output_format,
        },
        "dependencies": _dependency_versions(),
        "execution": execution_settings,
        "output": {
            "path": str(output_dir),
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "file_count_before_provenance": len(files),
            **structure,
        },
    }
    if recovery is not None:
        provenance["post_export_recovery"] = recovery
    provenance_path = output_dir / "quantization_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance_path


def _validate_calibration_token_coverage(
    prompts: list[str],
    tokenizer,
    *,
    seqlen: int,
) -> dict[str, int]:
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        truncation=False,
        return_length=True,
    )
    lengths = [int(value) for value in encoded["length"]]
    if len(lengths) != len(prompts):
        raise RuntimeError("tokenizer returned an invalid calibration batch")
    short = [
        (index, length)
        for index, length in enumerate(lengths)
        if length < seqlen
    ]
    if short:
        preview = ", ".join(
            f"{index}:{length}" for index, length in short[:8]
        )
        raise ValueError(
            f"{len(short)} calibration prompts are shorter than seqlen="
            f"{seqlen} tokens ({preview})"
        )
    return {
        "validated_prompt_count": len(lengths),
        "minimum_tokens_before_truncation": min(lengths),
        "maximum_tokens_before_truncation": max(lengths),
        "required_tokens": seqlen,
    }


def quantize_model(
    model_path: str,
    output_path: str,
    scheme: str = "W4A16",
    iters: int = 200,
    device: str = "auto",
    device_map: str | None = None,
    trust_remote_code: bool = False,
    batch_size: int | None = None,
    enable_torch_compile: bool = False,
    low_gpu_mem_usage: bool = False,
    calibration_jsonl: str | None = None,
    nsamples: int = 128,
    seqlen: int = 2048,
    seed: int = 42,
    platform: str = "hf",
    output_format: str = "auto_round",
    source_manifest: str | None = None,
):
    """Quantize a model using AutoRound (GPU by default when available)."""
    model_dir = Path(model_path).expanduser().resolve()
    output_dir = Path(output_path).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse an existing quantization output: {output_dir}"
        )
    manifest, manifest_path, manifest_sha256 = _load_source_manifest(
        source_manifest,
        model_path=model_dir,
    )
    try:
        from auto_round import AutoRound
    except ImportError:
        LOGGER.error(
            "Failed to import auto_round. Please install it: pip install auto-round"
        )
        raise

    resolved_device_map = _resolve_device_map(device, device_map)
    calibration_prompts, calibration_sha256 = _load_calibration_prompts(
        calibration_jsonl,
        nsamples=nsamples,
    )
    LOGGER.info("Starting quantization of %s", model_dir)
    LOGGER.info("Output path: %s", output_dir)
    LOGGER.info("Scheme: %s, Iterations: %s", scheme, iters)
    LOGGER.info("Output format: %s", output_format)
    LOGGER.info("Device map: %s (GPU when available)", resolved_device_map)
    if batch_size:
        LOGGER.info("Batch size: %s", batch_size)
    if calibration_prompts is not None:
        LOGGER.info(
            "Pinned calibration: %s prompts, SHA-256 %s",
            len(calibration_prompts),
            calibration_sha256,
        )
    if manifest is not None:
        LOGGER.info(
            "Pinned source manifest: %s, SHA-256 %s",
            manifest_path,
            manifest_sha256,
        )

    # AutoRound creates the leaf output directory. Only its parent may exist.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    calibration_token_validation = None
    tokenizer = None
    if calibration_prompts is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=trust_remote_code,
        )
        calibration_token_validation = _validate_calibration_token_coverage(
            calibration_prompts,
            tokenizer,
            seqlen=seqlen,
        )
        LOGGER.info(
            "Calibration token coverage: %s",
            calibration_token_validation,
        )

    # AutoRound uses device_map (int=GPU index, "cpu", "cuda:1", "auto"=all GPUs).
    # trust_remote_code is applied inside AutoRound's model loading (default True).
    ar_kwargs = {
        "model": str(model_dir),
        "scheme": scheme,
        "iters": iters,
        "low_cpu_mem_usage": True,
        "enable_torch_compile": enable_torch_compile,
        "device_map": resolved_device_map,
        "platform": platform,
        "trust_remote_code": trust_remote_code,
        "nsamples": nsamples,
        "seqlen": seqlen,
        "seed": seed,
    }
    if calibration_prompts is not None:
        ar_kwargs["dataset"] = calibration_prompts
        ar_kwargs["tokenizer"] = tokenizer
    if batch_size:
        ar_kwargs["batch_size"] = batch_size
    ar_kwargs["low_gpu_mem_usage"] = low_gpu_mem_usage

    # Initialize AutoRound (runs on GPU when device_map is 0 or "cuda:0" etc.)
    ar = AutoRound(**ar_kwargs)
    execution_settings = _execution_settings()

    # Quantize and save
    LOGGER.info("Starting quantization process...")
    ar.quantize_and_save(str(output_dir), format=output_format)

    provenance_path = _write_provenance(
        output_dir=output_dir,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        calibration_jsonl=calibration_jsonl,
        calibration_sha256=calibration_sha256,
        calibration_prompts=calibration_prompts,
        calibration_token_validation=calibration_token_validation,
        scheme=scheme,
        iters=iters,
        nsamples=nsamples,
        seqlen=seqlen,
        seed=seed,
        batch_size=batch_size,
        low_gpu_mem_usage=low_gpu_mem_usage,
        enable_torch_compile=enable_torch_compile,
        resolved_device_map=resolved_device_map,
        platform=platform,
        output_format=output_format,
        execution_settings=execution_settings,
    )
    LOGGER.info(
        "Quantization completed and structurally validated: %s",
        output_dir,
    )
    LOGGER.info("Provenance: %s", provenance_path)


def audit_existing_output(
    *,
    model_path: str,
    output_path: str,
    source_manifest: str | None,
    calibration_jsonl: str | None,
    scheme: str,
    iters: int,
    nsamples: int,
    seqlen: int,
    seed: int,
    batch_size: int | None,
    low_gpu_mem_usage: bool,
    enable_torch_compile: bool,
    device: str,
    device_map: str | None,
    platform: str,
    output_format: str,
    trust_remote_code: bool,
    executed_script_sha256: str,
) -> None:
    """Recover provenance when quantization succeeded but its audit failed."""
    if len(executed_script_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in executed_script_sha256.lower()
    ):
        raise ValueError("--executed-script-sha256 must be a SHA-256 digest")
    model_dir = Path(model_path).expanduser().resolve()
    output_dir = Path(output_path).expanduser().resolve()
    if not output_dir.is_dir():
        raise ValueError("existing quantization output does not exist")
    if (output_dir / "quantization_provenance.json").exists():
        raise FileExistsError(
            "refusing to overwrite existing quantization provenance"
        )
    manifest, manifest_path, manifest_sha256 = _load_source_manifest(
        source_manifest,
        model_path=model_dir,
    )
    calibration_prompts, calibration_sha256 = _load_calibration_prompts(
        calibration_jsonl,
        nsamples=nsamples,
    )
    calibration_token_validation = None
    if calibration_prompts is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=trust_remote_code,
        )
        calibration_token_validation = _validate_calibration_token_coverage(
            calibration_prompts,
            tokenizer,
            seqlen=seqlen,
        )
    resolved_device_map = _resolve_device_map(device, device_map)
    current_script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance_path = _write_provenance(
        output_dir=output_dir,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        calibration_jsonl=calibration_jsonl,
        calibration_sha256=calibration_sha256,
        calibration_prompts=calibration_prompts,
        calibration_token_validation=calibration_token_validation,
        scheme=scheme,
        iters=iters,
        nsamples=nsamples,
        seqlen=seqlen,
        seed=seed,
        batch_size=batch_size,
        low_gpu_mem_usage=low_gpu_mem_usage,
        enable_torch_compile=enable_torch_compile,
        resolved_device_map=resolved_device_map,
        platform=platform,
        output_format=output_format,
        execution_settings=_execution_settings(
            quantizer_script_sha256=executed_script_sha256.lower()
        ),
        recovery={
            "reason": "quantization_and_export_succeeded_but_post_export_audit_failed",
            "failure": "ModuleNotFoundError: No module named 'scripts'",
            "recovery_script_sha256": current_script_sha256,
            "recovered_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    LOGGER.info("Existing output structurally validated: %s", output_dir)
    LOGGER.info("Recovered provenance: %s", provenance_path)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize models using Intel AutoRound"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the input model",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the quantized model",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="W4A16",
        help="Quantization scheme (default: W4A16 for Int4)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Number of iterations for quantization (default: 200)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for quantization: auto (default, use GPU 0), cpu, cuda:0, cuda:1, etc.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Override device: '0' or '1' (single GPU), 'cuda:1', 'auto' (all GPUs), 'cpu'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for quantization (smaller values use less memory)",
    )
    parser.add_argument(
        "--enable-torch-compile",
        action="store_true",
        help="Enable torch.compile for faster quantization (may reduce memory)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow model-repository Python code. Disabled by default; prefer "
            "the pinned Transformers implementation."
        ),
    )
    parser.add_argument(
        "--low-gpu-mem-usage",
        action="store_true",
        help="Use lower GPU memory (slower, for OOM)",
    )
    parser.add_argument(
        "--calibration-jsonl",
        help=(
            "Pinned non-test JSONL with stable ids and prompt/text fields. "
            "If omitted, AutoRound's default calibration source is used."
        ),
    )
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--platform",
        choices=("hf", "model_scope"),
        default="hf",
    )
    parser.add_argument(
        "--output-format",
        default="auto_round",
        help="AutoRound export format (formal default: auto_round).",
    )
    parser.add_argument(
        "--source-manifest",
        help="Content-addressed manifest for --model-path.",
    )
    parser.add_argument(
        "--audit-existing-output",
        action="store_true",
        help=(
            "Audit an already exported model and recover provenance after a "
            "post-export failure; never reruns quantization."
        ),
    )
    parser.add_argument(
        "--executed-script-sha256",
        help=(
            "SHA-256 of the script that produced the existing output; required "
            "with --audit-existing-output."
        ),
    )

    args = parser.parse_args()

    try:
        if args.audit_existing_output:
            if args.executed_script_sha256 is None:
                parser.error(
                    "--audit-existing-output requires "
                    "--executed-script-sha256"
                )
            audit_existing_output(
                model_path=args.model_path,
                output_path=args.output_path,
                source_manifest=args.source_manifest,
                calibration_jsonl=args.calibration_jsonl,
                scheme=args.scheme,
                iters=args.iters,
                nsamples=args.nsamples,
                seqlen=args.seqlen,
                seed=args.seed,
                batch_size=args.batch_size,
                low_gpu_mem_usage=args.low_gpu_mem_usage,
                enable_torch_compile=args.enable_torch_compile,
                device=args.device,
                device_map=args.device_map,
                platform=args.platform,
                output_format=args.output_format,
                trust_remote_code=args.trust_remote_code,
                executed_script_sha256=args.executed_script_sha256,
            )
            return
        quantize_model(
            model_path=args.model_path,
            output_path=args.output_path,
            scheme=args.scheme,
            iters=args.iters,
            device=args.device,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            batch_size=args.batch_size,
            enable_torch_compile=args.enable_torch_compile,
            low_gpu_mem_usage=args.low_gpu_mem_usage,
            calibration_jsonl=args.calibration_jsonl,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            seed=args.seed,
            platform=args.platform,
            output_format=args.output_format,
            source_manifest=args.source_manifest,
        )
    except Exception as e:
        LOGGER.error(f"Quantization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
