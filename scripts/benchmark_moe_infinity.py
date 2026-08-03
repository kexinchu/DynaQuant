#!/usr/bin/env python3
"""Run the pinned MoE-Infinity baseline under the TC model protocol.

Only Qwen3-30B is admitted because that is the manuscript model supported by
the pinned official runtime.  The adapter uses the same isolated-model timing
function as the native baselines and refuses to write an artifact unless
offloaded expert tensors and measured-interval prefetch activity are observed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynaexq.baselines.moe_infinity import (  # noqa: E402
    PINNED_COMMIT,
    PrefetchTelemetry,
    count_offloaded_expert_tensors,
    validate_runtime_configuration,
    verify_import_from_checkout,
    verify_official_checkout,
)
from dynaexq.experiments.eval_perf import measure_latency  # noqa: E402
from dynaexq.experiments.eval_quality import (  # noqa: E402
    SCHEMA_VERSION,
    checkpoint_metadata,
    environment_metadata,
)


INPUT_TOKENS = 2048
OUTPUT_TOKENS = 256
WARMUPS = 5
REPEATS = 100
BATCHES = (1, 2, 4, 8, 16, 32)


def _inside_repository(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(
            f"formal artifact must be written inside repository: {resolved}"
        ) from error
    return resolved


def _resolve_checkpoint(reference: str) -> tuple[str, dict[str, Any], dict]:
    """Resolve a remote ID to an immutable local snapshot before loading."""
    source = checkpoint_metadata(reference, hash_weight_files=True)
    if source.get("local") is True:
        if not source.get("weight_hashes_included"):
            raise RuntimeError("local checkpoint weights were not hashed")
        return str(Path(reference).expanduser().resolve()), source, {
            "mode": "hashed_local_checkpoint",
            "remote_revision": None,
        }

    revision = source.get("revision")
    if not revision:
        raise RuntimeError("remote checkpoint revision could not be resolved")
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repo_id=reference, revision=revision)
    snapshot_path = Path(snapshot).resolve()
    if snapshot_path.name != revision:
        raise RuntimeError(
            "resolved Hugging Face snapshot path does not match revision"
        )
    return str(snapshot_path), source, {
        "mode": "pinned_huggingface_snapshot",
        "remote_revision": revision,
        "snapshot_commit_directory": snapshot_path.name,
    }


def _validate_hardware() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("paper protocol requires exactly one CUDA GPU")
    name = torch.cuda.get_device_name(0)
    total_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    if "A6000" not in name or total_bytes < 47 * 1024**3:
        raise RuntimeError(
            "paper protocol requires the declared 48 GB NVIDIA RTX A6000"
        )
    return {
        "device_count": 1,
        "device_name": name,
        "total_memory_bytes": total_bytes,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Qwen/Qwen3-30B-A3B-Instruct-2507 or a hashed local snapshot",
    )
    parser.add_argument(
        "--repo",
        required=True,
        type=Path,
        help=f"clean official MoE-Infinity checkout at {PINNED_COMMIT}",
    )
    parser.add_argument(
        "--offload-dir",
        required=True,
        type=Path,
        help="dedicated local SSD directory for this checkpoint",
    )
    parser.add_argument("--batch-size", required=True, type=int, choices=BATCHES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device-memory-ratio", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.seed != 42:
        raise ValueError("paper protocol requires seed 42")
    if not 0.1 <= args.device_memory_ratio <= 0.85:
        raise ValueError("--device-memory-ratio must be in [0.1, 0.85]")

    output = _inside_repository(args.output)
    source_identity = verify_official_checkout(args.repo)
    hardware = _validate_hardware()
    environment_before = environment_metadata()
    project_git = environment_before.get("git", {})
    if (
        not isinstance(project_git, dict)
        or not project_git.get("commit")
        or project_git.get("dirty") is not False
    ):
        raise RuntimeError("formal run requires a clean, committed DynaExQ tree")

    resolved_model, checkpoint, model_loading = _resolve_checkpoint(args.model)

    repo = args.repo.expanduser().resolve()
    sys.path.insert(0, str(repo))
    import moe_infinity
    from moe_infinity import MoE
    from transformers import AutoTokenizer

    module_path = verify_import_from_checkout(moe_infinity.__file__, repo)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    offload_dir = args.offload_dir.expanduser().resolve()
    offload_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = {
        "offload_path": str(offload_dir),
        "device_memory_ratio": args.device_memory_ratio,
        "prefetch": True,
        "speculative_prefetch": True,
        "speculative_prefetch_overlap": True,
        "use_native_engine": False,
        "kv_cache_memory_ratio": 0.0,
    }

    torch.manual_seed(args.seed)
    runtime = MoE(resolved_model, runtime_config)
    active_features = validate_runtime_configuration(runtime)
    total_tensors, offloaded_tensors = count_offloaded_expert_tensors(runtime)
    if offloaded_tensors <= 0:
        raise RuntimeError(
            "no expert tensor is offloaded; refusing mislabeled baseline"
        )

    telemetry = PrefetchTelemetry.install(runtime)
    try:
        benchmark = measure_latency(
            runtime.model,
            tokenizer,
            batch_size=args.batch_size,
            input_length=INPUT_TOKENS,
            output_length=OUTPUT_TOKENS,
            n_warmup=WARMUPS,
            n_repeats=REPEATS,
            iteration_setup=runtime._configure_hook,
            after_warmup=telemetry.reset,
            input_device="cuda:0",
            require_process_hbm_monitor=True,
        )
        runtime_stats = telemetry.snapshot(
            total_expert_tensors=total_tensors,
            offloaded_expert_tensors=offloaded_tensors,
        )
    finally:
        telemetry.close()

    if (
        runtime_stats["prefetch_calls"] <= 0
        or runtime_stats["prefetch_requested_experts"] <= 0
        or not runtime_stats["prefetch_layers_touched"]
    ):
        raise RuntimeError(
            "measured interval contains no external prefetch activity"
        )

    source_identity["imported_module"] = module_path
    source_identity["features"] = {
        "expert_offload": True,
        "activation_aware_cache": True,
        **active_features,
    }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "moe_infinity_performance",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "paper_model": "qwen30b",
        "paper_method": "moe_infinity",
        "method": "official_external_offload_runtime",
        "checkpoint": checkpoint,
        "model_loading": model_loading,
        "seed": args.seed,
        "evaluation_protocol": {
            "name": "tc_isolated_performance_v2",
            "seed": args.seed,
            "process_hbm_high_water": True,
        },
        "runtime_config": runtime_config,
        "baseline_implementation": source_identity,
        "baseline_runtime_stats": runtime_stats,
        "benchmark": benchmark,
        "hardware_contract": hardware,
        "environment": environment_metadata(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "claim_id": (
                    "performance:qwen30b:moe_infinity:"
                    f"bs{args.batch_size}"
                ),
                "source_commit": source_identity["commit"],
                "offloaded_expert_tensors": offloaded_tensors,
                "prefetch_calls": runtime_stats["prefetch_calls"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
