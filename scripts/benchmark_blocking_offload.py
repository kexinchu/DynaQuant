#!/usr/bin/env python3
"""Benchmark cold-cache blocking expert transfers from a registered trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynaexq.experiments.eval_quality import (
    PAPER_PROTOCOL,
    SCHEMA_VERSION,
    environment_metadata,
)


INPUT_LENGTHS = (
    16,
    32,
    64,
    96,
    128,
    192,
    256,
    320,
    384,
    448,
    512,
    640,
    768,
    896,
    1024,
    1536,
    2048,
)
WARMUP_TRIALS = 2
MEASURED_TRIALS = 10
MODEL_CONTRACTS = {
    "qwen30b": (48, 128),
    "qwen80b": (48, 512),
    "deepseek_v2_lite": (26, 64),
}
MODEL_TOPK = {
    "qwen30b": 8,
    "qwen80b": 10,
    "deepseek_v2_lite": 6,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_git(data: dict) -> bool:
    git = data.get("environment", {}).get("git", {})
    return bool(git.get("commit")) and git.get("dirty") is False


def _trial_work(
    trial: dict,
    *,
    layer_ids: list[int],
    experts_per_layer: int,
    expert_bytes_per_layer: list[int],
) -> tuple[int, int]:
    """Return cold-cache misses and transferred bytes for one raw trace."""
    active = trial.get("layer_active_experts")
    if not isinstance(active, dict):
        raise ValueError("trial has no layer_active_experts")
    misses = 0
    transferred = 0
    for offset, layer in enumerate(layer_ids):
        values = [int(value) for value in active.get(str(layer), [])]
        if (
            not values
            or values != sorted(set(values))
            or any(value < 0 or value >= experts_per_layer for value in values)
        ):
            raise ValueError(f"invalid active-expert set for layer {layer}")
        misses += len(values)
        transferred += len(values) * expert_bytes_per_layer[offset]
    if set(active) != {str(layer) for layer in layer_ids}:
        raise ValueError("trace layer set does not match the model contract")
    return misses, transferred


def _validate_expert_storage(
    data: dict,
    *,
    layer_ids: list[int],
    experts_per_layer: int,
    expert_bytes_per_layer: list[int],
) -> None:
    """Recompute each per-expert byte count from raw tensor metadata."""
    storage = data.get("expert_storage_tensors")
    if not isinstance(storage, dict) or set(storage) != {
        str(layer) for layer in layer_ids
    }:
        raise ValueError("trace expert-storage layer set is invalid")
    seen_names = set()
    for offset, layer in enumerate(layer_ids):
        records = storage[str(layer)]
        if not isinstance(records, list) or not records:
            raise ValueError("trace expert-storage records are missing")
        total = 0
        for record in records:
            name = str(record["name"])
            shape = [int(value) for value in record["shape"]]
            numel = int(record["numel"])
            element_size = int(record["element_size"])
            size_bytes = int(record["size_bytes"])
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
            if (
                name in seen_names
                or match is None
                or int(match.group(1)) != layer
                or "experts" not in name.split(".")
                or not str(record["dtype"])
                or any(value < 0 for value in shape)
                or math.prod(shape) != numel
                or element_size <= 0
                or size_bytes != numel * element_size
            ):
                raise ValueError("trace expert-storage tensor is invalid")
            seen_names.add(name)
            total += size_bytes
        if (
            total <= 0
            or total % experts_per_layer
            or total // experts_per_layer
            != expert_bytes_per_layer[offset]
        ):
            raise ValueError("trace expert-storage byte summary is invalid")


def load_trace(path: Path, *, paper_model: str) -> dict:
    """Load and fully validate the routing-active-set source artifact."""
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("routing trace must be stored inside the repository")
    data = json.loads(resolved.read_text(encoding="utf-8"))
    layers_expected, experts_expected = MODEL_CONTRACTS[paper_model]
    if (
        int(data.get("schema_version", 0)) < 2
        or data.get("artifact_type") != "routing_active_set_trace"
        or data.get("paper_model") != paper_model
        or not _clean_git(data)
    ):
        raise ValueError("trace identity or clean-code provenance is invalid")
    checkpoint = data.get("checkpoint", {})
    if checkpoint.get("local") is True:
        if not checkpoint.get("weight_hashes_included"):
            raise ValueError("local trace checkpoint lacks weight hashes")
    elif not checkpoint.get("revision"):
        raise ValueError("remote trace checkpoint is not pinned")
    protocol = data.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("name") != "tc_routing_active_set_v1"
        or protocol.get("input_lengths") != list(INPUT_LENGTHS)
        or protocol.get("warmup_trials") != WARMUP_TRIALS
        or protocol.get("measured_trials") != MEASURED_TRIALS
        or protocol.get("batch_size") != 1
        or protocol.get("padding") != "none"
        or protocol.get("prefix_policy")
        != "nested_prefix_per_disjoint_source_window"
        or protocol.get("router_metric")
        != "unique_selected_experts_per_layer"
        or protocol.get("topk") != MODEL_TOPK[paper_model]
        or protocol.get("causal_lm_logits_scope") != "last_position_only"
        or protocol.get("expert_payload_measurement")
        != "stored_routed_expert_parameter_and_buffer_bytes"
    ):
        raise ValueError("trace collection protocol is invalid")
    layer_ids = [int(value) for value in data.get("moe_layer_ids", [])]
    expert_bytes = [
        int(value) for value in data.get("expert_bytes_per_layer", [])
    ]
    if (
        len(layer_ids) != layers_expected
        or layer_ids != sorted(set(layer_ids))
        or len(expert_bytes) != layers_expected
        or any(value <= 0 for value in expert_bytes)
        or int(data.get("experts_per_layer", 0)) != experts_expected
    ):
        raise ValueError("trace model/payload contract is invalid")
    try:
        _validate_expert_storage(
            data,
            layer_ids=layer_ids,
            experts_per_layer=experts_expected,
            expert_bytes_per_layer=expert_bytes,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
    points = data.get("points")
    if (
        not isinstance(points, list)
        or tuple(int(point["input_tokens"]) for point in points)
        != INPUT_LENGTHS
    ):
        raise ValueError("trace input-length grid is invalid")
    for point in points:
        trials = point.get("trials")
        if (
            not isinstance(trials, list)
            or len(trials) != WARMUP_TRIALS + MEASURED_TRIALS
        ):
            raise ValueError("trace trial count is invalid")
        ids = [str(trial.get("trial_id", "")) for trial in trials]
        phases = [trial.get("phase") for trial in trials]
        if (
            any(not value for value in ids)
            or len(set(ids)) != len(ids)
            or phases
            != ["warmup"] * WARMUP_TRIALS
            + ["measured"] * MEASURED_TRIALS
        ):
            raise ValueError("trace trial identities or phases are invalid")
        for trial in trials:
            _trial_work(
                trial,
                layer_ids=layer_ids,
                experts_per_layer=experts_expected,
                expert_bytes_per_layer=expert_bytes,
            )
            max_unique = min(
                experts_expected,
                int(point["input_tokens"]) * MODEL_TOPK[paper_model],
            )
            if any(
                len(trial["layer_active_experts"][str(layer)])
                > max_unique
                for layer in layer_ids
            ):
                raise ValueError("trace has an impossible active-expert count")
    return data


def _copy_trial(
    trial: dict,
    *,
    layer_ids: list[int],
    experts_per_layer: int,
    expert_bytes_per_layer: list[int],
    host_payload: torch.Tensor,
    device_payload: torch.Tensor,
    device: torch.device,
) -> dict:
    misses, transferred = _trial_work(
        trial,
        layer_ids=layer_ids,
        experts_per_layer=experts_per_layer,
        expert_bytes_per_layer=expert_bytes_per_layer,
    )
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start_event.record()
    active = trial["layer_active_experts"]
    for offset, layer in enumerate(layer_ids):
        payload_bytes = expert_bytes_per_layer[offset]
        for _ in active[str(layer)]:
            device_payload[:payload_bytes].copy_(
                host_payload[:payload_bytes],
                non_blocking=True,
            )
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter() - wall_started) * 1000.0
    device_ms = float(start_event.elapsed_time(end_event))
    if not math.isfinite(wall_ms) or not math.isfinite(device_ms):
        raise RuntimeError("non-finite transfer timing")
    return {
        "trial_id": trial["trial_id"],
        "waiting_ms": wall_ms,
        "device_copy_ms": device_ms,
        "cache_misses": misses,
        "transferred_bytes": transferred,
    }


def benchmark(trace: dict, *, device: torch.device) -> dict:
    """Run two warmups and ten measured cold-cache transfers per point."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("blocking offload paper benchmark requires CUDA")
    layer_ids = [int(value) for value in trace["moe_layer_ids"]]
    experts = int(trace["experts_per_layer"])
    expert_bytes = [int(value) for value in trace["expert_bytes_per_layer"]]
    max_payload = max(expert_bytes)
    host_payload = torch.empty(
        max_payload,
        dtype=torch.uint8,
        pin_memory=True,
    )
    host_payload.zero_()
    device_payload = torch.empty(
        max_payload,
        dtype=torch.uint8,
        device=device,
    )
    points = []
    for source_point in trace["points"]:
        trials = source_point["trials"]
        for trial in trials[:WARMUP_TRIALS]:
            _copy_trial(
                trial,
                layer_ids=layer_ids,
                experts_per_layer=experts,
                expert_bytes_per_layer=expert_bytes,
                host_payload=host_payload,
                device_payload=device_payload,
                device=device,
            )
        samples = [
            _copy_trial(
                trial,
                layer_ids=layer_ids,
                experts_per_layer=experts,
                expert_bytes_per_layer=expert_bytes,
                host_payload=host_payload,
                device_payload=device_payload,
                device=device,
            )
            for trial in trials[WARMUP_TRIALS:]
        ]
        waiting = [sample["waiting_ms"] for sample in samples]
        points.append(
            {
                "input_tokens": int(source_point["input_tokens"]),
                "warmup_trials": WARMUP_TRIALS,
                "measured_trials": MEASURED_TRIALS,
                "samples": samples,
                "waiting_ms": waiting,
                "mean_waiting_ms": sum(waiting) / len(waiting),
            }
        )
    return {
        "protocol": {
            "name": "tc_blocking_offload_v1",
            "cache_start": "cold_per_trial",
            "transfer": "pinned_host_to_device",
            "execution": "serial_blocking_on_demand",
            "payload": "measured_packed_expert_bytes",
            "warmup_trials": WARMUP_TRIALS,
            "measured_trials": MEASURED_TRIALS,
            "input_lengths": list(INPUT_LENGTHS),
        },
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-model",
        required=True,
        choices=tuple(MODEL_CONTRACTS),
    )
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        trace = load_trace(args.trace, paper_model=args.paper_model)
        device = torch.device(args.device)
        result = benchmark(trace, device=device)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    device_index = (
        torch.cuda.current_device()
        if device.index is None
        else device.index
    )
    resolved_trace = args.trace.resolve()
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "blocking_offload_waiting",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_model": args.paper_model,
        "offload_method": "blocking_on_demand",
        "checkpoint": trace["checkpoint"],
        "seed": PAPER_PROTOCOL["seed"],
        "environment": environment_metadata(),
        "source_trace": {
            "path": resolved_trace.relative_to(ROOT).as_posix(),
            "sha256": sha256(resolved_trace),
        },
        "benchmark_device": {
            "type": "cuda",
            "index": device_index,
            "name": torch.cuda.get_device_name(device_index),
        },
        "moe_layer_ids": trace["moe_layer_ids"],
        "experts_per_layer": trace["experts_per_layer"],
        "expert_bytes_per_layer": trace["expert_bytes_per_layer"],
        "benchmark": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paper_model": args.paper_model,
                "points": len(result["points"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
