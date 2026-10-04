#!/usr/bin/env python3
"""Build provenance-rich Background motivation artifacts from measured inputs.

The script does not present replay output as an end-to-end runtime result.  It
combines registered routing sets, byte footprints, blocking H2D timings, and
model TTFT measurements in a deterministic trace replay.  The resulting JSON
records both its measured inputs and every replay assumption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = ROOT / "results/paper/qwen30b_routing_active_set_trace.json"
DEFAULT_OFFLOAD = ROOT / "results/paper/qwen30b_offload_waiting.json"
DEFAULT_DYNAMIC_PERF = ROOT / "results/paper/performance/qwen30b_dynaexq_bs1.json"
DEFAULT_ACTIVATION = ROOT / "results/paper/background/qwen30b_activation_density.json"
DEFAULT_PERF_DIR = ROOT / "results/paper/background/performance"
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_PPL_DIR = ROOT / "results/perplexity/Qwen3-30B"
DEFAULT_OUTPUT_DIR = ROOT / "results/paper/background"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _source(path: Path) -> dict:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"input must be stored inside the repository: {path}")
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "sha256": sha256(resolved),
    }


def _git_metadata() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _measured_trials(trace: dict, input_tokens: int) -> tuple[list[dict], list[dict]]:
    point = next(
        (
            item
            for item in trace.get("points", [])
            if int(item.get("input_tokens", -1)) == input_tokens
        ),
        None,
    )
    if point is None:
        raise ValueError(f"routing trace has no {input_tokens}-token point")
    trials = point.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("routing trace point has no trials")
    warmup = [trial for trial in trials if trial.get("phase") == "warmup"]
    measured = [trial for trial in trials if trial.get("phase") == "measured"]
    if not warmup or not measured:
        raise ValueError("routing trace needs warmup and measured trials")
    return warmup, measured


def _validate_trace(trace: dict) -> tuple[list[int], int, list[int]]:
    if (
        int(trace.get("schema_version", 0)) < 2
        or trace.get("artifact_type") != "routing_active_set_trace"
        or trace.get("paper_model") != "qwen30b"
    ):
        raise ValueError("unsupported routing trace")
    layers = [int(value) for value in trace.get("moe_layer_ids", [])]
    experts = int(trace.get("experts_per_layer", 0))
    expert_bytes = [int(value) for value in trace.get("expert_bytes_per_layer", [])]
    if (
        not layers
        or layers != sorted(set(layers))
        or experts <= 0
        or len(expert_bytes) != len(layers)
        or any(value <= 0 for value in expert_bytes)
    ):
        raise ValueError("invalid routing trace model contract")
    for point in trace.get("points", []):
        for trial in point.get("trials", []):
            active = trial.get("layer_active_experts", {})
            if set(active) != {str(layer) for layer in layers}:
                raise ValueError("routing trace layer set is incomplete")
            for layer in layers:
                values = [int(value) for value in active[str(layer)]]
                if values != sorted(set(values)) or any(
                    value < 0 or value >= experts for value in values
                ):
                    raise ValueError("routing trace active set is invalid")
    return layers, experts, expert_bytes


def _transfer_bandwidth_bytes_s(offload: dict, input_tokens: int) -> float:
    if (
        int(offload.get("schema_version", 0)) < 2
        or offload.get("artifact_type") != "blocking_offload_waiting"
        or offload.get("paper_model") != "qwen30b"
    ):
        raise ValueError("unsupported offload artifact")
    points = offload.get("benchmark", {}).get("points", [])
    point = next(
        (item for item in points if int(item.get("input_tokens", -1)) == input_tokens),
        None,
    )
    if point is None:
        raise ValueError(f"offload artifact has no {input_tokens}-token point")
    samples = point.get("samples", [])
    total_bytes = sum(int(sample["transferred_bytes"]) for sample in samples)
    total_ms = sum(float(sample["device_copy_ms"]) for sample in samples)
    if total_bytes <= 0 or not math.isfinite(total_ms) or total_ms <= 0:
        raise ValueError("offload artifact has invalid copy measurements")
    return total_bytes / (total_ms / 1000.0)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def _hot_sets(
    trials: list[dict],
    layers: list[int],
    capacity: int,
) -> dict[int, set[int]]:
    result = {}
    for layer in layers:
        counts: Counter[int] = Counter()
        for trial in trials:
            counts.update(
                int(value)
                for value in trial["layer_active_experts"][str(layer)]
            )
        result[layer] = {
            expert
            for expert, _ in sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:capacity]
        }
    return result


def build_transfer_pressure(
    trace: dict,
    offload: dict,
    static_perf: dict,
    *,
    input_tokens: int,
    resident_experts_per_layer: int,
) -> dict:
    """Derive per-layer demand and a conservative overlap bound."""
    layers, experts, expert_bytes = _validate_trace(trace)
    if not 0 <= resident_experts_per_layer <= experts:
        raise ValueError("resident expert count is outside the model contract")
    warmup, measured = _measured_trials(trace, input_tokens)
    resident = _hot_sets(warmup, layers, resident_experts_per_layer)
    bandwidth = _transfer_bandwidth_bytes_s(offload, input_tokens)
    try:
        ttft_ms = float(
            static_perf["benchmark"]["metrics"]["model_ttft_ms"]["mean"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("static performance artifact has no mean TTFT") from error
    if not math.isfinite(ttft_ms) or ttft_ms <= 0:
        raise ValueError("static performance artifact has invalid mean TTFT")
    mean_layer_window_ms = ttft_ms / len(layers)
    overlap_upper_bytes = bandwidth * mean_layer_window_ms / 1000.0

    layer_records = []
    for offset, layer in enumerate(layers):
        size = expert_bytes[offset]
        demand = []
        miss = []
        for trial in measured:
            active = {
                int(value)
                for value in trial["layer_active_experts"][str(layer)]
            }
            demand.append(len(active) * size)
            miss.append(len(active - resident[layer]) * size)
        mean_demand = _mean(demand)
        mean_miss = _mean(miss)
        service_ms = 1000.0 * mean_miss / bandwidth
        layer_records.append(
            {
                "layer": layer,
                "mean_demanded_bytes": mean_demand,
                "mean_miss_bytes": mean_miss,
                "overlap_capacity_upper_bytes": overlap_upper_bytes,
                "copy_service_ms": service_ms,
                "mean_layer_window_ms": mean_layer_window_ms,
                "exposed_wait_lower_bound_ms": max(
                    0.0,
                    service_ms - mean_layer_window_ms,
                ),
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "background_transfer_pressure_replay",
        "paper_model": "qwen30b",
        "input_tokens": input_tokens,
        "measured_trials": len(measured),
        "resident_policy": {
            "name": "warmup_frequency_topk",
            "experts_per_layer": resident_experts_per_layer,
            "warmup_trials": len(warmup),
        },
        "transfer_bandwidth_bytes_s": bandwidth,
        "compute_window": {
            "source": "mean_model_ttft_divided_by_routed_layer_count",
            "ttft_ms": ttft_ms,
            "routed_layers": len(layers),
            "mean_layer_window_ms": mean_layer_window_ms,
            "interpretation": "upper-bound diagnostic, not a deadline schedule",
        },
        "layers": layer_records,
    }


def build_stage_transfer_pressure(
    activation: dict,
    offload: dict,
    perf_by_batch: dict[int, dict],
    *,
    input_tokens: int,
    low_expert_bytes: int,
) -> dict:
    """Compare demanded and maximally hideable bytes in prefill and decode."""
    if (
        int(activation.get("schema_version", 0)) < 2
        or activation.get("artifact_type") != "activation_density"
        or activation.get("paper_model") != "qwen30b"
    ):
        raise ValueError("unsupported activation-density artifact")
    layers = [int(value) for value in activation.get("moe_layer_ids", [])]
    if not layers or low_expert_bytes <= 0:
        raise ValueError("invalid layer or expert-byte contract")
    bandwidth = _transfer_bandwidth_bytes_s(offload, input_tokens)
    stage_metric = {"prefill": "model_ttft_ms", "decode": "model_tpot_ms"}
    stage_records = {}
    batches = None
    for stage, metric in stage_metric.items():
        entries = activation.get("stages", {}).get(stage, [])
        current_batches = [int(entry["batch_size"]) for entry in entries]
        if not entries or current_batches != sorted(set(current_batches)):
            raise ValueError(f"invalid {stage} activation-density grid")
        if batches is None:
            batches = current_batches
        elif current_batches != batches:
            raise ValueError("prefill/decode batch grids differ")
        records = []
        for entry in entries:
            batch = int(entry["batch_size"])
            raw = entry.get("layer_active_counts")
            if not isinstance(raw, list) or not raw:
                raise ValueError(f"{stage} batch {batch} has no raw layer counts")
            flat = [int(value) for repeat in raw for value in repeat]
            experts = int(entry["experts_per_layer"])
            if (
                len(flat) != len(raw) * len(layers)
                or any(value < 0 or value > experts for value in flat)
            ):
                raise ValueError(f"{stage} batch {batch} raw counts are invalid")
            measured_ratio = 100.0 * _mean(flat) / experts
            if not math.isclose(
                measured_ratio,
                float(entry["ratio_pct"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"{stage} batch {batch} ratio is inconsistent")
            try:
                duration_ms = float(
                    perf_by_batch[batch]["benchmark"]["metrics"][metric]["mean"]
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"batch {batch} performance artifact lacks {metric}"
                ) from error
            layer_window_ms = duration_ms / len(layers)
            demanded_bytes = _mean(flat) * low_expert_bytes
            overlap_bytes = bandwidth * layer_window_ms / 1000.0
            records.append(
                {
                    "batch_size": batch,
                    "mean_active_experts_per_layer": _mean(flat),
                    "mean_demanded_bytes_per_layer": demanded_bytes,
                    "mean_layer_window_ms": layer_window_ms,
                    "overlap_capacity_upper_bytes": overlap_bytes,
                    "cold_demand_excess_bytes": max(0.0, demanded_bytes - overlap_bytes),
                    "cold_demand_exposed_lower_bound_ms": max(
                        0.0,
                        1000.0 * demanded_bytes / bandwidth - layer_window_ms,
                    ),
                }
            )
        stage_records[stage] = records
    return {
        "schema_version": 1,
        "artifact_type": "background_stage_transfer_pressure",
        "paper_model": "qwen30b",
        "input_tokens": input_tokens,
        "transfer_bandwidth_bytes_s": bandwidth,
        "expert_bytes": {
            "precision": "INT4",
            "bytes_per_expert": low_expert_bytes,
        },
        "capacity_interpretation": (
            "measured blocking-copy bandwidth multiplied by mean per-layer "
            "TTFT/TPOT; an aggregate upper-bound diagnostic, not a deadline schedule"
        ),
        "stages": stage_records,
    }


def load_perplexity_curve(directory: Path) -> tuple[list[dict], list[dict]]:
    points = []
    sources = []
    for path in sorted(directory.glob("30b-*.txt")):
        data = _load_json(path)
        low_ratio = int(path.stem.split("-")[-1])
        perplexity = float(data.get("perplexity", float("nan")))
        target_tokens = int(data.get("target_tokens", 0))
        if not math.isfinite(perplexity) or perplexity <= 0 or target_tokens <= 0:
            raise ValueError(f"invalid perplexity result: {path}")
        points.append(
            {
                "low_precision_ratio_pct": low_ratio,
                "high_precision_ratio_pct": 100 - low_ratio,
                "perplexity": perplexity,
                "target_tokens": target_tokens,
            }
        )
        sources.append(_source(path))
    points.sort(key=lambda point: point["high_precision_ratio_pct"])
    expected = {0, 10, 25, 40, 55, 70, 85, 100}
    if {point["high_precision_ratio_pct"] for point in points} != expected:
        raise ValueError("perplexity grid is incomplete")
    return points, sources


def _pool_contract(dynamic_perf: dict, layer_count: int) -> tuple[int, int, int]:
    try:
        initialization = dynamic_perf["runtime_initialization"]
        pool = initialization["bootstrap"]["pool"]
        hi_bytes = int(pool["hi"][0]["block_size_bytes"])
        lo_bytes = int(pool["lo"][0]["block_size_bytes"])
        total = int(initialization["pool_allocation_bytes"])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("dynamic artifact has no byte-accurate pool contract") from error
    if hi_bytes <= lo_bytes or lo_bytes <= 0 or total <= 0:
        raise ValueError("dynamic pool byte contract is invalid")
    return hi_bytes, lo_bytes, total // layer_count


def build_joint_surface(
    trace: dict,
    offload: dict,
    dynamic_perf: dict,
    perplexity_points: list[dict],
    *,
    input_tokens: int,
    staging_slots: list[int],
    max_perplexity_increase: float,
) -> dict:
    """Replay the shared-byte tradeoff over fidelity and lookahead."""
    layers, experts, _ = _validate_trace(trace)
    warmup, measured = _measured_trials(trace, input_tokens)
    bandwidth = _transfer_bandwidth_bytes_s(offload, input_tokens)
    hi_bytes, lo_bytes, budget_per_layer = _pool_contract(
        dynamic_perf,
        len(layers),
    )
    baseline = min(point["perplexity"] for point in perplexity_points)
    quality_limit = baseline + max_perplexity_increase
    quality_by_hi = {
        int(point["high_precision_ratio_pct"]): float(point["perplexity"])
        for point in perplexity_points
    }
    high_ratios = sorted(quality_by_hi)
    max_overlap_slots = max(
        1,
        math.floor(
            _mean(
                sample["transferred_bytes"] / sample["device_copy_ms"]
                for point in offload["benchmark"]["points"]
                if int(point["input_tokens"]) == input_tokens
                for sample in point["samples"]
            )
            / (lo_bytes / 1.0)
        ),
    )
    # The expression above is bytes/ms divided by bytes/expert.  It is the
    # number of low-tier experts transferable in a 1 ms scheduling window.

    cells = []
    for high_ratio in high_ratios:
        high_count = math.ceil(experts * high_ratio / 100.0)
        hot_high = _hot_sets(warmup, layers, high_count)
        for slots in staging_slots:
            fixed_bytes = high_count * hi_bytes + slots * lo_bytes
            memory_feasible = fixed_bytes <= budget_per_layer
            low_capacity = (
                min(
                    experts - high_count,
                    max(0, (budget_per_layer - fixed_bytes) // lo_bytes),
                )
                if memory_feasible
                else 0
            )
            resident = {}
            for layer in layers:
                counts: Counter[int] = Counter()
                for trial in warmup:
                    counts.update(
                        int(value)
                        for value in trial["layer_active_experts"][str(layer)]
                        if int(value) not in hot_high[layer]
                    )
                low_set = {
                    expert
                    for expert, _ in sorted(
                        counts.items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:low_capacity]
                }
                resident[layer] = hot_high[layer] | low_set

            exposed_bytes = []
            if memory_feasible:
                for trial in measured:
                    trial_exposed = 0
                    for layer in layers:
                        active = {
                            int(value)
                            for value in trial["layer_active_experts"][str(layer)]
                        }
                        misses = len(active - resident[layer])
                        hidden = min(misses, slots, max_overlap_slots)
                        trial_exposed += (misses - hidden) * lo_bytes
                    exposed_bytes.append(trial_exposed)
            mean_exposed = _mean(exposed_bytes) if exposed_bytes else None
            cells.append(
                {
                    "high_precision_ratio_pct": high_ratio,
                    "staging_experts_per_layer": slots,
                    "high_precision_experts_per_layer": high_count,
                    "low_precision_residents_per_layer": low_capacity,
                    "memory_feasible": memory_feasible,
                    "quality_feasible": quality_by_hi[high_ratio] <= quality_limit,
                    "perplexity": quality_by_hi[high_ratio],
                    "mean_exposed_bytes_per_request": mean_exposed,
                    "mean_exposed_transfer_ms": (
                        1000.0 * mean_exposed / bandwidth
                        if mean_exposed is not None
                        else None
                    ),
                }
            )
    feasible = [
        cell
        for cell in cells
        if cell["memory_feasible"]
        and cell["quality_feasible"]
        and cell["mean_exposed_transfer_ms"] is not None
    ]
    optimum = (
        min(feasible, key=lambda cell: cell["mean_exposed_transfer_ms"])
        if feasible
        else None
    )
    return {
        "schema_version": 1,
        "artifact_type": "background_joint_allocation_replay",
        "paper_model": "qwen30b",
        "input_tokens": input_tokens,
        "measured_trials": len(measured),
        "byte_contract": {
            "high_precision_expert_bytes": hi_bytes,
            "low_precision_expert_bytes": lo_bytes,
            "expert_budget_per_layer_bytes": budget_per_layer,
            "experts_per_layer": experts,
        },
        "quality_constraint": {
            "metric": "WikiText-2 perplexity",
            "all_high_reference": baseline,
            "maximum_increase": max_perplexity_increase,
            "limit": quality_limit,
        },
        "replay_policy": {
            "resident_ranking": "warmup_frequency",
            "lookahead": "oracle active set bounded by staging slots",
            "transfer_window_ms": 1.0,
            "max_transferable_low_experts_per_window": max_overlap_slots,
            "reported_metric": "copy-time equivalent of unhidden bytes",
        },
        "high_precision_ratios_pct": high_ratios,
        "staging_experts_per_layer": staging_slots,
        "cells": cells,
        "joint_optimum": optimum,
    }


def _write_artifact(path: Path, payload: dict, sources: list[dict], command: str) -> None:
    artifact = {
        **payload,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "command": command,
        "environment": {"git": _git_metadata()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--offload", type=Path, default=DEFAULT_OFFLOAD)
    parser.add_argument("--dynamic-perf", type=Path, default=DEFAULT_DYNAMIC_PERF)
    parser.add_argument("--activation-density", type=Path, default=DEFAULT_ACTIVATION)
    parser.add_argument("--performance-dir", type=Path, default=DEFAULT_PERF_DIR)
    parser.add_argument("--perplexity-dir", type=Path, default=DEFAULT_PPL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument(
        "--batches",
        default=",".join(str(value) for value in DEFAULT_BATCHES),
    )
    parser.add_argument(
        "--performance-pattern",
        default="qwen30b_static_int4_bs{batch}.json",
    )
    parser.add_argument(
        "--only",
        choices=("all", "transfer", "joint"),
        default="all",
    )
    parser.add_argument(
        "--staging-slots",
        default="0,4,8,12,16,20,24,28,32",
    )
    parser.add_argument("--max-perplexity-increase", type=float, default=2.1)
    args = parser.parse_args()

    trace = _load_json(args.trace)
    offload = _load_json(args.offload)
    dynamic_perf = _load_json(args.dynamic_perf)
    activation = _load_json(args.activation_density)
    ppl, ppl_sources = load_perplexity_curve(args.perplexity_dir)
    slots = [int(value) for value in args.staging_slots.split(",")]
    if not slots or slots != sorted(set(slots)) or any(value < 0 for value in slots):
        parser.error("--staging-slots must be sorted, unique non-negative integers")
    batches = tuple(int(value) for value in args.batches.split(",") if value.strip())
    if (
        not batches
        or batches != tuple(sorted(set(batches)))
        or any(value <= 0 for value in batches)
    ):
        parser.error("--batches must be sorted, unique positive integers")

    sources = [
        _source(args.trace),
        _source(args.offload),
        _source(args.dynamic_perf),
        _source(args.activation_density),
        *ppl_sources,
    ]
    perf_by_batch = {}
    for batch in batches:
        path = args.performance_dir / args.performance_pattern.format(batch=batch)
        perf_by_batch[batch] = _load_json(path)
        source = _source(path)
        if source not in sources:
            sources.append(source)
    command = "python " + " ".join(__import__("sys").argv)
    _, _, trace_expert_bytes = _validate_trace(trace)
    transfer = None
    joint = None
    if args.only in ("all", "transfer"):
        transfer = build_stage_transfer_pressure(
            activation,
            offload,
            perf_by_batch,
            input_tokens=args.input_tokens,
            low_expert_bytes=round(_mean(trace_expert_bytes)),
        )
    if args.only in ("all", "joint"):
        joint = build_joint_surface(
            trace,
            offload,
            dynamic_perf,
            ppl,
            input_tokens=args.input_tokens,
            staging_slots=slots,
            max_perplexity_increase=args.max_perplexity_increase,
        )
    written = {}
    if transfer is not None:
        transfer_path = args.output_dir / "qwen30b_transfer_pressure.json"
        _write_artifact(transfer_path, transfer, sources, command)
        written["transfer_pressure"] = str(transfer_path)
    if joint is not None:
        joint_path = args.output_dir / "qwen30b_joint_allocation_replay.json"
        _write_artifact(joint_path, joint, sources, command)
        written["joint_allocation"] = str(joint_path)
    print(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()
