from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import audit_paper_results as audit
from scripts import benchmark_blocking_offload as offload


def test_phi35_offload_contract_matches_checkpoint_architecture():
    assert offload.MODEL_CONTRACTS["phi35"] == (32, 16)
    assert offload.MODEL_TOPK["phi35"] == 2


def _trace() -> dict:
    layer_ids = list(range(48))
    expert_bytes = [1000 + layer for layer in layer_ids]
    expert_storage = {
        str(layer): [
            {
                "name": f"model.layers.{layer}.mlp.experts.weight",
                "shape": [128, expert_bytes[layer]],
                "dtype": "torch.uint8",
                "numel": 128 * expert_bytes[layer],
                "element_size": 1,
                "size_bytes": 128 * expert_bytes[layer],
            }
        ]
        for layer in layer_ids
    }
    points = []
    for point_index, input_tokens in enumerate(offload.INPUT_LENGTHS):
        trials = []
        for trial_index in range(12):
            trials.append(
                {
                    "trial_id": f"p{point_index}-t{trial_index}",
                    "phase": "warmup" if trial_index < 2 else "measured",
                    "layer_active_experts": {
                        str(layer): [
                            (point_index + trial_index + layer) % 128
                        ]
                        for layer in layer_ids
                    },
                }
            )
        points.append({"input_tokens": input_tokens, "trials": trials})
    return {
        "schema_version": 2,
        "artifact_type": "routing_active_set_trace",
        "paper_model": "qwen30b",
        "checkpoint": {"local": False, "revision": "checkpoint-sha"},
        "environment": {
            "git": {"commit": "code-sha", "dirty": False},
            "process_max_rss_bytes": 1024,
        },
        "created_at": "2026-01-01T00:00:00+00:00",
        "seed": 42,
        "protocol": {
            "name": "tc_routing_active_set_v1",
            "input_lengths": list(offload.INPUT_LENGTHS),
            "warmup_trials": 2,
            "measured_trials": 10,
            "batch_size": 1,
            "padding": "none",
            "prefix_policy": "nested_prefix_per_disjoint_source_window",
            "router_metric": "unique_selected_experts_per_layer",
            "topk": 8,
            "causal_lm_logits_scope": "last_position_only",
            "expert_payload_measurement": (
                "stored_routed_expert_parameter_and_buffer_bytes"
            ),
        },
        "moe_layer_ids": layer_ids,
        "experts_per_layer": 128,
        "expert_bytes_per_layer": expert_bytes,
        "expert_storage_tensors": expert_storage,
        "points": points,
    }


def _artifact(trace: dict, trace_path: Path) -> dict:
    layer_ids = trace["moe_layer_ids"]
    expert_bytes = trace["expert_bytes_per_layer"]
    transferred = sum(expert_bytes)
    points = []
    for source_point in trace["points"]:
        samples = []
        for sample_index, trial in enumerate(source_point["trials"][2:]):
            waiting_ms = float(sample_index + 1)
            samples.append(
                {
                    "trial_id": trial["trial_id"],
                    "waiting_ms": waiting_ms,
                    "device_copy_ms": waiting_ms / 2,
                    "cache_misses": len(layer_ids),
                    "transferred_bytes": transferred,
                }
            )
        waiting = [sample["waiting_ms"] for sample in samples]
        points.append(
            {
                "input_tokens": source_point["input_tokens"],
                "warmup_trials": 2,
                "measured_trials": 10,
                "samples": samples,
                "waiting_ms": waiting,
                "mean_waiting_ms": sum(waiting) / len(waiting),
            }
        )
    return {
        "schema_version": 2,
        "artifact_type": "blocking_offload_waiting",
        "created_at": "2026-01-01T00:00:00+00:00",
        "paper_model": "qwen30b",
        "offload_method": "blocking_on_demand",
        "checkpoint": trace["checkpoint"],
        "seed": 42,
        "environment": {
            "git": {"commit": "code-sha", "dirty": False},
            "process_max_rss_bytes": 1024,
        },
        "source_trace": {
            "path": trace_path.name,
            "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        },
        "benchmark_device": {
            "type": "cuda",
            "index": 0,
            "name": "NVIDIA RTX A6000",
        },
        "moe_layer_ids": layer_ids,
        "experts_per_layer": trace["experts_per_layer"],
        "expert_bytes_per_layer": expert_bytes,
        "benchmark": {
            "protocol": {
                "name": "tc_blocking_offload_v1",
                "cache_start": "cold_per_trial",
                "transfer": "pinned_host_to_device",
                "execution": "serial_blocking_on_demand",
                "payload": "measured_packed_expert_bytes",
                "warmup_trials": 2,
                "measured_trials": 10,
                "input_lengths": list(offload.INPUT_LENGTHS),
            },
            "points": points,
        },
    }


def test_trace_loader_validates_model_grid_and_raw_work(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(offload, "ROOT", tmp_path)
    trace_path = tmp_path / "trace.json"
    trace = _trace()
    trace_path.write_text(json.dumps(trace), encoding="utf-8")

    loaded = offload.load_trace(trace_path, paper_model="qwen30b")
    misses, transferred = offload._trial_work(
        loaded["points"][0]["trials"][0],
        layer_ids=loaded["moe_layer_ids"],
        experts_per_layer=loaded["experts_per_layer"],
        expert_bytes_per_layer=loaded["expert_bytes_per_layer"],
    )
    assert misses == 48
    assert transferred == sum(trace["expert_bytes_per_layer"])

    trace["points"][0]["trials"][0]["layer_active_experts"]["0"] = [1, 1]
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    with pytest.raises(ValueError, match="active-expert"):
        offload.load_trace(trace_path, paper_model="qwen30b")

    trace = _trace()
    trace["expert_storage_tensors"]["0"][0]["size_bytes"] += 1
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    with pytest.raises(ValueError, match="expert-storage"):
        offload.load_trace(trace_path, paper_model="qwen30b")


def test_audit_recomputes_offload_work_from_hashed_trace(
    tmp_path,
    monkeypatch,
):
    prompts_path = tmp_path / "prompts.jsonl"
    prompt_ids = ["prompt-0"]
    prompts_path.write_text(
        json.dumps({"id": prompt_ids[0], "prompt": "text"}) + "\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "trace.json"
    trace = _trace()
    trace["protocol"]["prompt_source"] = {
        "path": prompts_path.name,
        "source_sha256": hashlib.sha256(
            prompts_path.read_bytes()
        ).hexdigest(),
        "selected_row_count": 1,
        "selected_ids_sha256": hashlib.sha256(
            json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "selected_token_count": 12 * 2048,
        "selection": (
            "concatenated_eos_separated_disjoint_2048_token_blocks"
        ),
    }
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    artifact = _artifact(trace, trace_path)
    monkeypatch.setattr(audit, "ROOT", tmp_path)

    assert audit.validate_manifest_artifact(
        "offload_waiting",
        "offload_waiting[0]",
        artifact,
        "offload_waiting:qwen30b",
    ) == []

    artifact["benchmark"]["points"][0]["samples"][0][
        "transferred_bytes"
    ] += 1
    problems = audit.validate_manifest_artifact(
        "offload_waiting",
        "offload_waiting[0]",
        artifact,
        "offload_waiting:qwen30b",
    )
    assert "INVALID OFFLOAD WAITING SAMPLES: offload_waiting[0]" in problems


def test_offload_benchmark_script_is_directly_executable():
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "benchmark_blocking_offload.py"),
            "--help",
        ],
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
