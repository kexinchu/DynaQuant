#!/usr/bin/env python3
"""Measure prefill TTFT at 128, 512, and 2048 tokens on one A6000.

The registered replay divides the 2048-token DynaExq TTFT by the layer count
and scales that window linearly for shorter prompts. This probe loads the
uniform-low runtime once, so all three lengths share a stack. Output length
is 1 because only time-to-first-token is the overlap window.

The checkpoint on disk is the unpacked Qwen3-30B tree, not the missing
BF16-pinned path recorded in qwen30b_dynaexq_bs1.json. Results are a new
artifact. They do not overwrite the registered 2048-token window.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from dynaexq.core.config import DynaExqConfig
from dynaexq.experiments.eval_perf import measure_latency
from dynaexq.experiments.run_shift import initialize_dynaexq, load_model
from dynaexq.integration.moe_wrapper import MoEWrapper
from dynaexq.policy import RuntimeExchangePlanner


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper/a6000_wikitext_replay/e2_layer_window.json"
LENGTHS = (128, 512, 2048)
N_WARMUP = 2
N_REPEATS = 10


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("E2 requires a CUDA device")
    config = DynaExqConfig.from_yaml(ROOT / "dynaexq/configs/qwen30b_uniform_low_a6000.yaml")
    model_path = os.environ.get(
        "E2_MODEL",
        "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507",
    )
    model, tokenizer = load_model(model_path, device)
    (
        observer,
        tracker,
        scheduler,
        registry,
        transition_engine,
        _,
        initialization,
    ) = initialize_dynaexq(config, model, device, high_precision_ratio=0.0)
    wrapper = MoEWrapper(
        model=model,
        router_observer=observer,
        hotness_tracker=tracker,
        scheduler=scheduler,
        registry=registry,
        transition_engine=transition_engine,
        exchange_planner=RuntimeExchangePlanner(registry, transition_engine),
        residency_controller=None,
        num_layers=config.model.layers,
        experts_per_layer=config.model.experts_per_layer,
        topk=config.model.topk,
        scheduler_enabled=False,
    )
    wrapper.validate_integration()
    n_layers = config.model.layers
    points = []
    for tokens in LENGTHS:
        print(f"measure input_tokens={tokens}", flush=True)
        benchmark = measure_latency(
            wrapper,
            tokenizer,
            batch_size=1,
            input_length=tokens,
            output_length=1,
            n_warmup=N_WARMUP,
            n_repeats=N_REPEATS,
        )
        mean_ttft = float(benchmark["metrics"]["model_ttft_ms"]["mean"])
        points.append(
            {
                "input_tokens": tokens,
                "ttft_ms": benchmark["metrics"]["model_ttft_ms"],
                "layer_time_ms": mean_ttft / n_layers,
                "layer_time_source": "measured_ttft_over_layers",
                "warmup_iterations": N_WARMUP,
                "measured_iterations": N_REPEATS,
                "output_tokens": 1,
            }
        )
        print(
            f"tokens={tokens} ttft_mean_ms={mean_ttft:.2f} "
            f"layer_ms={mean_ttft / n_layers:.3f}",
            flush=True,
        )
    reference = next(item for item in points if item["input_tokens"] == 2048)
    for item in points:
        item["linear_scale_from_this_2048_ms"] = (
            reference["layer_time_ms"] * item["input_tokens"] / 2048
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "e2_layer_window_probe",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_path": model_path,
        "config": "dynaexq/configs/qwen30b_uniform_low_a6000.yaml",
        "policy": "uniform_low",
        "high_precision_ratio": 0.0,
        "note": (
            "Resident low-tier prefill. This is the compute overlap window, "
            "not the offload-inflated TTFT in qwen30b_dynaexq_bs1.json."
        ),
        "initialization_pool_bytes": initialization.get("pool_allocation_bytes"),
        "points": points,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
