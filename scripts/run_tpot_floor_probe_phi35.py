#!/usr/bin/env python3
"""Measure resident low-tier decode TPOT for Phi-3.5-MoE on one A6000.

Same definition as the Qwen3-30B floor probe: uniform-low, scheduler off,
greedy decode, model_tpot_ms = (e2e - ttft) / (output_tokens - 1).
The artifact does not overwrite a paper table.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper/a6000_wikitext_replay/tpot_floor_probe_phi35.json"
MODEL_PATH = "/home/kec23008/Models/Phi-3.5-MoE-instruct"
CONFIG = "dynaexq/configs/phi35_moe.yaml"
INPUT_TOKENS = 128
OUTPUT_TOKENS = 32
N_WARMUP = 1
N_REPEATS = 5
BATCHES = (1, 8)


def _gpu_state(index: int) -> tuple[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    used, util = [int(part.strip()) for part in result.stdout.strip().split(",")]
    return used, util


def _wait_for_weights() -> None:
    root = Path(MODEL_PATH)
    while True:
        weights = list(root.glob("*.safetensors")) + list(root.glob("model.safetensors.index.json"))
        nbytes = sum(path.stat().st_size for path in root.glob("*.safetensors"))
        print(
            json.dumps(
                {
                    "stage": "wait_weights",
                    "safetensors": len(list(root.glob("*.safetensors"))),
                    "bytes": nbytes,
                    "index": (root / "model.safetensors.index.json").exists(),
                }
            ),
            flush=True,
        )
        if (root / "model.safetensors.index.json").exists() and nbytes > 70 * 1024**3:
            return
        if not weights and not root.exists():
            pass
        time.sleep(20)


def _wait_for_idle_gpu() -> int:
    preferred = (1, 0)
    streak = {index: 0 for index in preferred}
    while True:
        for index in preferred:
            used, util = _gpu_state(index)
            if used <= 1024 and util == 0:
                streak[index] += 1
            else:
                streak[index] = 0
            print(
                json.dumps(
                    {
                        "stage": "wait_idle",
                        "gpu": index,
                        "used_mib": used,
                        "util": util,
                        "streak": streak[index],
                    }
                ),
                flush=True,
            )
            if streak[index] >= 4:
                return index
        time.sleep(10)


def _point(benchmark: dict, batch_size: int) -> dict:
    tpot = benchmark["metrics"]["model_tpot_ms"]
    mean_tpot_ms = float(tpot["mean"])
    decode_tok_s = (
        batch_size / (mean_tpot_ms / 1000.0) if mean_tpot_ms > 0 else 0.0
    )
    return {
        "batch_size": batch_size,
        "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "decode_steps": OUTPUT_TOKENS - 1,
        "warmup_iterations": N_WARMUP,
        "measured_iterations": N_REPEATS,
        "model_tpot_ms": tpot,
        "model_ttft_ms": benchmark["metrics"]["model_ttft_ms"],
        "model_e2e_ms": benchmark["metrics"]["model_e2e_ms"],
        "e2e_throughput_tokens_s": benchmark["metrics"]["throughput_tokens_s"],
        "decode_throughput_tokens_s_from_mean_tpot": decode_tok_s,
        "peak_allocated_bytes": benchmark["metrics"]["peak_allocated_bytes"],
        "status": "measured",
    }


def main() -> None:
    _wait_for_weights()
    physical_gpu = _wait_for_idle_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    print(json.dumps({"stage": "gpu_selected", "gpu": physical_gpu}), flush=True)

    import torch

    from dynaexq.core.config import DynaExqConfig
    from dynaexq.experiments.eval_perf import measure_latency
    from dynaexq.experiments.run_shift import initialize_dynaexq, load_model
    from dynaexq.integration.moe_wrapper import MoEWrapper
    from dynaexq.policy import RuntimeExchangePlanner

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("TPOT probe requires a CUDA device")
    config = DynaExqConfig.from_yaml(ROOT / CONFIG)
    model_path = os.environ.get("TPOT_MODEL", MODEL_PATH)
    print("loading model", flush=True)
    model, tokenizer = load_model(model_path, device)
    print("initializing uniform-low runtime", flush=True)
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
    points = []
    for batch_size in BATCHES:
        print(f"measure batch={batch_size}", flush=True)
        try:
            benchmark = measure_latency(
                wrapper,
                tokenizer,
                batch_size=batch_size,
                input_length=INPUT_TOKENS,
                output_length=OUTPUT_TOKENS,
                n_warmup=N_WARMUP,
                n_repeats=N_REPEATS,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            points.append(
                {
                    "batch_size": batch_size,
                    "status": "oom",
                    "error": traceback.format_exc(limit=2),
                }
            )
            print(f"batch={batch_size} oom", flush=True)
            continue
        point = _point(benchmark, batch_size)
        points.append(point)
        print(
            f"batch={batch_size} tpot_mean_ms={point['model_tpot_ms']['mean']:.2f} "
            f"tpot_p95_ms={point['model_tpot_ms']['p95']:.2f} "
            f"decode_tok_s={point['decode_throughput_tokens_s_from_mean_tpot']:.2f}",
            flush=True,
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "tpot_floor_probe",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_path": model_path,
        "config": CONFIG,
        "policy": "uniform_low",
        "high_precision_ratio": 0.0,
        "scheduler_enabled": False,
        "definition": (
            "model_tpot_ms is (e2e - ttft) / (output_tokens - 1) inside "
            "one greedy request. It is not a 1,000-request p95 and it is "
            "not the Table 4 cell."
        ),
        "initialization_pool_bytes": initialization.get("pool_allocation_bytes"),
        "points": points,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
