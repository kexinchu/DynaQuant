"""Model-level latency benchmark with raw samples and provenance metadata.

This benchmark measures an isolated model, not a serving stack: queueing,
tokenization, network transport, and response serialization are excluded.
The distinction is recorded in every artifact and must be preserved in paper
captions.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from .gpu_memory import NvmlProcessMemoryMonitor
from .eval_quality import (
    SCHEMA_VERSION,
    autoround_load_config,
    checkpoint_metadata,
    environment_metadata,
)


def percentile(values: list[float], quantile: float) -> float:
    """Nearest-rank percentile; valid for small and large sample sets."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def fixed_length_inputs(
    tokenizer,
    text: str,
    *,
    input_length: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create exact-length, non-padded inputs for repeatable prefill timing."""
    if input_length <= 0 or batch_size <= 0:
        raise ValueError("input_length and batch_size must be positive")
    payload = tokenizer.encode(text, add_special_tokens=False)
    if not payload:
        raise ValueError("input_text tokenizes to an empty sequence")
    prefix = []
    if tokenizer.bos_token_id is not None:
        prefix.append(tokenizer.bos_token_id)
    remaining = input_length - len(prefix)
    if remaining < 0:
        prefix = prefix[:input_length]
        remaining = 0
    repeated = list(itertools.islice(itertools.cycle(payload), remaining))
    one = torch.tensor(prefix + repeated, dtype=torch.long)
    input_ids = one.unsqueeze(0).repeat(batch_size, 1)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_memory() -> None:
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)


def _peak_memory_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return sum(
        torch.cuda.max_memory_allocated(index)
        for index in range(torch.cuda.device_count())
    )


def _peak_reserved_memory_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return sum(
        torch.cuda.max_memory_reserved(index)
        for index in range(torch.cuda.device_count())
    )


def _model_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _one_generation(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    output_length: int,
    process_memory_monitor: NvmlProcessMemoryMonitor | None = None,
) -> dict[str, float]:
    if output_length <= 0:
        raise ValueError("output_length must be positive")

    _sync_cuda()
    _reset_peak_memory()
    if process_memory_monitor is not None:
        process_memory_monitor.start()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat((input_ids, next_token), dim=1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_token)), dim=1
            )
            past = outputs.past_key_values
            _sync_cuda()
            first_token_at = time.perf_counter()

            for _ in range(output_length - 1):
                # The model helper supplies architecture-specific
                # position/cache arguments and slices ``generated`` to the
                # uncached token.
                prepared = model.prepare_inputs_for_generation(
                    generated,
                    past_key_values=past,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                outputs = model(**prepared)
                next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
                past = outputs.past_key_values
                generated = torch.cat((generated, next_token), dim=1)
                attention_mask = torch.cat(
                    (attention_mask, torch.ones_like(next_token)), dim=1
                )

        _sync_cuda()
        finished = time.perf_counter()
    finally:
        process_memory = (
            process_memory_monitor.stop()
            if process_memory_monitor is not None
            else {}
        )

    ttft_ms = (first_token_at - started) * 1000.0
    e2e_ms = (finished - started) * 1000.0
    decode_tokens_after_first = max(output_length - 1, 0)
    tpot_ms = (
        (e2e_ms - ttft_ms) / decode_tokens_after_first
        if decode_tokens_after_first
        else 0.0
    )
    batch_size = input_ids.shape[0]
    return {
        "model_ttft_ms": ttft_ms,
        "model_tpot_ms": tpot_ms,
        "model_e2e_ms": e2e_ms,
        "throughput_tokens_s": batch_size * output_length / ((finished - started)),
        "peak_allocated_bytes": float(_peak_memory_bytes()),
        "peak_reserved_bytes": float(_peak_reserved_memory_bytes()),
        **process_memory,
    }


def _summarize(samples: list[dict[str, float]], key: str) -> dict[str, float]:
    values = [sample[key] for sample in samples]
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def measure_latency(
    model: torch.nn.Module,
    tokenizer,
    *,
    input_text: str = "The quick brown fox jumps over the lazy dog. ",
    batch_size: int = 1,
    input_length: int = 128,
    output_length: int = 64,
    n_warmup: int = 5,
    n_repeats: int = 100,
    iteration_setup: Callable[[torch.Tensor], None] | None = None,
    after_warmup: Callable[[], None] | None = None,
    input_device: torch.device | str | None = None,
    require_process_hbm_monitor: bool = False,
) -> dict[str, Any]:
    """Measure prefill/decode latency using exact-length, non-padded inputs.

    ``iteration_setup`` supports pinned external runtimes that must attach
    request-local tracing state before every forward.  It is deliberately
    called outside the timed region.  ``after_warmup`` lets those adapters
    reset their own counters so formal telemetry covers measured iterations
    only.  Native DynaExQ and Transformers runs leave both callbacks unset.
    """
    if n_warmup < 0 or n_repeats <= 0:
        raise ValueError("n_warmup must be non-negative and n_repeats positive")
    model.eval()
    input_ids, attention_mask = fixed_length_inputs(
        tokenizer,
        input_text,
        input_length=input_length,
        batch_size=batch_size,
    )
    device = (
        torch.device(input_device)
        if input_device is not None
        else _model_input_device(model)
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    process_memory_monitor = None
    if require_process_hbm_monitor:
        if device.type != "cuda":
            raise ValueError(
                "formal process-HBM monitoring requires a CUDA input device"
            )
        device_index = (
            torch.cuda.current_device()
            if device.index is None
            else device.index
        )
        process_memory_monitor = NvmlProcessMemoryMonitor([device_index])

    def run_once() -> dict[str, float]:
        if iteration_setup is not None:
            iteration_setup(input_ids)
        return _one_generation(
            model,
            input_ids,
            attention_mask,
            output_length,
            process_memory_monitor,
        )

    try:
        for _ in range(n_warmup):
            run_once()
        if after_warmup is not None:
            after_warmup()

        samples = [run_once() for _ in range(n_repeats)]
    finally:
        if process_memory_monitor is not None:
            process_memory_monitor.close()
    metric_names = [
        "model_ttft_ms",
        "model_tpot_ms",
        "model_e2e_ms",
        "throughput_tokens_s",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    ]
    if require_process_hbm_monitor:
        metric_names.append("process_hbm_used_peak_bytes")
    result = {
        "scope": "isolated_model",
        "excludes": [
            "request_queueing",
            "tokenization",
            "network_transport",
            "response_serialization",
        ],
        "batch_size": batch_size,
        "input_tokens": input_length,
        "output_tokens_per_sequence": output_length,
        "warmup_iterations": n_warmup,
        "measured_iterations": n_repeats,
        "metrics": {
            key: _summarize(samples, key)
            for key in metric_names
        },
        "samples": samples,
    }
    if process_memory_monitor is not None:
        result["process_hbm_monitor"] = process_memory_monitor.metadata()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--paper-model",
        choices=("qwen30b", "qwen80b", "phi35"),
        help="Canonical manuscript model key required for claim registration",
    )
    parser.add_argument(
        "--paper-protocol",
        action="store_true",
        help=(
            "Enforce the exact TC latency grid and whole-process NVML HBM "
            "high-water monitoring"
        ),
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=("reference_fp16", "reference_bf16", "quantized_checkpoint"),
    )
    parser.add_argument("--quantization", choices=("int2", "int4"))
    parser.add_argument(
        "--autoround-backend",
        choices=("triton",),
        help=(
            "Explicit AutoRound inference backend. Required for formal "
            "quantized-checkpoint runs."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-length", type=int, default=128)
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hash-model-files", action="store_true")
    args = parser.parse_args()

    if (args.method == "quantized_checkpoint") != (args.quantization is not None):
        parser.error(
            "--quantization is required exactly when --method=quantized_checkpoint"
        )
    if args.autoround_backend is not None and args.method != "quantized_checkpoint":
        parser.error("--autoround-backend requires --method=quantized_checkpoint")
    if (
        args.paper_protocol
        and args.method == "quantized_checkpoint"
        and args.autoround_backend != "triton"
    ):
        parser.error(
            "formal quantized-checkpoint runs require "
            "--autoround-backend=triton"
        )
    if args.paper_protocol:
        if args.paper_model is None:
            parser.error("--paper-protocol requires --paper-model")
        if args.batch_size not in (1, 2, 4, 8, 16, 32):
            parser.error(
                "--paper-protocol requires batch size in 1,2,4,8,16,32"
            )
        if args.device_map not in {"cuda", "cuda:0"}:
            parser.error(
                "--paper-protocol requires --device-map=cuda or cuda:0"
            )
        expected = {
            "input_length": 2048,
            "output_length": 256,
            "n_warmup": 5,
            "n_repeats": 100,
            "seed": 42,
        }
        mismatches = [
            f"--{name.replace('_', '-')}={getattr(args, name)}"
            for name, value in expected.items()
            if getattr(args, name) != value
        ]
        if mismatches:
            parser.error(
                "paper protocol requires "
                + ", ".join(
                    f"--{name.replace('_', '-')}={value}"
                    for name, value in expected.items()
                )
                + "; got "
                + ", ".join(mismatches)
            )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.method == "reference_bf16" else torch.float16
    checkpoint = checkpoint_metadata(
        args.model,
        hash_weight_files=args.hash_model_files,
    )
    revision = checkpoint.get("revision")
    if checkpoint.get("local") is False and not revision:
        parser.error(
            "remote checkpoint revision could not be resolved; refusing an "
            "unpinned benchmark"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=revision,
        trust_remote_code=False,
    )
    quantization_config = autoround_load_config(
        args.model,
        args.autoround_backend,
    )
    model_kwargs = {}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=revision,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=False,
        **model_kwargs,
    )
    result = measure_latency(
        model,
        tokenizer,
        batch_size=args.batch_size,
        input_length=args.input_length,
        output_length=args.output_length,
        n_warmup=args.n_warmup,
        n_repeats=args.n_repeats,
        require_process_hbm_monitor=args.paper_protocol,
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "paper_model": args.paper_model,
        "paper_method": (
            "static_ptq"
            if args.method == "quantized_checkpoint"
            else "reference"
        ),
        "checkpoint": checkpoint,
        "method": args.method,
        "quantization": args.quantization,
        "inference_backend": args.autoround_backend,
        "device_map": args.device_map,
        "seed": args.seed,
        "evaluation_protocol": {
            "name": (
                "tc_isolated_performance_v2"
                if args.paper_protocol
                else "custom"
            ),
            "seed": args.seed,
            "process_hbm_high_water": args.paper_protocol,
        },
        "benchmark": result,
        "environment": environment_metadata(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({**artifact, "benchmark": {
        key: value for key, value in result.items() if key != "samples"
    }}, indent=2))


if __name__ == "__main__":
    main()
