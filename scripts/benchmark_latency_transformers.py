#!/usr/bin/env python3
"""
Benchmark Qwen3 latency/throughput using vanilla Hugging Face Transformers.

This alternative to `benchmark_latency.py` avoids vLLM and therefore supports
custom AutoRound Int2 checkpoints (or any HF-compatible weights).  It replays
prompts from a dataset, measures:

  * Prefill latency (seconds) – forward pass over the prompt (TTFT proxy)
  * Prefill throughput (tokens / second)
  * Decode throughput (tokens / second) for tokens generated after the first
  * Time per output token (ms/token)

The CLI mirrors the vLLM benchmark so you can selectively run any combination
of FP/Int4/Int2 checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger("dynaexq.transformers_latency")


@dataclass
class PromptSample:
    sample_id: str
    text: str


@dataclass
class Scenario:
    name: str
    model_path: str
    dtype: str = "auto"
    device: str = "auto"
    trust_remote_code: bool = False


@dataclass
class RequestMeasurement:
    sample_id: str
    prompt_tokens: int
    decode_tokens: int
    prefill_latency: float
    prefill_throughput: float
    decode_throughput: float
    tpop: float


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> Optional[torch.dtype]:
    if dtype_arg == "auto":
        if device.type == "cuda":
            return torch.float16
        return torch.float32

    normalized = dtype_arg.lower()
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float8", "fp8"}:
        return torch.float8_e5m2
    if normalized in {"int8"}:
        return torch.int8
    if normalized in {"int4"}:
        return torch.int4 if hasattr(torch, "int4") else None
    return getattr(torch, dtype_arg, None)


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def load_prompts(
    dataset_path: Path,
    prompt_key: str,
    required: int,
) -> List[PromptSample]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    prompts: List[PromptSample] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(prompts) >= required:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                LOGGER.warning(
                    "Skipping malformed JSON line: %s", stripped[:120])
                continue
            prompt = obj.get(prompt_key)
            if not prompt:
                LOGGER.warning(
                    "Dataset entry missing key '%s': %s", prompt_key, stripped[:120]
                )
                continue
            sample_id = str(obj.get("id", f"sample_{len(prompts):04d}"))
            prompts.append(PromptSample(sample_id=sample_id, text=str(prompt)))

    if len(prompts) < required:
        raise ValueError(
            f"Dataset {dataset_path} only yielded {len(prompts)} prompts; "
            f"{required} required (max_prompts + warmup_prompts)."
        )

    LOGGER.info("Loaded %d prompts from %s", len(
        prompts), dataset_path.resolve())
    return prompts


def load_model_and_tokenizer(
    model_path: str,
    *,
    dtype: Optional[torch.dtype],
    device: torch.device,
    trust_remote_code: bool,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    LOGGER.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading model from %s (%s, device=%s)",
                model_path, dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=None,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if not 0 <= q <= 1:
        raise ValueError("Percentile q must be within [0, 1].")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * q
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return float(ordered[lower])
    fraction = idx - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def measure_prompt(
    *,
    model,
    tokenizer,
    prompt: PromptSample,
    device: torch.device,
    max_new_tokens: int,
    stop_on_eos: bool,
) -> Optional[RequestMeasurement]:
    encoded = tokenizer(
        prompt.text,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_tokens = encoded["input_ids"].shape[1]

    maybe_sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            use_cache=True,
        )
    maybe_sync(device)
    prefill_latency = time.perf_counter() - start

    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]

    generated_tokens: List[int] = []
    decode_durations: List[float] = []
    next_token = torch.argmax(logits, dim=-1)
    generated_tokens.append(int(next_token.item()))

    for _ in range(max_new_tokens - 1):
        if (
            stop_on_eos
            and tokenizer.eos_token_id is not None
            and generated_tokens[-1] == tokenizer.eos_token_id
        ):
            break

        input_ids = torch.tensor([[generated_tokens[-1]]], device=device)
        maybe_sync(device)
        decode_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
        maybe_sync(device)
        decode_durations.append(time.perf_counter() - decode_start)

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
        generated_tokens.append(int(next_token.item()))

    decode_tokens = len(generated_tokens)
    timed_decode_tokens = len(decode_durations)
    decode_time = sum(decode_durations)

    if prompt_tokens == 0 or prefill_latency <= 0:
        LOGGER.warning(
            "Prompt %s produced invalid prefill metrics (tokens=%d, latency=%.4f)",
            prompt.sample_id,
            prompt_tokens,
            prefill_latency,
        )
        return None

    prefill_throughput = prompt_tokens / prefill_latency
    if timed_decode_tokens > 0 and decode_time > 0:
        decode_throughput = timed_decode_tokens / decode_time
        tpop = (decode_time / timed_decode_tokens) * 1000.0
    else:
        decode_throughput = 0.0
        tpop = 0.0

    return RequestMeasurement(
        sample_id=prompt.sample_id,
        prompt_tokens=prompt_tokens,
        decode_tokens=decode_tokens,
        prefill_latency=prefill_latency,
        prefill_throughput=prefill_throughput,
        decode_throughput=decode_throughput,
        tpop=tpop,
    )


def summarize_measurements(
    scenario: Scenario,
    measurements: Sequence[RequestMeasurement],
) -> Dict[str, object]:
    prefill_latencies = [m.prefill_latency for m in measurements]
    prefill_throughputs = [m.prefill_throughput for m in measurements]
    decode_throughputs = [m.decode_throughput for m in measurements]
    tpops = [m.tpop for m in measurements if m.tpop > 0]
    decode_tokens = [m.decode_tokens for m in measurements]

    summary: Dict[str, object] = {
        "scenario": scenario.name,
        "model_path": scenario.model_path,
        "dtype": scenario.dtype,
        "device": scenario.device,
        "samples": len(measurements),
        "prefill_latency_avg": statistics.fmean(prefill_latencies)
        if prefill_latencies
        else None,
        "prefill_latency_p95": percentile(prefill_latencies, 0.95),
        "prefill_throughput_avg": statistics.fmean(prefill_throughputs)
        if prefill_throughputs
        else None,
        "decode_throughput_avg": statistics.fmean(decode_throughputs)
        if decode_throughputs
        else None,
        "tpop_avg": statistics.fmean(tpops) if tpops else None,
        "tpop_p95": percentile(tpops, 0.95),
        "avg_decode_tokens": statistics.fmean(decode_tokens)
        if decode_tokens
        else None,
    }
    return summary


def run_scenario(
    scenario: Scenario,
    *,
    prompts: Sequence[PromptSample],
    args: argparse.Namespace,
) -> Dict[str, object]:
    device = resolve_device(scenario.device)
    dtype = resolve_dtype(scenario.dtype, device)
    model, tokenizer = load_model_and_tokenizer(
        scenario.model_path,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )

    warmup = args.warmup_prompts
    max_total = warmup + args.max_prompts
    measurements: List[RequestMeasurement] = []
    failures: List[str] = []

    try:
        for idx, sample in enumerate(prompts[:max_total]):
            measurement = measure_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=sample,
                device=device,
                max_new_tokens=args.max_new_tokens,
                stop_on_eos=not args.disable_eos_stop,
            )
            if measurement is None:
                failures.append(sample.sample_id)
                continue

            if idx < warmup:
                LOGGER.debug("Warm-up sample %s completed", sample.sample_id)
                continue

            measurements.append(measurement)
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize_measurements(scenario, measurements)
    summary["warmup_prompts"] = warmup
    summary["failures"] = failures
    if args.keep_details:
        summary["details"] = [asdict(m) for m in measurements]

    LOGGER.info(
        "%s → prefill %.2f tok/s, decode %.2f tok/s, TPOP %.2f ms "
        "(samples=%d, failures=%d)",
        scenario.name,
        summary.get("prefill_throughput_avg") or 0.0,
        summary.get("decode_throughput_avg") or 0.0,
        summary.get("tpop_avg") or 0.0,
        len(measurements),
        len(failures),
    )
    return summary


def build_scenarios(args: argparse.Namespace) -> List[Scenario]:
    scenarios: List[Scenario] = []

    if args.qwen30_fp16:
        scenarios.append(
            Scenario(
                name="Qwen3-30B-A3B FP16 (transformers)",
                model_path=args.qwen30_fp16,
                dtype=args.qwen30_fp16_dtype,
                device=args.device,
                trust_remote_code=args.trust_remote_code,
            )
        )
    if args.qwen30_int4:
        scenarios.append(
            Scenario(
                name="Qwen3-30B-A3B Int4 (transformers)",
                model_path=args.qwen30_int4,
                dtype=args.qwen30_int4_dtype,
                device=args.device,
                trust_remote_code=args.trust_remote_code,
            )
        )
    if args.qwen80_int4:
        scenarios.append(
            Scenario(
                name="Qwen3-80B-A3B Int4 (transformers)",
                model_path=args.qwen80_int4,
                dtype=args.qwen80_int4_dtype,
                device=args.device,
                trust_remote_code=args.trust_remote_code,
            )
        )
    if args.qwen80_int2:
        scenarios.append(
            Scenario(
                name="Qwen3-80B-A3B Int2 (transformers)",
                model_path=args.qwen80_int2,
                dtype=args.qwen80_int2_dtype,
                device=args.device,
                trust_remote_code=args.trust_remote_code,
            )
        )

    if not scenarios:
        raise ValueError(
            "No benchmark scenarios configured. Provide at least one model path "
            "via --qwen30-fp16/--qwen30-int4/--qwen80-int4/--qwen80-int2."
        )

    return scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3 latency with HF Transformers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("calibration_datasets/requests/wikitext2_128x2048.jsonl"),
        help="JSONL dataset containing prompt entries.",
    )
    parser.add_argument(
        "--prompt-key",
        type=str,
        default="prompt",
        help="Field to read from JSONL objects.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=8,
        help="Number of prompts (excluding warm-up) to benchmark per scenario.",
    )
    parser.add_argument(
        "--warmup-prompts",
        type=int,
        default=1,
        help="Number of prompts to run (and discard) before recording metrics.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum decode tokens per request.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device to place models/tensors on (e.g., 'cuda:0').",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoModel/AutoTokenizer.",
    )
    parser.add_argument(
        "--disable-eos-stop",
        action="store_true",
        help="Do not stop decoding early when EOS is generated.",
    )
    parser.add_argument(
        "--keep-details",
        action="store_true",
        help="Include per-sample metrics in the JSON summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save JSON results.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging verbosity.",
    )
    # Scenario-specific arguments reuse the vLLM CLI shape but allow dtype overrides.
    parser.add_argument("--qwen30-fp16", type=str, default=None,
                        help="Path or repo id for Qwen3-30B-A3B FP16 weights.")
    parser.add_argument("--qwen30-fp16-dtype", type=str, default="float16",
                        help="Torch dtype hint for the FP16 baseline.")
    parser.add_argument("--qwen30-int4", type=str, default=None,
                        help="Path or repo id for Qwen3-30B-A3B Int4 weights.")
    parser.add_argument("--qwen30-int4-dtype", type=str, default="float16",
                        help="Torch dtype hint for the 30B Int4 checkpoint.")
    parser.add_argument("--qwen80-int4", type=str, default=None,
                        help="Path or repo id for Qwen3-80B-A3B Int4 weights.")
    parser.add_argument("--qwen80-int4-dtype", type=str, default="float16",
                        help="Torch dtype hint for the 80B Int4 checkpoint.")
    parser.add_argument("--qwen80-int2", type=str, default=None,
                        help="Path or repo id for Qwen3-80B-A3B Int2 weights.")
    parser.add_argument("--qwen80-int2-dtype", type=str, default="auto",
                        help="Torch dtype hint for the 80B Int2 checkpoint.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive.")
    if args.warmup_prompts < 0:
        raise ValueError("--warmup-prompts cannot be negative.")

    total_required = args.max_prompts + args.warmup_prompts
    prompts = load_prompts(
        args.dataset,
        args.prompt_key,
        total_required,
    )

    summaries: List[Dict[str, object]] = []
    for scenario in build_scenarios(args):
        LOGGER.info("Running scenario: %s", scenario.name)
        summary = run_scenario(scenario, prompts=prompts, args=args)
        summaries.append(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            summaries, indent=2), encoding="utf-8")
        LOGGER.info("Wrote results to %s", args.output.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
