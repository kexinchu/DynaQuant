#!/usr/bin/env python3
"""
Benchmark Qwen3 prefill/decode latency and throughput across preset scenarios.

This script replays calibration prompts (default: wikitext2_128x2048.jsonl)
through vLLM and records:
  * Prefill latency (seconds)      – a proxy for TTFT
  * Prefill throughput (tokens/s)
  * Decode throughput (tokens/s)
  * TPOP (ms/token)

Scenarios covered by default:
  1. Qwen3-30B-A3B FP16 (TP=2)
  2. Qwen3-30B-A3B Int4 (single GPU)
  3. Qwen3-80B-A3B Int4 (TP=2)
  4. Qwen3-80B-A3B Int2 (single GPU)

Example
-------
python scripts/benchmark_latency.py \
  --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl \
  --qwen30-fp16 /workspace/Models/Qwen3-30B-A3B-Instruct-2507 \
  --qwen30-int4 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound \
  --qwen80-int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
  --qwen80-int2 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
  --max-prompts 8 \
  --max-new-tokens 256 \
  --output scripts/results/latency_summary.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    from vllm import LLM, SamplingParams
except ImportError as exc:  # pragma: no cover - surfaced to user at runtime
    raise ImportError(
        "benchmark_latency.py requires vLLM. Install via `pip install vllm`."
    ) from exc

try:  # vLLM ≥ 0.6 (V1 logging)
    from vllm.v1.metrics.loggers import StatLoggerBase as VLLMStatLoggerBase  # type: ignore
    VLLM_USES_V1_LOGGER = True
except ImportError:  # pragma: no cover - fallback for older versions
    from vllm.engine.metrics import StatLoggerBase as VLLMStatLoggerBase  # type: ignore
    VLLM_USES_V1_LOGGER = False


LOGGER = logging.getLogger("dynaexq.latency_benchmark")


@dataclass
class PromptSample:
    """Lightweight wrapper for dataset entries."""

    sample_id: str
    text: str


@dataclass
class Scenario:
    """Describe a single benchmark configuration."""

    name: str
    model_path: str
    tensor_parallel: int
    dtype: str = "auto"
    quantization: Optional[str] = None


@dataclass
class RequestMeasurement:
    """Per-request latency/throughput statistics."""

    sample_id: str
    prompt_tokens: int
    decode_tokens: int
    prefill_latency: float
    prefill_throughput: float
    decode_throughput: float
    tpop: float


if VLLM_USES_V1_LOGGER:

    class _EngineBenchmarkLogger(VLLMStatLoggerBase):
        """Per-engine logger that captures finished-request stats."""

        def __init__(self, vllm_config, engine_index: int = 0) -> None:
            self.engine_index = engine_index
            self.reset()

        def reset(self) -> None:
            self._ttft: List[float] = []
            self._tpot: List[float] = []
            self._prompt_tokens: List[int] = []
            self._decode_tokens: List[int] = []

        def record(self, scheduler_stats, iteration_stats, engine_idx: int = 0):
            if iteration_stats is None:
                return
            self._ttft.extend(iteration_stats.time_to_first_tokens_iter)
            self._tpot.extend(iteration_stats.inter_token_latencies_iter)
            for finished in iteration_stats.finished_requests:
                self._prompt_tokens.append(finished.num_prompt_tokens)
                self._decode_tokens.append(finished.num_generation_tokens)

        def log_engine_initialized(self):
            return

        def consume(self) -> Dict[str, List[float]]:
            payload = {
                "ttft": list(self._ttft),
                "tpot": list(self._tpot),
                "prompt_tokens": list(self._prompt_tokens),
                "decode_tokens": list(self._decode_tokens),
            }
            self.reset()
            return payload

    class BenchmarkStatLogger:
        """Aggregate stats across all engine shards via attached loggers."""

        def __init__(self, llm_engine) -> None:
            manager = getattr(llm_engine, "logger_manager", None)
            if manager is None:
                raise RuntimeError(
                    "vLLM stats logging is disabled. Re-create LLM with "
                    "`disable_log_stats=False` to collect benchmark metrics."
                )

            self._loggers: List[_EngineBenchmarkLogger] = []
            vllm_config = llm_engine.vllm_config
            for engine_idx, loggers in manager.per_engine_logger_dict.items():
                logger = _EngineBenchmarkLogger(vllm_config, engine_idx)
                loggers.append(logger)
                self._loggers.append(logger)

        def reset(self) -> None:
            for logger in self._loggers:
                logger.reset()

        def consume(self) -> Dict[str, List[float]]:
            payload = {
                "ttft": [],
                "tpot": [],
                "prompt_tokens": [],
                "decode_tokens": [],
            }
            for logger in self._loggers:
                stats = logger.consume()
                for key in payload:
                    payload[key].extend(stats[key])
            return payload


else:

    class BenchmarkStatLogger(VLLMStatLoggerBase):
        """Collect per-iteration stats directly from legacy vLLM."""

        def __init__(self) -> None:
            super().__init__(local_interval=1e9)  # disable periodic stdout logging
            self.reset()

        def info(self, type: str, obj) -> None:  # pragma: no cover - no-op hook
            return

        def reset(self) -> None:
            self._ttft: List[float] = []
            self._tpot: List[float] = []
            self._prompt_tokens: List[int] = []
            self._decode_tokens: List[int] = []

        def log(self, stats) -> None:  # type: ignore[override]
            self._ttft.extend(stats.time_to_first_tokens_iter)
            self._tpot.extend(stats.time_per_output_tokens_iter)
            self._prompt_tokens.extend(stats.num_prompt_tokens_requests)
            self._decode_tokens.extend(stats.num_generation_tokens_requests)

        def consume(self) -> Dict[str, List[float]]:
            payload = {
                "ttft": list(self._ttft),
                "tpot": list(self._tpot),
                "prompt_tokens": list(self._prompt_tokens),
                "decode_tokens": list(self._decode_tokens),
            }
            self.reset()
            return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3 prefill/decode latency using vLLM.",
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
        "--max-context-length",
        type=int,
        default=4096,
        help="Maximum context length to allocate in vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization hint (0-1).",
    )
    parser.add_argument(
        "--swap-space-gb",
        type=int,
        default=4,
        help="CPU swap space (GB) per GPU for vLLM KV cache spills.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to vLLM when loading models.",
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
        "--show-progress",
        action="store_true",
        help="Enable vLLM tqdm progress bars for each request.",
    )
    # Scenario-specific arguments
    parser.add_argument("--qwen30-fp16", type=str, default=None,
                        help="Path or repo id for Qwen3-30B-A3B FP16 weights.")
    parser.add_argument("--qwen30-fp16-quant", type=str, default=None,
                        help="Optional quantization flag for the FP16 baseline.")
    parser.add_argument("--qwen30-int4", type=str, default=None,
                        help="Path or repo id for Qwen3-30B-A3B Int4 weights.")
    parser.add_argument("--qwen30-int4-quant", type=str, default="awq",
                        help="Quantization flag for the 30B Int4 checkpoint.")
    parser.add_argument("--qwen80-int4", type=str, default=None,
                        help="Path or repo id for Qwen3-80B-A3B Int4 weights.")
    parser.add_argument("--qwen80-int4-quant", type=str, default="awq",
                        help="Quantization flag for the 80B Int4 checkpoint.")
    parser.add_argument("--qwen80-int2", type=str, default=None,
                        help="Path or repo id for Qwen3-80B-A3B Int2 weights.")
    parser.add_argument("--qwen80-int2-quant", type=str, default=None,
                        help="Quantization flag for the 80B Int2 checkpoint.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging verbosity.",
    )
    return parser.parse_args()


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
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed JSON line: %s", line[:80])
                continue
            prompt = obj.get(prompt_key)
            if not prompt:
                LOGGER.warning("Dataset entry missing key '%s': %s",
                               prompt_key, line[:80])
                continue
            sample_id = str(obj.get("id", f"sample_{len(prompts):04d}"))
            prompts.append(PromptSample(sample_id=sample_id, text=str(prompt)))

    if len(prompts) < required:
        raise ValueError(
            f"Dataset {dataset_path} only yielded {len(prompts)} prompts; "
            f"{required} required (max_prompts + warmup_prompts)."
        )

    LOGGER.info("Loaded %d prompts from %s",
                len(prompts), dataset_path.resolve())
    return prompts


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


def build_measurement(
    sample: PromptSample,
    stats_payload: Dict[str, List[float]],
    request_output,
) -> Optional[RequestMeasurement]:
    prompt_tokens = sum(int(t) for t in stats_payload.get("prompt_tokens", []))
    decode_tokens = sum(int(t) for t in stats_payload.get("decode_tokens", []))
    ttft = sum(stats_payload.get("ttft", []))
    decode_times = stats_payload.get("tpot", [])

    if request_output is not None:
        prompt_tokens = len(request_output.prompt_token_ids)
        decode_tokens = sum(len(output.token_ids)
                            for output in request_output.outputs)
        metrics = request_output.metrics
        if metrics is not None and metrics.first_token_time and metrics.arrival_time:
            ttft = metrics.first_token_time - metrics.arrival_time
        if metrics is not None and metrics.finished_time and metrics.first_token_time:
            decode_times = [
                metrics.finished_time - metrics.first_token_time
            ]

    if ttft <= 0 or prompt_tokens <= 0:
        LOGGER.warning(
            "Missing TTFT or prompt tokens for sample %s",
            sample.sample_id,
        )
        return None

    prefill_throughput = prompt_tokens / ttft

    decode_time = sum(decode_times)
    if decode_tokens > 0 and decode_time > 0:
        decode_throughput = decode_tokens / decode_time
        tpop = (decode_time / decode_tokens) * 1000.0
    else:
        decode_throughput = 0.0
        tpop = 0.0

    return RequestMeasurement(
        sample_id=sample.sample_id,
        prompt_tokens=prompt_tokens,
        decode_tokens=decode_tokens,
        prefill_latency=ttft,
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
        "tensor_parallel": scenario.tensor_parallel,
        "dtype": scenario.dtype,
        "quantization": scenario.quantization,
        "samples": len(measurements),
        "prefill_latency_avg": statistics.fmean(prefill_latencies)
        if prefill_latencies else None,
        "prefill_latency_p95": percentile(prefill_latencies, 0.95),
        "prefill_throughput_avg": statistics.fmean(prefill_throughputs)
        if prefill_throughputs else None,
        "decode_throughput_avg": statistics.fmean(decode_throughputs)
        if decode_throughputs else None,
        "tpop_avg": statistics.fmean(tpops) if tpops else None,
        "tpop_p95": percentile(tpops, 0.95),
        "avg_decode_tokens": statistics.fmean(decode_tokens)
        if decode_tokens else None,
    }
    return summary


def run_scenario(
    scenario: Scenario,
    *,
    prompts: Sequence[PromptSample],
    args: argparse.Namespace,
) -> Dict[str, object]:
    LOGGER.info(
        "Booting %s (TP=%d, dtype=%s, quant=%s)",
        scenario.name,
        scenario.tensor_parallel,
        scenario.dtype,
        scenario.quantization or "<none>",
    )
    llm = LLM(
        model=scenario.model_path,
        tensor_parallel_size=scenario.tensor_parallel,
        dtype=scenario.dtype,
        quantization=scenario.quantization,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space_gb,
        max_model_len=args.max_context_length,
        disable_log_stats=False,
    )
    if VLLM_USES_V1_LOGGER:
        stat_logger = BenchmarkStatLogger(llm.llm_engine)
    else:  # pragma: no cover - legacy vLLM fallback
        stat_logger = BenchmarkStatLogger()
        llm.llm_engine.add_logger("benchmark_latency", stat_logger)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    warmup = args.warmup_prompts
    max_total = warmup + args.max_prompts
    measurements: List[RequestMeasurement] = []
    failures: List[str] = []

    try:
        for idx, sample in enumerate(prompts[:max_total]):
            stat_logger.reset()
            try:
                request_outputs = llm.generate(
                    sample.text,
                    sampling_params=sampling_params,
                    use_tqdm=args.show_progress,
                )
            except Exception as exc:  # pragma: no cover - execution-time failure path
                LOGGER.exception("Generation failed for %s: %s",
                                 sample.sample_id, exc)
                failures.append(sample.sample_id)
                continue

            stats_payload = stat_logger.consume()
            request_output = request_outputs[0] if request_outputs else None
            measurement = build_measurement(
                sample, stats_payload, request_output)
            if measurement is None:
                failures.append(sample.sample_id)
                continue

            if idx < warmup:
                LOGGER.debug("Warm-up sample %s completed", sample.sample_id)
                continue

            measurements.append(measurement)
    finally:
        if not VLLM_USES_V1_LOGGER:
            llm.llm_engine.remove_logger("benchmark_latency")
        del llm
        gc.collect()

    summary = summarize_measurements(scenario, measurements)
    summary["warmup_prompts"] = warmup
    summary["failures"] = failures
    if args.keep_details:
        summary["details"] = [asdict(m) for m in measurements]

    LOGGER.info(
        "%s → prefill %.2f tok/s, decode %.2f tok/s, TTFT %.2f s (p95 %.2f), "
        "TPOP %.2f ms (samples=%d, failures=%d)",
        scenario.name,
        summary.get("prefill_throughput_avg") or 0.0,
        summary.get("decode_throughput_avg") or 0.0,
        summary.get("prefill_latency_avg") or 0.0,
        summary.get("prefill_latency_p95") or 0.0,
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
                name="Qwen3-30B-A3B FP16 (TP=2)",
                model_path=args.qwen30_fp16,
                tensor_parallel=2,
                dtype="float16",
                quantization=args.qwen30_fp16_quant,
            )
        )
    if args.qwen30_int4:
        scenarios.append(
            Scenario(
                name="Qwen3-30B-A3B Int4 (cuda:0)",
                model_path=args.qwen30_int4,
                tensor_parallel=1,
                dtype="float16",
                quantization=args.qwen30_int4_quant,
            )
        )
    if args.qwen80_int4:
        scenarios.append(
            Scenario(
                name="Qwen3-80B-A3B Int4 (TP=2)",
                model_path=args.qwen80_int4,
                tensor_parallel=2,
                dtype="float16",
                quantization=args.qwen80_int4_quant,
            )
        )
    if args.qwen80_int2:
        scenarios.append(
            Scenario(
                name="Qwen3-80B-A3B Int2 (cuda:0)",
                model_path=args.qwen80_int2,
                tensor_parallel=1,
                dtype="float16",
                quantization=args.qwen80_int2_quant,
            )
        )

    if not scenarios:
        raise ValueError(
            "No benchmark scenarios configured. Provide at least one model path "
            "via --qwen30-fp16/--qwen30-int4/--qwen80-int4/--qwen80-int2."
        )

    return scenarios


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
