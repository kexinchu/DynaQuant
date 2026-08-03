#!/usr/bin/env python3
"""
Performance test: TTFT, TPOP, End2End latency across different batch sizes.

Reads prompts from ShareGPT_V3_unfiltered_cleaned_split.json, fixes each request
to 512 tokens (pad if shorter, truncate if longer), and measures:
  - TTFT (Time To First Token)
  - TPOP (Time Per Output Token)
  - End2End latency

Supports model as CLI argument and sweeps over multiple batch sizes.

Example:
  python scripts/perf_test_base.py --model /path/to/Qwen3-30B-A3B-Instruct-2507 \\
    --batch-sizes 1 2 4 8 --max-new-tokens 128 --output results/perf.json
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
from typing import Any, Dict, List, Optional, Sequence

VLLM_AVAILABLE = False
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    pass

VLLM_USES_V1_LOGGER = False
if VLLM_AVAILABLE:
    try:
        from vllm.v1.metrics.loggers import StatLoggerBase as VLLMStatLoggerBase
        VLLM_USES_V1_LOGGER = True
    except ImportError:
        from vllm.engine.metrics import StatLoggerBase as VLLMStatLoggerBase

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger("perf_test_base")

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "ShareGPT_V3_unfiltered_cleaned_split.json"
FIXED_PROMPT_LENGTH = 512


@dataclass
class PromptSample:
    sample_id: str
    text: str
    token_count: int


@dataclass
class RequestMeasurement:
    sample_id: str
    prompt_tokens: int
    decode_tokens: int
    ttft_ms: float
    tpop_ms: float
    end2end_ms: float


def load_sharegpt_prompts(
    dataset_path: Path,
    tokenizer: AutoTokenizer,
    target_length: int,
    max_requests: int,
) -> List[PromptSample]:
    """
    Load prompts from ShareGPT JSON. Each request = first human message per item.
    Fix to target_length tokens: pad if shorter, truncate if longer.
    """
    LOGGER.info("Loading ShareGPT from %s (target_length=%d, max=%d)",
                dataset_path, target_length, max_requests)

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    samples: List[PromptSample] = []
    for item in data:
        if len(samples) >= max_requests:
            break
        if "conversations" not in item:
            continue
        text = None
        for conv in item["conversations"]:
            if conv.get("from") == "human" and "value" in conv:
                text = conv["value"]
                break
        if not text or not text.strip():
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < target_length:
            ids = ids + [pad_id] * (target_length - len(ids))
        else:
            ids = ids[:target_length]

        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        sample_id = str(item.get("id", f"req_{len(samples):05d}"))
        samples.append(PromptSample(sample_id=sample_id, text=decoded, token_count=len(ids)))

    LOGGER.info("Loaded %d prompts (fixed to %d tokens)", len(samples), target_length)
    return samples


if VLLM_AVAILABLE and VLLM_USES_V1_LOGGER:

    class _EngineBenchmarkLogger(VLLMStatLoggerBase):
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
        def __init__(self, llm_engine) -> None:
            manager = getattr(llm_engine, "logger_manager", None)
            if manager is None:
                raise RuntimeError(
                    "vLLM stats logging is disabled. Re-create LLM with "
                    "disable_log_stats=False to collect benchmark metrics."
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
            payload = {"ttft": [], "tpot": [], "prompt_tokens": [], "decode_tokens": []}
            for logger in self._loggers:
                stats = logger.consume()
                for key in payload:
                    payload[key].extend(stats[key])
            return payload
elif VLLM_AVAILABLE:

    class BenchmarkStatLogger(VLLMStatLoggerBase):
        def __init__(self) -> None:
            super().__init__(local_interval=1e9)
            self.reset()

        def info(self, type: str, obj) -> None:
            return

        def reset(self) -> None:
            self._ttft: List[float] = []
            self._tpot: List[float] = []
            self._prompt_tokens: List[int] = []
            self._decode_tokens: List[int] = []

        def log(self, stats) -> None:
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
else:
    BenchmarkStatLogger = None  # Not used when VLLM unavailable


def run_batch_hf(
    model,
    tokenizer,
    samples: Sequence[PromptSample],
    device: torch.device,
    max_new_tokens: int,
) -> List[RequestMeasurement]:
    """Run batched generation with HF Transformers, return measurements."""
    batch_texts = [s.text for s in samples]
    encoded = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    batch_size = input_ids.shape[0]
    prompt_tokens = attention_mask.sum(dim=1).cpu().tolist() if attention_mask is not None else [input_ids.shape[1]] * batch_size

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    sync()
    ttft_s = time.perf_counter() - t0
    ttft_ms = ttft_s * 1000.0

    past_kv = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
    generated = next_tokens.clone()
    eos_id = tokenizer.eos_token_id

    decode_times: List[float] = []
    for _ in range(max_new_tokens - 1):
        sync()
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=next_tokens, past_key_values=past_kv, use_cache=True)
        sync()
        decode_times.append(time.perf_counter() - t1)
        past_kv = out.past_key_values
        logits = out.logits[:, -1, :]
        next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tokens], dim=1)
        if eos_id is not None and (next_tokens == eos_id).all().item():
            break

    num_decode = len(decode_times) + 1
    total_decode_s = sum(decode_times)
    tpop_ms = (total_decode_s / num_decode) * 1000.0 if num_decode > 0 else 0.0
    end2end_ms = ttft_ms + total_decode_s * 1000.0

    return [
        RequestMeasurement(
            sample_id=s.sample_id,
            prompt_tokens=prompt_tokens[i] if i < len(prompt_tokens) else input_ids.shape[1],
            decode_tokens=num_decode,
            ttft_ms=ttft_ms / batch_size,
            tpop_ms=tpop_ms,
            end2end_ms=end2end_ms,
        )
        for i, s in enumerate(samples)
    ]


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


def run_batch(
    llm: LLM,
    stat_logger: BenchmarkStatLogger,
    samples: Sequence[PromptSample],
    sampling_params: SamplingParams,
    use_tqdm: bool = False,
) -> List[RequestMeasurement]:
    prompts = [s.text for s in samples]
    stat_logger.reset()

    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
    stats = stat_logger.consume()
    ttft_list = stats.get("ttft", [])
    tpot_list = stats.get("tpot", [])
    prompt_tok_list = stats.get("prompt_tokens", [])
    decode_tok_list = stats.get("decode_tokens", [])

    measurements: List[RequestMeasurement] = []
    for i, sample in enumerate(samples):
        req_output = outputs[i] if i < len(outputs) else None
        prompt_tokens = int(prompt_tok_list[i]) if i < len(prompt_tok_list) else sample.token_count
        decode_tokens = int(decode_tok_list[i]) if i < len(decode_tok_list) else 0
        if req_output is not None:
            prompt_tokens = len(req_output.prompt_token_ids)
            decode_tokens = sum(len(o.token_ids) for o in req_output.outputs)

        ttft_ms = 0.0
        tpop_ms = 0.0
        end2end_ms = 0.0

        if req_output is not None and hasattr(req_output, "metrics") and req_output.metrics:
            m = req_output.metrics
            if m.arrival_time is not None and m.first_token_time is not None:
                ttft_ms = (m.first_token_time - m.arrival_time) * 1000.0
            if m.finished_time is not None and m.arrival_time is not None:
                end2end_ms = (m.finished_time - m.arrival_time) * 1000.0
            if decode_tokens > 0 and m.first_token_time is not None and m.finished_time is not None:
                decode_time_s = m.finished_time - m.first_token_time
                tpop_ms = (decode_time_s / decode_tokens) * 1000.0

        if ttft_ms <= 0 and i < len(ttft_list):
            ttft_s = ttft_list[i]
            if isinstance(ttft_s, (int, float)):
                ttft_ms = float(ttft_s) * 1000.0
        if tpop_ms <= 0 and i < len(tpot_list):
            tpot_raw = tpot_list[i]
            if isinstance(tpot_raw, (list, tuple)):
                tpot_values = [float(x) for x in tpot_raw if isinstance(x, (int, float))]
            else:
                tpot_values = [float(tpot_raw)] if isinstance(tpot_raw, (int, float)) else []
            if tpot_values:
                tpop_ms = statistics.fmean(tpot_values) * 1000.0
        if end2end_ms <= 0 and ttft_ms > 0:
            decode_time_ms = decode_tokens * tpop_ms if tpop_ms > 0 else 0.0
            end2end_ms = ttft_ms + decode_time_ms

        measurements.append(RequestMeasurement(
            sample_id=sample.sample_id,
            prompt_tokens=prompt_tokens,
            decode_tokens=decode_tokens,
            ttft_ms=ttft_ms,
            tpop_ms=tpop_ms,
            end2end_ms=end2end_ms,
        ))

    return measurements


def summarize_batch_size(
    batch_size: int,
    measurements: Sequence[RequestMeasurement],
) -> Dict[str, Any]:
    ttft = [m.ttft_ms for m in measurements]
    tpop = [m.tpop_ms for m in measurements if m.tpop_ms > 0]
    e2e = [m.end2end_ms for m in measurements]

    return {
        "batch_size": batch_size,
        "num_requests": len(measurements),
        "ttft_avg_ms": statistics.fmean(ttft) if ttft else None,
        "ttft_p95_ms": percentile(ttft, 0.95),
        "ttft_p99_ms": percentile(ttft, 0.99),
        "tpop_avg_ms": statistics.fmean(tpop) if tpop else None,
        "tpop_p95_ms": percentile(tpop, 0.95) if tpop else None,
        "tpop_p99_ms": percentile(tpop, 0.99) if tpop else None,
        "end2end_avg_ms": statistics.fmean(e2e) if e2e else None,
        "end2end_p95_ms": percentile(e2e, 0.95),
        "end2end_p99_ms": percentile(e2e, 0.99),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TTFT, TPOP, End2End latency across batch sizes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or HuggingFace model id (e.g. Qwen3-30B-A3B-Instruct).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to ShareGPT JSON dataset.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Batch sizes to sweep.",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=FIXED_PROMPT_LENGTH,
        help="Fixed prompt length in tokens (pad/truncate).",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=64,
        help="Maximum number of requests to load from dataset.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens per request.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Number of warmup batches (discarded) per batch size.",
    )
    parser.add_argument(
        "--batches-per-size",
        type=int,
        default=10,
        help="Number of batches to run per batch size (after warmup).",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method (e.g. awq, gptq, fp8) for vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization (0-1).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for model loading.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path for results.",
    )
    parser.add_argument(
        "--keep-details",
        action="store_true",
        help="Include per-request details in output.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show vLLM tqdm progress.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    LOGGER.info("Loading tokenizer from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = load_sharegpt_prompts(
        args.dataset,
        tokenizer,
        target_length=args.prompt_length,
        max_requests=max(args.batch_sizes) * (args.warmup_batches + args.batches_per_size) + 16,
    )
    if not samples:
        raise ValueError("No valid prompts loaded from dataset.")

    results: List[Dict[str, Any]] = []
    sample_idx = 0

    if VLLM_AVAILABLE:
        load_kwargs = dict(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.prompt_length + args.max_new_tokens + 64,
            disable_log_stats=False,
        )
        if getattr(args, "quantization", None):
            load_kwargs["quantization"] = args.quantization
        LOGGER.info("Loading vLLM model: %s", args.model)
        llm = LLM(**load_kwargs)
        if VLLM_USES_V1_LOGGER:
            stat_logger = BenchmarkStatLogger(llm.llm_engine)
        else:
            stat_logger = BenchmarkStatLogger()
            llm.llm_engine.add_logger("perf_test", stat_logger)
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
        try:
            for batch_size in sorted(args.batch_sizes):
                LOGGER.info("=== Batch size %d ===", batch_size)
                all_measurements = []
                for b in range(args.warmup_batches + args.batches_per_size):
                    batch_samples = samples[sample_idx : sample_idx + batch_size]
                    if len(batch_samples) < batch_size:
                        sample_idx = 0
                        batch_samples = samples[sample_idx : sample_idx + batch_size]
                    sample_idx += len(batch_samples)
                    if not batch_samples:
                        break
                    ms = run_batch(llm, stat_logger, batch_samples, sampling_params, use_tqdm=args.show_progress)
                    if b >= args.warmup_batches:
                        all_measurements.extend(ms)
                if all_measurements:
                    summary = summarize_batch_size(batch_size, all_measurements)
                    summary["model"] = args.model
                    if args.keep_details:
                        summary["details"] = [asdict(m) for m in all_measurements]
                    results.append(summary)
                    LOGGER.info("batch_size=%d | TTFT: avg=%.1f p95=%.1f p99=%.1f ms | TPOP: avg=%.1f | End2End: avg=%.1f ms",
                        batch_size, summary.get("ttft_avg_ms") or 0, summary.get("ttft_p95_ms") or 0,
                        summary.get("ttft_p99_ms") or 0, summary.get("tpop_avg_ms") or 0,
                        summary.get("end2end_avg_ms") or 0)
        finally:
            if not VLLM_USES_V1_LOGGER:
                try:
                    llm.llm_engine.remove_logger("perf_test")
                except Exception:
                    pass
            del llm
            gc.collect()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Using HF Transformers backend (vLLM not available)")
        LOGGER.info("Loading model: %s", args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", trust_remote_code=args.trust_remote_code, torch_dtype=torch.float16
        )
        model.eval()
        try:
            for batch_size in sorted(args.batch_sizes):
                LOGGER.info("=== Batch size %d ===", batch_size)
                all_measurements = []
                for b in range(args.warmup_batches + args.batches_per_size):
                    batch_samples = samples[sample_idx : sample_idx + batch_size]
                    if len(batch_samples) < batch_size:
                        sample_idx = 0
                        batch_samples = samples[sample_idx : sample_idx + batch_size]
                    sample_idx += len(batch_samples)
                    if not batch_samples:
                        break
                    ms = run_batch_hf(model, tokenizer, batch_samples, device, args.max_new_tokens)
                    if b >= args.warmup_batches:
                        all_measurements.extend(ms)
                if all_measurements:
                    summary = summarize_batch_size(batch_size, all_measurements)
                    summary["model"] = args.model
                    if args.keep_details:
                        summary["details"] = [asdict(m) for m in all_measurements]
                    results.append(summary)
                    LOGGER.info("batch_size=%d | TTFT: avg=%.1f p95=%.1f p99=%.1f ms | TPOP: avg=%.1f | End2End: avg=%.1f ms",
                        batch_size, summary.get("ttft_avg_ms") or 0, summary.get("ttft_p95_ms") or 0,
                        summary.get("ttft_p99_ms") or 0, summary.get("tpop_avg_ms") or 0,
                        summary.get("end2end_avg_ms") or 0)
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "prompt_length": args.prompt_length,
                    "max_new_tokens": args.max_new_tokens,
                    "batch_sizes": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER.info("Results written to %s", args.output.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
