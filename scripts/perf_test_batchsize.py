#!/usr/bin/env python3
"""
Performance test: TTFT (Time To First Token) across different prompt token lengths.

Tests different models at different num_of_tokens (prompt lengths).
Outputs TTFT: AVG, P95, P99. Each configuration runs 10 groups.

Token lengths (default): 1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384,
                        512, 576, 640, 704, 768, 832, 896, 960, 1024

Example:
  python scripts/perf_test_batchsize.py --model /path/to/Qwen3-30B-A3B \\
    --output results/ttft_vs_tokens.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    SamplingParams = None
    VLLM_AVAILABLE = False

if VLLM_AVAILABLE:
    try:
        from vllm.v1.metrics.loggers import StatLoggerBase as VLLMStatLoggerBase
        VLLM_USES_V1_LOGGER = True
    except ImportError:
        from vllm.engine.metrics import StatLoggerBase as VLLMStatLoggerBase
        VLLM_USES_V1_LOGGER = False
else:
    VLLM_USES_V1_LOGGER = False
    VLLMStatLoggerBase = object

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger("perf_test_batchsize")

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "ShareGPT_V3_unfiltered_cleaned_split.json"
DEFAULT_NUM_TOKENS = [
    1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384,
    512, 576, 640, 704, 768, 832, 896, 960, 1024,
]
GROUPS_PER_CONFIG = 10


@dataclass
class PromptSample:
    sample_id: str
    text: str
    token_count: int


def load_sharegpt_prompts(
    dataset_path: Path,
    tokenizer: AutoTokenizer,
    target_length: int,
    max_requests: int,
) -> List[PromptSample]:
    """Load prompts from ShareGPT, fix each to target_length tokens (pad/truncate)."""
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
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
    return samples


if VLLM_USES_V1_LOGGER:

    class _EngineBenchmarkLogger(VLLMStatLoggerBase):
        def __init__(self, vllm_config, engine_index: int = 0) -> None:
            self.engine_index = engine_index
            self.reset()

        def reset(self) -> None:
            self._ttft: List[float] = []
            self._prompt_tokens: List[int] = []

        def record(self, scheduler_stats, iteration_stats, engine_idx: int = 0):
            if iteration_stats is None:
                return
            self._ttft.extend(iteration_stats.time_to_first_tokens_iter)
            for finished in iteration_stats.finished_requests:
                self._prompt_tokens.append(finished.num_prompt_tokens)

        def log_engine_initialized(self):
            return

        def consume(self) -> Dict[str, List[float]]:
            payload = {"ttft": list(self._ttft), "prompt_tokens": list(self._prompt_tokens)}
            self.reset()
            return payload

    class BenchmarkStatLogger:
        def __init__(self, llm_engine) -> None:
            manager = getattr(llm_engine, "logger_manager", None)
            if manager is None:
                raise RuntimeError("vLLM stats logging disabled. Use disable_log_stats=False.")
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
            payload = {"ttft": [], "prompt_tokens": []}
            for logger in self._loggers:
                stats = logger.consume()
                for key in payload:
                    payload[key].extend(stats[key])
            return payload
else:

    class BenchmarkStatLogger(VLLMStatLoggerBase):
        def __init__(self) -> None:
            super().__init__(local_interval=1e9)
            self.reset()

        def info(self, type: str, obj) -> None:
            return

        def reset(self) -> None:
            self._ttft: List[float] = []
            self._prompt_tokens: List[int] = []

        def log(self, stats) -> None:
            self._ttft.extend(stats.time_to_first_tokens_iter)
            self._prompt_tokens.extend(stats.num_prompt_tokens_requests)

        def consume(self) -> Dict[str, List[float]]:
            payload = {"ttft": list(self._ttft), "prompt_tokens": list(self._prompt_tokens)}
            self.reset()
            return payload


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo))


def run_ttft_batch(
    llm: LLM,
    stat_logger: BenchmarkStatLogger,
    samples: Sequence[PromptSample],
    sampling_params: SamplingParams,
) -> List[float]:
    """Run batch, return list of TTFT values (seconds, convert to ms in summary)."""
    prompts = [s.text for s in samples]
    stat_logger.reset()
    llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    stats = stat_logger.consume()
    ttft_seconds = stats.get("ttft", [])
    return [float(t) * 1000.0 for t in ttft_seconds]  # ms


def run_ttft_batch_transformers(
    model: Any,
    tokenizer: AutoTokenizer,
    samples: Sequence[PromptSample],
    device: torch.device,
    max_new_tokens: int = 1,
) -> List[float]:
    """Run each prompt with generate(max_new_tokens=1), return TTFT in ms (wall-clock)."""
    ttft_ms: List[float] = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    for s in samples:
        inputs = tokenizer(
            s.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=s.token_count,
            return_attention_mask=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        ttft_ms.append((t1 - t0) * 1000.0)
    return ttft_ms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark TTFT across prompt token lengths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", type=str, required=True, help="Model path or HF id")
    p.add_argument(
        "--backend",
        type=str,
        choices=("vllm", "transformers"),
        default="transformers",
        help="Backend: transformers (no vLLM) or vllm",
    )
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Multiple models to test (overrides --model if set)",
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=DEFAULT_NUM_TOKENS,
        help="Prompt token lengths to test",
    )
    p.add_argument(
        "--groups-per-config",
        type=int,
        default=GROUPS_PER_CONFIG,
        help="Number of groups (runs) per (model, num_tokens)",
    )
    p.add_argument("--max-new-tokens", type=int, default=1, help="Tokens to generate (1 for TTFT)")
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--output", type=Path, help="Output JSON")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    if args.backend == "vllm" and not VLLM_AVAILABLE:
        raise RuntimeError("--backend vllm requires vLLM. Install via: pip install vllm")

    models = args.models if args.models else [args.model]
    num_tokens_list = sorted(set(args.num_tokens))
    max_tokens = max(num_tokens_list)

    LOGGER.info("Loading tokenizer from %s", models[0])
    tokenizer = AutoTokenizer.from_pretrained(models[0], trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results: List[Dict[str, Any]] = []

    if args.backend == "transformers":
        device = torch.device("cuda:0")
        for model_path in models:
            LOGGER.info("Loading model (transformers): %s", model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=args.trust_remote_code,
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
            )
            model.eval()
            try:
                for num_tokens in num_tokens_list:
                    samples = load_sharegpt_prompts(
                        args.dataset,
                        tokenizer,
                        target_length=num_tokens,
                        max_requests=args.groups_per_config + 4,
                    )
                    samples = samples[: args.groups_per_config]
                    if not samples:
                        continue
                    ttft_ms_list = run_ttft_batch_transformers(
                        model, tokenizer, samples, device, max_new_tokens=args.max_new_tokens
                    )
                    if not ttft_ms_list:
                        continue
                    summary = {
                        "model": model_path,
                        "num_tokens": num_tokens,
                        "groups": len(ttft_ms_list),
                        "ttft_avg_ms": sum(ttft_ms_list) / len(ttft_ms_list),
                        "ttft_p95_ms": percentile(ttft_ms_list, 0.95),
                        "ttft_p99_ms": percentile(ttft_ms_list, 0.99),
                    }
                    all_results.append(summary)
                    LOGGER.info(
                        "model=%s num_tokens=%d | TTFT: avg=%.1f p95=%.1f p99=%.1f ms",
                        model_path, num_tokens,
                        summary["ttft_avg_ms"],
                        summary["ttft_p95_ms"] or 0,
                        summary["ttft_p99_ms"] or 0,
                    )
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
    else:
        for model_path in models:
            LOGGER.info("Loading model: %s", model_path)
            llm = LLM(
                model=model_path,
                tensor_parallel_size=args.tensor_parallel,
                trust_remote_code=args.trust_remote_code,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=max_tokens + args.max_new_tokens + 64,
                disable_log_stats=False,
            )

            if VLLM_USES_V1_LOGGER:
                stat_logger = BenchmarkStatLogger(llm.llm_engine)
            else:
                stat_logger = BenchmarkStatLogger()
                llm.llm_engine.add_logger("perf_batchsize", stat_logger)

            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=args.max_new_tokens,
            )

            try:
                for num_tokens in num_tokens_list:
                    samples = load_sharegpt_prompts(
                        args.dataset,
                        tokenizer,
                        target_length=num_tokens,
                        max_requests=args.groups_per_config + 4,
                    )
                    if len(samples) < args.groups_per_config:
                        LOGGER.warning(
                            "Only %d prompts for num_tokens=%d, need %d",
                            len(samples), num_tokens, args.groups_per_config,
                        )
                    samples = samples[: args.groups_per_config]
                    if not samples:
                        continue

                    ttft_ms_list = run_ttft_batch(
                        llm, stat_logger, samples, sampling_params
                    )
                    if not ttft_ms_list:
                        LOGGER.warning("No TTFT data for num_tokens=%d", num_tokens)
                        continue

                    summary = {
                        "model": model_path,
                        "num_tokens": num_tokens,
                        "groups": len(ttft_ms_list),
                        "ttft_avg_ms": sum(ttft_ms_list) / len(ttft_ms_list),
                        "ttft_p95_ms": percentile(ttft_ms_list, 0.95),
                        "ttft_p99_ms": percentile(ttft_ms_list, 0.99),
                    }
                    all_results.append(summary)
                    LOGGER.info(
                        "model=%s num_tokens=%d | TTFT: avg=%.1f p95=%.1f p99=%.1f ms",
                        model_path, num_tokens,
                        summary["ttft_avg_ms"],
                        summary["ttft_p95_ms"] or 0,
                        summary["ttft_p99_ms"] or 0,
                    )
            finally:
                if not VLLM_USES_V1_LOGGER:
                    try:
                        llm.llm_engine.remove_logger("perf_batchsize")
                    except Exception:
                        pass
                del llm
                gc.collect()

    out = {
        "backend": args.backend,
        "models": models,
        "num_tokens_tested": num_tokens_list,
        "groups_per_config": args.groups_per_config,
        "max_new_tokens": args.max_new_tokens,
        "results": all_results,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        LOGGER.info("Results saved to %s", args.output.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
