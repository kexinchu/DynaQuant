#!/usr/bin/env python3
"""
Performance test: Prefill latency (TTFT) vs prompt length with MoE expert CPU/GPU migration.

Uses the same expert offload/prefetch scenario as perf_test_moe_offload.py.
Tests different prompt lengths (num_tokens); for each length runs 10 trials and reports
TTFT: AVG, P95, P99.

Default models:
  - Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound
  - Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound

Default token lengths: 1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384,
                      512, 576, 640, 704, 768, 832, 896, 960, 1024

Example:
  python scripts/perf_test_moe_offload_tokens.py \\
    --models /path/to/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound \\
             /path/to/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \\
    --output results/moe_offload_tokens.json
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure scripts/ is on path when run from project root
_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from perf_test_moe_offload import (
    DEFAULT_DATASET,
    ExpertMemoryManager,
    compute_gpu_memory_limit_bytes,
    load_sharegpt_prompts,
    percentile,
    run_batch_generation,
    setup_router_hooks,
)

LOGGER = logging.getLogger("perf_test_moe_offload_tokens")

# Same token lengths as perf_test_batchsize.py
DEFAULT_NUM_TOKENS = [
    1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384,
    512, 576, 640, 704, 768, 832, 896, 960, 1024,
]

TRIALS_PER_NUM_TOKENS = 10

DEFAULT_MODELS = [
    "Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound",
    "Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound",
]


def _run_one_model(
    model_path: str,
    device_str: str,
    tokenizer: AutoTokenizer,
    num_tokens_list: List[int],
    trials_per_config: int,
    dataset_path: Path,
    max_new_tokens: int,
    output_path: Optional[Path],
) -> Dict[str, Any]:
    """Run prefill (TTFT) test for one model across all num_tokens. Returns result dict."""
    device = torch.device(device_str)

    LOGGER.info("Loading model from %s on %s", model_path, device_str)
    load_kwargs: Dict[str, Any] = dict(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # Phi-3.5-MoE: avoid OOM and avoid dynamic module (flash_attn). Use built-in phimoe + CPU offload.
    if "Phi-3.5-MoE" in model_path or "Phi-3.5-MoE-instruct" in model_path:
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with config_path.open() as f:
                config_dict = json.load(f)
            config_dict.pop("auto_map", None)
            from transformers import AutoConfig
            config = AutoConfig.for_model(**config_dict)
            load_kwargs["config"] = config
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = {0: "44GiB", "cpu": "60GiB"}
            load_kwargs["trust_remote_code"] = False
    else:
        load_kwargs["device_map"] = device_str
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()

    expert_budget_bytes, gpu_meta = compute_gpu_memory_limit_bytes(model)
    experts_per_layer = gpu_meta.get("experts_per_layer", 0)
    thres_expert = experts_per_layer // 2 if experts_per_layer else 1

    mem_manager = ExpertMemoryManager(model, device, expert_budget_bytes)
    hooks = setup_router_hooks(model, mem_manager, thres_expert)

    model_kwargs_base: Dict[str, Any] = {}
    model_type = getattr(model.config, "model_type", "").lower()
    if "deepseek" not in model_type:
        try:
            sig = inspect.signature(model.forward)
            if "output_router_logits" in sig.parameters:
                model_kwargs_base["output_router_logits"] = True
        except (TypeError, AttributeError):
            pass

    results: List[Dict[str, Any]] = []

    try:
        for num_tokens in num_tokens_list:
            LOGGER.info("=== num_tokens=%d ===", num_tokens)
            samples = load_sharegpt_prompts(
                dataset_path,
                tokenizer,
                target_length=num_tokens,
                max_requests=trials_per_config + 4,
            )
            if len(samples) < trials_per_config:
                LOGGER.warning(
                    "Only %d prompts for num_tokens=%d, need %d",
                    len(samples), num_tokens, trials_per_config,
                )
            samples = samples[:trials_per_config]
            if not samples:
                results.append({
                    "num_tokens": num_tokens,
                    "trials": 0,
                    "ttft_avg_ms": None,
                    "ttft_p95_ms": None,
                    "ttft_p99_ms": None,
                })
                continue

            ttft_list: List[float] = []
            for i in range(len(samples)):
                batch_token_ids = [samples[i][1]]
                ttft_ms, _, _ = run_batch_generation(
                    model,
                    tokenizer,
                    batch_token_ids,
                    device,
                    max_new_tokens,
                    model_kwargs_base,
                )
                ttft_list.append(ttft_ms)

            summary = {
                "num_tokens": num_tokens,
                "trials": len(ttft_list),
                "ttft_avg_ms": sum(ttft_list) / len(ttft_list) if ttft_list else None,
                "ttft_p95_ms": percentile(ttft_list, 0.95),
                "ttft_p99_ms": percentile(ttft_list, 0.99),
            }
            results.append(summary)
            LOGGER.info(
                "num_tokens=%d | TTFT: avg=%.1f p95=%.1f p99=%.1f ms",
                num_tokens,
                summary["ttft_avg_ms"] or 0,
                summary["ttft_p95_ms"] or 0,
                summary["ttft_p99_ms"] or 0,
            )

            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                out = {
                    "model": model_path,
                    "device": device_str,
                    "num_tokens_tested": num_tokens_list,
                    "trials_per_config": trials_per_config,
                    "max_new_tokens": max_new_tokens,
                    "gpu_metadata": gpu_meta,
                    "results": results,
                }
                output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    finally:
        for h in hooks:
            h.remove()
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "model": model_path,
        "device": device_str,
        "num_tokens_tested": num_tokens_list,
        "trials_per_config": trials_per_config,
        "max_new_tokens": max_new_tokens,
        "gpu_metadata": gpu_meta,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MoE offload: Prefill latency (TTFT) vs prompt length.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model paths (default: Qwen3-30B and Qwen3-80B A3B)",
    )
    p.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cuda:0", "cuda:1"],
        help="Devices for each model",
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="ShareGPT JSON path")
    p.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=DEFAULT_NUM_TOKENS,
        help="Prompt token lengths to test",
    )
    p.add_argument(
        "--trials-per-config",
        type=int,
        default=TRIALS_PER_NUM_TOKENS,
        help="Number of trials per (model, num_tokens)",
    )
    p.add_argument("--max-new-tokens", type=int, default=1, help="Tokens to generate (1 for TTFT only)")
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

    model_list = args.models
    device_list = (args.devices + [args.devices[-1]] * len(model_list))[: len(model_list)]
    num_tokens_list = sorted(set(args.num_tokens))

    all_results: List[Dict[str, Any]] = []

    for model_path, device_str in zip(model_list, device_list):
        LOGGER.info("Loading tokenizer from %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        single_out = args.output
        if single_out and len(model_list) > 1:
            stem = "moe_offload_tokens_" + ("qwen30b" if "30B" in model_path else "qwen80b")
            single_out = single_out.parent / f"{stem}.json"
        try:
            out = _run_one_model(
                model_path=model_path,
                device_str=device_str,
                tokenizer=tokenizer,
                num_tokens_list=num_tokens_list,
                trials_per_config=args.trials_per_config,
                dataset_path=args.dataset,
                max_new_tokens=args.max_new_tokens,
                output_path=single_out,
            )
            all_results.append(out)
            if single_out:
                single_out.parent.mkdir(parents=True, exist_ok=True)
                single_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
                LOGGER.info("Results saved to %s", single_out.resolve())
        except Exception as e:
            LOGGER.exception("Failed for %s: %s", model_path, e)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.output and len(all_results) > 1:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({
                "num_tokens_tested": num_tokens_list,
                "trials_per_config": args.trials_per_config,
                "max_new_tokens": args.max_new_tokens,
                "models": all_results,
            }, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Combined results saved to %s", args.output.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
