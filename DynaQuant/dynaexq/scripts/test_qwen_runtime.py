#!/usr/bin/env python3
"""
Quick smoke test for DynaExQ using Qwen3-30B-A3B W4A16 / W2A16 checkpoints.

The script reuses the test utilities shipped with DynaExQ to exercise:
  • Mixed-precision expert switching and swap pipeline
  • On-demand loading of W4/W2 expert weights from the provided model folders
  • Telemetry / statistics collection from the runtime

Usage:
    python dynaexq/scripts/test_qwen_runtime.py \
        --mode hf \
        --w4-model /workspace/Models/Qwen3-30B-A3B-W4A16 \
        --w2-model /workspace/Models/Qwen3-30B-A3B-W2A16 \
        --prompt-file calibration_datasets/requests/mmlu_pro_200.jsonl \
        --max-prompts 5 \
        --num-tokens 32 \
        --save-report hf_outputs.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import torch


# Ensure repository root is on PYTHONPATH so `dynaexq` package is importable when the
# script is invoked directly (e.g. via `python dynaexq/scripts/test_qwen_runtime.py`).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the comprehensive test harness for the mock MoE + runtime plumbing.
from dynaexq.tests.test_end_to_end import DynaExQTestModel  # noqa: E402


logger = logging.getLogger("dynaexq.smoke_test")


def _default_prompts() -> List[str]:
    """Short list of prompts to exercise the runtime."""
    return [
        "Explain why expert mixing helps large language models.",
        "Summarise the main differences between FP16, W4A16 and W2A16 quantisation.",
        "Write a short Python function that computes the Fibonacci numbers.",
    ]


def _load_prompts_from_file(path: Path, prompt_key: str = "prompt") -> List[str]:
    """
    Load prompts from a plain-text file or JSON/JSONL dataset.

    * For .txt/.md/... files, every non-empty line becomes one prompt.
    * For .json/.jsonl, each entry is expected to contain `prompt_key`.
    """
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    suffix = path.suffix.lower()
    prompts: List[str] = []

    if suffix in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Invalid JSON line in {path}: {line[:80]}…")

                if prompt_key not in data:
                    raise KeyError(
                        f"Key '{prompt_key}' not found in JSON object: {list(data.keys())}"
                    )
                value = str(data[prompt_key]).strip()
                if value:
                    prompts.append(value)
    else:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    prompts.append(line)

    if not prompts:
        raise ValueError(f"No prompts loaded from {path}")

    logging.getLogger("dynaexq.smoke_test").info(
        "Loaded %d prompts from %s", len(prompts), path
    )
    return prompts


def run_smoke_test(args: argparse.Namespace) -> dict:
    """
    Execute a lightweight runtime simulation:
      1. Instantiate the mock MoE model backed by the provided checkpoints.
      2. Run a few synthetic inference passes (router outputs are random).
      3. Collect runtime / weight-loader statistics for inspection.
    """
    if args.prompt_file:
        prompts = _load_prompts_from_file(
            Path(args.prompt_file), prompt_key=args.prompt_key
        )
    elif args.prompts:
        prompts = list(args.prompts)
    else:
        prompts = _default_prompts()

    if args.mode == "hf":
        return run_hf_inference(args, prompts)

    logger.info("Initialising DynaExQTestModel …")

    model = DynaExQTestModel(
        w4a16_path=args.w4_model,
        w2a16_path=args.w2_model,
        num_layers=args.num_layers,
        num_experts_per_layer=args.experts_per_layer,
        device=args.device,
        config_path=args.runtime_config,
    )

    try:
        for idx, prompt in enumerate(prompts[: args.max_prompts], start=1):
            logger.info("-" * 80)
            logger.info("Processing prompt %d/%d: %s",
                        idx, len(prompts), prompt)

            encoded = model.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=args.seq_length,
                truncation=True,
            )

            input_ids = encoded["input_ids"].to(device=args.device)
            logger.info("  input_ids shape: %s", tuple(input_ids.shape))

            with torch.no_grad():
                output = model.forward(
                    input_ids=input_ids,
                    num_tokens=args.num_tokens,
                )

            logger.info("  simulated output shape: %s", tuple(output.shape))

        stats = model.get_statistics()
        runtime_stats = stats["runtime"]
        weight_stats = stats["weights"]

        logger.info("=" * 80)
        logger.info("DynaExQ runtime summary")
        logger.info("=" * 80)
        monitor = runtime_stats["monitor"]
        swap_stats = runtime_stats["swap_engine"]
        memory_stats = runtime_stats["memory"]

        logger.info("Monitored experts: %d", monitor["total_experts_tracked"])
        logger.info(
            "Hotness mean/max: %.4f / %.4f",
            monitor["mean_hotness"],
            monitor["max_hotness"],
        )
        logger.info(
            "Swaps (upgrade / downgrade): %d / %d",
            swap_stats["upgrade_count"],
            swap_stats["downgrade_count"],
        )
        logger.info("Swap misses: %d", swap_stats["miss_count"])
        logger.info("Ready-before-use ratio: %.2f%%",
                    swap_stats["ready_ratio"] * 100)
        logger.info(
            "HBM pressure: %.2f%%",
            memory_stats["hbm_pressure"] * 100,
        )
        logger.info(
            "Experts currently cached (W4 / W2): %d / %d",
            stats["expert_precision_counts"].get("W4", 0),
            stats["expert_precision_counts"].get("W2", 0),
        )
        logger.info(
            "Weights loaded (W4 / W2 / total): %d / %d / %d",
            weight_stats["w4_experts_loaded"],
            weight_stats["w2_experts_loaded"],
            weight_stats["total_experts_loaded"],
        )

        return {
            "runtime": runtime_stats,
            "weights": weight_stats,
            "expert_precision_counts": stats["expert_precision_counts"],
        }

    finally:
        model.cleanup()


def run_hf_inference(args: argparse.Namespace, prompts: Sequence[str]) -> dict:
    """
    Load the quantised W4A16 checkpoint via HuggingFace Transformers and run genuine
    generation for the provided prompts. This validates that the model can be loaded
    end-to-end after quantisation.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading HF model from %s", args.w4_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.w4_model, trust_remote_code=True
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.w4_model,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map="auto",
        )
    except ValueError as exc:
        if "autoawq" in str(exc).lower():
            logger.error(
                "Failed to load AWQ checkpoint. Please install/upgrade autoawq >= 0.1.8 "
                "(e.g. `pip install --upgrade autoawq`). Original error: %s",
                exc,
            )
        raise

    generated_texts = []

    for idx, prompt in enumerate(prompts[: args.max_prompts], start=1):
        logger.info("-" * 80)
        logger.info("HF inference prompt %d/%d: %s", idx, len(prompts), prompt)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.seq_length,
        )
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=inputs.get(
                    "attention_mask", None).to(model.device)
                if "attention_mask" in inputs
                else None,
                max_new_tokens=args.num_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_texts.append(
            {
                "prompt": prompt,
                "output": text,
            }
        )
        logger.info("HF output: %s", text)

    if args.save_report:
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(
            generated_texts, indent=2), encoding="utf-8")
        logger.info("Saved HF inference outputs to %s", report_path)

    return {"completions": generated_texts}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DynaExQ smoke test or HF inference against Qwen3-30B-A3B checkpoints."
    )
    parser.add_argument(
        "--w4-model",
        type=str,
        required=True,
        help="Path to the W4A16 quantised model (directory created by quantisation pipeline).",
    )
    parser.add_argument(
        "--w2-model",
        type=str,
        required=True,
        help="Path to the W2A16 quantised model directory.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of MoE layers to simulate (reduces runtime for smoke test).",
    )
    parser.add_argument(
        "--experts-per-layer",
        type=int,
        default=64,
        help="Number of experts per layer to simulate.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
        help="Sequence length to use when tokenising prompts.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=16,
        help="Number of tokens to generate in the mock forward pass.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for the mock model (the test harness operates on CPU by default).",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=3,
        help="Maximum number of prompts to feed through the simulation.",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Optional list of custom prompts. Defaults to a built-in trio.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional path to a prompt dataset (.txt/.json/.jsonl). "
        "For JSON lines, each object must contain --prompt-key (default 'prompt').",
    )
    parser.add_argument(
        "--prompt-key",
        type=str,
        default="prompt",
        help="Key name to read from JSON/JSONL prompt files (default: 'prompt').",
    )
    parser.add_argument(
        "--runtime-config",
        type=str,
        default=None,
        help="Optional path to a DynaExQ runtime YAML config (defaults to configs/default.yaml).",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Optional path to dump the collected statistics as JSON.",
    )
    parser.add_argument(
        "--mode",
        choices=["runtime", "hf"],
        default="runtime",
        help="Testing mode: 'runtime' uses the mock MoE/DynaExQ harness (default). "
        "'hf' performs genuine HF model.generate() calls on the W4 checkpoint.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging output.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling when using --mode hf (defaults to greedy).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for HF mode (ignored if not sampling).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling parameter for HF mode (ignored if not sampling).",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Sanity checks for supplied paths
    for name, path in (("W4A16", args.w4_model), ("W2A16", args.w2_model)):
        if not os.path.isdir(path):
            logger.error("%s model directory not found: %s", name, path)
            return 1

    try:
        stats = run_smoke_test(args)
    except Exception as exc:  # pragma: no cover - runtime failure path
        logger.exception("Smoke test failed: %s", exc)
        return 1

    if args.mode == "runtime" and args.save_report:
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        logger.info("Saved runtime report to %s", report_path)

    logger.info("DynaExQ smoke test finished successfully.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
