#!/usr/bin/env python3
"""Evaluate MMLU perplexity for a DynaExQ mixed-precision model.

Example
-------
python scripts/evaluate_mmlu_perplexity_mixed.py \
  --fp16 /workspace/Models/Qwen3-30B-A3B-Instruct-2507 \
  --int4 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound \
  --activation-file ./activations/activation_qwen30b_mmlu_pro_sorted.json \
  --tail-count 64 \
  --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl
"""

from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from evaluate_mmlu_perplexity import (  # type: ignore
    SUPPORTED_TASKS,
    AggregatePerplexity,
    SamplePerplexity,
    compute_sample_neg_log_likelihood,
    evaluate_mmlu_perplexity,
    iter_jsonl,
    resolve_model_device,
)
from dynaexq.runtime import ExpertID
from auto_round.inference.convert_model import convert_hf_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import argparse
import gc
import json
import logging
import math
import re
from typing import Dict, Iterable, List, Optional, Set


LOGGER = logging.getLogger("dynaexq.mixed_perplexity")


def dataset_has_reference(dataset_path: Path) -> bool:
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            return "answer" in sample
    return False


def evaluate_plaintext_perplexity(
    *,
    model,
    tokenizer,
    dataset_path: Path,
    max_samples: Optional[int],
    keep_details: bool,
) -> AggregatePerplexity:
    model_device = resolve_model_device(model)
    total_tokens = 0
    total_nll = 0.0
    consumed = 0
    skipped = 0
    details: List[SamplePerplexity] = [] if keep_details else None

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    for entry in iter_jsonl(dataset_path):
        if max_samples is not None and consumed >= max_samples:
            break

        prompt_id = entry.get("id")
        text = entry.get("prompt") or entry.get("text") or entry.get("content")
        if not text:
            skipped += 1
            continue

        neg_log_likelihood, token_count = compute_sample_neg_log_likelihood(
            model=model,
            tokenizer=tokenizer,
            formatted_prompt="",
            target_text=text,
            device=model_device,
        )

        if token_count <= 0 or not math.isfinite(neg_log_likelihood):
            skipped += 1
            continue

        total_tokens += token_count
        total_nll += neg_log_likelihood
        consumed += 1

        if keep_details and details is not None:
            details.append(
                SamplePerplexity(
                    prompt_id=str(
                        prompt_id) if prompt_id is not None else None,
                    reference=text,
                    target_token_count=token_count,
                    neg_log_likelihood=neg_log_likelihood,
                )
            )

    return AggregatePerplexity(
        samples=consumed,
        skipped=skipped,
        target_tokens=total_tokens,
        total_neg_log_likelihood=total_nll,
        details=details,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default="calibration_datasets/requests/mmlu_pro_200.jsonl",
        help="Path to the MMLU-Pro dataset JSONL file.",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default="mmlu_pro_200.jsonl",
        help="Dataset key registered in SUPPORTED_TASKS.",
    )
    parser.add_argument(
        "--fp16",
        required=True,
        help="Directory containing the FP16 (high precision) checkpoint.",
    )
    parser.add_argument(
        "--int4",
        required=True,
        help="Directory containing the INT4 (low precision) checkpoint.",
    )
    parser.add_argument(
        "--activation-file",
        required=True,
        help=(
            "JSON file containing per-layer expert activation rankings. "
            "Expected format: {\"0\": [expert_indices...], ...}."
        ),
    )
    parser.add_argument(
        "--tail-count",
        type=int,
        default=1,
        help="Number of tail experts per layer to downgrade to INT4.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device placement for the model (auto|cpu|cuda|cuda:0...).",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default=None,
        help="Optional torch dtype override (e.g., 'float16', 'bfloat16').",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of dataset samples to evaluate.",
    )
    parser.add_argument(
        "--include-eos",
        action="store_true",
        help="Append EOS token to references when scoring.",
    )
    parser.add_argument(
        "--user-prefix",
        type=str,
        default="",
        help="Optional prefix prepended to each user prompt.",
    )
    parser.add_argument(
        "--user-suffix",
        type=str,
        default="",
        help="Optional suffix appended to each user prompt.",
    )
    parser.add_argument(
        "--keep-details",
        action="store_true",
        help="Export per-sample perplexity diagnostics.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save aggregated results as JSON.",
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Merge results into existing JSON when --output exists.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, ...).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform all preprocessing without loading the model or running perplexity.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: Optional[str], device: torch.device) -> torch.dtype:
    if dtype_arg is None or dtype_arg.lower() == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    normalized = dtype_arg.lower()
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype argument: {dtype_arg}")


def load_activation_spec(path: Path) -> Dict[int, List[int]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    mapping: Dict[int, List[int]] = {}
    for layer_str, experts in data.items():
        match = re.search(r"\d+", layer_str)
        if match is None:
            raise ValueError(
                f"Invalid layer id '{layer_str}' in activation file")
        layer = int(match.group(0))
        if not isinstance(experts, Iterable):
            raise ValueError(
                f"Expected iterable expert list for layer {layer}")
        mapping[layer] = [int(idx) for idx in experts]
    return mapping


def select_tail_experts(activation_map: Dict[int, List[int]], tail_count: int) -> Set[ExpertID]:
    selected: Set[ExpertID] = set()
    if tail_count <= 0:
        return selected
    for layer, experts in activation_map.items():
        if not experts:
            continue
        count = min(tail_count, len(experts))
        tail = experts[-count:]
        for idx in tail:
            selected.add(ExpertID(layer=layer, idx=int(idx)))
    return selected


def apply_quantized_experts(
    target_model: AutoModelForCausalLM,
    quant_model: AutoModelForCausalLM,
    experts: Set[ExpertID],
) -> None:
    if not experts:
        return

    for expert in sorted(experts, key=lambda e: (e.layer, e.idx)):
        try:
            target_layer = target_model.model.layers[expert.layer]
            quant_layer = quant_model.model.layers[expert.layer]
            target_layer.mlp.experts[expert.idx] = quant_layer.mlp.experts[expert.idx]
        except IndexError as exc:
            raise ValueError(
                f"Expert {expert} is out of bounds for the loaded model") from exc


def load_mixed_precision_model(
    *,
    fp16_path: str,
    int4_path: str,
    low_precision_experts: Set[ExpertID],
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    LOGGER.info("Loading FP16 base model from %s", fp16_path)
    model = AutoModelForCausalLM.from_pretrained(
        fp16_path,
        torch_dtype=dtype,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        fp16_path, trust_remote_code=trust_remote_code)

    if low_precision_experts:
        LOGGER.info("Loading AutoRound INT4 model from %s", int4_path)
        quant_model = AutoModelForCausalLM.from_pretrained(
            int4_path,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )
        LOGGER.info("Converting AutoRound model to quantized modules")
        quant_model, _ = convert_hf_model(quant_model, target_device="cpu")

        apply_quantized_experts(model, quant_model, low_precision_experts)

        del quant_model
        gc.collect()

    model.to(device=device)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    has_reference = dataset_has_reference(dataset_path)
    task = None
    if has_reference:
        try:
            task = SUPPORTED_TASKS[args.dataset_key]
        except KeyError as exc:
            raise KeyError(
                f"Dataset key {args.dataset_key} is not registered in SUPPORTED_TASKS"
            ) from exc

    activation_map = load_activation_spec(Path(args.activation_file))
    low_precision_experts = select_tail_experts(
        activation_map, args.tail_count)
    LOGGER.info(
        "Downgrading %d experts to INT4 precision (tail count %d per layer)",
        len(low_precision_experts),
        args.tail_count,
    )

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.torch_dtype, device)

    summary_base: Dict[str, object] = {
        "fp16_path": args.fp16,
        "int4_path": args.int4,
        "device": str(device),
        "dtype": str(dtype),
        "tail_count": args.tail_count,
        "downgraded_experts": [
            {"layer": expert.layer, "idx": expert.idx}
            for expert in sorted(low_precision_experts, key=lambda e: (e.layer, e.idx))
        ],
    }

    if args.dry_run:
        LOGGER.info(
            "Dry-run enabled; skipping model loading and perplexity computation.")
        summary_base["dry_run"] = True
        return summary_base

    model, tokenizer = load_mixed_precision_model(
        fp16_path=args.fp16,
        int4_path=args.int4,
        low_precision_experts=low_precision_experts,
        device=device,
        dtype=dtype,
    )

    if has_reference and task is not None:
        aggregate: AggregatePerplexity = evaluate_mmlu_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            task=task,
            max_samples=args.max_samples,
            include_eos=args.include_eos,
            user_prefix=args.user_prefix,
            user_suffix=args.user_suffix,
            keep_details=args.keep_details,
        )
    else:
        aggregate = evaluate_plaintext_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            max_samples=args.max_samples,
            keep_details=args.keep_details,
        )

    summary: Dict[str, object] = dict(summary_base)
    summary.update(
        {
            "samples": aggregate.samples,
            "skipped": aggregate.skipped,
            "target_tokens": aggregate.target_tokens,
            "avg_neg_log_likelihood": aggregate.avg_nll,
            "perplexity": aggregate.perplexity,
        }
    )

    if args.keep_details and aggregate.details is not None:
        summary["details"] = [
            {
                "id": detail.prompt_id,
                "reference": detail.reference,
                "target_tokens": detail.target_token_count,
                "neg_log_likelihood": detail.neg_log_likelihood,
                "avg_neg_log_likelihood": detail.avg_nll,
                "perplexity": detail.perplexity,
            }
            for detail in aggregate.details
        ]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        result = evaluate(args)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Mixed-precision evaluation failed: %s", exc)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {}
        if args.append_output and output_path.exists():
            try:
                with output_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except json.JSONDecodeError:
                LOGGER.warning(
                    "Existing output file %s is not valid JSON. Overwriting.", output_path)
        if not isinstance(payload, dict):
            payload = {}
        payload_key = Path(args.fp16).name
        payload[payload_key] = result
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        LOGGER.info("Saved results to %s", output_path)

    LOGGER.info("Mixed-precision perplexity evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
