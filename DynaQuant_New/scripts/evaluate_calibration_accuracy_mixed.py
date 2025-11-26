#!/usr/bin/env python3
"""
Evaluate calibration dataset accuracy for a mixed-precision Qwen3-30B model.

This helper mirrors ``evaluate_calibration_accuracy.py`` but accepts two
checkpoints (FP16 + INT4), an expert activation ranking, and downgrades the
least-active experts to the low-precision model before running accuracy tests.

Example
-------
python scripts/evaluate_calibration_accuracy_mixed.py \
    --fp16 /home/chuke/Models/Qwen3-30B-A3B-Instruct-2507 \
    --int4 /home/chuke/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound \
    --activation-file ./activations/activation_qwen30b_mmlu_pro_sorted.json \
    --tail-count 64 \
    --dataset-dir calibration_datasets/requests \
    --datasets mmlu_pro_200.jsonl gsm8k_200.jsonl \
    --max-samples 200 \
    --output mixed_accuracy.json
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import logging
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch as _torch
from auto_round.inference.convert_model import convert_hf_model
from dynaexq.runtime import ExpertID
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_calibration_accuracy import (  # type: ignore  # noqa: E402
    SUPPORTED_TASKS,
    UNSUPPORTED_FILES,
    DatasetSummary,
    evaluate_dataset,
)


def _ensure_real_torch(module: ModuleType) -> ModuleType:
    """
    AutoRound installs helper packages that include a stub ``torch`` package
    under ``auto_round_extension/torch``.  If that directory is inserted ahead
    of the real PyTorch installation on ``sys.path`` (for example via a custom
    ``PYTHONPATH``), then ``import torch`` resolves to the stub and any access
    to ``torch.Tensor`` raises ``AttributeError``.

    When we detect this situation, demote the shadowing directory to the end of
    ``sys.path`` and re-import the canonical PyTorch package.
    """

    if hasattr(module, "Tensor"):
        return module

    module_file = getattr(module, "__file__", "") or ""
    logger = logging.getLogger("calibration_eval_mixed")

    if module_file and "auto_round_extension" in module_file:
        offending_dir = str(Path(module_file).resolve().parents[1])
        try:
            sys.path.remove(offending_dir)
        except ValueError:
            pass
        sys.path.append(offending_dir)
        sys.modules.pop("torch", None)
        fixed_module = importlib.import_module("torch")
        if hasattr(fixed_module, "Tensor"):
            logger.warning(
                "Resolved PyTorch shadowing from %s by reloading the real torch package.",
                module_file,
            )
            return fixed_module

    raise RuntimeError(
        "Failed to import PyTorch with tensor support. "
        "Ensure that auto_round_extension is not shadowing the real torch package."
    )


torch = _ensure_real_torch(_torch)


LOGGER = logging.getLogger("calibration_eval_mixed")


# -----------------------------------------------------------------------------
# Activation helpers
# -----------------------------------------------------------------------------


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


def select_tail_experts(
    activation_map: Dict[int, List[int]], tail_count: int
) -> Set[ExpertID]:
    selected: Set[ExpertID] = set()
    if tail_count <= 0:
        return selected
    for layer, experts in activation_map.items():
        if not experts:
            continue
        count = min(tail_count, len(experts))
        for idx in experts[-count:]:
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
        except IndexError as exc:  # pragma: no cover - defensive guard
            raise ValueError(
                f"Expert {expert} is out of bounds for the loaded model"
            ) from exc


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def load_mixed_precision_model(
    *,
    fp16_path: str,
    int4_path: str,
    low_precision_experts: Set[ExpertID],
    device: str,
    torch_dtype: Optional[str],
    base_quantization: str,
    low_precision_quantization: str,
    trust_remote_code: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    def _load_checkpoint(
        path: str,
        quantization: str,
        dtype_hint: Optional[str],
    ) -> AutoModelForCausalLM:
        dtype_obj = None
        if quantization.lower() != "none":
            dtype_obj = torch.float16
        elif dtype_hint and dtype_hint.lower() != "auto":
            dtype_obj = getattr(torch, dtype_hint)
        elif device.startswith("cuda"):
            dtype_obj = torch.float16

        LOGGER.info(
            "Loading %s model from %s (quantization=%s)",
            "base" if path == fp16_path else "low-precision",
            path,
            quantization,
        )
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype_obj,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )

        quantization = quantization.lower()
        if quantization in {"autoround-int4", "autoround-int2"}:
            LOGGER.info("Converting AutoRound (%s) modules to quantized kernels",
                        quantization)
            model, _ = convert_hf_model(model, target_device="cpu")
        elif quantization != "none":
            raise ValueError(
                f"Unsupported quantization mode '{quantization}'. "
                "Valid options: none, autoround-int4, autoround-int2."
            )

        return model

    model = _load_checkpoint(
        fp16_path,
        base_quantization,
        torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        fp16_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if low_precision_experts:
        quant_model = _load_checkpoint(
            int4_path,
            low_precision_quantization,
            torch_dtype,
        )
        apply_quantized_experts(model, quant_model, low_precision_experts)

        del quant_model
        gc.collect()

    LOGGER.info("Moving mixed-precision model to %s", device)
    model.to(device)
    model.eval()

    return model, tokenizer


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def build_generation_defaults(args: argparse.Namespace) -> Dict[str, object]:
    sampling_requested = (args.temperature is not None and args.temperature > 0.0) or (
        args.top_p is not None and args.top_p < 1.0
    )
    generation_defaults: Dict[str, object] = {"do_sample": sampling_requested}
    if sampling_requested:
        if args.temperature is not None and args.temperature > 0.0:
            generation_defaults["temperature"] = args.temperature
        if args.top_p is not None and args.top_p < 1.0:
            generation_defaults["top_p"] = args.top_p
    if args.max_new_tokens is not None:
        generation_defaults["max_new_tokens"] = args.max_new_tokens
    return generation_defaults


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    activation_map = load_activation_spec(Path(args.activation_file))
    downgraded = select_tail_experts(activation_map, args.tail_count)
    LOGGER.info(
        "Downgrading %d experts to INT4 precision (tail count %d)",
        len(downgraded),
        args.tail_count,
    )

    model, tokenizer = load_mixed_precision_model(
        fp16_path=args.fp16,
        int4_path=args.int4,
        low_precision_experts=downgraded,
        device=args.device,
        torch_dtype=args.torch_dtype,
        base_quantization=args.base_quantization,
        low_precision_quantization=args.low_precision_quantization,
        trust_remote_code=True,
    )

    dataset_names = args.datasets or list(SUPPORTED_TASKS.keys())
    generation_defaults = build_generation_defaults(args)
    if args.user_prefix or args.user_suffix:
        LOGGER.info(
            "Applying user prompt wrappers (prefix=%r, suffix=%r)",
            args.user_prefix,
            args.user_suffix,
        )

    results: Dict[str, object] = {
        "fp16_path": args.fp16,
        "int4_path": args.int4,
        "device": args.device,
        "tail_count": args.tail_count,
        "base_quantization": args.base_quantization,
        "low_precision_quantization": args.low_precision_quantization,
        "downgraded_experts": [
            {"layer": expert.layer, "idx": expert.idx}
            for expert in sorted(downgraded, key=lambda e: (e.layer, e.idx))
        ],
        "datasets": {},
    }

    summaries: List[DatasetSummary] = []
    for dataset_name in dataset_names:
        dataset_path = dataset_dir / dataset_name
        if dataset_name in UNSUPPORTED_FILES:
            LOGGER.warning(
                "Skipping dataset %s (automatic scoring not implemented).", dataset_name
            )
            summary = DatasetSummary(
                name=dataset_name,
                path=dataset_path,
                total=0,
                correct=0,
                skipped=0,
                accuracy=None,
                unsupported=True,
            )
            summaries.append(summary)
            continue

        task = SUPPORTED_TASKS.get(dataset_name)
        if task is None:
            LOGGER.warning(
                "No task handler registered for %s, skipping.", dataset_name)
            summary = DatasetSummary(
                name=dataset_name,
                path=dataset_path,
                total=0,
                correct=0,
                skipped=0,
                accuracy=None,
                unsupported=True,
            )
            summaries.append(summary)
            continue

        if not dataset_path.exists():
            LOGGER.warning("Dataset file not found: %s", dataset_path)
            summary = DatasetSummary(
                name=task.name,
                path=dataset_path,
                total=0,
                correct=0,
                skipped=0,
                accuracy=None,
                unsupported=True,
            )
            summaries.append(summary)
            continue

        summary = evaluate_dataset(
            task=task,
            path=dataset_path,
            model=model,
            tokenizer=tokenizer,
            max_samples=args.max_samples,
            generation_defaults=generation_defaults,
            keep_details=args.keep_details,
            user_prefix=args.user_prefix,
            user_suffix=args.user_suffix,
        )
        summaries.append(summary)

    for summary in summaries:
        if summary.unsupported:
            LOGGER.info("%s: unsupported (skipped)", summary.name)
        else:
            LOGGER.info(
                "%s: accuracy=%s (%d/%d), skipped=%d",
                summary.name,
                f"{summary.accuracy:.3f}" if summary.accuracy is not None else "n/a",
                summary.correct,
                summary.total - summary.skipped,
                summary.skipped,
            )
        payload: Dict[str, object] = {
            "path": str(summary.path),
            "total": summary.total,
            "correct": summary.correct,
            "skipped": summary.skipped,
            "accuracy": summary.accuracy,
            "unsupported": summary.unsupported,
        }
        if args.keep_details and summary.details is not None:
            payload["samples"] = [
                {
                    "id": sample.prompt_id,
                    "reference": sample.reference,
                    "raw_output": sample.raw_output,
                    "parsed_prediction": sample.parsed_prediction,
                    "correct": sample.correct,
                    "notes": sample.notes,
                }
                for sample in summary.details
            ]
        results["datasets"][summary.name] = payload

    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate mixed FP16/INT4 accuracy on calibration datasets."
    )
    parser.add_argument(
        "--fp16",
        required=True,
        help="Path to the FP16 (or BF16) full-precision model directory.",
    )
    parser.add_argument(
        "--int4",
        required=True,
        help="Path to the AutoRound INT4 checkpoint to draw low-precision experts from.",
    )
    parser.add_argument(
        "--activation-file",
        required=True,
        help="JSON file with per-layer expert activation rankings.",
    )
    parser.add_argument(
        "--tail-count",
        type=int,
        default=32,
        help="Number of least-active experts per layer to downgrade to INT4.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="calibration_datasets/requests",
        help="Directory containing calibration JSONL files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific dataset filenames to evaluate (defaults to all supported).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on samples per dataset.",
    )
    parser.add_argument(
        "--keep-details",
        action="store_true",
        help="Include per-sample outputs in the JSON results.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling parameter.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens for all tasks.",
    )
    parser.add_argument(
        "--user-prefix",
        type=str,
        default="",
        help="Optional prefix prepended to every user prompt.",
    )
    parser.add_argument(
        "--user-suffix",
        type=str,
        default="",
        help="Optional suffix appended to every user prompt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device placement for the resulting mixed model (e.g., 'cuda:0').",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default=None,
        help="Optional torch dtype override for the FP16 base model.",
    )
    parser.add_argument(
        "--base-quantization",
        type=str,
        choices=["none", "autoround-int4", "autoround-int2"],
        default="none",
        help="Quantization format for --fp16 checkpoint (allows Int4 base models).",
    )
    parser.add_argument(
        "--low-precision-quantization",
        type=str,
        choices=["none", "autoround-int4", "autoround-int2"],
        default="autoround-int4",
        help="Quantization format for --int4 checkpoint (e.g., autoround-int2).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save aggregated JSON results.",
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="When set, merge results into an existing JSON file instead of overwriting.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, ...).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        results = evaluate(args)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Mixed accuracy evaluation failed: %s", exc)
        return 1

    print(json.dumps(results, indent=2, ensure_ascii=False))

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
                    "Existing output file %s is not valid JSON. Overwriting.", output_path
                )
        if not isinstance(payload, dict):
            payload = {}
        payload_key = Path(args.fp16).name
        payload[payload_key] = results
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        LOGGER.info("Saved results to %s", output_path)

    LOGGER.info("Mixed-precision accuracy evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
