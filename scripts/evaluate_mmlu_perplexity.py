#!/usr/bin/env python3
"""
Evaluate token-level perplexity for MMLU-Pro prompts across multiple model checkpoints.

This script reuses the prompt construction logic from ``evaluate_calibration_accuracy.py``
to ensure identical formatting when computing log-likelihoods.  For each supplied model,
it measures the average negative log-likelihood (in nats) assigned to the ground-truth
answer token and reports the corresponding perplexity.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from evaluate_calibration_accuracy import (  # type: ignore
    SUPPORTED_TASKS,
    iter_jsonl,
    load_model_and_tokenizer,
    resolve_model_device,
)

LOGGER = logging.getLogger("dynaexq.perplexity")


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass
class SamplePerplexity:
    """Container for per-sample perplexity diagnostics."""

    prompt_id: Optional[str]
    reference: str
    target_token_count: int
    neg_log_likelihood: float

    @property
    def avg_nll(self) -> float:
        if self.target_token_count == 0:
            return math.inf
        return self.neg_log_likelihood / self.target_token_count

    @property
    def perplexity(self) -> float:
        avg = self.avg_nll
        if not math.isfinite(avg):
            return math.inf
        return math.exp(avg)


@dataclass
class AggregatePerplexity:
    """Aggregated perplexity statistics for a dataset/model pair."""

    samples: int
    skipped: int
    target_tokens: int
    total_neg_log_likelihood: float
    details: Optional[List[SamplePerplexity]] = None

    @property
    def avg_nll(self) -> Optional[float]:
        if self.target_tokens == 0:
            return None
        return self.total_neg_log_likelihood / self.target_tokens

    @property
    def perplexity(self) -> Optional[float]:
        avg = self.avg_nll
        if avg is None:
            return None
        try:
            return math.exp(avg)
        except OverflowError:
            return math.inf


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def parse_model_specs(specs: Iterable[str]) -> List[Tuple[str, str]]:
    """
    Parse CLI model specifications of the form ``label=/abs/path``.

    If the label is omitted, the basename of the path is used.
    """
    parsed: List[Tuple[str, str]] = []
    for spec in specs:
        if "=" in spec:
            label, path = spec.split("=", 1)
            label = label.strip()
            path = path.strip()
            if not label:
                raise ValueError(f"Invalid model label in spec: {spec!r}")
        else:
            path = spec.strip()
            label = Path(path).name
            LOGGER.warning(
                "No label provided for model %s; using '%s'.", path, label
            )
        if not path:
            raise ValueError(f"Missing model path in spec: {spec!r}")
        parsed.append((label, path))
    return parsed


def build_prompt(
    *,
    task,
    entry: Dict,
    tokenizer,
    user_prefix: str,
    user_suffix: str,
) -> Tuple[str, str]:
    """
    Construct the formatted prompt string and the ground-truth answer for a sample.
    """
    user_prompt = task.build_user_prompt(entry)
    reference = task.extract_reference(entry)

    messages: List[Dict[str, str]] = []
    if task.system_prompt:
        messages.append({"role": "system", "content": task.system_prompt})

    if user_prefix or user_suffix:
        composed = f"{user_prefix}{user_prompt}{user_suffix}"
    else:
        composed = user_prompt

    messages.append({"role": "user", "content": composed})

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return formatted_prompt, reference


def compute_sample_neg_log_likelihood(
    *,
    model,
    tokenizer,
    formatted_prompt: str,
    target_text: str,
    device: torch.device,
) -> Tuple[float, int]:
    """
    Compute the negative log-likelihood for ``target_text`` appended to ``formatted_prompt``.
    """
    prompt_encoding = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )
    full_encoding = tokenizer(
        formatted_prompt + target_text,
        return_tensors="pt",
    )

    prompt_len = prompt_encoding["input_ids"].shape[1]
    total_len = full_encoding["input_ids"].shape[1]
    target_len = total_len - prompt_len
    if target_len <= 0:
        return math.nan, 0

    inputs = {
        key: value.to(device)
        for key, value in full_encoding.items()
    }
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(
            **inputs,
            labels=labels,
        )
    loss = float(outputs.loss)
    neg_log_likelihood = loss * target_len
    return neg_log_likelihood, target_len


def evaluate_mmlu_perplexity(
    *,
    model,
    tokenizer,
    dataset_path: Path,
    task,
    max_samples: Optional[int],
    include_eos: bool,
    user_prefix: str,
    user_suffix: str,
    keep_details: bool,
) -> AggregatePerplexity:
    """
    Iterate over the dataset to compute aggregate perplexity statistics.
    """
    model_device = resolve_model_device(model)
    total_tokens = 0
    total_nll = 0.0
    consumed = 0
    skipped = 0
    details: List[SamplePerplexity] = [] if keep_details else None

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.debug("Evaluating perplexity on %s", dataset_path)

    for entry in iter_jsonl(dataset_path):
        if max_samples is not None and consumed >= max_samples:
            break

        prompt_id = entry.get("id")
        try:
            formatted_prompt, reference = build_prompt(
                task=task,
                entry=entry,
                tokenizer=tokenizer,
                user_prefix=user_prefix,
                user_suffix=user_suffix,
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning(
                "Skipping entry %s due to prompt construction failure: %s",
                prompt_id,
                exc,
            )
            skipped += 1
            continue

        if not reference:
            LOGGER.warning(
                "Skipping entry %s: empty reference answer.", prompt_id
            )
            skipped += 1
            continue

        target_text = reference
        if include_eos and tokenizer.eos_token:
            target_text = f"{reference}{tokenizer.eos_token}"

        neg_log_likelihood, token_count = compute_sample_neg_log_likelihood(
            model=model,
            tokenizer=tokenizer,
            formatted_prompt=formatted_prompt,
            target_text=target_text,
            device=model_device,
        )

        if token_count <= 0 or not math.isfinite(neg_log_likelihood):
            LOGGER.warning(
                "Skipping entry %s: invalid token count (%s) or NLL (%s).",
                prompt_id,
                token_count,
                neg_log_likelihood,
            )
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
                    reference=reference,
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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


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
        help=(
            "Dataset key registered in SUPPORTED_TASKS. "
            "Defaults to 'mmlu_pro_200.jsonl' for MMLU-Pro."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Model specification of the form label=/abs/path. "
            "Can be repeated to compare multiple checkpoints."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device placement hint passed to the model loader.",
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
        help="Append the tokenizer EOS token to the reference answer when scoring.",
    )
    parser.add_argument(
        "--user-prefix",
        type=str,
        default="",
        help="Optional prefix to prepend to each user prompt (e.g., '<think>').",
    )
    parser.add_argument(
        "--user-suffix",
        type=str,
        default="",
        help="Optional suffix to append to each user prompt (e.g., '</think>').",
    )
    parser.add_argument(
        "--keep-details",
        action="store_true",
        help="Retain per-sample perplexity diagnostics in the output JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save aggregated results as JSON.",
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help=(
            "When --output is specified, merge the new results into the existing JSON "
            "instead of overwriting it."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, ...).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        LOGGER.error("Dataset file not found: %s", dataset_path)
        return 1

    try:
        task = SUPPORTED_TASKS[args.dataset_key]
    except KeyError:
        LOGGER.error(
            "Dataset key %s is not registered in SUPPORTED_TASKS.", args.dataset_key
        )
        return 1

    model_specs = parse_model_specs(args.model)
    results: Dict[str, Dict[str, object]] = {}

    for label, model_path in model_specs:
        LOGGER.info("Loading model (%s) from %s", label, model_path)
        try:
            model, tokenizer = load_model_and_tokenizer(
                model_path,
                device=args.device,
                torch_dtype=args.torch_dtype,
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error(
                "Failed to load model %s from %s: %s", label, model_path, exc
            )
            results[label] = {
                "model_path": model_path,
                "error": str(exc),
            }
            continue

        aggregate = evaluate_mmlu_perplexity(
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

        summary: Dict[str, object] = {
            "model_path": model_path,
            "samples": aggregate.samples,
            "skipped": aggregate.skipped,
            "target_tokens": aggregate.target_tokens,
            "avg_neg_log_likelihood": aggregate.avg_nll,
            "perplexity": aggregate.perplexity,
        }
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

        results[label] = summary

        # Free model memory before loading the next checkpoint.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_payload = results

    if args.output and args.append_output:
        output_path = Path(args.output)
        if output_path.exists():
            try:
                with output_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
            except json.JSONDecodeError as exc:
                LOGGER.warning(
                    "Failed to parse existing output file %s: %s. Overwriting.",
                    output_path,
                    exc,
                )
                combined = results
            else:
                if not isinstance(existing, dict):
                    LOGGER.warning(
                        "Existing output file %s does not contain a dict. Overwriting.",
                        output_path,
                    )
                    combined = results
                else:
                    combined = dict(existing)
                    combined.update(results)
        else:
            combined = results
        output_payload = combined

    print(json.dumps(output_payload, indent=2, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output_payload, handle, indent=2, ensure_ascii=False)
        LOGGER.info("Saved perplexity results to %s", output_path)

    LOGGER.info("Perplexity evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
