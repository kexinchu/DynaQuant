#!/usr/bin/env python3
"""Collect exact per-layer active sets for the blocking-offload benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynaexq.experiments.eval_quality import (
    PAPER_PROTOCOL,
    SCHEMA_VERSION,
    checkpoint_metadata,
    environment_metadata,
)
from scripts.benchmark_blocking_offload import (
    INPUT_LENGTHS,
    MEASURED_TRIALS,
    MODEL_CONTRACTS,
    WARMUP_TRIALS,
)
from scripts.collect_activation_density import (
    ActivationDensityCollector,
    MODEL_CONTRACTS as ROUTER_CONTRACTS,
    last_logit_only_kwargs,
)


TRIALS = WARMUP_TRIALS + MEASURED_TRIALS
MAX_INPUT_TOKENS = max(INPUT_LENGTHS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_rows(path: Path) -> tuple[list[dict], dict]:
    """Load a repository-local JSONL corpus with stable unique IDs."""
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("routing-trace prompts must be inside the repository")
    rows = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL line {line_number}") from error
        if (
            not isinstance(row, dict)
            or not str(row.get("id", "")).strip()
            or not str(row.get("prompt", "")).strip()
        ):
            raise ValueError(
                f"line {line_number} requires nonempty id and prompt"
            )
        rows.append({"id": str(row["id"]), "prompt": str(row["prompt"])})
    if not rows:
        raise ValueError("routing-trace prompt corpus is empty")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("routing-trace prompt IDs must be unique")
    return rows, {
        "path": resolved.relative_to(ROOT).as_posix(),
        "source_sha256": _sha256(resolved),
    }


def build_token_windows(
    tokenizer,
    rows: list[dict],
) -> tuple[list[list[int]], dict]:
    """Build 12 disjoint 2,048-token blocks from a deterministic text stream."""
    required = TRIALS * MAX_INPUT_TOKENS
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must define eos_token_id")
    token_stream: list[int] = []
    used_ids = []
    for row in rows:
        encoded = tokenizer(
            row["prompt"],
            add_special_tokens=False,
        )
        ids = encoded["input_ids"]
        if (
            not isinstance(ids, list)
            or any(not isinstance(value, int) for value in ids)
        ):
            raise ValueError("tokenizer returned invalid input_ids")
        token_stream.extend(ids)
        token_stream.append(int(eos))
        used_ids.append(row["id"])
        if len(token_stream) >= required:
            break
    if len(token_stream) < required:
        raise ValueError(
            "routing-trace corpus has "
            f"{len(token_stream)} tokens; {required} are required"
        )
    selected = token_stream[:required]
    windows = [
        selected[index * MAX_INPUT_TOKENS : (index + 1) * MAX_INPUT_TOKENS]
        for index in range(TRIALS)
    ]
    if any(len(window) != MAX_INPUT_TOKENS for window in windows):
        raise RuntimeError("failed to construct exact routing-trace windows")
    return windows, {
        "selected_row_count": len(used_ids),
        "selected_ids_sha256": hashlib.sha256(
            json.dumps(used_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "selected_token_count": required,
        "selection": "concatenated_eos_separated_disjoint_2048_token_blocks",
    }


def measure_expert_storage(
    model: torch.nn.Module,
    *,
    layer_ids: list[int],
    experts_per_layer: int,
) -> tuple[list[int], dict[str, list[dict]]]:
    """Measure routed-expert tensor storage and retain raw byte evidence."""
    records = {layer: [] for layer in layer_ids}
    tensors = [*model.named_parameters(), *model.named_buffers()]
    for name, tensor in tensors:
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
        if match is None or "experts" not in name.split("."):
            continue
        layer = int(match.group(1))
        if layer in records:
            numel = int(tensor.numel())
            element_size = int(tensor.element_size())
            records[layer].append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": numel,
                    "element_size": element_size,
                    "size_bytes": numel * element_size,
                }
            )
    per_expert = []
    for layer in layer_ids:
        total = sum(record["size_bytes"] for record in records[layer])
        if total <= 0 or total % experts_per_layer:
            raise RuntimeError(
                f"layer {layer} routed-expert storage ({total} bytes) "
                f"is not positive and divisible by {experts_per_layer}"
            )
        per_expert.append(total // experts_per_layer)
    return per_expert, {
        str(layer): records[layer]
        for layer in layer_ids
    }


def measure_expert_bytes(
    model: torch.nn.Module,
    *,
    layer_ids: list[int],
    experts_per_layer: int,
) -> list[int]:
    """Compatibility helper returning one routed expert's bytes per layer."""
    per_expert, _ = measure_expert_storage(
        model,
        layer_ids=layer_ids,
        experts_per_layer=experts_per_layer,
    )
    return per_expert


def collect_trace(
    model,
    *,
    windows: list[list[int]],
    device: torch.device,
    experts_per_layer: int,
    topk: int,
) -> tuple[list[dict], list[int]]:
    """Run nested prefixes and retain canonical raw active-expert sets."""
    if len(windows) != TRIALS:
        raise ValueError(f"routing trace requires exactly {TRIALS} windows")
    collector = ActivationDensityCollector(
        model,
        experts_per_layer=experts_per_layer,
        topk=topk,
    )
    last_logit_kwargs = last_logit_only_kwargs(model)
    model.eval()
    points = []
    try:
        for input_tokens in INPUT_LENGTHS:
            trials = []
            for trial_index, window in enumerate(windows):
                print(
                    json.dumps(
                        {
                            "progress": "routing_active_set_trace",
                            "input_tokens": input_tokens,
                            "trial": trial_index + 1,
                            "trials": TRIALS,
                        }
                    ),
                    flush=True,
                )
                input_ids = torch.tensor(
                    [window[:input_tokens]],
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.ones_like(input_ids)
                collector.begin(attention_mask)
                with torch.inference_mode():
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                        **last_logit_kwargs,
                    )
                trials.append(
                    {
                        "trial_id": (
                            f"tokens{input_tokens}-trial{trial_index:02d}"
                        ),
                        "phase": (
                            "warmup"
                            if trial_index < WARMUP_TRIALS
                            else "measured"
                        ),
                        "layer_active_experts": (
                            collector.snapshot_active_experts()
                        ),
                    }
                )
            points.append(
                {
                    "input_tokens": input_tokens,
                    "trials": trials,
                }
            )
    finally:
        collector.close()
    return points, collector.layer_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-model",
        required=True,
        choices=tuple(MODEL_CONTRACTS),
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="One device such as cuda:0, or auto for model sharding",
    )
    parser.add_argument("--hash-model-files", action="store_true")
    args = parser.parse_args()

    checkpoint = checkpoint_metadata(
        args.model_path,
        hash_weight_files=args.hash_model_files,
    )
    if checkpoint.get("local") is True and not checkpoint.get(
        "weight_hashes_included"
    ):
        parser.error("local paper checkpoints require --hash-model-files")
    if checkpoint.get("local") is False and not checkpoint.get("revision"):
        parser.error("remote paper checkpoints require an immutable revision")
    try:
        rows, prompt_source = load_prompt_rows(args.prompts)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = checkpoint.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        revision=revision,
        trust_remote_code=True,
    )
    try:
        windows, window_source = build_token_windows(tokenizer, rows)
    except ValueError as error:
        parser.error(str(error))
    device_map: str | dict[str, str] = (
        "auto" if args.device == "auto" else {"": args.device}
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        revision=revision,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map=device_map,
    )
    device = model.get_input_embeddings().weight.device
    router_contract = ROUTER_CONTRACTS[args.paper_model]
    points, layer_ids = collect_trace(
        model,
        windows=windows,
        device=device,
        experts_per_layer=router_contract["experts"],
        topk=router_contract["topk"],
    )
    layers_expected, experts_expected = MODEL_CONTRACTS[args.paper_model]
    if (
        len(layer_ids) != layers_expected
        or router_contract["experts"] != experts_expected
    ):
        parser.error("observed routers do not match the paper model contract")
    try:
        expert_bytes, expert_storage = measure_expert_storage(
            model,
            layer_ids=layer_ids,
            experts_per_layer=experts_expected,
        )
    except RuntimeError as error:
        parser.error(str(error))

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "routing_active_set_trace",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_model": args.paper_model,
        "checkpoint": checkpoint,
        "seed": PAPER_PROTOCOL["seed"],
        "environment": environment_metadata(),
        "protocol": {
            "name": "tc_routing_active_set_v1",
            "input_lengths": list(INPUT_LENGTHS),
            "warmup_trials": WARMUP_TRIALS,
            "measured_trials": MEASURED_TRIALS,
            "batch_size": 1,
            "padding": "none",
            "prefix_policy": "nested_prefix_per_disjoint_source_window",
            "router_metric": "unique_selected_experts_per_layer",
            "topk": router_contract["topk"],
            "causal_lm_logits_scope": "last_position_only",
            "expert_payload_measurement": (
                "stored_routed_expert_parameter_and_buffer_bytes"
            ),
            "prompt_source": {**prompt_source, **window_source},
        },
        "moe_layer_ids": layer_ids,
        "experts_per_layer": experts_expected,
        "expert_bytes_per_layer": expert_bytes,
        "expert_storage_tensors": expert_storage,
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paper_model": args.paper_model,
                "moe_layers": len(layer_ids),
                "points": len(points),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
