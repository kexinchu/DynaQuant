#!/usr/bin/env python3
"""Collect raw per-layer MoE activation density for the motivation table."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


BATCH_SIZES = (1, 2, 4, 8, 16, 32)
MODEL_CONTRACTS = {
    "qwen30b": {"experts": 128, "topk": 8},
    "qwen80b": {"experts": 512, "topk": 10},
    "deepseek_v2_lite": {"experts": 64, "topk": 6},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompts(path: Path, *, repeats: int) -> tuple[list[dict], dict]:
    """Load deterministic 32-prompt blocks with stable, unique IDs."""
    required = repeats * max(BATCH_SIZES)
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
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
        if len(rows) == required:
            break
    if len(rows) < required:
        raise ValueError(
            f"activation protocol needs {required} prompts, found {len(rows)}"
        )
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("activation prompt IDs must be unique")
    selected_hash = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(
            "activation prompts must be stored inside the repository"
        )
    return rows, {
        "path": resolved.relative_to(ROOT).as_posix(),
        "source_sha256": _sha256(resolved),
        "selected_ids_sha256": selected_hash,
        "selected_prompt_count": len(rows),
        "selection": "ordered_blocks_of_32_nested_by_batch_size",
    }


class ActivationDensityCollector:
    """Hook model routers and count unique selected experts per MoE layer."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        experts_per_layer: int,
        topk: int,
    ) -> None:
        self.experts_per_layer = experts_per_layer
        self.topk = topk
        self._mask: torch.Tensor | None = None
        self._active: dict[int, set[int]] = {}
        self._hooks = []
        self.layer_ids: list[int] = []
        hooked_layers: set[int] = set()
        for name, module in model.named_modules():
            match = re.search(r"layers\.(\d+)", name)
            if match is None:
                continue
            layer = int(match.group(1))
            if layer in hooked_layers:
                continue
            class_name = type(module).__name__.lower()
            if not (
                name.endswith(".gate")
                or "router" in class_name
                or "router" in name.rsplit(".", 1)[-1].lower()
            ):
                continue
            self._hooks.append(
                module.register_forward_hook(self._make_hook(layer))
            )
            hooked_layers.add(layer)
        self.layer_ids = sorted(hooked_layers)
        if not self.layer_ids:
            raise RuntimeError("no MoE router modules were found")

    def _make_hook(self, layer: int):
        def hook(module, inputs, output) -> None:
            del module, inputs
            selected = self._extract_selected(output)
            self._record(layer, selected)

        return hook

    def _extract_selected(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            integer_tensors = [
                value
                for value in output
                if isinstance(value, torch.Tensor)
                and not value.is_floating_point()
                and value.ndim >= 1
            ]
            if integer_tensors:
                selected = integer_tensors[-1]
                if (
                    selected.ndim >= 2
                    and selected.shape[-1] != self.topk
                    and integer_tensors[0].shape[-1] == self.topk
                ):
                    selected = integer_tensors[0]
                return selected
            floating = [
                value
                for value in output
                if isinstance(value, torch.Tensor)
                and value.is_floating_point()
                and value.shape[-1] == self.experts_per_layer
            ]
            if floating:
                return torch.topk(
                    floating[0],
                    k=self.topk,
                    dim=-1,
                    sorted=False,
                ).indices
        if isinstance(output, torch.Tensor):
            if (
                output.is_floating_point()
                and output.shape[-1] == self.experts_per_layer
            ):
                return torch.topk(
                    output,
                    k=self.topk,
                    dim=-1,
                    sorted=False,
                ).indices
            if not output.is_floating_point():
                return output
        raise RuntimeError("router hook could not identify selected experts")

    def begin(self, attention_mask: torch.Tensor) -> None:
        if attention_mask.ndim != 2:
            raise ValueError("routing attention mask must be rank two")
        self._mask = attention_mask.detach().bool().reshape(-1)
        self._active = {layer: set() for layer in self.layer_ids}

    def _record(self, layer: int, selected: torch.Tensor) -> None:
        if self._mask is None:
            raise RuntimeError("collector.begin must precede model forward")
        selected = selected.detach().reshape(-1, selected.shape[-1])
        if selected.shape[-1] != self.topk:
            raise RuntimeError(
                f"router returned top-{selected.shape[-1]}, expected {self.topk}"
            )
        if selected.shape[0] != self._mask.numel():
            raise RuntimeError(
                "router token rows do not match the attention mask; "
                "refusing to count padding ambiguously"
            )
        selected = selected[self._mask.to(selected.device)]
        if selected.numel() == 0:
            raise RuntimeError("router produced no unmasked dispatches")
        if (
            int(selected.min().item()) < 0
            or int(selected.max().item()) >= self.experts_per_layer
        ):
            raise RuntimeError("router selected an out-of-range expert")
        self._active[layer].update(
            int(value) for value in selected.unique().cpu().tolist()
        )

    def snapshot(self) -> list[int]:
        if self._mask is None:
            raise RuntimeError("collector has not started")
        missing = [
            layer for layer in self.layer_ids if not self._active[layer]
        ]
        if missing:
            raise RuntimeError(f"routers produced no counts for layers {missing}")
        return [len(self._active[layer]) for layer in self.layer_ids]

    def snapshot_active_experts(self) -> dict[str, list[int]]:
        """Return canonical raw active sets for trace-driven experiments."""
        self.snapshot()
        return {
            str(layer): sorted(self._active[layer])
            for layer in self.layer_ids
        }

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def _ratio(samples: list[list[int]], experts: int) -> float:
    values = [count for sample in samples for count in sample]
    if not values:
        raise ValueError("activation samples are empty")
    return 100.0 * sum(values) / (len(values) * experts)


def prepare_one_token_decode(
    model,
    *,
    generated: torch.Tensor,
    next_token: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
) -> dict:
    """Normalize model-specific generation inputs to one cached decode token."""
    if past_key_values is None:
        raise RuntimeError(
            "checkpoint returned no KV cache; one-token decode is unavailable"
        )
    prepared = dict(
        model.prepare_inputs_for_generation(
            generated,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
        )
    )
    input_ids = prepared.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.shape[-1] != 1:
        prepared["input_ids"] = next_token
        prepared.pop("inputs_embeds", None)
    for name in ("position_ids", "cache_position"):
        value = prepared.get(name)
        if isinstance(value, torch.Tensor) and value.shape[-1] != 1:
            prepared[name] = value[..., -1:]
    prepared["past_key_values"] = past_key_values
    prepared["attention_mask"] = attention_mask
    prepared["use_cache"] = True
    return prepared


def last_logit_only_kwargs(model) -> dict[str, int]:
    """Return the model-specific argument that suppresses full-vocab logits."""
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError) as error:
        raise RuntimeError("cannot inspect causal-LM forward signature") from error
    if "logits_to_keep" in parameters:
        return {"logits_to_keep": 1}
    if "num_logits_to_keep" in parameters:
        return {"num_logits_to_keep": 1}
    raise RuntimeError(
        "model cannot request last-position-only logits; refusing a "
        "memory-ambiguous activation run"
    )


def collect(
    model,
    tokenizer,
    prompts: list[dict],
    *,
    device: torch.device,
    experts_per_layer: int,
    topk: int,
    repeats: int,
    max_input_tokens: int,
) -> tuple[dict, list[int]]:
    """Run nested batches and return raw prefill/decode density samples."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    collector = ActivationDensityCollector(
        model,
        experts_per_layer=experts_per_layer,
        topk=topk,
    )
    stages = {"prefill": [], "decode": []}
    last_logit_kwargs = last_logit_only_kwargs(model)
    model.eval()
    try:
        for batch_size in BATCH_SIZES:
            raw = {"prefill": [], "decode": []}
            for repeat in range(repeats):
                print(
                    json.dumps(
                        {
                            "progress": "activation_density",
                            "batch_size": batch_size,
                            "repeat": repeat + 1,
                            "repeats": repeats,
                        }
                    ),
                    flush=True,
                )
                begin = repeat * max(BATCH_SIZES)
                texts = [
                    row["prompt"]
                    for row in prompts[begin : begin + batch_size]
                ]
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_tokens,
                )
                encoded = {
                    key: value.to(device) for key, value in encoded.items()
                }
                collector.begin(encoded["attention_mask"])
                with torch.inference_mode():
                    output = model(
                        **encoded,
                        use_cache=True,
                        return_dict=True,
                        **last_logit_kwargs,
                    )
                raw["prefill"].append(collector.snapshot())

                next_token = output.logits[:, -1:].argmax(dim=-1)
                generated = torch.cat((encoded["input_ids"], next_token), dim=1)
                attention_mask = torch.cat(
                    (
                        encoded["attention_mask"],
                        torch.ones_like(next_token),
                    ),
                    dim=1,
                )
                prepared = prepare_one_token_decode(
                    model,
                    generated=generated,
                    next_token=next_token,
                    attention_mask=attention_mask,
                    past_key_values=output.past_key_values,
                )
                collector.begin(
                    torch.ones(
                        (batch_size, 1),
                        dtype=torch.bool,
                        device=device,
                    )
                )
                prepared.update(last_logit_kwargs)
                with torch.inference_mode():
                    model(**prepared, return_dict=True)
                raw["decode"].append(collector.snapshot())
            for stage in ("prefill", "decode"):
                stages[stage].append(
                    {
                        "batch_size": batch_size,
                        "experts_per_layer": experts_per_layer,
                        "layer_active_counts": raw[stage],
                        "ratio_pct": _ratio(
                            raw[stage],
                            experts_per_layer,
                        ),
                    }
                )
    finally:
        collector.close()
    return stages, collector.layer_ids


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
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="One device such as cuda:0, or auto for model sharding",
    )
    parser.add_argument("--hash-model-files", action="store_true")
    args = parser.parse_args()
    if args.repeats != 5:
        parser.error("the paper protocol requires --repeats=5")
    if args.max_input_tokens != 2048:
        parser.error("the paper protocol requires --max-input-tokens=2048")

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
        prompts, prompt_provenance = load_prompts(
            args.prompts,
            repeats=args.repeats,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = checkpoint.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        revision=revision,
        trust_remote_code=True,
    )
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
    contract = MODEL_CONTRACTS[args.paper_model]
    stages, layer_ids = collect(
        model,
        tokenizer,
        prompts,
        device=device,
        experts_per_layer=contract["experts"],
        topk=contract["topk"],
        repeats=args.repeats,
        max_input_tokens=args.max_input_tokens,
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "activation_density",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_model": args.paper_model,
        "checkpoint": checkpoint,
        "seed": PAPER_PROTOCOL["seed"],
        "environment": environment_metadata(),
        "protocol": {
            "name": "tc_activation_density_v1",
            "batch_sizes": list(BATCH_SIZES),
            "repeats": args.repeats,
            "max_input_tokens": args.max_input_tokens,
            "padding_side": "left",
            "prefill_scope": "all_nonpadding_prompt_tokens",
            "decode_scope": "first_single_token_step_after_prefill",
            "decode_token_selection": "greedy_argmax_last_position",
            "causal_lm_logits_scope": "last_position_only",
            "aggregation": "mean_unique_experts_across_moe_layers_and_repeats",
            "experts_per_layer": contract["experts"],
            "topk": contract["topk"],
            "prompt_source": prompt_provenance,
        },
        "moe_layer_ids": layer_ids,
        "stages": stages,
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
