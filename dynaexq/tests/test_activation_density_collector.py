from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts import collect_activation_density as collector_module
from scripts.collect_activation_density import (
    ActivationDensityCollector,
    last_logit_only_kwargs,
    load_prompts,
    prepare_one_token_decode,
)


class _SelectedGate(torch.nn.Module):
    def forward(self, selected: torch.Tensor):
        weights = torch.ones_like(selected, dtype=torch.float32)
        probabilities = torch.zeros(
            (*selected.shape[:-1], 4),
            dtype=torch.float32,
        )
        return probabilities, weights, selected


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = _SelectedGate()


class _TinyMoE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(), _Layer()])


class _FullSequenceGenerationInputs:
    def prepare_inputs_for_generation(self, generated, **kwargs):
        return {
            "input_ids": generated,
            "position_ids": torch.arange(generated.shape[1])[None, :],
            "cache_position": torch.arange(generated.shape[1]),
            **kwargs,
        }


class _ModernCausalLM:
    def forward(self, input_ids, logits_to_keep=0):
        del input_ids, logits_to_keep


class _LegacyCausalLM:
    def forward(self, input_ids, num_logits_to_keep=0):
        del input_ids, num_logits_to_keep


class _UnboundedCausalLM:
    def forward(self, input_ids):
        del input_ids


def test_phi35_router_contract_matches_checkpoint_architecture():
    assert collector_module.MODEL_CONTRACTS["phi35"] == {
        "experts": 16,
        "topk": 2,
    }


def test_collector_masks_padding_and_resets_between_stages():
    model = _TinyMoE()
    collector = ActivationDensityCollector(
        model,
        experts_per_layer=4,
        topk=2,
    )
    selected = torch.tensor([[0, 1], [1, 2], [2, 3]])
    collector.begin(torch.tensor([[1, 1, 0]]))
    for layer in model.layers:
        layer.gate(selected)
    assert collector.layer_ids == [0, 1]
    assert collector.snapshot() == [3, 3]
    assert collector.snapshot_active_experts() == {
        "0": [0, 1, 2],
        "1": [0, 1, 2],
    }

    collector.begin(torch.tensor([[1]]))
    for layer in model.layers:
        layer.gate(torch.tensor([[3, 0]]))
    assert collector.snapshot() == [2, 2]
    collector.close()


def test_prompt_loader_requires_160_unique_repository_rows(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(collector_module, "ROOT", tmp_path)
    path = tmp_path / "prompts.jsonl"
    rows = [
        {"id": f"sample-{index}", "prompt": f"prompt {index}"}
        for index in range(160)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    selected, provenance = load_prompts(path, repeats=5)
    assert len(selected) == 160
    assert provenance["path"] == "prompts.jsonl"
    assert len(provenance["source_sha256"]) == 64
    assert len(provenance["selected_ids_sha256"]) == 64

    rows[-1]["id"] = rows[0]["id"]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_prompts(path, repeats=5)


def test_decode_input_normalization_keeps_exactly_one_token():
    generated = torch.tensor([[1, 2, 3]])
    next_token = generated[:, -1:]
    attention_mask = torch.ones_like(generated)
    past = object()
    prepared = prepare_one_token_decode(
        _FullSequenceGenerationInputs(),
        generated=generated,
        next_token=next_token,
        attention_mask=attention_mask,
        past_key_values=past,
    )
    assert prepared["input_ids"].tolist() == [[3]]
    assert prepared["position_ids"].shape == (1, 1)
    assert prepared["cache_position"].shape == (1,)
    assert prepared["past_key_values"] is past

    with pytest.raises(RuntimeError, match="no KV cache"):
        prepare_one_token_decode(
            _FullSequenceGenerationInputs(),
            generated=generated,
            next_token=next_token,
            attention_mask=attention_mask,
            past_key_values=None,
        )


def test_last_logit_scope_is_required_and_model_specific():
    assert last_logit_only_kwargs(_ModernCausalLM()) == {
        "logits_to_keep": 1
    }
    assert last_logit_only_kwargs(_LegacyCausalLM()) == {
        "num_logits_to_keep": 1
    }
    with pytest.raises(RuntimeError, match="last-position-only"):
        last_logit_only_kwargs(_UnboundedCausalLM())


def test_collector_script_is_directly_executable():
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "collect_activation_density.py"),
            "--help",
        ],
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
