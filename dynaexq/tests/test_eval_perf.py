from __future__ import annotations

import json

import pytest

import torch

from dynaexq.experiments import eval_perf
from dynaexq.experiments.eval_perf import fixed_length_inputs, percentile


class _Tokenizer:
    bos_token_id = 1

    @staticmethod
    def encode(text, add_special_tokens=False):
        assert not add_special_tokens
        return [2, 3, 4]


def test_fixed_length_inputs_have_no_padding():
    input_ids, mask = fixed_length_inputs(
        _Tokenizer(), "ignored", input_length=8, batch_size=2
    )
    assert input_ids.tolist() == [
        [1, 2, 3, 4, 2, 3, 4, 2],
        [1, 2, 3, 4, 2, 3, 4, 2],
    ]
    assert mask.sum().item() == 16


def test_percentile_uses_nearest_rank():
    values = list(range(1, 101))
    assert percentile(values, 0.50) == 50
    assert percentile(values, 0.95) == 95
    assert percentile(values, 0.99) == 99


def test_percentile_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile([1], 1.1)


def test_external_iteration_callbacks_exclude_warmup_telemetry(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

    calls = []
    resets = []

    def fake_generation(
        model,
        input_ids,
        attention_mask,
        output_length,
        process_memory_monitor=None,
    ):
        assert process_memory_monitor is None
        calls.append(input_ids.shape)
        return {
            key: 1.0
            for key in (
                "model_ttft_ms",
                "model_tpot_ms",
                "model_e2e_ms",
                "throughput_tokens_s",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            )
        }

    monkeypatch.setattr(eval_perf, "_one_generation", fake_generation)
    result = eval_perf.measure_latency(
        Model(),
        _Tokenizer(),
        batch_size=2,
        input_length=8,
        output_length=3,
        n_warmup=2,
        n_repeats=3,
        iteration_setup=lambda input_ids: calls.append("setup"),
        after_warmup=lambda: resets.append(len(calls)),
    )
    assert resets == [4]
    assert len(calls) == 10
    assert len(result["samples"]) == 3


def test_sharegpt_trace_selection_is_deterministic_and_hashed(tmp_path):
    class Tokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            tokens = list(range(len(text.split())))
            return ([99] if add_special_tokens else []) + tokens

    rows = [
        {
            "id": "long-input",
            "conversations": [
                {"from": "human", "value": " ".join(["x"] * 80)},
                {"from": "gpt", "value": " ".join(["y"] * 20)},
            ],
        },
        {
            "id": "long-output",
            "conversations": [
                {"from": "human", "value": " ".join(["x"] * 20)},
                {"from": "gpt", "value": " ".join(["y"] * 80)},
            ],
        },
    ]
    source = tmp_path / "sharegpt.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    prefill, metadata = eval_perf.load_sharegpt_trace(
        source,
        Tokenizer(),
        regime="prefill",
        request_count=1,
        max_input_tokens=128,
        max_output_tokens=128,
    )
    decode, _ = eval_perf.load_sharegpt_trace(
        source,
        Tokenizer(),
        regime="decode",
        request_count=1,
        max_input_tokens=128,
        max_output_tokens=128,
    )
    assert prefill[0]["request_id"] == "long-input"
    assert decode[0]["request_id"] == "long-output"
    assert metadata["source_sha256"]
    assert metadata["selected_ids_sha256"]


def test_trace_latency_preserves_request_level_lengths(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def eval(self):
            return self

    def fake_generation(
        model,
        input_ids,
        attention_mask,
        output_length,
        process_memory_monitor=None,
    ):
        assert process_memory_monitor is None
        return {
            "model_ttft_ms": float(input_ids.shape[1]),
            "model_tpot_ms": float(output_length),
            "model_e2e_ms": 3.0,
            "throughput_tokens_s": 4.0,
            "peak_allocated_bytes": 5.0,
            "peak_reserved_bytes": 6.0,
        }

    monkeypatch.setattr(eval_perf, "_one_generation", fake_generation)
    result = eval_perf.measure_trace_latency(
        Model(),
        _Tokenizer(),
        [
            {
                "request_id": "a",
                "input_ids": [1, 2, 3],
                "input_tokens": 3,
                "output_tokens": 7,
            }
        ],
        n_warmup=1,
    )
    assert result["samples"][0]["request_id"] == "a"
    assert result["samples"][0]["model_ttft_ms"] == 3.0
    assert result["samples"][0]["model_tpot_ms"] == 7.0
