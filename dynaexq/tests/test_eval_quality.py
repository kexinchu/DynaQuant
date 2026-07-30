from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from dynaexq.experiments.datasets import (
    EvalRequest,
    _aime25_requests,
    dataset_provenance,
)
from dynaexq.experiments.eval_quality import (
    PAPER_PROTOCOL,
    _continuation_logprobs,
    autoround_load_config,
    checkpoint_metadata,
    compute_perplexity,
    evaluate,
    execute_humaneval,
    extract_final_integer,
    paper_quality_method,
    wilson_interval,
)


def test_autoround_load_config_pins_checkpoint_format_and_backend(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "auto-round",
                    "bits": 2,
                    "group_size": 64,
                    "sym": True,
                    "packing_format": "auto_round:auto_gptq",
                    "autoround_version": "0.12.0",
                    "extra_config": {
                        "model.layers.0.self_attn.q_proj": {"bits": 8},
                        "model.layers.0.mlp.gate": {
                            "bits": 16,
                            "data_type": "float",
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    config = autoround_load_config(str(tmp_path), "triton")
    assert config.bits == 2
    assert config.group_size == 64
    assert config.backend == "triton"
    assert config.packing_format == "auto_round:auto_gptq"
    assert config.autoround_version == "0.12.0"
    assert config.extra_config["model.layers.0.self_attn.q_proj"]["bits"] == 8
    assert config.extra_config["model.layers.0.mlp.gate"]["bits"] == 16


def test_extract_final_integer_prefers_explicit_final_answer():
    text = "We first try 18, but correct it later.\nAnswer: 21"
    assert extract_final_integer(text) == "21"


def test_extract_final_integer_handles_commas_and_boxed_answers():
    assert extract_final_integer(r"Therefore \boxed{1,024}.") == "1024"


def test_extract_final_integer_does_not_accept_decimal_as_integer():
    assert extract_final_integer("Answer: 3.14") is None


def test_humaneval_executes_tests_not_only_compile():
    request = EvalRequest(
        prompt="def add_one(x):\n",
        task_type="code",
        sample_id="synthetic/0",
        metadata={
            "test": "def check(candidate):\n    assert candidate(2) == 3\n",
            "entry_point": "add_one",
        },
    )
    passed, status = execute_humaneval("    return x + 1\n", request)
    assert passed
    assert status == "pass"

    passed, status = execute_humaneval("    return x\n", request)
    assert not passed
    assert status == "failed_tests"


def test_humaneval_rejects_missing_test_metadata():
    request = EvalRequest(prompt="def f():\n", task_type="code")
    assert execute_humaneval("    return 1\n", request) == (
        False,
        "missing_test_metadata",
    )


def test_checkpoint_metadata_hashes_controls_and_optionally_weights(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"tiny"}')
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    light = checkpoint_metadata(str(tmp_path))
    assert light["control_files"][0]["sha256"]
    assert "sha256" not in light["weight_files"][0]

    strong = checkpoint_metadata(str(tmp_path), hash_weight_files=True)
    assert strong["weight_hashes_included"] is True
    assert strong["weight_files"][0]["sha256"]


@pytest.mark.parametrize(
    ("paper_model", "method", "quantization", "expected"),
    [
        ("qwen30b", "reference_fp16", None, "reference_fp16"),
        ("qwen30b", "quantized_checkpoint", "int4", "static_int4"),
        ("qwen80b", "quantized_checkpoint", "int4", "static_int4"),
        ("qwen80b", "quantized_checkpoint", "int2", "static_int2"),
        ("phi35", "reference_fp16", None, "reference_fp16"),
        ("phi35", "quantized_checkpoint", "int4", "static_int4"),
    ],
)
def test_paper_quality_method_has_explicit_model_contract(
    paper_model,
    method,
    quantization,
    expected,
):
    assert (
        paper_quality_method(paper_model, method, quantization)
        == expected
    )


def test_paper_quality_method_rejects_unreported_precision():
    with pytest.raises(ValueError, match="not a reported"):
        paper_quality_method("qwen30b", "quantized_checkpoint", "int2")


def test_aime25_rows_are_validated_and_keep_provenance():
    identity = {
        "repository": "math-ai/aime25",
        "revision": "immutable",
        "config": None,
        "split": "test",
        "source_rows": 30,
        "fingerprint": "fingerprint",
    }
    requests = _aime25_requests(
        [{"problem": "Compute 1+1.", "answer": "002", "id": "aime-I-1"}],
        identity,
    )
    assert requests[0].target == "002"
    assert requests[0].sample_id == "aime25/aime-I-1"
    assert dataset_provenance(requests) == {**identity, "evaluated_rows": 1}


@pytest.mark.parametrize(
    "row",
    [
        {"problem": "", "answer": "1", "id": "blank"},
        {"problem": "x", "answer": "1000", "id": "range"},
        {"problem": "x", "answer": "3.5", "id": "decimal"},
    ],
)
def test_aime25_rejects_invalid_rows(row):
    with pytest.raises(ValueError):
        _aime25_requests([row], {"repository": "test"})


def test_dataset_provenance_rejects_missing_identity():
    with pytest.raises(ValueError, match="must carry"):
        dataset_provenance([EvalRequest(prompt="x")])


def test_paper_protocol_applies_per_benchmark_limits(monkeypatch):
    calls = []

    def fake_load(name, **kwargs):
        calls.append((name, kwargs))
        return [
            EvalRequest(
                prompt="text",
                metadata={
                    "dataset": {
                        "repository": "test",
                        "revision": "sha",
                        "config": None,
                        "split": "test",
                        "source_rows": 1,
                        "fingerprint": "fp",
                    }
                },
            )
        ]

    monkeypatch.setattr(
        "dynaexq.experiments.eval_quality.load_dataset",
        fake_load,
    )
    monkeypatch.setattr(
        "dynaexq.experiments.eval_quality.compute_mc_accuracy",
        lambda *args, **kwargs: {
            "metric": "accuracy",
            "score": 0.0,
            "total": 1,
            "evaluated": 1,
            "failed": 0,
        },
    )
    monkeypatch.setattr(
        "dynaexq.experiments.eval_quality.compute_generation_accuracy",
        lambda *args, **kwargs: {
            "metric": "accuracy",
            "score": 0.0,
            "total": 1,
            "evaluated": 1,
            "failed": 0,
        },
    )

    evaluate(
        object(),
        object(),
        ["mmlu_pro", "gpqa", "gsm8k"],
        sample_limits=PAPER_PROTOCOL["sample_limits"],
    )
    assert calls == [
        ("mmlu_pro", {}),
        ("gpqa", {}),
        ("gsm8k", {}),
    ]


def test_evaluate_rejects_conflicting_sample_controls():
    with pytest.raises(ValueError, match="mutually exclusive"):
        evaluate(
            object(),
            object(),
            [],
            n_samples=1,
            sample_limits={"mmlu_pro": 200},
        )


def test_wilson_interval_is_bounded_and_contains_observed_rate():
    interval = wilson_interval(75, 100)
    assert interval["method"] == "wilson"
    assert 0.0 <= interval["low"] < 0.75 < interval["high"] <= 1.0


@pytest.mark.parametrize("correct,total", [(-1, 2), (3, 2), (0, 0)])
def test_wilson_interval_rejects_invalid_counts(correct, total):
    with pytest.raises(ValueError):
        wilson_interval(correct, total)


def test_multiple_choice_labels_share_one_unpadded_forward():
    class Tokenizer:
        def __call__(self, text, *, add_special_tokens, return_tensors):
            assert return_tensors == "pt"
            ids = [1, 2] if add_special_tokens else [3 if text == " A" else 4]
            return SimpleNamespace(input_ids=torch.tensor([ids]))

    class Model:
        calls = 0

        def __call__(self, input_ids):
            self.calls += 1
            assert input_ids.tolist() == [[1, 2, 3], [1, 2, 4]]
            logits = torch.zeros((2, 3, 5))
            logits[0, 1, 3] = 3.0
            logits[1, 1, 4] = 1.0
            return SimpleNamespace(logits=logits)

    model = Model()
    scores = _continuation_logprobs(
        model,
        Tokenizer(),
        "question",
        [" A", " B"],
        torch.device("cpu"),
    )
    assert model.calls == 1
    assert scores[0] > scores[1]


def test_perplexity_emits_recomputable_raw_window_nll():
    class Tokenizer:
        def __call__(self, text, *, return_tensors):
            assert return_tensors == "pt"
            return SimpleNamespace(input_ids=torch.arange(10).unsqueeze(0))

    class Model:
        calls = 0

        def eval(self):
            return self

        def __call__(self, input_ids, labels):
            self.calls += 1
            return SimpleNamespace(loss=torch.tensor(float(self.calls)))

    result = compute_perplexity(
        Model(),
        Tokenizer(),
        [EvalRequest(prompt="corpus")],
        max_length=4,
        stride=4,
        max_windows=3,
        device="cpu",
    )
    assert result["windows"] == 3
    assert result["total_tokens"] == 7
    assert result["total_nll"] == pytest.approx(12.0)
    assert result["score"] == pytest.approx(torch.exp(torch.tensor(12 / 7)).item())
    assert [window["target_tokens"] for window in result["window_details"]] == [
        3,
        3,
        1,
    ]
    assert sum(
        window["nll"] for window in result["window_details"]
    ) == pytest.approx(result["total_nll"])
