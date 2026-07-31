from __future__ import annotations

import pytest
import sys
import torch
from types import SimpleNamespace

from dynaexq.core import (
    ExpertHandle,
    ExpertKey,
    ExpertRegistry,
    HotnessTracker,
    ModelWeightStore,
    PrecisionScheduler,
    RouterObserver,
    Tier,
    pack,
)
from dynaexq.core.quant import QuantFormat
from dynaexq.integration.moe_wrapper import MoEWrapper
from dynaexq.integration.qwen3_next_adapter import attach_qwen3_next_experts
from dynaexq.experiments.run_shift import load_model

transformers = pytest.importorskip("transformers")
Qwen3NextConfig = pytest.importorskip(
    "transformers.models.qwen3_next.configuration_qwen3_next"
).Qwen3NextConfig
Qwen3NextForCausalLM = pytest.importorskip(
    "transformers.models.qwen3_next.modeling_qwen3_next"
).Qwen3NextForCausalLM


def _tiny_model() -> torch.nn.Module:
    config = Qwen3NextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=8,
        full_attention_interval=1,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return Qwen3NextForCausalLM(config).eval().half()


def _registry_from_native_experts(model: torch.nn.Module) -> ExpertRegistry:
    registry = ExpertRegistry()
    experts = model.model.layers[0].mlp.experts
    for expert in range(experts.num_experts):
        registry.register(
            ExpertKey(0, expert),
            ExpertHandle(
                tier=Tier.HI,
                quant_meta={
                    "gate_up_proj": pack(
                        experts.gate_up_proj[expert].detach(),
                        QuantFormat.FP16,
                    ),
                    "down_proj": pack(
                        experts.down_proj[expert].detach(),
                        QuantFormat.FP16,
                    ),
                },
            ),
        )
    return registry


def _wrapper(
    model: torch.nn.Module,
    registry: ExpertRegistry,
) -> MoEWrapper:
    return MoEWrapper(
        model=model,
        router_observer=RouterObserver(use_probabilities=True),
        hotness_tracker=HotnessTracker(1, 4),
        scheduler=PrecisionScheduler(1, 4, [4], update_period_steps=100),
        registry=registry,
        num_layers=1,
        experts_per_layer=4,
        topk=2,
    )


def test_qwen3_next_fp16_handles_match_native_forward_and_release_leases():
    torch.manual_seed(17)
    model = _tiny_model()
    input_ids = torch.tensor([[1, 3, 4]])
    with torch.inference_mode():
        baseline = model(input_ids=input_ids, use_cache=False).logits

    registry = _registry_from_native_experts(model)
    wrapper = _wrapper(model, registry)
    wrapper.validate_integration()
    with torch.inference_mode():
        handled = wrapper(input_ids=input_ids, use_cache=False).logits
    wrapper.remove_hooks()

    torch.testing.assert_close(handled, baseline, rtol=1e-3, atol=1e-3)
    assert wrapper._attached_layers == 1
    assert wrapper._router_layers == 1
    assert (wrapper.hotness_tracker.get_layer_scores(0) > 0).any()
    assert all(
        handle.active_readers == 0
        for handle in registry.handle_snapshot().values()
    )


def test_qwen3_next_adapter_fails_closed_on_missing_handle():
    model = _tiny_model()
    wrapper = _wrapper(model, ExpertRegistry())
    wrapper.validate_integration()
    with pytest.raises(RuntimeError, match="has no registered handle"):
        with torch.inference_mode():
            wrapper(
                input_ids=torch.tensor([[1, 3, 4]]),
                use_cache=False,
            )
    wrapper.remove_hooks()


def test_qwen3_next_loader_forwards_revision_and_disables_remote_code(
    monkeypatch,
):
    calls: list[tuple[str, str | None, bool | None]] = []

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(
                (
                    "tokenizer",
                    kwargs.get("revision"),
                    kwargs.get("trust_remote_code"),
                )
            )
            return object()

    class _AutoConfig:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(
                (
                    "discovery",
                    kwargs.get("revision"),
                    kwargs.get("trust_remote_code"),
                )
            )
            return SimpleNamespace(model_type="qwen3_next")

    class _Config:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(
                (
                    "config",
                    kwargs.get("revision"),
                    kwargs.get("trust_remote_code"),
                )
            )
            return object()

    class _Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(
                (
                    "weights",
                    kwargs.get("revision"),
                    kwargs.get("trust_remote_code"),
                )
            )
            return cls()

        def eval(self):
            return self

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=_Tokenizer,
            AutoConfig=_AutoConfig,
            Qwen3NextConfig=_Config,
            Qwen3NextForCausalLM=_Model,
        ),
    )

    model, _ = load_model(
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        torch.device("cpu"),
        revision="immutable-sha",
    )
    assert isinstance(model, _Model)
    assert calls == [
        ("tokenizer", "immutable-sha", False),
        ("discovery", "immutable-sha", False),
        ("config", "immutable-sha", None),
        ("weights", "immutable-sha", None),
    ]


def test_qwen3_next_weight_store_extracts_and_releases_only_routed_experts():
    model = _tiny_model()
    experts = model.model.layers[0].mlp.experts
    shared_parameter_ids = {
        id(parameter)
        for parameter in model.model.layers[0].mlp.shared_expert.parameters()
    }
    expected_gate_up = experts.gate_up_proj.detach().clone()
    expected_down = experts.down_proj.detach().clone()

    store = ModelWeightStore(
        model=model,
        hi_format="fp16",
        lo_format="fp16",
    )
    packed = store.load_weights(ExpertKey(0, 2), Tier.HI)
    assert isinstance(packed, dict)
    torch.testing.assert_close(
        packed["gate_up_proj"].qweight,
        expected_gate_up[2],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        packed["down_proj"].qweight,
        expected_down[2],
        rtol=0,
        atol=0,
    )

    expected_released = (
        expected_gate_up.numel() + expected_down.numel()
    ) * expected_gate_up.element_size()
    stats = store.preload_and_release_all(1, 4)
    assert stats["released_native_expert_bytes"] == expected_released
    assert experts.gate_up_proj.numel() == 0
    assert experts.down_proj.numel() == 0
    assert shared_parameter_ids == {
        id(parameter)
        for parameter in model.model.layers[0].mlp.shared_expert.parameters()
    }

    registry = ExpertRegistry()
    for expert in range(4):
        registry.register(
            ExpertKey(0, expert),
            ExpertHandle(
                tier=Tier.HI,
                quant_meta=store.load_weights(ExpertKey(0, expert), Tier.HI),
            ),
        )
    wrapper = _wrapper(model, registry)
    wrapper.validate_integration()
    with torch.inference_mode():
        output = wrapper(
            input_ids=torch.tensor([[1, 3, 4]]),
            use_cache=False,
        ).logits
    assert torch.isfinite(output).all()
    wrapper.remove_hooks()


def test_qwen3_next_fused_chunk_preload_is_exact_for_fp16():
    model = _tiny_model()
    experts = model.model.layers[0].mlp.experts
    expected_gate_up = experts.gate_up_proj.detach().clone()
    expected_down = experts.down_proj.detach().clone()
    store = ModelWeightStore(
        model=model,
        hi_format="fp16",
        lo_format="fp16",
        fused_pack_chunk_experts=2,
    )
    summary = store.preload_and_release_all(1, 4)
    assert summary["entries"] == 8
    for expert in range(4):
        for tier in (Tier.HI, Tier.LO):
            packed = store.load_weights(ExpertKey(0, expert), tier)
            assert isinstance(packed, dict)
            torch.testing.assert_close(
                packed["gate_up_proj"].qweight,
                expected_gate_up[expert],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                packed["down_proj"].qweight,
                expected_down[expert],
                rtol=0,
                atol=0,
            )


class _FakeAutoRoundProjection(torch.nn.Module):
    def __init__(self, out_features: int, in_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.float16)
        )
        self.register_buffer(
            "qweight",
            torch.zeros(in_features // 8, out_features, dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(1, max(1, out_features // 8), dtype=torch.int32),
        )
        self.register_buffer(
            "scales",
            torch.ones(1, out_features, dtype=torch.float16),
        )


class Qwen3NextMLP(torch.nn.Module):
    """AutoRound-like unfused expert; class name is part of the adapter ABI."""

    def __init__(self):
        super().__init__()
        self.gate_proj = _FakeAutoRoundProjection(8, 16)
        self.up_proj = _FakeAutoRoundProjection(8, 16)
        self.down_proj = _FakeAutoRoundProjection(16, 8)
        self.act_fn = torch.nn.functional.silu
        self.native_calls = 0

    def forward(self, hidden_states):
        self.native_calls += 1
        gate = torch.nn.functional.linear(
            hidden_states,
            self.gate_proj.weight,
        )
        up = torch.nn.functional.linear(hidden_states, self.up_proj.weight)
        return torch.nn.functional.linear(
            self.act_fn(gate) * up,
            self.down_proj.weight,
        )


def test_qwen3_next_unfused_autoround_expert_uses_registry_handle():
    torch.manual_seed(31)
    expert = Qwen3NextMLP()
    experts = torch.nn.ModuleList([expert])
    hidden_states = torch.randn(3, 16, dtype=torch.float16)
    with torch.inference_mode():
        expected = expert(hidden_states)
    assert expert.native_calls == 1

    registry = ExpertRegistry()
    registry.register(
        ExpertKey(0, 0),
        ExpertHandle(
            tier=Tier.HI,
            quant_meta={
                "gate_proj": pack(
                    expert.gate_proj.weight.detach(),
                    QuantFormat.FP16,
                ),
                "up_proj": pack(
                    expert.up_proj.weight.detach(),
                    QuantFormat.FP16,
                ),
                "down_proj": pack(
                    expert.down_proj.weight.detach(),
                    QuantFormat.FP16,
                ),
            },
        ),
    )
    assert attach_qwen3_next_experts(experts, registry, 0)
    with torch.inference_mode():
        actual = expert(hidden_states)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert expert.native_calls == 1
    assert (
        registry.handle_snapshot()[ExpertKey(0, 0)].active_readers == 0
    )
