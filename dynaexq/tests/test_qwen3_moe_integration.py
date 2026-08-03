from __future__ import annotations

import pytest
import torch

from dynaexq.core import (
    ExpertHandle,
    ExpertKey,
    ExpertRegistry,
    HotnessTracker,
    PrecisionScheduler,
    RouterObserver,
    Tier,
    pack,
)
from dynaexq.core.quant import QuantFormat
from dynaexq.integration.moe_wrapper import MoEWrapper
from dynaexq.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM


def _tiny_model() -> Qwen3MoeForCausalLM:
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return Qwen3MoeForCausalLM(config).eval()


def test_qwen3_local_adapter_forward_and_fp16_handle_mode_match():
    torch.manual_seed(7)
    model = _tiny_model()
    input_ids = torch.tensor([[1, 3, 4]])
    with torch.inference_mode():
        baseline = model(input_ids=input_ids).logits

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

    wrapper = MoEWrapper(
        model=model,
        router_observer=RouterObserver(use_probabilities=True),
        hotness_tracker=HotnessTracker(1, 4),
        scheduler=PrecisionScheduler(1, 4, [4], update_period_steps=100),
        registry=registry,
        num_layers=1,
        experts_per_layer=4,
        topk=2,
        routing_profile_enabled=True,
    )
    wrapper.validate_integration()
    with torch.inference_mode():
        handled = wrapper.forward(input_ids=input_ids).logits
        profile = wrapper.get_routing_profile()
        generated = wrapper.generate(
            input_ids=input_ids,
            max_new_tokens=2,
            do_sample=False,
            eos_token_id=None,
        )
    wrapper.remove_hooks()

    torch.testing.assert_close(handled, baseline, rtol=1e-3, atol=1e-3)
    assert len(profile[0]) == 4
    assert sum(profile[0]) == input_ids.numel() * 2
    assert generated.shape == (1, input_ids.shape[1] + 2)

    wrapper.reset_routing_profile()
    assert wrapper.get_routing_profile() == {}


def test_qwen3_local_adapter_fails_closed_on_missing_handle():
    model = _tiny_model()
    wrapper = MoEWrapper(
        model=model,
        router_observer=RouterObserver(use_probabilities=True),
        hotness_tracker=HotnessTracker(1, 4),
        scheduler=PrecisionScheduler(1, 4, [4], update_period_steps=100),
        registry=ExpertRegistry(),
        num_layers=1,
        experts_per_layer=4,
        topk=2,
    )
    wrapper.validate_integration()
    with pytest.raises(RuntimeError, match="has no registered handle"):
        with torch.inference_mode():
            wrapper(input_ids=torch.tensor([[1, 3, 4]]))
    wrapper.remove_hooks()
