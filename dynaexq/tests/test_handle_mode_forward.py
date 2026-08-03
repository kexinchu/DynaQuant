"""
Tests for handle-mode forward (plan §4.2 / §4.5 second acceptance criterion).

The contract: when an ``ExpertHandle`` with multi-linear ``quant_meta``
(a ``dict[str, PackedTensor]`` for w1/w2/w3) is registered in the
``ExpertRegistry`` and the model's ``PhimoeSparseMoeBlock`` has
``attach_dynaexq(registry, layer_idx)`` called, the forward path MUST
use ``fused_linear(x, packed)`` instead of ``nn.Linear.forward(x)`` and
the numerical output must stay close to the original fp16 forward.

For Phi-MoE:
    expert.w1  (gate)  — shape (intermediate, hidden)
    expert.w3  (up)    — shape (intermediate, hidden)
    expert.w2  (down)  — shape (hidden, intermediate)

We register the model's own weights as multi-linear PackedTensors
(``fmt=FP16``, zero quant error) and verify the handle-mode forward
produces outputs identical to the no-handle forward.
"""

from __future__ import annotations

import warnings

import pytest
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

from dynaexq.core.config import Tier
from dynaexq.core.quant import QuantFormat, pack
from dynaexq.core.registry import ExpertHandle, ExpertKey, ExpertRegistry
from dynaexq.models.phimoe import PhimoeConfig, PhimoeForCausalLM, PhimoeSparseMoeBlock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_config() -> PhimoeConfig:
    return PhimoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        router_jitter_noise=0.0,
        input_jitter_noise=0.0,
    )


def _register_all_expert_handles_fp16(
    model: PhimoeForCausalLM,
    registry: ExpertRegistry,
) -> None:
    """Register every expert in every MoE layer as an FP16 multi-linear
    handle with the model's own nn.Linear weights packed as
    ``PackedTensor(fmt=FP16)``. This gives zero quantization error,
    so handle-mode forward should reproduce no-handle forward exactly."""
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        moe = decoder_layer.block_sparse_moe
        for expert_idx, expert in enumerate(moe.experts):
            key = ExpertKey(layer_idx, expert_idx)
            quant_meta = {
                "w1": pack(expert.w1.weight.data, QuantFormat.FP16),
                "w3": pack(expert.w3.weight.data, QuantFormat.FP16),
                "w2": pack(expert.w2.weight.data, QuantFormat.FP16),
            }
            handle = ExpertHandle(tier=Tier.HI, quant_meta=quant_meta)
            registry.register(key, handle)


def _attach_registry_to_model(
    model: PhimoeForCausalLM,
    registry: ExpertRegistry,
) -> None:
    """Call ``attach_dynaexq`` on every MoE block in the model."""
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        moe = decoder_layer.block_sparse_moe
        assert isinstance(moe, PhimoeSparseMoeBlock)
        moe.attach_dynaexq(registry, layer_idx)


# ---------------------------------------------------------------------------
# ExpertHandle multi-linear quant_meta unit tests
# ---------------------------------------------------------------------------


def test_expert_handle_multi_linear_format_and_bytes():
    """A dict quant_meta should derive format from the common format
    and bytes from the sum of all slots."""
    # Use dims divisible by default INT4 group_size=128.
    w1_pt = pack(torch.randn(64, 128, dtype=torch.float16), QuantFormat.INT4)
    w2_pt = pack(torch.randn(128, 128, dtype=torch.float16), QuantFormat.INT4)
    w3_pt = pack(torch.randn(64, 128, dtype=torch.float16), QuantFormat.INT4)
    h = ExpertHandle(
        tier=Tier.LO,
        quant_meta={"w1": w1_pt, "w2": w2_pt, "w3": w3_pt},
    )
    assert h.format == "int4"
    assert h.bytes == w1_pt.nbytes + w2_pt.nbytes + w3_pt.nbytes


def test_expert_handle_get_packed_multi_slots():
    w1 = pack(torch.randn(4, 128, dtype=torch.float16), QuantFormat.FP16)
    w2 = pack(torch.randn(128, 4, dtype=torch.float16), QuantFormat.FP16)
    h = ExpertHandle(tier=Tier.HI, quant_meta={"w1": w1, "w2": w2})
    assert h.get_packed("w1") is w1
    assert h.get_packed("w2") is w2
    assert h.get_packed("nonexistent") is None


def test_expert_handle_get_packed_single_backward_compat():
    """Single PackedTensor quant_meta still works via get_packed("weight")."""
    pt = pack(torch.randn(8, 128, dtype=torch.float16), QuantFormat.FP16)
    h = ExpertHandle(tier=Tier.HI, quant_meta=pt)
    assert h.get_packed("weight") is pt
    assert h.get_packed("w1") is None


def test_expert_handle_mixed_format_dict_raises():
    """All slots in a dict must share the same QuantFormat."""
    w_int4 = pack(torch.randn(8, 128, dtype=torch.float16), QuantFormat.INT4)
    w_fp16 = pack(torch.randn(8, 128, dtype=torch.float16), QuantFormat.FP16)
    with pytest.raises(ValueError, match="same QuantFormat"):
        ExpertHandle(tier=Tier.HI, quant_meta={"a": w_int4, "b": w_fp16})


# ---------------------------------------------------------------------------
# Handle-mode forward: Phi-MoE
# ---------------------------------------------------------------------------


def test_phimoe_handle_mode_fp16_matches_no_handle():
    """
    Plan §4.5 acceptance (second part): "注册 handle 后 forward 数值与
    原始 HF 权重一致".

    Register every expert with FP16 PackedTensors of its own weights.
    The handle-mode forward path (fused_linear with FP16) should produce
    logits identical to the no-handle path (nn.Linear.forward).
    """
    cfg = _tiny_config()
    model = PhimoeForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))

    # 1) No-handle forward (baseline).
    with torch.no_grad():
        baseline = model(x).logits.clone()

    # 2) Register handles + attach registry → handle-mode.
    registry = ExpertRegistry()
    _register_all_expert_handles_fp16(model, registry)
    _attach_registry_to_model(model, registry)

    with torch.no_grad():
        handle_out = model(x).logits

    assert torch.allclose(handle_out, baseline, atol=1e-4), (
        f"handle-mode fp16 output differs from no-handle baseline; "
        f"max diff = {(handle_out - baseline).abs().max().item():.6f}"
    )


def test_phimoe_handle_mode_int4_produces_different_but_valid_output():
    """
    Register INT4 packed versions of every expert's weights. The output
    should differ from fp16 (quantization error) but still be finite and
    the right shape.
    """
    cfg = _tiny_config()
    model = PhimoeForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))

    registry = ExpertRegistry()
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        moe = decoder_layer.block_sparse_moe
        for expert_idx, expert in enumerate(moe.experts):
            key = ExpertKey(layer_idx, expert_idx)
            # Tiny model dims (32/64) need small group_size to divide evenly.
            quant_meta = {
                "w1": pack(expert.w1.weight.data, QuantFormat.INT4, group_size=32),
                "w3": pack(expert.w3.weight.data, QuantFormat.INT4, group_size=32),
                "w2": pack(expert.w2.weight.data, QuantFormat.INT4, group_size=32),
            }
            handle = ExpertHandle(tier=Tier.LO, quant_meta=quant_meta)
            registry.register(key, handle)

    _attach_registry_to_model(model, registry)

    with torch.no_grad():
        out = model(x)

    assert out.logits.shape == (1, 8, cfg.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_phimoe_handle_mode_detach_restores_no_handle():
    """After detaching the registry (setting to None), the model should
    revert to the original nn.Linear path and produce baseline output."""
    cfg = _tiny_config()
    model = PhimoeForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))

    with torch.no_grad():
        baseline = model(x).logits.clone()

    # Attach → detach cycle.
    registry = ExpertRegistry()
    _register_all_expert_handles_fp16(model, registry)
    _attach_registry_to_model(model, registry)

    # Detach: set registry to None on all MoE blocks.
    for decoder_layer in model.model.layers:
        moe = decoder_layer.block_sparse_moe
        moe._dynaexq_registry = None
        moe._refresh_expert_handles()

    with torch.no_grad():
        restored = model(x).logits

    assert torch.equal(restored, baseline)


def test_phimoe_partial_handle_registration_fails_closed():
    """An attached runtime must never mix registered and native experts."""
    cfg = _tiny_config()
    model = PhimoeForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))

    registry = ExpertRegistry()
    # Only register layer 0 expert 0.
    expert = model.model.layers[0].block_sparse_moe.experts[0]
    key = ExpertKey(0, 0)
    quant_meta = {
        "w1": pack(expert.w1.weight.data, QuantFormat.FP16),
        "w3": pack(expert.w3.weight.data, QuantFormat.FP16),
        "w2": pack(expert.w2.weight.data, QuantFormat.FP16),
    }
    registry.register(key, ExpertHandle(tier=Tier.HI, quant_meta=quant_meta))
    _attach_registry_to_model(model, registry)

    with pytest.raises(RuntimeError, match="has no registered handle"):
        with torch.no_grad():
            model(x)
