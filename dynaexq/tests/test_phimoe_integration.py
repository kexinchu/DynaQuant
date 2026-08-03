"""
Integration smoke tests for the Phi-3.5-MoE port (Phase 3).

These tests verify the minimum viable success condition from plan §4.5:

    python -c "from dynaexq.models.phimoe import PhimoeForCausalLM; ..."
    can run a forward pass in no-handle mode (registry is None, fall back
    to nn.Linear).

The test model is intentionally tiny (4 experts, 2 layers, hidden=32) so
it builds and runs in well under a second without needing a GPU. We
verify:

* Config and top-level module import cleanly from ``dynaexq.models.phimoe``
* ``PhimoeForCausalLM`` constructs with a hand-built PhimoeConfig
* A forward pass on random inputs produces logits of the expected shape
* ``output_router_logits=True`` produces one router_logits tensor per
  MoE layer with shape ``(seq_len, num_experts)``
* The existing MoEWrapper hook path (forward_hook + router_logits
  extraction) picks up PhimoeSparseMoeBlock without any model-code
  modifications
* A Phi-MoE expert exposes the three-linear ``w1/w2/w3`` structure that
  Phase 6 will need when it adds multi-linear ``quant_meta`` support.

Handle-mode forward (replacing nn.Linear with ``quant.fused_linear``) is
NOT tested here — per the Phase 3 plan it's deferred to Phase 6 when we
have multi-linear quant_meta. The Phase 3 deliverable is specifically
that no-handle mode works end-to-end and ExpertPrecisionManager has been
fully removed.
"""

from __future__ import annotations

import warnings

import pytest
import torch

# Silence transformers deprecation chatter — not the object of this test.
warnings.filterwarnings("ignore", category=FutureWarning)

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
from dynaexq.models.phimoe import (
    PhimoeBlockSparseTop2MLP,
    PhimoeConfig,
    PhimoeForCausalLM,
    PhimoeSparseMoeBlock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_config(
    num_hidden_layers: int = 2,
    num_local_experts: int = 4,
    num_experts_per_tok: int = 2,
) -> PhimoeConfig:
    return PhimoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        router_jitter_noise=0.0,
        input_jitter_noise=0.0,
    )


# ---------------------------------------------------------------------------
# Import + construction
# ---------------------------------------------------------------------------


def test_phimoe_package_imports():
    """Plan §4.5 acceptance: the package must import cleanly."""
    assert PhimoeConfig is not None
    assert PhimoeForCausalLM is not None
    assert PhimoeSparseMoeBlock is not None
    assert PhimoeBlockSparseTop2MLP is not None


def test_tiny_phimoe_builds():
    cfg = _make_tiny_config()
    model = PhimoeForCausalLM(cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())
    # Sanity: the tiny model should be small enough to fit in CPU memory
    # but large enough to actually have MoE experts wired.
    assert 20_000 < n_params < 200_000


def test_expert_has_three_linears_w1_w2_w3():
    """
    Phi-MoE experts use the ``w1`` / ``w2`` / ``w3`` linear naming
    (Mixtral-style). Phase 6 will need to enumerate these when it adds
    multi-linear ``quant_meta`` to ExpertHandle — this test pins the
    contract so the later refactor knows the attribute names.
    """
    cfg = _make_tiny_config()
    model = PhimoeForCausalLM(cfg).eval()
    # Reach into the first MoE layer's first expert.
    moe_block = model.model.layers[0].block_sparse_moe
    assert isinstance(moe_block, PhimoeSparseMoeBlock)
    expert = moe_block.experts[0]
    assert isinstance(expert, PhimoeBlockSparseTop2MLP)
    assert isinstance(expert.w1, torch.nn.Linear)
    assert isinstance(expert.w2, torch.nn.Linear)
    assert isinstance(expert.w3, torch.nn.Linear)
    # Gate / up share hidden_dim → ffn_dim; down is the reverse.
    assert expert.w1.in_features == cfg.hidden_size
    assert expert.w1.out_features == cfg.intermediate_size
    assert expert.w2.in_features == cfg.intermediate_size
    assert expert.w2.out_features == cfg.hidden_size
    assert expert.w3.in_features == cfg.hidden_size
    assert expert.w3.out_features == cfg.intermediate_size


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def test_tiny_phimoe_forward_logits_shape():
    cfg = _make_tiny_config()
    model = PhimoeForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(x)
    assert out.logits.shape == (1, 8, cfg.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_tiny_phimoe_forward_router_logits():
    cfg = _make_tiny_config(num_hidden_layers=2, num_local_experts=4)
    model = PhimoeForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(x, output_router_logits=True)
    # One router_logits per layer
    assert out.router_logits is not None
    assert len(out.router_logits) == cfg.num_hidden_layers
    # Each is (seq_len, num_experts)
    for rl in out.router_logits:
        assert rl.shape[-1] == cfg.num_local_experts
        # rl is (total_tokens, num_experts); total_tokens may include
        # padding from the model's internal flattening.
        assert rl.dim() == 2


def test_tiny_phimoe_forward_deterministic_without_jitter():
    """With ``router_jitter_noise=0`` the forward should be deterministic.
    This rules out hidden state pollution between calls (which would
    break the ExpertRegistry handle contract)."""
    cfg = _make_tiny_config()
    model = PhimoeForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        y1 = model(x).logits
        y2 = model(x).logits
    assert torch.equal(y1, y2)


def test_tiny_phimoe_batch_forward():
    cfg = _make_tiny_config()
    model = PhimoeForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.no_grad():
        out = model(x)
    assert out.logits.shape == (2, 6, cfg.vocab_size)


# ---------------------------------------------------------------------------
# ExpertPrecisionManager removal
# ---------------------------------------------------------------------------


def test_expert_precision_manager_is_fully_removed():
    """Plan §4.4: the legacy ExpertPrecisionManager must be gone from
    the core namespace, and no Phi-MoE / Qwen3 / DeepSeek expert module
    may carry a ``precision_manager`` attribute."""
    import dynaexq.core as core

    assert not hasattr(core, "ExpertPrecisionManager")
    with pytest.raises(ModuleNotFoundError):
        import dynaexq.core.expert_precision_manager  # noqa: F401

    cfg = _make_tiny_config()
    model = PhimoeForCausalLM(cfg).eval()
    for m in model.modules():
        assert not hasattr(m, "precision_manager"), (
            f"module {type(m).__name__} still has a precision_manager attribute"
        )


# ---------------------------------------------------------------------------
# MoEWrapper hook path: no model-code changes required
# ---------------------------------------------------------------------------


def test_moe_wrapper_hook_finds_phimoe_blocks():
    """
    The existing MoEWrapper walks named_modules looking for anything
    whose name contains 'moe' or 'expert'. PhimoeSparseMoeBlock is
    named ``block_sparse_moe`` in the Phi-MoE layer, so it must be
    matched by the 'moe' substring test.

    This locks the property that Phi-MoE doesn't need a special wiring
    pass — it plugs straight into the existing hook-based observer
    path. If someone renames that attribute in the future, this test
    fails loudly rather than silently losing observations.
    """
    cfg = _make_tiny_config()
    model = PhimoeForCausalLM(cfg).eval()

    moe_names = [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, PhimoeSparseMoeBlock)
    ]
    assert len(moe_names) == cfg.num_hidden_layers
    for name in moe_names:
        # The wrapper's is_moe_layer test looks for 'moe' / 'expert' /
        # ('mlp' AND hasattr experts). At least one must fire.
        lower = name.lower()
        moe_keyword = "moe" in lower or "expert" in lower
        assert moe_keyword, f"MoEWrapper would fail to match {name!r}"


def test_moe_wrapper_attaches_to_phimoe_end_to_end():
    """
    Full round-trip: construct a tiny Phi-MoE, wrap it with a MoEWrapper
    (router_observer + hotness_tracker + scheduler + registry), run a
    forward, and verify the hotness tracker recorded non-zero scores
    for at least one expert in layer 0. This proves the hook path
    actually fires for Phi-MoE with zero model-code modifications — the
    whole point of not having to edit modeling_phimoe.py for Phase 3.
    """
    from dynaexq.integration.moe_wrapper import MoEWrapper

    cfg = _make_tiny_config()
    model = PhimoeForCausalLM(cfg).eval()

    num_layers = cfg.num_hidden_layers
    experts_per_layer = cfg.num_local_experts

    registry = ExpertRegistry()
    for layer_index, layer in enumerate(model.model.layers):
        for expert_index, expert in enumerate(
            layer.block_sparse_moe.experts
        ):
            registry.register(
                ExpertKey(layer_index, expert_index),
                ExpertHandle(
                    tier=Tier.HI,
                    quant_meta={
                        "w1": pack(expert.w1.weight, QuantFormat.FP16),
                        "w2": pack(expert.w2.weight, QuantFormat.FP16),
                        "w3": pack(expert.w3.weight, QuantFormat.FP16),
                    },
                ),
            )
    observer = RouterObserver(use_probabilities=True)
    tracker = HotnessTracker(num_layers, experts_per_layer, alpha=0.5)
    scheduler = PrecisionScheduler(
        num_layers=num_layers,
        experts_per_layer=experts_per_layer,
        n_hi=[2] * num_layers,
        update_period_steps=1000,  # never update during this test
    )

    wrapper = MoEWrapper(
        model=model,
        router_observer=observer,
        hotness_tracker=tracker,
        scheduler=scheduler,
        registry=registry,
        num_layers=num_layers,
        experts_per_layer=experts_per_layer,
    )
    # Use enough tokens that the top-2 routing actually spreads across
    # several experts per layer.
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        _ = wrapper.forward(x)

    # At least one layer must have picked up non-zero hotness.
    any_layer_has_signal = any(
        tracker.get_layer_scores(l).sum() > 0 for l in range(num_layers)
    )
    assert any_layer_has_signal, (
        "MoEWrapper hook never fired for any Phi-MoE layer — the module "
        "name-match or router_logits extraction path is broken"
    )

    wrapper.remove_hooks()
