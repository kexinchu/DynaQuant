"""
Tests for ``dynaexq.core.weight_store`` after the Plan A rewrite.

These tests guard the new contract:

* ``load_weights(key, tier)`` returns a ``PackedTensor`` (not a raw tensor),
  with ``fmt`` matching the configured ``hi_format`` / ``lo_format``.
* ``get_byte_size`` agrees with the actual ``PackedTensor.nbytes`` and with
  ``compute_packed_nbytes`` — there is exactly one source of truth for the
  byte counter that the BudgetTracker reserves against.
* The packed cache is hit on the second call (no double-quantization) and
  invalidates correctly when ``register_expert`` overwrites a key.
* The HF-style model walk finds experts via ``layers.X.experts[Y]``.
"""

from __future__ import annotations

import pytest
import torch

from dynaexq.core.config import Tier
from dynaexq.core.quant import (
    PackedTensor,
    QuantFormat,
    compute_packed_nbytes,
    pack,
)
from dynaexq.core.registry import ExpertKey
from dynaexq.core.weight_store import ModelWeightStore, _parse_format


# ---------------------------------------------------------------------------
# Helpers — fake MoE model
# ---------------------------------------------------------------------------


class _FakeExpert(torch.nn.Module):
    """Single-linear expert. The Plan A loader picks the largest 2-D float
    parameter, which here is just ``self.weight``."""

    def __init__(self, out_features: int, in_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.float16) * 0.05
        )


class _FakeMoELayer(torch.nn.Module):
    def __init__(self, n_experts: int, out_features: int, in_features: int):
        super().__init__()
        self.experts = torch.nn.ModuleList(
            [_FakeExpert(out_features, in_features) for _ in range(n_experts)]
        )


class _FakeMoEModel(torch.nn.Module):
    def __init__(self, n_layers: int, n_experts: int, out_features: int, in_features: int):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_FakeMoELayer(n_experts, out_features, in_features) for _ in range(n_layers)]
        )


class _FakeFusedExperts(torch.nn.Module):
    def __init__(self, n_experts: int):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(
            torch.randn(n_experts, 16, 128, dtype=torch.float16)
        )
        self.down_proj = torch.nn.Parameter(
            torch.randn(n_experts, 8, 128, dtype=torch.float16)
        )


class _FakeFusedModel(torch.nn.Module):
    def __init__(self, n_experts: int):
        super().__init__()
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = _FakeFusedExperts(n_experts)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([layer])


def _make_store(hi: str = "fp16", lo: str = "int4") -> ModelWeightStore:
    model = _FakeMoEModel(n_layers=2, n_experts=4, out_features=16, in_features=128)
    return ModelWeightStore(model=model, hi_format=hi, lo_format=lo)


# ---------------------------------------------------------------------------
# _parse_format
# ---------------------------------------------------------------------------


def test_parse_format_accepts_known_strings():
    assert _parse_format("fp16") == QuantFormat.FP16
    assert _parse_format("FP16") == QuantFormat.FP16
    assert _parse_format(" int4 ") == QuantFormat.INT4
    assert _parse_format("int2") == QuantFormat.INT2


def test_parse_format_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown quant format"):
        _parse_format("int8")


# ---------------------------------------------------------------------------
# load_weights returns PackedTensor with correct fmt
# ---------------------------------------------------------------------------


def test_load_weights_hi_returns_fp16_packed():
    store = _make_store(hi="fp16", lo="int4")
    p = store.load_weights(ExpertKey(0, 0), Tier.HI)
    assert isinstance(p, PackedTensor)
    assert p.fmt == QuantFormat.FP16
    assert p.scales is None
    assert p.qweight.dtype == torch.float16
    assert p.qweight.shape == (16, 128)


def test_load_weights_lo_returns_int4_packed():
    store = _make_store(hi="fp16", lo="int4")
    p = store.load_weights(ExpertKey(0, 0), Tier.LO)
    assert isinstance(p, PackedTensor)
    assert p.fmt == QuantFormat.INT4
    assert p.scales is not None
    assert p.qweight.dtype == torch.uint8


def test_load_weights_lo_int2_returns_int2_packed():
    store = _make_store(hi="fp16", lo="int2")
    p = store.load_weights(ExpertKey(0, 0), Tier.LO)
    assert p.fmt == QuantFormat.INT2
    assert p.scales is not None


def test_load_weights_distinct_keys_return_distinct_packed():
    store = _make_store()
    p_a = store.load_weights(ExpertKey(0, 0), Tier.LO)
    p_b = store.load_weights(ExpertKey(0, 1), Tier.LO)
    p_c = store.load_weights(ExpertKey(1, 0), Tier.LO)
    # Different objects, different underlying weights
    assert p_a is not p_b is not p_c
    assert not torch.equal(p_a.qweight, p_b.qweight)
    assert not torch.equal(p_a.qweight, p_c.qweight)


# ---------------------------------------------------------------------------
# get_byte_size matches the canonical formula and the cached object
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier,fmt", [(Tier.HI, QuantFormat.FP16), (Tier.LO, QuantFormat.INT4)])
def test_get_byte_size_matches_compute_packed_nbytes(tier, fmt):
    store = _make_store()
    key = ExpertKey(0, 0)
    n = store.get_byte_size(key, tier)
    out_features, in_features = 16, 128
    if fmt == QuantFormat.FP16:
        expected = compute_packed_nbytes(out_features, in_features, fmt, in_features)
    else:
        expected = compute_packed_nbytes(out_features, in_features, fmt, 128)  # default
    assert n == expected


def test_get_byte_size_matches_loaded_packed_nbytes():
    """The byte counter MUST agree with the PackedTensor that load_weights
    returns. The BudgetTracker reserves against get_byte_size and the
    runtime copies packed.nbytes — any drift breaks the HBM envelope."""
    store = _make_store(hi="fp16", lo="int4")
    key = ExpertKey(0, 0)
    for tier in (Tier.HI, Tier.LO):
        size_via_method = store.get_byte_size(key, tier)
        packed = store.load_weights(key, tier)
        assert size_via_method == packed.nbytes, (
            f"get_byte_size({tier})={size_via_method} != "
            f"load_weights({tier}).nbytes={packed.nbytes}"
        )


def test_get_byte_size_does_not_force_eager_pack():
    """Reading a size for a key that has not been loaded yet should not
    populate the packed cache (it should derive from the raw tensor shape).
    Otherwise BudgetTracker probes during scheduling would quantize every
    expert at startup."""
    store = _make_store(lo="int4")
    key = ExpertKey(0, 2)
    _ = store.get_byte_size(key, Tier.LO)
    assert (key, Tier.LO) not in store._packed_cache


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_load_weights_is_memoized():
    store = _make_store()
    key = ExpertKey(0, 0)
    p1 = store.load_weights(key, Tier.LO)
    p2 = store.load_weights(key, Tier.LO)
    assert p1 is p2  # identity, not equality — second call is a cache hit


def test_clear_cache_evicts_packed_but_keeps_overrides():
    store = _make_store()
    key = ExpertKey(0, 0)
    _ = store.load_weights(key, Tier.LO)
    assert (key, Tier.LO) in store._packed_cache
    store.clear_cache()
    assert (key, Tier.LO) not in store._packed_cache


def test_register_expert_invalidates_cache():
    store = _make_store()
    key = ExpertKey(0, 0)
    p1 = store.load_weights(key, Tier.LO)

    new_weight = torch.randn(16, 128, dtype=torch.float16) * 0.5
    store.register_expert(key, new_weight)

    p2 = store.load_weights(key, Tier.LO)
    assert p2 is not p1, "register_expert must drop the stale packed cache"
    assert not torch.equal(p1.qweight, p2.qweight)


def test_register_expert_rejects_non_2d():
    store = _make_store()
    with pytest.raises(ValueError, match="2D"):
        store.register_expert(ExpertKey(0, 0), torch.zeros(16, 16, 16))


# ---------------------------------------------------------------------------
# Model walk: standard HF MoE layout
# ---------------------------------------------------------------------------


def test_walk_finds_expert_via_layers_experts_list():
    store = _make_store()
    # Confirm both layers and all experts within each layer are reachable.
    for l in range(2):
        for e in range(4):
            packed = store.load_weights(ExpertKey(l, e), Tier.HI)
            assert packed.qweight.shape == (16, 128)


def test_load_weights_missing_expert_raises():
    store = _make_store()
    with pytest.raises(ValueError, match="not found"):
        store.load_weights(ExpertKey(99, 0), Tier.HI)


def test_load_weights_with_no_model_requires_register_expert():
    store = ModelWeightStore(model=None, hi_format="fp16", lo_format="int4")
    key = ExpertKey(0, 0)
    with pytest.raises(ValueError, match="not found"):
        store.load_weights(key, Tier.HI)
    # Now register and retry
    store.register_expert(key, torch.randn(8, 128, dtype=torch.float16))
    p = store.load_weights(key, Tier.LO)
    assert p.fmt == QuantFormat.INT4
    assert p.qweight.shape == (8, 128 // 2)


def test_preload_then_release_native_sources_is_self_contained():
    store = _make_store()
    summary = store.preload_all(2, 4)
    assert summary["entries"] == 16
    assert summary["host_packed_bytes"] > 0

    released = store.release_model_expert_sources(2, 4)
    assert released == 2 * 4 * 16 * 128 * 2
    assert store.model is None

    # Both tiers remain available after the source model has been detached.
    for tier in (Tier.HI, Tier.LO):
        packed = store.load_weights(ExpertKey(1, 3), tier)
        assert packed.qweight.device.type == "cpu"


def test_release_requires_complete_dual_tier_cache():
    store = _make_store()
    store.load_weights(ExpertKey(0, 0), Tier.LO)
    with pytest.raises(RuntimeError, match="before caching"):
        store.release_model_expert_sources(2, 4)


def test_layerwise_preload_and_release_bounds_source_lifetime():
    store = _make_store()
    summary = store.preload_and_release_all(2, 4)
    assert summary["entries"] == 16
    assert summary["released_native_expert_bytes"] == 2 * 4 * 16 * 128 * 2
    assert len(summary["layer_pack_release_seconds"]) == 2
    assert all(value >= 0 for value in summary["layer_pack_release_seconds"])
    assert sum(summary["layer_host_packed_bytes"]) == summary["host_packed_bytes"]
    assert (
        sum(summary["layer_released_native_expert_bytes"])
        == summary["released_native_expert_bytes"]
    )
    assert summary["host_allocator_trim_attempts"] == 2
    assert store.model is None
    assert store.load_weights(ExpertKey(0, 0), Tier.HI).qweight.device.type == "cpu"


def test_fused_chunk_pack_matches_individual_reference_bits():
    model = _FakeFusedModel(4)
    container = model.model.layers[0].mlp.experts
    source = {
        "gate_up_proj": container.gate_up_proj.detach().clone(),
        "down_proj": container.down_proj.detach().clone(),
    }
    store = ModelWeightStore(
        model=model,
        hi_format="int4",
        lo_format="int2",
        fused_pack_chunk_experts=2,
    )
    summary = store.preload_and_release_all(1, 4)
    assert summary["entries"] == 8
    for expert in range(4):
        for tier, fmt, group_size in (
            (Tier.HI, QuantFormat.INT4, 128),
            (Tier.LO, QuantFormat.INT2, 64),
        ):
            actual = store.load_weights(ExpertKey(0, expert), tier)
            assert isinstance(actual, dict)
            for slot, weights in source.items():
                expected = pack(
                    weights[expert],
                    fmt,
                    group_size=group_size,
                )
                torch.testing.assert_close(
                    actual[slot].qweight,
                    expected.qweight,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    actual[slot].scales,
                    expected.scales,
                    rtol=0,
                    atol=0,
                )


def test_int4_kernel_cache_is_included_in_resident_byte_size():
    store = ModelWeightStore(
        model=None,
        hi_format="fp16",
        lo_format="int4",
        enable_int4_kernel_cache=True,
    )
    key = ExpertKey(0, 0)
    store.register_expert(key, torch.randn(16, 128, dtype=torch.float16))
    packed = store.load_weights(key, Tier.LO)
    assert packed.resident_nbytes > packed.nbytes
    assert store.get_byte_size(key, Tier.LO) == packed.resident_nbytes
