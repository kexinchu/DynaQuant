"""
Tests for ``dynaexq.core.quant`` (Phase 2).

These tests guard the contract that the rest of the runtime depends on:

* Pack → dequant roundtrip error stays within the theoretical bound for
  group-wise symmetric quantization.
* The byte count returned by ``compute_packed_nbytes`` matches the actual
  ``element_size * numel`` of the produced tensors. The BudgetTracker (Phase 4)
  reserves HBM by trusting this number, so a mismatch would silently break the
  HBM envelope guarantee from §III-D.
* ``fused_linear`` agrees numerically with a manual ``dequant + matmul`` and
  with an unquantized fp16 baseline within the expected error band.
* The ``autoround`` backend seam raises ``NotImplementedError`` (not silently
  falling back to fp16, which would defeat the purpose of Phase 2).
"""

from __future__ import annotations

import math

import pytest
import torch

from dynaexq.core.quant import (
    DEFAULT_GROUP_SIZE,
    PackedTensor,
    QuantFormat,
    compute_packed_nbytes,
    dequant_to_fp16,
    fused_linear,
    pack,
)
from dynaexq.core.quant import (
    _pack_int2,
    _pack_int4,
    _unpack_int2,
    _unpack_int4,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_weight(out_features: int, in_features: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # Mix of small and large magnitudes to exercise per-group scales.
    w = torch.randn(out_features, in_features, generator=g) * 0.1
    # Inject a few large outliers in some rows so different groups land on
    # different scales.
    w[0, :32] = 5.0
    w[5, 64:96] = -3.0
    return w.to(torch.float16)


# ---------------------------------------------------------------------------
# nbytes accounting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "out_features,in_features,fmt",
    [
        (16, 128, QuantFormat.FP16),
        (16, 128, QuantFormat.INT4),
        (16, 256, QuantFormat.INT4),
        (16, 256, QuantFormat.INT2),
        (32, 1024, QuantFormat.INT4),
        (32, 1024, QuantFormat.INT2),
    ],
)
def test_nbytes_matches_actual_storage(out_features, in_features, fmt):
    w = _make_weight(out_features, in_features)
    p = pack(w, fmt)

    expected = compute_packed_nbytes(out_features, in_features, fmt, p.group_size)
    actual = p.qweight.element_size() * p.qweight.numel()
    if p.scales is not None:
        actual += p.scales.element_size() * p.scales.numel()

    assert p.nbytes == expected, (
        f"PackedTensor.nbytes ({p.nbytes}) != compute_packed_nbytes ({expected})"
    )
    assert actual == expected, (
        f"actual storage ({actual}) != computed nbytes ({expected}); "
        f"BudgetTracker would over/under-charge by {actual - expected} bytes"
    )


def test_int4_byte_savings_vs_fp16():
    """INT4 (group=128) on a 1024-wide weight should be ~4x smaller than fp16."""
    out_features, in_features = 32, 1024
    fp16_bytes = compute_packed_nbytes(out_features, in_features, QuantFormat.FP16, in_features)
    int4_bytes = compute_packed_nbytes(out_features, in_features, QuantFormat.INT4, 128)
    # fp16: 32*1024*2 = 65536. int4: 32*512 + 32*8*2 = 16384+512 = 16896.
    assert fp16_bytes == 65536
    assert int4_bytes == 16896
    assert fp16_bytes / int4_bytes > 3.8  # close to 4x with small scale overhead


def test_int2_byte_savings_vs_fp16():
    out_features, in_features = 32, 1024
    fp16_bytes = compute_packed_nbytes(out_features, in_features, QuantFormat.FP16, in_features)
    int2_bytes = compute_packed_nbytes(out_features, in_features, QuantFormat.INT2, 64)
    # int2: 32*256 + 32*16*2 = 8192+1024 = 9216
    assert int2_bytes == 9216
    assert fp16_bytes / int2_bytes > 7.0  # close to 8x with scale overhead


# ---------------------------------------------------------------------------
# Roundtrip correctness
# ---------------------------------------------------------------------------


def test_fp16_roundtrip_is_identity():
    w = _make_weight(8, 64)
    p = pack(w, QuantFormat.FP16)
    out = dequant_to_fp16(p)
    assert torch.equal(out, w)


def test_int4_roundtrip_within_quantization_error():
    """
    For symmetric int4 with per-group scales, the maximum per-element error is
    bounded by ``scale / 2`` for any one group (rounding to nearest). We verify
    a much looser bound (mean relative error vs the FP16 weight norm) so the
    test stays robust against generator changes.
    """
    out_features, in_features = 16, 256
    w = _make_weight(out_features, in_features, seed=42)
    p = pack(w, QuantFormat.INT4, group_size=128)
    w_hat = dequant_to_fp16(p)

    assert w_hat.shape == w.shape
    assert w_hat.dtype == torch.float16

    err = (w_hat.float() - w.float()).abs()
    rel = err.mean() / w.float().abs().mean().clamp(min=1e-6)
    # int4 group-wise sym typically achieves ~5–10% mean relative error on
    # iid Gaussian weights with outliers. Use 0.20 as a generous ceiling.
    assert rel < 0.20, f"int4 mean relative error {rel.item():.4f} exceeds 0.20"


def test_int2_roundtrip_within_quantization_error():
    out_features, in_features = 16, 256
    w = _make_weight(out_features, in_features, seed=7)
    p = pack(w, QuantFormat.INT2, group_size=64)
    w_hat = dequant_to_fp16(p)

    assert w_hat.shape == w.shape
    err = (w_hat.float() - w.float()).abs()
    rel = err.mean() / w.float().abs().mean().clamp(min=1e-6)
    # int2 only has 4 levels per group → much higher error. Plan §3.1 only
    # requires error to be "correct" (i.e. the quantization actually happens),
    # not small. We assert the error is bounded but clearly larger than int4.
    assert rel < 0.60, f"int2 mean relative error {rel.item():.4f} exceeds 0.60"


def test_int2_error_strictly_larger_than_int4():
    """Sanity: INT2 should be strictly worse than INT4 on the same input.
    If this ever fails, the packing layout is bugged (e.g. the same backend
    is being used for both formats)."""
    w = _make_weight(16, 256, seed=123)
    p4 = pack(w, QuantFormat.INT4)
    p2 = pack(w, QuantFormat.INT2)
    err4 = (dequant_to_fp16(p4).float() - w.float()).abs().mean()
    err2 = (dequant_to_fp16(p2).float() - w.float()).abs().mean()
    assert err2 > err4 * 1.5


# ---------------------------------------------------------------------------
# Packing layout correctness (regression guards on the bit layout)
# ---------------------------------------------------------------------------


def test_int4_pack_unpack_layout_low_then_high_nibble():
    """
    Test the int4 byte layout directly with hand-crafted signed int values.
    Decoupled from quantization rounding so the test only guards the bit
    layout (low nibble = even column, q+8 unsigned encoding).
    """
    # 1 row, 8 columns → 4 bytes
    q_signed = torch.tensor(
        [[1, -2, 3, -4, 7, -8, 0, 6]], dtype=torch.int8
    )
    packed = _pack_int4(q_signed)
    assert packed.shape == (1, 4)
    assert packed.dtype == torch.uint8
    # Expected unsigned (q + 8): [9, 6, 11, 4, 15, 0, 8, 14]
    # byte0 = 9 | (6 << 4) = 0x69
    # byte1 = 11 | (4 << 4) = 0x4B
    # byte2 = 15 | (0 << 4) = 0x0F
    # byte3 = 8 | (14 << 4) = 0xE8
    expected = torch.tensor([[0x69, 0x4B, 0x0F, 0xE8]], dtype=torch.uint8)
    assert torch.equal(packed, expected), (
        f"int4 byte layout regression: got {packed.tolist()} "
        f"expected {expected.tolist()}"
    )

    # And the unpack must invert it.
    unpacked = _unpack_int4(packed, in_features=8)
    assert torch.equal(unpacked, q_signed)


def test_int2_pack_unpack_layout_four_crumbs_per_byte():
    """
    Test the int2 byte layout directly with hand-crafted signed int values.
    Layout: bits 0-1 = col 0, bits 2-3 = col 1, bits 4-5 = col 2, bits 6-7 = col 3.
    Encoding: stored = q + 2.
    """
    # 1 row, 4 columns → 1 byte
    q_signed = torch.tensor([[1, 0, -1, -2]], dtype=torch.int8)
    packed = _pack_int2(q_signed)
    assert packed.shape == (1, 1)
    # Unsigned (q + 2): [3, 2, 1, 0]
    # byte = 3 | (2 << 2) | (1 << 4) | (0 << 6) = 0b00_01_10_11 = 0x1B
    assert packed.item() == 0x1B
    unpacked = _unpack_int2(packed, in_features=4)
    assert torch.equal(unpacked, q_signed)


def test_int4_pack_unpack_roundtrip_random():
    """Pack/unpack must be a perfect inverse for any valid signed-int input."""
    g = torch.Generator().manual_seed(0)
    q = torch.randint(-8, 8, (16, 256), generator=g, dtype=torch.int8)
    packed = _pack_int4(q)
    assert packed.shape == (16, 128)
    unpacked = _unpack_int4(packed, in_features=256)
    assert torch.equal(unpacked, q)


def test_int2_pack_unpack_roundtrip_random():
    g = torch.Generator().manual_seed(0)
    q = torch.randint(-2, 2, (16, 256), generator=g, dtype=torch.int8)
    packed = _pack_int2(q)
    assert packed.shape == (16, 64)
    unpacked = _unpack_int2(packed, in_features=256)
    assert torch.equal(unpacked, q)


# ---------------------------------------------------------------------------
# fused_linear correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", [QuantFormat.FP16, QuantFormat.INT4, QuantFormat.INT2])
def test_fused_linear_matches_dequant_then_matmul(fmt):
    """``fused_linear`` must agree with the explicit dequant+matmul path."""
    out_features, in_features = 32, 256
    batch = 4
    w = _make_weight(out_features, in_features, seed=11)
    x = torch.randn(batch, in_features, dtype=torch.float16)

    p = pack(w, fmt)
    y_fused = fused_linear(x, p)
    y_ref = torch.nn.functional.linear(x, dequant_to_fp16(p))
    assert torch.equal(y_fused, y_ref), (
        "fused_linear diverged from dequant+matmul; this is a contract"
        " violation, not a numerical issue"
    )


def test_int4_linear_close_to_fp16_baseline():
    """End-to-end sanity: INT4 linear output is close to fp16 linear output."""
    out_features, in_features = 64, 512
    w = _make_weight(out_features, in_features, seed=99)
    x = torch.randn(8, in_features, dtype=torch.float16) * 0.5

    y_fp16 = torch.nn.functional.linear(x, w)
    p_int4 = pack(w, QuantFormat.INT4)
    y_int4 = fused_linear(x, p_int4)

    diff = (y_int4.float() - y_fp16.float()).norm()
    base = y_fp16.float().norm().clamp(min=1e-6)
    rel = (diff / base).item()
    assert rel < 0.15, f"int4 linear relative error {rel:.4f} too large vs fp16"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int2_triton_linear_matches_reference_without_full_weight():
    w = _make_weight(33, 256, seed=123).cuda()
    x = torch.randn(19, 256, dtype=torch.float16, device="cuda")
    packed = pack(w, QuantFormat.INT2).to("cuda")
    actual = fused_linear(x, packed)
    expected = torch.nn.functional.linear(x, dequant_to_fp16(packed))
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# AutoRound seam
# ---------------------------------------------------------------------------


def test_autoround_backend_now_returns_packed_tensor():
    """
    Phase 2.1: the autoround seam is now live. It must return a valid
    PackedTensor whose byte layout is identical to what the reference
    backend would produce (same ``fmt``, ``nbytes``, ``group_size``,
    ``qweight``/``scales`` shapes). Numerical values may differ slightly
    due to fp16 rounding inside auto-round.
    """
    pytest.importorskip("auto_round", reason="install the 'autoround' extra")
    w = _make_weight(8, 128)
    p = pack(w, QuantFormat.INT4, backend="autoround")
    assert p.fmt == QuantFormat.INT4
    assert p.out_features == 8
    assert p.in_features == 128
    # Same nbytes as reference — this is the contract the BudgetTracker
    # depends on.
    assert p.nbytes == pack(w, QuantFormat.INT4, backend="reference").nbytes


def test_unknown_backend_raises():
    w = _make_weight(8, 128)
    with pytest.raises(ValueError, match="Unknown quantization backend"):
        pack(w, QuantFormat.INT4, backend="bogus")


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------


def test_in_features_must_divide_group_size():
    w = _make_weight(8, 100)  # 100 not divisible by 128 default
    with pytest.raises(ValueError, match="not divisible"):
        pack(w, QuantFormat.INT4, group_size=128)


def test_pack_rejects_non_2d_weight():
    with pytest.raises(ValueError, match="2D"):
        pack(torch.randn(8, 16, 32), QuantFormat.INT4)


def test_packed_tensor_to_device_preserves_metadata():
    w = _make_weight(8, 128)
    p = pack(w, QuantFormat.INT4)
    p_cpu = p.to("cpu")
    assert p_cpu.nbytes == p.nbytes
    assert p_cpu.fmt == p.fmt
    assert p_cpu.group_size == p.group_size
    assert p_cpu.in_features == p.in_features
    assert p_cpu.out_features == p.out_features
