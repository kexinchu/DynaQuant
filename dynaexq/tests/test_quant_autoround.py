"""
Tests for ``dynaexq.core.quant_autoround`` (Phase 2.1 — AutoRound backend).

These tests guard the seam between ``quant.pack(backend="autoround")`` and
``auto_round.data_type.get_quant_func``. The most important property being
verified is **byte-layout parity** with the reference backend: the
BudgetTracker reserves bytes based on ``packed.nbytes``, and the
TransitionEngine copies the raw byte stream via ``_packed_to_bytes``, so
any divergence in layout between backends would be a silent runtime bug.

Tests are automatically skipped if ``auto_round`` is not importable, so
the suite remains portable to environments without the library.
"""

from __future__ import annotations

import pytest
import torch

from dynaexq.core.quant import (
    PackedTensor,
    QuantFormat,
    compute_packed_nbytes,
    dequant_to_fp16,
    pack,
)

# Skip the whole module if auto_round isn't installed. We probe the import
# at collection time rather than inside each test so the skip reason is
# visible at the top of the pytest output.
pytest.importorskip("auto_round")

from dynaexq.core.quant_autoround import (  # noqa: E402
    _compute_imatrix,
    pack_with_autoround,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_weight(out_features: int, in_features: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out_features, in_features, generator=g, dtype=torch.float16) * 0.1
    # Inject a few outliers so per-group scales actually vary.
    w[0, :32] = 2.0
    return w


# ---------------------------------------------------------------------------
# Output shape / type contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt,group_size",
    [
        (QuantFormat.INT4, 128),
        (QuantFormat.INT4, 64),
        (QuantFormat.INT2, 64),
    ],
)
def test_autoround_returns_valid_packed_tensor(fmt, group_size):
    w = _make_weight(16, 256)
    p = pack_with_autoround(w, fmt, group_size=group_size)
    assert isinstance(p, PackedTensor)
    assert p.fmt == fmt
    assert p.out_features == 16
    assert p.in_features == 256
    assert p.group_size == group_size
    # qweight is uint8
    assert p.qweight.dtype == torch.uint8
    # scales is fp16 with shape (out, n_groups)
    assert p.scales is not None
    assert p.scales.dtype == torch.float16
    assert p.scales.shape == (16, 256 // group_size)


def test_autoround_defaults_group_size_from_format():
    w = _make_weight(8, 128)
    p = pack_with_autoround(w, QuantFormat.INT4)
    # Default for INT4 is 128; our in_features is 128, so n_groups=1.
    assert p.group_size == 128
    assert p.scales.shape == (8, 1)


# ---------------------------------------------------------------------------
# Byte-layout parity with the reference backend (the critical invariant)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt,group_size,out,inp",
    [
        (QuantFormat.INT4, 128, 16, 256),
        (QuantFormat.INT4, 64, 16, 256),
        (QuantFormat.INT2, 64, 16, 256),
        (QuantFormat.INT4, 128, 32, 1024),
    ],
)
def test_autoround_nbytes_matches_reference(fmt, group_size, out, inp):
    """
    The BudgetTracker reserves bytes by calling ``compute_packed_nbytes``
    and trusts that both backends produce tensors of that exact size.
    This test locks that guarantee per (fmt, group_size, shape).
    """
    w = _make_weight(out, inp)
    p_ref = pack(w, fmt, group_size=group_size, backend="reference")
    p_ar = pack_with_autoround(w, fmt, group_size=group_size)

    expected = compute_packed_nbytes(out, inp, fmt, group_size)
    assert p_ref.nbytes == expected
    assert p_ar.nbytes == expected
    # And the underlying tensor storage sizes also match.
    assert p_ar.qweight.shape == p_ref.qweight.shape
    assert p_ar.scales.shape == p_ref.scales.shape


def test_autoround_output_is_a_valid_dequantizable_packed_tensor():
    """``dequant_to_fp16`` must work on the autoround output with zero
    special-casing — this proves the byte layout is semantically
    compatible with the reference backend's unpacker."""
    w = _make_weight(8, 256)
    p = pack_with_autoround(w, QuantFormat.INT4, group_size=128)
    recovered = dequant_to_fp16(p)
    assert recovered.shape == w.shape
    assert recovered.dtype == torch.float16


def test_autoround_error_bounded_relative_to_reference():
    """
    On the same weight, autoround and reference should produce similar
    reconstruction error (both are RTN under the hood in the current
    auto-round release). We allow a generous 3× slack so this doesn't
    break if auto-round switches to a non-RTN default in a future
    release that improves error.
    """
    w = _make_weight(16, 256, seed=7)
    err_ref = (
        (dequant_to_fp16(pack(w, QuantFormat.INT4, group_size=128, backend="reference")).float() - w.float())
        .abs()
        .mean()
    )
    err_ar = (
        (dequant_to_fp16(pack_with_autoround(w, QuantFormat.INT4, group_size=128)).float() - w.float())
        .abs()
        .mean()
    )
    # Both must be finite and reasonably close.
    assert torch.isfinite(err_ref) and torch.isfinite(err_ar)
    assert err_ar < err_ref * 3.0 + 1e-4


# ---------------------------------------------------------------------------
# Calibration inputs — the Phase 6 forward-compatibility hook
# ---------------------------------------------------------------------------


def test_compute_imatrix_shape_and_positivity():
    x = torch.randn(32, 128)  # (n_samples, in_features)
    imx = _compute_imatrix(x)
    assert imx.shape == (128,)
    assert imx.dtype == torch.float32
    assert (imx > 0).all(), "imatrix should be strictly positive (includes 1e-12 floor)"


def test_compute_imatrix_flattens_multidim_inputs():
    """Real forward-pass activations are often (batch, seq, in_features);
    the helper must flatten to (-1, in_features) before reducing."""
    x = torch.randn(4, 8, 128)  # (batch, seq, in_features)
    imx = _compute_imatrix(x)
    assert imx.shape == (128,)


def test_compute_imatrix_rejects_1d():
    with pytest.raises(ValueError, match="at least 2 dims"):
        _compute_imatrix(torch.randn(128))


def test_autoround_accepts_calibration_inputs_with_correct_shape():
    w = _make_weight(8, 128)
    calib = torch.randn(64, 128)
    p = pack_with_autoround(w, QuantFormat.INT4, calibration_inputs=calib)
    assert p.fmt == QuantFormat.INT4
    assert p.nbytes == compute_packed_nbytes(8, 128, QuantFormat.INT4, 128)


def test_autoround_rejects_mismatched_calibration_input_shape():
    w = _make_weight(8, 128)
    calib = torch.randn(64, 256)  # wrong in_features
    with pytest.raises(ValueError, match="calibration_inputs"):
        pack_with_autoround(w, QuantFormat.INT4, calibration_inputs=calib)


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------


def test_autoround_rejects_non_2d_weight():
    with pytest.raises(ValueError, match="2D"):
        pack_with_autoround(torch.randn(8, 16, 32), QuantFormat.INT4)


def test_autoround_rejects_integer_weight():
    with pytest.raises(ValueError, match="float"):
        pack_with_autoround(torch.zeros(8, 128, dtype=torch.int32), QuantFormat.INT4)


def test_autoround_rejects_fp16_format():
    """FP16 should go through the reference backend's passthrough, not
    auto-round — the autoround path is only meaningful for INT4/INT2."""
    w = _make_weight(8, 128)
    with pytest.raises(ValueError, match="FP16"):
        pack_with_autoround(w, QuantFormat.FP16)


def test_autoround_rejects_indivisible_group_size():
    w = _make_weight(8, 100)
    with pytest.raises(ValueError, match="not divisible"):
        pack_with_autoround(w, QuantFormat.INT4, group_size=128)


# ---------------------------------------------------------------------------
# End-to-end via pack() dispatch (the public-facing API)
# ---------------------------------------------------------------------------


def test_pack_public_api_routes_to_autoround():
    w = _make_weight(8, 128)
    p = pack(w, QuantFormat.INT4, backend="autoround")
    assert isinstance(p, PackedTensor)
    assert p.fmt == QuantFormat.INT4


def test_pack_public_api_forwards_calibration_inputs():
    w = _make_weight(8, 128)
    calib = torch.randn(64, 128)
    p = pack(w, QuantFormat.INT4, backend="autoround", calibration_inputs=calib)
    assert p.fmt == QuantFormat.INT4
