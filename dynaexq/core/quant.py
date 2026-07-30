"""
Quantization primitives for DynaExq (Phase 2 of plan.md).

Provides a unified interface for INT4 / INT2 group-wise symmetric quantization
of expert MLP weights, plus pack / dequant / fused_linear operations.

Two backends are exposed:

1. ``"reference"`` — pure PyTorch group-wise symmetric quantization. Always
   available. Used in unit tests and as a correctness baseline. Slow but
   produces a faithful PackedTensor (correct quantization error, correct byte
   accounting), which is what the BudgetTracker and quality eval depend on.

2. ``"autoround"`` — wrapper around AutoRound's per-tensor symmetric integer
   quantizer. It produces the same packed runtime layout, but it does not run
   AutoRound's full activation-calibrated optimization loop.

Conventions
-----------
* Weight tensors follow ``nn.Linear``: shape ``(out_features, in_features)``.
* Quantization groups partition along the **input** dimension. ``group_size``
  must divide ``in_features``.
* Symmetric quantization: ``q = round(w / scale)``, ``w_hat = q * scale``.
  ``qmin = -(2 ** (nbits - 1))``, ``qmax = 2 ** (nbits - 1) - 1``.
* Packed storage uses ``torch.uint8``:
  - INT4: 2 nibbles per byte, low nibble = even column. Stored as
    unsigned ``q + 8`` in ``[0, 15]``.
  - INT2: 4 crumbs per byte, lowest 2 bits = column 0. Stored as unsigned
    ``q + 2`` in ``[0, 3]``.
* Scales are kept in ``fp16`` and counted toward ``nbytes`` (per plan §3.1).

Why group-wise symmetric (not asymmetric)?
------------------------------------------
The plan §3.1 specifies "group-wise symmetric"; symmetric removes the
zero-point, which (a) saves bytes in the budget, (b) matches what AutoRound
produces by default for INT4/INT2, and (c) keeps the reference backend
trivially auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Runtime capability check for the fused INT4 matmul kernel.
# Requires PyTorch >= 2.1 and a CUDA device with sm80+ (Ampere or newer).
# ---------------------------------------------------------------------------
_INT4PACK_MM_AVAILABLE: bool = (
    hasattr(torch.ops.aten, "_weight_int4pack_mm")
    and hasattr(torch.ops.aten, "_convert_weight_to_int4pack")
)


def budget_safe_dispatch_available(
    fmt: "QuantFormat",
    device: torch.device | str,
) -> bool:
    """Whether dispatch avoids materializing a full dequantized weight."""
    device_type = torch.device(device).type
    if fmt == QuantFormat.FP16:
        return True
    if device_type != "cuda":
        return False
    if fmt == QuantFormat.INT4:
        return _INT4PACK_MM_AVAILABLE
    if fmt == QuantFormat.INT2:
        try:
            from .int2_kernel import TRITON_INT2_AVAILABLE

            return TRITON_INT2_AVAILABLE
        except ImportError:
            return False
    return False


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class QuantFormat(Enum):
    """Supported precision tiers for expert MLP weights."""

    FP16 = "fp16"
    INT4 = "int4"  # group-wise symmetric, default group_size=128
    INT2 = "int2"  # group-wise symmetric, default group_size=64

    @property
    def nbits(self) -> int:
        return {QuantFormat.FP16: 16, QuantFormat.INT4: 4, QuantFormat.INT2: 2}[self]

    @property
    def is_quantized(self) -> bool:
        return self != QuantFormat.FP16


# Default group sizes per plan §3.1.
DEFAULT_GROUP_SIZE: dict[QuantFormat, int] = {
    QuantFormat.INT4: 128,
    QuantFormat.INT2: 64,
}


@dataclass
class PackedTensor:
    """
    Packed representation of a quantized weight tensor.

    For ``fmt == FP16`` only ``qweight`` is populated (shape ``(out, in)``,
    dtype fp16) and ``scales`` is ``None``. This lets the runtime carry HI-tier
    weights through the same data path without a special case.

    For INT4/INT2 the layout is described in the module docstring. ``nbytes``
    is the *real* on-device byte count (qweight + scales) and is what the
    BudgetTracker must account for.
    """

    qweight: torch.Tensor
    scales: Optional[torch.Tensor]
    group_size: int
    out_features: int
    in_features: int
    fmt: QuantFormat
    nbytes: int
    # Total bytes required by the on-device representation. This may differ
    # from ``nbytes`` when a kernel-native layout replaces the canonical
    # transfer layout. Host artifacts still contain only ``nbytes``.
    resident_nbytes: int = 0
    canonical_valid: bool = True
    # Reserved for future asymmetric backends; always None for the reference
    # symmetric path.
    zeros: Optional[torch.Tensor] = None
    # Cache for the native int4pack kernel format (INT4 only, CUDA only).
    # Populated lazily on first call to fused_linear; cleared on device transfer.
    # Excluded from equality checks and repr since it is a derived cache, not
    # authoritative state.
    int4pack_weight: Optional[torch.Tensor] = field(
        default=None, repr=False, compare=False
    )
    int4pack_scales_and_zeros: Optional[torch.Tensor] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.resident_nbytes == 0:
            self.resident_nbytes = self.nbytes
        if self.resident_nbytes < self.nbytes:
            raise ValueError("resident_nbytes cannot be smaller than nbytes")
        if self.fmt == QuantFormat.FP16:
            if self.qweight.dtype != torch.float16:
                raise ValueError(
                    f"FP16 PackedTensor expects fp16 qweight, got {self.qweight.dtype}"
                )
            if self.scales is not None:
                raise ValueError("FP16 PackedTensor must not carry scales")
        else:
            if self.qweight.dtype != torch.uint8:
                raise ValueError(
                    f"{self.fmt.value} PackedTensor expects uint8 qweight, "
                    f"got {self.qweight.dtype}"
                )
            if self.scales is None:
                raise ValueError(f"{self.fmt.value} PackedTensor requires scales")
            if self.scales.dtype != torch.float16:
                raise ValueError(
                    f"scales must be fp16, got {self.scales.dtype}"
                )
            if self.in_features % self.group_size != 0:
                raise ValueError(
                    f"in_features={self.in_features} not divisible by "
                    f"group_size={self.group_size}"
                )

    @property
    def device(self) -> torch.device:
        return self.qweight.device

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "PackedTensor":
        """Move the packed tensor to a new device (qweight + scales together)."""
        new_qweight = self.qweight.to(device, non_blocking=non_blocking)
        new_scales = (
            self.scales.to(device, non_blocking=non_blocking)
            if self.scales is not None
            else None
        )
        return PackedTensor(
            qweight=new_qweight,
            scales=new_scales,
            group_size=self.group_size,
            out_features=self.out_features,
            in_features=self.in_features,
            fmt=self.fmt,
            nbytes=self.nbytes,
            resident_nbytes=self.resident_nbytes,
            canonical_valid=self.canonical_valid,
            zeros=self.zeros,
        )


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def compute_packed_nbytes(
    out_features: int,
    in_features: int,
    fmt: QuantFormat,
    group_size: int,
) -> int:
    """
    Return the on-device byte footprint of a packed tensor (qweight + scales).

    This is what the BudgetTracker (Phase 4) must charge against the HBM
    envelope, so it MUST agree with what ``pack`` actually allocates.
    """
    if fmt == QuantFormat.FP16:
        return out_features * in_features * 2

    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} not divisible by group_size={group_size}"
        )

    if fmt == QuantFormat.INT4:
        qweight_bytes = out_features * (in_features // 2)  # 2 nibbles per byte
    elif fmt == QuantFormat.INT2:
        qweight_bytes = out_features * (in_features // 4)  # 4 crumbs per byte
    else:
        raise ValueError(f"Unsupported QuantFormat: {fmt}")

    n_groups = in_features // group_size
    scales_bytes = out_features * n_groups * 2  # fp16 scales
    return qweight_bytes + scales_bytes


# ---------------------------------------------------------------------------
# Reference backend (pure PyTorch)
# ---------------------------------------------------------------------------


def _qrange(nbits: int) -> tuple[int, int]:
    qmax = (1 << (nbits - 1)) - 1
    qmin = -(1 << (nbits - 1))
    return qmin, qmax


def _quantize_groupwise_symmetric(
    weight: torch.Tensor,
    group_size: int,
    nbits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Group-wise symmetric quantization along the input dimension.

    Args:
        weight: shape ``(out, in)``, any float dtype.
        group_size: must divide ``in``.
        nbits: 4 or 2.

    Returns:
        q_signed: int8 tensor with shape ``(out, in)`` holding values in
            ``[qmin, qmax]``. Caller is responsible for packing into uint8.
        scales: fp16 tensor with shape ``(out, in/group_size)``.
    """
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    qmin, qmax = _qrange(nbits)

    w = weight.to(torch.float32)
    grouped = w.view(out_features, in_features // group_size, group_size)
    # Symmetric scale: max(|w|) / qmax. Use qmax (positive) so qmin is reachable.
    max_abs = grouped.abs().amax(dim=-1)  # (out, n_groups)
    # Avoid divide-by-zero on dead groups.
    scales = (max_abs / qmax).clamp(min=1e-8)
    scales_expanded = scales.unsqueeze(-1)  # (out, n_groups, 1)
    q = torch.round(grouped / scales_expanded).clamp(qmin, qmax).to(torch.int8)
    q = q.view(out_features, in_features)
    return q, scales.to(torch.float16)


def _pack_int4(q_signed: torch.Tensor) -> torch.Tensor:
    """
    Pack signed int4 values into uint8: 2 nibbles per byte.

    Encoding: stored = q + 8 (so range maps to [0, 15]).
    Layout: low nibble = even column, high nibble = odd column.
    """
    out_features, in_features = q_signed.shape
    assert in_features % 2 == 0
    unsigned = (q_signed.to(torch.int16) + 8).to(torch.uint8)
    # (out, in) → (out, in/2, 2)
    pairs = unsigned.view(out_features, in_features // 2, 2)
    low = pairs[..., 0] & 0x0F
    high = (pairs[..., 1] & 0x0F) << 4
    return (low | high).contiguous()


def _unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Inverse of ``_pack_int4``. Returns int8 in ``[-8, 7]``."""
    out_features = packed.shape[0]
    assert packed.shape[1] == in_features // 2
    low = (packed & 0x0F).to(torch.int16)
    high = ((packed >> 4) & 0x0F).to(torch.int16)
    # Interleave: column 2k = low, column 2k+1 = high
    interleaved = torch.stack((low, high), dim=-1).view(out_features, in_features)
    return (interleaved - 8).to(torch.int8)


def _pack_int2(q_signed: torch.Tensor) -> torch.Tensor:
    """
    Pack signed int2 values into uint8: 4 crumbs per byte.

    Encoding: stored = q + 2 (so range maps to [0, 3]).
    Layout: bits 0-1 = column 0, bits 2-3 = column 1, ..., bits 6-7 = column 3.
    """
    out_features, in_features = q_signed.shape
    assert in_features % 4 == 0
    unsigned = (q_signed.to(torch.int16) + 2).to(torch.uint8)
    quads = unsigned.view(out_features, in_features // 4, 4)
    byte = (
        (quads[..., 0] & 0x03)
        | ((quads[..., 1] & 0x03) << 2)
        | ((quads[..., 2] & 0x03) << 4)
        | ((quads[..., 3] & 0x03) << 6)
    )
    return byte.contiguous()


def _unpack_int2(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Inverse of ``_pack_int2``. Returns int8 in ``[-2, 1]``."""
    out_features = packed.shape[0]
    assert packed.shape[1] == in_features // 4
    c0 = (packed & 0x03).to(torch.int16)
    c1 = ((packed >> 2) & 0x03).to(torch.int16)
    c2 = ((packed >> 4) & 0x03).to(torch.int16)
    c3 = ((packed >> 6) & 0x03).to(torch.int16)
    interleaved = torch.stack((c0, c1, c2, c3), dim=-1).view(out_features, in_features)
    return (interleaved - 2).to(torch.int8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pack(
    weight: torch.Tensor,
    fmt: QuantFormat,
    group_size: Optional[int] = None,
    backend: str = "reference",
    calibration_inputs: Optional[torch.Tensor] = None,
) -> PackedTensor:
    """
    Quantize and pack a weight tensor.

    Args:
        weight: ``(out_features, in_features)`` float tensor (fp16/fp32/bf16).
        fmt: target quantization format.
        group_size: defaults to ``DEFAULT_GROUP_SIZE[fmt]`` for INT4/INT2.
        backend: ``"reference"`` (default) or ``"autoround"`` (Phase 2.1).
        calibration_inputs: Only used when ``backend="autoround"``. Activation
            samples ``(n, in_features)`` used to derive an importance matrix
            (imatrix). Silently ignored by the reference backend. Silently
            accepted by the autoround backend for forward compatibility —
            the current per-tensor autoround entry point is RTN-based and
            ignores imatrix numerically; Phase 6 will upgrade to the full
            WrapperLinear optimization loop.

    Returns:
        PackedTensor with ``nbytes`` matching the on-device footprint.
    """
    if weight.dim() != 2:
        raise ValueError(f"pack expects a 2D weight tensor, got shape {tuple(weight.shape)}")

    out_features, in_features = weight.shape

    if fmt == QuantFormat.FP16:
        return PackedTensor(
            qweight=weight.to(torch.float16).contiguous(),
            scales=None,
            group_size=in_features,
            out_features=out_features,
            in_features=in_features,
            fmt=fmt,
            nbytes=compute_packed_nbytes(out_features, in_features, fmt, in_features),
        )

    if group_size is None:
        group_size = DEFAULT_GROUP_SIZE[fmt]

    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} not divisible by group_size={group_size}"
        )

    if backend == "reference":
        q_signed, scales = _quantize_groupwise_symmetric(
            weight, group_size, fmt.nbits
        )
        if fmt == QuantFormat.INT4:
            qweight = _pack_int4(q_signed)
        elif fmt == QuantFormat.INT2:
            qweight = _pack_int2(q_signed)
        else:
            raise ValueError(f"Unsupported QuantFormat for reference backend: {fmt}")

        return PackedTensor(
            qweight=qweight,
            scales=scales.contiguous(),
            group_size=group_size,
            out_features=out_features,
            in_features=in_features,
            fmt=fmt,
            nbytes=compute_packed_nbytes(out_features, in_features, fmt, group_size),
        )

    if backend == "autoround":
        # Phase 2.1 live path. See :mod:`quant_autoround` for the full
        # rationale — in short, we call auto_round's per-tensor
        # symmetric int quantizer and repack into our byte layout so
        # the runtime stays backend-agnostic.
        from .quant_autoround import pack_with_autoround

        return pack_with_autoround(
            weight, fmt, group_size=group_size, calibration_inputs=calibration_inputs
        )

    raise ValueError(f"Unknown quantization backend: {backend!r}")


def dequant_to_fp16(packed: PackedTensor) -> torch.Tensor:
    """
    Reconstruct a fp16 weight tensor of shape ``(out_features, in_features)``.

    For FP16 inputs this is a no-op view; for INT4/INT2 it unpacks and applies
    the per-group scale. Used in tests, in fallback forward paths, and as the
    correctness reference for any future fused kernel.
    """
    if not packed.canonical_valid:
        raise RuntimeError(
            "canonical packed bytes were replaced by a kernel-native layout"
        )
    if packed.fmt == QuantFormat.FP16:
        return packed.qweight

    if packed.fmt == QuantFormat.INT4:
        q_signed = _unpack_int4(packed.qweight, packed.in_features)
    elif packed.fmt == QuantFormat.INT2:
        q_signed = _unpack_int2(packed.qweight, packed.in_features)
    else:
        raise ValueError(f"Unsupported QuantFormat for dequant: {packed.fmt}")

    out_features = packed.out_features
    in_features = packed.in_features
    n_groups = in_features // packed.group_size

    q_grouped = q_signed.to(torch.float16).view(out_features, n_groups, packed.group_size)
    scales = packed.scales.view(out_features, n_groups, 1)
    w = (q_grouped * scales).view(out_features, in_features)
    return w.contiguous()


def _prepare_int4pack_mm(
    packed: PackedTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert an INT4 ``PackedTensor`` to the format required by
    ``torch.ops.aten._weight_int4pack_mm``.

    Our ``qweight`` is ``(out_features, in_features // 2)`` uint8 with the
    low-nibble-first layout that ``_convert_weight_to_int4pack`` expects:
    bits [3:0] hold the even column, bits [7:4] hold the odd column.

    The kernel dequantizes via::

        w = (uint4_val - 8) * scale + zero_bias

    Our symmetric encoding stores ``uint4_val = q_signed + 8``, so for a
    faithful reconstruction ``w = q_signed * scale`` we need ``zero_bias = 0``.
    The kernel already handles the ``-8`` offset internally; the caller must
    NOT fold it into ``zero_bias`` a second time.

    Args:
        packed: INT4 ``PackedTensor`` on a CUDA device.

    Returns:
        w_int4pack: kernel-internal packed tensor on the same device.
        scales_and_zeros: ``(n_groups, out_features, 2)`` bfloat16 tensor
            where ``[g, n, 0] = scale`` and ``[g, n, 1] = 0`` (symmetric).
    """
    assert packed.fmt == QuantFormat.INT4, "Only INT4 is supported"
    # Our _pack_int4 convention: low nibble (bits 0-3) = even column,
    # high nibble (bits 4-7) = odd column.
    # _convert_weight_to_int4pack uses the OPPOSITE convention:
    # low nibble = odd column, high nibble = even column.
    # Swap nibbles before handing off to the kernel.
    q_swapped = ((packed.qweight & 0x0F) << 4) | ((packed.qweight >> 4) & 0x0F)
    w_int4pack = torch.ops.aten._convert_weight_to_int4pack(q_swapped, innerKTiles=2)
    n_groups = packed.in_features // packed.group_size
    # packed.scales: (out_features, n_groups) → kernel needs (n_groups, out_features)
    scales_t = packed.scales.view(packed.out_features, n_groups).T.contiguous()
    # zero_bias = 0 for symmetric quantization (kernel already subtracts 8 internally)
    zeros_fp = torch.zeros_like(scales_t)
    scales_and_zeros = torch.stack(
        [scales_t.to(torch.bfloat16), zeros_fp.to(torch.bfloat16)], dim=-1
    ).contiguous()
    return w_int4pack, scales_and_zeros


def fused_linear(
    x: torch.Tensor,
    packed: PackedTensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute ``x @ W.T + bias`` where ``W`` is held in packed form.

    **Fast path (INT4 on CUDA, PyTorch ≥ 2.1 / sm80+)**:
    Uses ``torch.ops.aten._weight_int4pack_mm`` — a fused dequant + matmul
    kernel that never materialises a full fp16 weight tensor. The kernel
    format is prepared once and cached on the ``PackedTensor``.
    Output is cast to **fp16** so it can be added directly to FP16 expert
    outputs in mixed-precision MoE blocks without a dtype mismatch.

    **Fallback**: reference dequant → FP16 matmul (always available).
    """
    if (
        _INT4PACK_MM_AVAILABLE
        and packed.fmt == QuantFormat.INT4
        and packed.qweight.is_cuda
    ):
        # Lazily prepare and cache the kernel-format weights.
        if packed.int4pack_weight is None:
            packed.int4pack_weight, packed.int4pack_scales_and_zeros = (
                _prepare_int4pack_mm(packed)
            )
        out = torch.ops.aten._weight_int4pack_mm(
            x.to(torch.bfloat16),
            packed.int4pack_weight,
            packed.group_size,
            packed.int4pack_scales_and_zeros,
        )
        if bias is not None:
            out = out + bias
        # Cast to fp16 so the result is compatible with FP16 expert outputs
        # in mixed-precision MoE accumulation (INT4 result + FP16 result).
        return out.to(torch.float16)

    if packed.fmt == QuantFormat.INT2 and packed.qweight.is_cuda:
        from .int2_kernel import int2_linear

        return int2_linear(
            x,
            packed.qweight,
            packed.scales,
            in_features=packed.in_features,
            out_features=packed.out_features,
            group_size=packed.group_size,
            bias=bias,
        )

    # Fallback: reference dequant + FP16 matmul.
    w_fp16 = dequant_to_fp16(packed)
    return torch.nn.functional.linear(x.to(torch.float16), w_fp16, bias)


__all__ = [
    "QuantFormat",
    "PackedTensor",
    "DEFAULT_GROUP_SIZE",
    "compute_packed_nbytes",
    "pack",
    "dequant_to_fp16",
    "fused_linear",
    "budget_safe_dispatch_available",
    "_INT4PACK_MM_AVAILABLE",
]
