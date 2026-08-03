"""Audited conversion between AutoGPTQ and DynaExQ packed layouts.

AutoRound checkpoints use AutoGPTQ's int32 packing convention: weights are
packed along the input axis and zero points along the output axis.  DynaExQ
uses a row-major uint8 representation without an explicit zero-point tensor.
The helpers below perform a lossless W4 layout conversion and the same
deterministic W4-to-W2 requantization used by the checkpoint derivation tool.
"""

from __future__ import annotations

import torch

from .quant import PackedTensor, QuantFormat, compute_packed_nbytes


def unpack_axis0(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack AutoGPTQ qweight to unsigned values shaped ``[in, out]``."""
    pack_factor = 32 // bits
    shifts = torch.arange(pack_factor, dtype=torch.int64) * bits
    values = (
        packed.to(device="cpu", dtype=torch.int64).unsqueeze(1)
        >> shifts.view(1, pack_factor, 1)
    ) & ((1 << bits) - 1)
    return values.reshape(packed.shape[0] * pack_factor, packed.shape[1])


def unpack_axis1(
    packed: torch.Tensor,
    bits: int,
    *,
    out_features: int,
) -> torch.Tensor:
    """Unpack AutoGPTQ qzeros to stored values shaped ``[groups, out]``."""
    pack_factor = 32 // bits
    shifts = torch.arange(pack_factor, dtype=torch.int64) * bits
    values = (
        packed.to(device="cpu", dtype=torch.int64).unsqueeze(-1)
        >> shifts.view(1, 1, pack_factor)
    ) & ((1 << bits) - 1)
    return values.reshape(packed.shape[0], -1)[:, :out_features]


def pack_axis0(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned values shaped ``[in, out]`` as AutoGPTQ qweight."""
    pack_factor = 32 // bits
    if values.shape[0] % pack_factor:
        raise ValueError("input features are not divisible by the pack factor")
    grouped = values.to(torch.int64).reshape(
        values.shape[0] // pack_factor,
        pack_factor,
        values.shape[1],
    )
    shifts = torch.arange(pack_factor, dtype=torch.int64) * bits
    packed = torch.sum(grouped << shifts.view(1, pack_factor, 1), dim=1)
    return packed.to(torch.int32)


def pack_axis1(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack stored zero points shaped ``[groups, out]`` as AutoGPTQ qzeros."""
    pack_factor = 32 // bits
    if values.shape[1] % pack_factor:
        raise ValueError("output features are not divisible by the pack factor")
    grouped = values.to(torch.int64).reshape(
        values.shape[0],
        values.shape[1] // pack_factor,
        pack_factor,
    )
    shifts = torch.arange(pack_factor, dtype=torch.int64) * bits
    packed = torch.sum(grouped << shifts.view(1, 1, pack_factor), dim=2)
    return packed.to(torch.int32)


def requantize_autogptq_tensor(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    source_bits: int = 4,
    source_group_size: int = 128,
    target_bits: int = 2,
    target_group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Requantize one AutoGPTQ matrix without materializing an FP16 model."""
    if source_bits != 4 or target_bits != 2:
        raise ValueError("this audited converter only supports W4-to-W2")
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise ValueError("AutoGPTQ qweight and qzeros must be int32")
    if scales.ndim != 2 or qweight.ndim != 2 or qzeros.ndim != 2:
        raise ValueError("invalid AutoGPTQ tensor rank")

    unsigned = unpack_axis0(qweight, source_bits)
    in_features, out_features = unsigned.shape
    if source_group_size % target_group_size:
        raise ValueError("target groups must evenly partition source groups")
    if in_features % source_group_size or in_features % target_group_size:
        raise ValueError("input features do not satisfy source/target groups")
    if out_features % (32 // target_bits):
        raise ValueError("output features do not satisfy W2 packing")
    source_groups = in_features // source_group_size
    if tuple(scales.shape) != (source_groups, out_features):
        raise ValueError("source scale shape does not match qweight")
    if qzeros.shape[0] != source_groups:
        raise ValueError("source zero-point groups do not match qweight")

    zero_points = unpack_axis1(
        qzeros,
        source_bits,
        out_features=out_features,
    ) + 1
    signed_source = (
        unsigned.reshape(source_groups, source_group_size, out_features)
        - zero_points.unsqueeze(1)
    ).to(torch.int8)

    target_groups = in_features // target_group_size
    grouped_codes = signed_source.reshape(
        target_groups,
        target_group_size,
        out_features,
    )
    code_absmax = grouped_codes.abs().amax(dim=1)
    safe_code_absmax = code_absmax.clamp(min=1)
    source_scales_for_target = scales.to(torch.float32).repeat_interleave(
        source_group_size // target_group_size,
        dim=0,
    )
    target_scales = (
        source_scales_for_target.abs() * code_absmax.to(torch.float32)
    ).clamp(min=1e-10)
    effective_codes = torch.where(
        source_scales_for_target.unsqueeze(1) < 0,
        -grouped_codes,
        grouped_codes,
    )
    signed = torch.round(
        effective_codes.to(torch.float32)
        / safe_code_absmax.unsqueeze(1).to(torch.float32)
    ).clamp(-2, 1)
    unsigned_target = (signed.to(torch.int64) + 2).reshape(
        in_features,
        out_features,
    )
    target_qweight = pack_axis0(unsigned_target, target_bits)

    # AutoGPTQ stores zero_point - 1, hence W2 symmetric zp=2 is stored as 1.
    stored_zeros = torch.ones(
        (target_groups, out_features),
        dtype=torch.int64,
    )
    target_qzeros = pack_axis1(stored_zeros, target_bits)
    return target_qweight, target_qzeros, target_scales.to(torch.float16)


def _pack_runtime_signed(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack signed ``[out, in]`` codes into DynaExQ's uint8 layout."""
    out_features, in_features = values.shape
    values = values.to(torch.int16)
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    if values.numel() and (
        int(values.min().item()) < qmin or int(values.max().item()) > qmax
    ):
        raise ValueError(
            f"AutoGPTQ codes do not fit signed W{bits}: "
            f"[{int(values.min().item())}, {int(values.max().item())}]"
        )
    unsigned = (values + (1 << (bits - 1))).to(torch.uint8)
    pack_factor = 8 // bits
    if in_features % pack_factor:
        raise ValueError("input features do not satisfy runtime packing")
    grouped = unsigned.reshape(out_features, in_features // pack_factor, pack_factor)
    shifts = torch.arange(pack_factor, dtype=torch.uint8) * bits
    return torch.sum(
        grouped << shifts.view(1, 1, pack_factor),
        dim=-1,
        dtype=torch.uint8,
    ).contiguous()


def _validate_sequential_groups(
    g_idx: torch.Tensor | None,
    *,
    in_features: int,
    group_size: int,
) -> None:
    """Reject act-order/permuted layouts that the runtime cannot represent."""
    if g_idx is None or g_idx.numel() == 0:
        return
    actual = g_idx.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    expected = torch.arange(in_features, dtype=torch.int64) // group_size
    if actual.shape != expected.shape or not torch.equal(actual, expected):
        raise ValueError(
            "AutoGPTQ g_idx is not sequential; act-order layouts are unsupported"
        )


def packed_from_autogptq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    source_bits: int,
    source_group_size: int,
    target_format: QuantFormat,
    g_idx: torch.Tensor | None = None,
) -> PackedTensor:
    """Convert an AutoGPTQ matrix to a DynaExQ ``PackedTensor``.

    W4 is a layout-only, bit-exact conversion. W2 invokes the audited
    calibration-free integer-domain derivation. FP16 reconstructs the W4
    checkpoint values and is intended only for diagnostics.
    """
    qweight = qweight.detach().to(device="cpu")
    qzeros = qzeros.detach().to(device="cpu")
    scales = scales.detach().to(device="cpu", dtype=torch.float16)
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise ValueError("AutoGPTQ qweight and qzeros must be int32")
    if source_bits != 4:
        raise ValueError(f"expected an AutoRound W4 source, got W{source_bits}")

    source_unsigned = unpack_axis0(qweight, source_bits)
    in_features, out_features = source_unsigned.shape
    _validate_sequential_groups(
        g_idx,
        in_features=in_features,
        group_size=source_group_size,
    )

    if target_format == QuantFormat.INT2:
        qweight, qzeros, scales = requantize_autogptq_tensor(
            qweight,
            qzeros,
            scales,
            source_bits=source_bits,
            source_group_size=source_group_size,
            target_bits=2,
            target_group_size=64,
        )
        bits = 2
        group_size = 64
        unsigned = unpack_axis0(qweight, bits)
    else:
        bits = source_bits
        group_size = source_group_size
        unsigned = source_unsigned

    groups = in_features // group_size
    if tuple(scales.shape) != (groups, out_features):
        raise ValueError(
            "AutoGPTQ scale shape does not match packed weight dimensions"
        )
    zero_points = unpack_axis1(
        qzeros,
        bits,
        out_features=out_features,
    ) + 1
    signed = (
        unsigned.reshape(groups, group_size, out_features)
        - zero_points.unsqueeze(1)
    )
    signed_rows = signed.reshape(in_features, out_features).transpose(0, 1)
    runtime_scales = scales.transpose(0, 1).contiguous()

    if target_format == QuantFormat.FP16:
        reconstructed = (
            signed_rows.to(torch.float16)
            .reshape(out_features, groups, group_size)
            * runtime_scales.unsqueeze(-1)
        ).reshape(out_features, in_features).contiguous()
        return PackedTensor(
            qweight=reconstructed,
            scales=None,
            group_size=in_features,
            out_features=out_features,
            in_features=in_features,
            fmt=QuantFormat.FP16,
            nbytes=compute_packed_nbytes(
                out_features,
                in_features,
                QuantFormat.FP16,
                in_features,
            ),
        )
    if target_format not in {QuantFormat.INT4, QuantFormat.INT2}:
        raise ValueError(f"unsupported target format: {target_format}")
    if target_format == QuantFormat.INT4 and bits != 4:
        raise ValueError("W4 target requires W4 source codes")

    runtime_qweight = _pack_runtime_signed(signed_rows, bits)
    return PackedTensor(
        qweight=runtime_qweight,
        scales=runtime_scales,
        group_size=group_size,
        out_features=out_features,
        in_features=in_features,
        fmt=target_format,
        nbytes=compute_packed_nbytes(
            out_features,
            in_features,
            target_format,
            group_size,
        ),
    )
