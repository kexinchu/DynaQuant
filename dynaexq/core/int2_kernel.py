"""Budget-safe Triton kernel for group-wise symmetric INT2 linear layers."""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    TRITON_INT2_AVAILABLE = torch.cuda.is_available()
except ImportError:  # pragma: no cover - depends on the PyTorch distribution
    triton = None
    tl = None
    TRITON_INT2_AVAILABLE = False


if triton is not None:

    @triton.jit
    def _int2_linear_kernel(
        x_ptr,
        q_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        m_size: tl.constexpr,
        n_size: tl.constexpr,
        k_size: tl.constexpr,
        groups_per_row: tl.constexpr,
        group_size: tl.constexpr,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, k_size, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            x = tl.load(
                x_ptr + offs_m[:, None] * k_size + offs_k[None, :],
                mask=(offs_m[:, None] < m_size)
                & (offs_k[None, :] < k_size),
                other=0.0,
            )
            byte_index = offs_k // 4
            shift = (offs_k % 4) * 2
            packed = tl.load(
                q_ptr
                + offs_n[:, None] * (k_size // 4)
                + byte_index[None, :],
                mask=(offs_n[:, None] < n_size)
                & (offs_k[None, :] < k_size),
                other=0,
            ).to(tl.int32)
            signed = ((packed >> shift[None, :]) & 0x3) - 2
            scales = tl.load(
                scale_ptr
                + offs_n[:, None] * groups_per_row
                + (offs_k // group_size)[None, :],
                mask=(offs_n[:, None] < n_size)
                & (offs_k[None, :] < k_size),
                other=0.0,
            )
            weight = signed.to(tl.float16) * scales
            accumulator += tl.dot(x, tl.trans(weight))

        if has_bias:
            bias = tl.load(
                bias_ptr + offs_n,
                mask=offs_n < n_size,
                other=0.0,
            )
            accumulator += bias[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * n_size + offs_n[None, :],
            accumulator,
            mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size),
        )


def int2_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    in_features: int,
    out_features: int,
    group_size: int,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``x @ W.T`` without materializing an FP16 weight matrix."""
    if not TRITON_INT2_AVAILABLE:
        raise RuntimeError("Triton INT2 CUDA kernel is unavailable")
    if not (x.is_cuda and qweight.is_cuda and scales.is_cuda):
        raise ValueError("INT2 Triton inputs must all reside on CUDA")
    if x.shape[-1] != in_features:
        raise ValueError(
            f"input last dimension {x.shape[-1]} != {in_features}"
        )
    leading_shape = x.shape[:-1]
    x_2d = x.reshape(-1, in_features).contiguous()
    output = torch.empty(
        (x_2d.shape[0], out_features),
        dtype=torch.float16,
        device=x.device,
    )
    grid = (
        triton.cdiv(x_2d.shape[0], 16),
        triton.cdiv(out_features, 32),
    )
    bias_arg = bias if bias is not None else output
    _int2_linear_kernel[grid](
        x_2d,
        qweight,
        scales,
        bias_arg,
        output,
        x_2d.shape[0],
        out_features,
        in_features,
        in_features // group_size,
        group_size,
        bias is not None,
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )
    return output.reshape(*leading_shape, out_features)


__all__ = ["TRITON_INT2_AVAILABLE", "int2_linear"]
