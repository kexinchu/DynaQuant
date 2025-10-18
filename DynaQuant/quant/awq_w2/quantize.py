"""
W2 Symmetric Quantization
==========================
2-bit symmetric per-group quantization with AWQ-style scaling.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def symmetric_quantize(
    weight: torch.Tensor,
    n_bits: int = 2,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-group quantization.

    Args:
        weight: Float weight tensor, shape [out_features, in_features]
        n_bits: Number of bits (default: 2)
        group_size: Group size for per-group quantization

    Returns:
        weight_q: Quantized weights in int8, values in [-2^(n_bits-1), 2^(n_bits-1)-1]
        scale: Per-group scales in fp16, shape [out_features, in_features // group_size]
    """
    assert n_bits == 2, f"Only 2-bit quantization supported, got {n_bits}"

    out_features, in_features = weight.shape
    assert in_features % group_size == 0, \
        f"in_features ({in_features}) must be divisible by group_size ({group_size})"

    num_groups = in_features // group_size

    # Reshape to [out_features, num_groups, group_size]
    weight_grouped = weight.reshape(out_features, num_groups, group_size)

    # Compute scale per group: max absolute value
    scale = weight_grouped.abs().max(dim=-1, keepdim=True)[0]

    # Avoid division by zero
    scale = torch.clamp(scale, min=1e-5)

    # Quantization bounds for 2-bit signed: [-2, 1]
    qmin, qmax = -2, 1

    # Quantize: q = clamp(round(w / scale), qmin, qmax)
    weight_q = weight_grouped / scale
    weight_q = torch.clamp(torch.round(weight_q), qmin, qmax)
    weight_q = weight_q.reshape(out_features, in_features).to(torch.int8)

    # Reshape scale: [out_features, num_groups, 1] -> [out_features, num_groups]
    scale = scale.squeeze(-1).to(torch.float16)

    return weight_q, scale


def dequantize_weight(
    weight_q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Dequantize 2-bit weights.

    Args:
        weight_q: Quantized weights, shape [out_features, in_features]
        scale: Per-group scales, shape [out_features, num_groups]
        group_size: Group size
        dtype: Output dtype (default: fp16)

    Returns:
        Dequantized weights in specified dtype
    """
    out_features, in_features = weight_q.shape
    num_groups = in_features // group_size

    # Reshape weight_q to [out_features, num_groups, group_size]
    weight_grouped = weight_q.reshape(
        out_features, num_groups, group_size).to(dtype)

    # Broadcast scale: [out_features, num_groups] -> [out_features, num_groups, 1]
    scale_expanded = scale.unsqueeze(-1).to(dtype)

    # Dequantize: w = q * scale
    weight_deq = weight_grouped * scale_expanded
    weight_deq = weight_deq.reshape(out_features, in_features)

    return weight_deq


def quantize_weight_w2(
    weight: torch.Tensor,
    group_size: int = 128,
    alpha: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    2-bit weight quantization with optional per-group clipping (AWQ alpha).

    Args:
        weight: Float weight tensor, shape [out_features, in_features]
        group_size: Group size for quantization
        alpha: Optional per-group clipping factor, shape [out_features, num_groups]
               If provided, clips weights to [-alpha*max, alpha*max] before quantization

    Returns:
        weight_q: Quantized weights in int8
        scale: Per-group scales in fp16
    """
    out_features, in_features = weight.shape
    num_groups = in_features // group_size

    weight_grouped = weight.reshape(out_features, num_groups, group_size)

    # Apply alpha clipping (AWQ-style)
    if alpha is not None:
        assert alpha.shape == (out_features, num_groups), \
            f"Alpha shape mismatch: expected {(out_features, num_groups)}, got {alpha.shape}"

        # Compute max per group
        max_val = weight_grouped.abs().max(dim=-1, keepdim=True)[0]

        # Clip: w_clipped = clamp(w, -alpha*max, alpha*max)
        alpha_expanded = alpha.unsqueeze(-1)
        clip_val = alpha_expanded * max_val
        weight_grouped = torch.clamp(weight_grouped, -clip_val, clip_val)

    # Compute scale
    scale = weight_grouped.abs().max(dim=-1, keepdim=True)[0]
    scale = torch.clamp(scale, min=1e-5)

    # Quantize
    qmin, qmax = -2, 1
    weight_q = weight_grouped / scale
    weight_q = torch.clamp(torch.round(weight_q), qmin, qmax)
    weight_q = weight_q.reshape(out_features, in_features).to(torch.int8)

    scale = scale.squeeze(-1).to(torch.float16)

    return weight_q, scale


def compute_quantization_error(
    weight_orig: torch.Tensor,
    weight_q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    X: Optional[torch.Tensor] = None
) -> float:
    """
    Compute quantization error.

    Args:
        weight_orig: Original float weights
        weight_q: Quantized weights
        scale: Quantization scales
        group_size: Group size
        X: Optional activation tensor for weighted error (AWQ-style)
           Shape: [batch, in_features]

    Returns:
        Reconstruction error (MSE or activation-weighted MSE)
    """
    # Dequantize
    weight_deq = dequantize_weight(
        weight_q, scale, group_size, weight_orig.dtype)

    if X is None:
        # Simple MSE
        error = ((weight_orig - weight_deq) ** 2).mean()
    else:
        # Activation-weighted error: ||X @ W - X @ W_q||^2
        out_orig = X @ weight_orig.T  # [batch, out_features]
        out_deq = X @ weight_deq.T
        error = ((out_orig - out_deq) ** 2).mean()

    return error.item()


class QuantizationConfig:
    """Configuration for W2A16 quantization."""

    def __init__(
        self,
        algorithm: str = "awq",
        bits: int = 2,
        group_size: int = 128,
        symmetric: bool = True,
        packed_layout: str = "4x2bit_per_byte",
        preserve_dtype: str = "float16",
        version: str = "1.0"
    ):
        self.algorithm = algorithm
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.packed_layout = packed_layout
        self.preserve_dtype = preserve_dtype
        self.version = version

    def to_dict(self):
        return {
            "algorithm": self.algorithm,
            "bits": self.bits,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "packed_layout": self.packed_layout,
            "preserve_dtype": self.preserve_dtype,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


def test_quantization():
    """Test quantization functions."""
    print("Testing 2-bit quantization...")

    torch.manual_seed(42)

    # Test 1: Basic symmetric quantization
    weight = torch.randn(64, 512, dtype=torch.float32)
    weight_q, scale = symmetric_quantize(weight, n_bits=2, group_size=128)

    print(f"Weight shape: {weight.shape}")
    print(f"Quantized shape: {weight_q.shape}")
    print(f"Scale shape: {scale.shape}")
    print(f"Quantized range: [{weight_q.min()}, {weight_q.max()}]")

    # Dequantize and compute error
    weight_deq = dequantize_weight(
        weight_q, scale, group_size=128, dtype=torch.float32)
    mse = ((weight - weight_deq) ** 2).mean()
    rel_error = (mse.sqrt() / weight.abs().mean()).item()

    print(f"MSE: {mse:.6f}")
    print(f"Relative error: {rel_error:.6f}")
    assert rel_error < 0.1, f"Relative error too high: {rel_error}"

    # Test 2: With AWQ alpha clipping
    alpha = torch.ones(64, 4) * 0.5  # Aggressive clipping
    weight_q2, scale2 = quantize_weight_w2(weight, group_size=128, alpha=alpha)

    weight_deq2 = dequantize_weight(
        weight_q2, scale2, group_size=128, dtype=torch.float32)
    mse2 = ((weight - weight_deq2) ** 2).mean()
    print(f"\nWith alpha clipping:")
    print(f"MSE: {mse2:.6f}")

    # Test 3: Activation-weighted error
    X = torch.randn(32, 512, dtype=torch.float32)
    error_weighted = compute_quantization_error(
        weight, weight_q, scale, group_size=128, X=X)
    print(f"\nActivation-weighted error: {error_weighted:.6f}")

    # Test 4: Different group sizes
    for gs in [64, 128]:
        w_q, s = symmetric_quantize(weight, n_bits=2, group_size=gs)
        w_deq = dequantize_weight(w_q, s, group_size=gs, dtype=torch.float32)
        error = ((weight - w_deq) ** 2).mean().item()
        print(f"Group size {gs}: error = {error:.6f}")

    print("\n✅ All quantization tests passed!")


if __name__ == "__main__":
    test_quantization()
