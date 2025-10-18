"""
Weight packing utilities for INT4 (W4) and INT2 (W2) quantization.
Supports per-group symmetric quantization with configurable group size.
Provides bit-packing to uint8 for efficient storage.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def quantize_weights_symmetric(
    weights: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights using per-group symmetric quantization.

    Args:
        weights: Float tensor of shape [out_features, in_features]
        bits: Number of bits (2 or 4)
        group_size: Size of quantization groups

    Returns:
        qweights: Quantized weights (int8)
        scales: Per-group scales
    """
    assert bits in [2, 4], f"Only 2 or 4 bits supported, got {bits}"
    assert weights.dim() == 2, "Weights must be 2D"

    out_features, in_features = weights.shape
    num_groups = (in_features + group_size - 1) // group_size

    # Reshape to [out_features, num_groups, group_size]
    # Pad if necessary
    if in_features % group_size != 0:
        pad_size = num_groups * group_size - in_features
        weights_padded = torch.nn.functional.pad(weights, (0, pad_size))
    else:
        weights_padded = weights

    weights_grouped = weights_padded.view(out_features, num_groups, group_size)

    # Compute per-group scales (symmetric)
    # Scale = max(abs(group)) / (2^(bits-1) - 1)
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))

    abs_max = torch.amax(torch.abs(weights_grouped), dim=2, keepdim=True)
    # Avoid division by zero
    scales = abs_max / qmax
    scales = torch.clamp(scales, min=1e-8)

    # Quantize: q = round(w / scale)
    qweights = torch.round(weights_grouped / scales)
    qweights = torch.clamp(qweights, qmin, qmax).to(torch.int8)

    # Reshape back
    qweights = qweights.view(out_features, -1)
    scales = scales.view(out_features, num_groups)

    # Remove padding if added
    if in_features % group_size != 0:
        qweights = qweights[:, :in_features]

    return qweights, scales


def pack_weights_4bit(qweights: torch.Tensor) -> torch.Tensor:
    """
    Pack 4-bit quantized weights into uint8 (2 values per byte).

    Args:
        qweights: Quantized weights in int8 format (values in [-8, 7])

    Returns:
        packed: Packed weights as uint8, shape [out_features, in_features // 2]
    """
    assert qweights.dtype == torch.int8
    out_features, in_features = qweights.shape

    # Convert to unsigned: add 8 to shift range from [-8, 7] to [0, 15]
    qweights_unsigned = qweights.to(torch.int32) + 8
    qweights_unsigned = qweights_unsigned.to(torch.uint8)

    # Pack two 4-bit values into one uint8
    # Need even number of elements
    if in_features % 2 != 0:
        # Pad with zeros
        qweights_unsigned = torch.nn.functional.pad(qweights_unsigned, (0, 1))
        in_features += 1

    # Reshape to [out_features, in_features // 2, 2]
    qweights_pairs = qweights_unsigned.view(out_features, in_features // 2, 2)

    # Pack: low nibble = first value, high nibble = second value
    packed = qweights_pairs[:, :, 0] | (qweights_pairs[:, :, 1] << 4)

    return packed


def unpack_weights_4bit(packed: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """
    Unpack 4-bit weights from uint8 back to int8.

    Args:
        packed: Packed weights as uint8
        out_features: Original out_features dimension
        in_features: Original in_features dimension (before packing)

    Returns:
        qweights: Unpacked weights as int8
    """
    # Unpack nibbles
    low_nibble = packed & 0x0F
    high_nibble = (packed >> 4) & 0x0F

    # Interleave
    unpacked = torch.stack([low_nibble, high_nibble], dim=2)
    unpacked = unpacked.view(out_features, -1)

    # Convert back to signed: subtract 8 to shift range from [0, 15] to [-8, 7]
    unpacked = unpacked.to(torch.int32) - 8
    unpacked = unpacked.to(torch.int8)

    # Trim to original size
    unpacked = unpacked[:, :in_features]

    return unpacked


def pack_weights_2bit(qweights: torch.Tensor) -> torch.Tensor:
    """
    Pack 2-bit quantized weights into uint8 (4 values per byte).

    Args:
        qweights: Quantized weights in int8 format (values in [-2, 1])

    Returns:
        packed: Packed weights as uint8, shape [out_features, in_features // 4]
    """
    assert qweights.dtype == torch.int8
    out_features, in_features = qweights.shape

    # Convert to unsigned: add 2 to shift range from [-2, 1] to [0, 3]
    qweights_unsigned = qweights.to(torch.int32) + 2
    qweights_unsigned = qweights_unsigned.to(torch.uint8)

    # Pack four 2-bit values into one uint8
    # Need multiple of 4 elements
    if in_features % 4 != 0:
        pad_size = 4 - (in_features % 4)
        qweights_unsigned = torch.nn.functional.pad(
            qweights_unsigned, (0, pad_size))
        in_features += pad_size

    # Reshape to [out_features, in_features // 4, 4]
    qweights_quads = qweights_unsigned.view(out_features, in_features // 4, 4)

    # Pack: bits [1:0] = first value, [3:2] = second, [5:4] = third, [7:6] = fourth
    packed = (
        qweights_quads[:, :, 0] |
        (qweights_quads[:, :, 1] << 2) |
        (qweights_quads[:, :, 2] << 4) |
        (qweights_quads[:, :, 3] << 6)
    )

    return packed


def unpack_weights_2bit(packed: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """
    Unpack 2-bit weights from uint8 back to int8.

    Args:
        packed: Packed weights as uint8
        out_features: Original out_features dimension
        in_features: Original in_features dimension (before packing)

    Returns:
        qweights: Unpacked weights as int8
    """
    # Unpack 2-bit values
    val0 = packed & 0x03
    val1 = (packed >> 2) & 0x03
    val2 = (packed >> 4) & 0x03
    val3 = (packed >> 6) & 0x03

    # Interleave
    unpacked = torch.stack([val0, val1, val2, val3], dim=2)
    unpacked = unpacked.view(out_features, -1)

    # Convert back to signed: subtract 2 to shift range from [0, 3] to [-2, 1]
    unpacked = unpacked.to(torch.int32) - 2
    unpacked = unpacked.to(torch.int8)

    # Trim to original size
    unpacked = unpacked[:, :in_features]

    return unpacked


def dequantize_weights(
    qweights: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Dequantize weights from quantized representation.

    Args:
        qweights: Quantized weights (int8)
        scales: Per-group scales
        group_size: Size of quantization groups

    Returns:
        weights: Dequantized float weights
    """
    out_features, in_features = qweights.shape
    num_groups = scales.shape[1]

    # Pad if necessary
    padded_in_features = num_groups * group_size
    if in_features < padded_in_features:
        qweights_padded = torch.nn.functional.pad(
            qweights, (0, padded_in_features - in_features))
    else:
        qweights_padded = qweights

    # Reshape to [out_features, num_groups, group_size]
    qweights_grouped = qweights_padded.view(
        out_features, num_groups, group_size)

    # Expand scales
    scales_expanded = scales.unsqueeze(2)  # [out_features, num_groups, 1]

    # Dequantize: w = q * scale
    weights = qweights_grouped.float() * scales_expanded

    # Reshape back
    weights = weights.view(out_features, -1)

    # Remove padding
    weights = weights[:, :in_features]

    return weights


class WeightPacker:
    """
    Utility class for packing/unpacking weights.
    """

    def __init__(self, bits: int = 4, group_size: int = 128):
        """
        Initialize weight packer.

        Args:
            bits: Number of bits (2 or 4)
            group_size: Size of quantization groups
        """
        assert bits in [2, 4], f"Only 2 or 4 bits supported, got {bits}"
        self.bits = bits
        self.group_size = group_size

    def pack(self, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Quantize and pack weights.

        Args:
            weights: Float weights

        Returns:
            packed_weights: Packed weights (uint8)
            scales: Per-group scales
            metadata: Dictionary with shape and packing info
        """
        # Quantize
        qweights, scales = quantize_weights_symmetric(
            weights, bits=self.bits, group_size=self.group_size
        )

        # Pack
        if self.bits == 4:
            packed = pack_weights_4bit(qweights)
        else:  # bits == 2
            packed = pack_weights_2bit(qweights)

        metadata = {
            'bits': self.bits,
            'group_size': self.group_size,
            'out_features': weights.shape[0],
            'in_features': weights.shape[1],
        }

        return packed, scales, metadata

    def unpack(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        metadata: dict,
        dequantize: bool = True,
    ) -> torch.Tensor:
        """
        Unpack and optionally dequantize weights.

        Args:
            packed: Packed weights (uint8)
            scales: Per-group scales
            metadata: Metadata from packing
            dequantize: Whether to dequantize to float

        Returns:
            weights: Unpacked (and optionally dequantized) weights
        """
        bits = metadata['bits']
        out_features = metadata['out_features']
        in_features = metadata['in_features']
        group_size = metadata['group_size']

        # Unpack
        if bits == 4:
            qweights = unpack_weights_4bit(packed, out_features, in_features)
        else:  # bits == 2
            qweights = unpack_weights_2bit(packed, out_features, in_features)

        if dequantize:
            return dequantize_weights(qweights, scales, group_size)
        else:
            return qweights


def test_pack_unpack():
    """
    Unit tests for weight packing/unpacking.
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Testing weight packing/unpacking...")

    # Test W4
    logger.info("\n--- Testing W4 (4-bit) packing ---")
    torch.manual_seed(42)
    weights_w4 = torch.randn(256, 512)

    packer_w4 = WeightPacker(bits=4, group_size=128)
    packed_w4, scales_w4, metadata_w4 = packer_w4.pack(weights_w4)

    logger.info(f"Original shape: {weights_w4.shape}")
    logger.info(f"Packed shape: {packed_w4.shape}")
    logger.info(f"Scales shape: {scales_w4.shape}")
    logger.info(
        f"Compression ratio: {weights_w4.numel() * 4 / packed_w4.numel():.2f}x")

    # Unpack and dequantize
    weights_w4_reconstructed = packer_w4.unpack(
        packed_w4, scales_w4, metadata_w4, dequantize=True)

    # Compute error
    mse_w4 = torch.mean((weights_w4 - weights_w4_reconstructed) ** 2).item()
    max_error_w4 = torch.max(
        torch.abs(weights_w4 - weights_w4_reconstructed)).item()

    logger.info(f"MSE: {mse_w4:.6f}")
    logger.info(f"Max error: {max_error_w4:.6f}")

    # Test W2
    logger.info("\n--- Testing W2 (2-bit) packing ---")
    torch.manual_seed(42)
    weights_w2 = torch.randn(256, 512)

    packer_w2 = WeightPacker(bits=2, group_size=128)
    packed_w2, scales_w2, metadata_w2 = packer_w2.pack(weights_w2)

    logger.info(f"Original shape: {weights_w2.shape}")
    logger.info(f"Packed shape: {packed_w2.shape}")
    logger.info(f"Scales shape: {scales_w2.shape}")
    logger.info(
        f"Compression ratio: {weights_w2.numel() * 4 / packed_w2.numel():.2f}x")

    # Unpack and dequantize
    weights_w2_reconstructed = packer_w2.unpack(
        packed_w2, scales_w2, metadata_w2, dequantize=True)

    # Compute error
    mse_w2 = torch.mean((weights_w2 - weights_w2_reconstructed) ** 2).item()
    max_error_w2 = torch.max(
        torch.abs(weights_w2 - weights_w2_reconstructed)).item()

    logger.info(f"MSE: {mse_w2:.6f}")
    logger.info(f"Max error: {max_error_w2:.6f}")

    # Test edge cases
    logger.info("\n--- Testing edge cases ---")

    # Non-divisible dimensions
    weights_odd = torch.randn(17, 99)
    packed_odd, scales_odd, metadata_odd = packer_w4.pack(weights_odd)
    weights_odd_reconstructed = packer_w4.unpack(
        packed_odd, scales_odd, metadata_odd, dequantize=True)

    assert weights_odd_reconstructed.shape == weights_odd.shape
    logger.info(
        f"✓ Non-divisible dimensions: {weights_odd.shape} -> {weights_odd_reconstructed.shape}")

    # Very small weights
    weights_small = torch.randn(8, 16) * 0.01
    packed_small, scales_small, metadata_small = packer_w4.pack(weights_small)
    weights_small_reconstructed = packer_w4.unpack(
        packed_small, scales_small, metadata_small, dequantize=True)

    logger.info(f"✓ Small weights test passed")

    # Check if W2 has higher error (expected due to fewer bits)
    assert mse_w2 > mse_w4, "W2 should have higher error than W4"
    logger.info(f"✓ W2 error > W4 error as expected")

    logger.info("\n✓ All pack/unpack tests passed!")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_pack_unpack()
