"""
2-bit Weight Packing/Unpacking
================================
Packs 4 x 2-bit weights into 1 byte (little-endian).

Format: byte = w0 | (w1 << 2) | (w2 << 4) | (w3 << 6)
where each w_i is in range [-2, 1] (2-bit signed) or [0, 3] (2-bit unsigned)
"""

import torch
import numpy as np


def pack_2bit(weights_int: torch.Tensor) -> torch.Tensor:
    """
    Pack 2-bit quantized weights into uint8 format (4 weights per byte).

    Args:
        weights_int: Quantized weights in int8 format, values in [-2, 1] for symmetric
                    Shape: [out_features, in_features]

    Returns:
        Packed weights in uint8 format
        Shape: [out_features, in_features // 4]

    Note:
        - For symmetric quantization: maps [-2, -1, 0, 1] -> [0, 1, 2, 3]
        - Packs 4 consecutive weights along in_features dimension
        - Little-endian: byte = w0 | (w1<<2) | (w2<<4) | (w3<<6)
    """
    assert weights_int.dtype in [torch.int8, torch.int32, torch.int64], \
        f"Expected integer tensor, got {weights_int.dtype}"

    out_features, in_features = weights_int.shape
    assert in_features % 4 == 0, \
        f"in_features must be divisible by 4, got {in_features}"

    # Convert signed [-2, 1] to unsigned [0, 3]
    weights_uint = (weights_int + 2).to(torch.uint8)

    # Reshape to [out_features, in_features // 4, 4]
    weights_reshaped = weights_uint.reshape(out_features, in_features // 4, 4)

    # Pack: byte = w0 | (w1 << 2) | (w2 << 4) | (w3 << 6)
    packed = (
        weights_reshaped[:, :, 0] |
        (weights_reshaped[:, :, 1] << 2) |
        (weights_reshaped[:, :, 2] << 4) |
        (weights_reshaped[:, :, 3] << 6)
    )

    return packed


def unpack_2bit(packed_weights: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """
    Unpack 2-bit weights from uint8 format.

    Args:
        packed_weights: Packed weights in uint8 format
                       Shape: [out_features, in_features // 4]
        out_features: Number of output features
        in_features: Number of input features (must be divisible by 4)

    Returns:
        Unpacked weights in int8 format, values in [-2, 1]
        Shape: [out_features, in_features]
    """
    assert packed_weights.dtype == torch.uint8, \
        f"Expected uint8 tensor, got {packed_weights.dtype}"
    assert in_features % 4 == 0, \
        f"in_features must be divisible by 4, got {in_features}"

    device = packed_weights.device

    # Extract 4 weights from each byte
    w0 = (packed_weights & 0b00000011)
    w1 = (packed_weights & 0b00001100) >> 2
    w2 = (packed_weights & 0b00110000) >> 4
    w3 = (packed_weights & 0b11000000) >> 6

    # Stack and reshape
    # [out_features, in_features//4, 4]
    unpacked = torch.stack([w0, w1, w2, w3], dim=-1)
    unpacked = unpacked.reshape(out_features, in_features)

    # Convert unsigned [0, 3] back to signed [-2, 1]
    unpacked = unpacked.to(torch.int8) - 2

    return unpacked


def pack_2bit_vectorized(weights_int: torch.Tensor) -> torch.Tensor:
    """
    Vectorized version of pack_2bit for better performance on large tensors.

    Args:
        weights_int: Quantized weights, shape [out_features, in_features]

    Returns:
        Packed weights, shape [out_features, in_features // 4]
    """
    out_features, in_features = weights_int.shape
    assert in_features % 4 == 0

    # Convert to uint8 range [0, 3]
    weights_uint = (weights_int + 2).to(torch.uint8).contiguous()

    # Use view to group by 4
    weights_view = weights_uint.view(out_features, in_features // 4, 4)

    # Pack using bit operations
    packed = torch.zeros(out_features, in_features // 4,
                         dtype=torch.uint8, device=weights_int.device)
    packed = (weights_view[:, :, 0] |
              (weights_view[:, :, 1] << 2) |
              (weights_view[:, :, 2] << 4) |
              (weights_view[:, :, 3] << 6))

    return packed


def unpack_2bit_vectorized(packed_weights: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """
    Vectorized version of unpack_2bit.

    Args:
        packed_weights: Packed weights, shape [out_features, in_features // 4]
        out_features: Number of output features
        in_features: Number of input features

    Returns:
        Unpacked weights, shape [out_features, in_features]
    """
    assert in_features % 4 == 0
    device = packed_weights.device

    # Create output tensor
    unpacked = torch.zeros(out_features, in_features //
                           4, 4, dtype=torch.uint8, device=device)

    # Extract all 4 weights at once
    unpacked[:, :, 0] = packed_weights & 0b11
    unpacked[:, :, 1] = (packed_weights >> 2) & 0b11
    unpacked[:, :, 2] = (packed_weights >> 4) & 0b11
    unpacked[:, :, 3] = (packed_weights >> 6) & 0b11

    # Reshape and convert
    unpacked = unpacked.view(out_features, in_features).to(torch.int8) - 2

    return unpacked


def test_pack_unpack():
    """Test packing/unpacking correctness."""
    print("Testing 2-bit packing/unpacking...")

    # Test 1: Small random tensor
    torch.manual_seed(42)
    weights = torch.randint(-2, 2, (4, 16), dtype=torch.int8)
    print(f"Original shape: {weights.shape}")
    print(f"Original weights (first row):\n{weights[0]}")

    # Pack
    packed = pack_2bit(weights)
    print(f"Packed shape: {packed.shape}")
    print(f"Packed (first row): {packed[0]}")
    print(f"Compression ratio: {weights.numel() / packed.numel()}x")

    # Unpack
    unpacked = unpack_2bit(packed, *weights.shape)
    print(f"Unpacked shape: {unpacked.shape}")
    print(f"Unpacked (first row):\n{unpacked[0]}")

    # Verify
    matches = torch.all(weights == unpacked)
    print(f"Pack/Unpack matches: {matches}")
    assert matches, "Pack/unpack mismatch!"

    # Test 2: Vectorized version
    packed_vec = pack_2bit_vectorized(weights)
    unpacked_vec = unpack_2bit_vectorized(packed_vec, *weights.shape)
    assert torch.all(packed == packed_vec), "Vectorized pack mismatch!"
    assert torch.all(weights == unpacked_vec), "Vectorized unpack mismatch!"
    print("✓ Vectorized versions match!")

    # Test 3: Large tensor performance
    large_weights = torch.randint(-2, 2, (4096, 4096), dtype=torch.int8)
    packed_large = pack_2bit(large_weights)
    unpacked_large = unpack_2bit(packed_large, *large_weights.shape)
    assert torch.all(large_weights == unpacked_large), "Large tensor mismatch!"
    print(f"✓ Large tensor ({large_weights.shape}) test passed!")

    print("\n✅ All packing tests passed!")


if __name__ == "__main__":
    test_pack_unpack()
