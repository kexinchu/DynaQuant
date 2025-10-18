"""
W2AWQLinear Runtime Module
===========================
Quantized linear layer for inference with 2-bit weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .pack import unpack_2bit
from .quantize import dequantize_weight


class W2AWQLinear(nn.Module):
    """
    2-bit AWQ quantized linear layer for inference.

    This module stores weights in packed 2-bit format and unpacks/dequantizes
    them during forward pass.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to include bias
        group_size: Group size for quantization
        device: Device to place tensors
        dtype: Data type for activations (fp16/bf16)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype or torch.float16

        assert in_features % 4 == 0, "in_features must be divisible by 4 for packing"
        assert in_features % group_size == 0, "in_features must be divisible by group_size"

        # Register packed weights (uint8)
        self.register_buffer(
            'weight_packed',
            torch.zeros(out_features, in_features // 4,
                        dtype=torch.uint8, device=self.device)
        )

        # Register scales (fp16)
        num_groups = in_features // group_size
        self.register_buffer(
            'scale',
            torch.ones(out_features, num_groups,
                       dtype=torch.float16, device=self.device)
        )

        # Bias (optional)
        if bias:
            self.register_buffer(
                'bias',
                torch.zeros(out_features, dtype=self.dtype, device=self.device)
            )
        else:
            self.bias = None

    def load_weights(
        self,
        weight_q: torch.Tensor,
        scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        packed: bool = False
    ):
        """
        Load quantized weights into the module.

        Args:
            weight_q: Quantized weights (packed uint8 or unpacked int8)
            scale: Quantization scales
            bias: Optional bias
            packed: Whether weight_q is already packed
        """
        if packed:
            assert weight_q.dtype == torch.uint8
            assert weight_q.shape == (self.out_features, self.in_features // 4)
            self.weight_packed.copy_(weight_q)
        else:
            # Pack the weights
            from .pack import pack_2bit
            assert weight_q.shape == (self.out_features, self.in_features)
            weight_packed = pack_2bit(weight_q)
            self.weight_packed.copy_(weight_packed)

        self.scale.copy_(scale)

        if bias is not None and self.bias is not None:
            self.bias.copy_(bias.to(self.dtype))

    def unpack_and_dequantize(self) -> torch.Tensor:
        """
        Unpack and dequantize weights.

        Returns:
            Dequantized weights in self.dtype
        """
        # Unpack
        weight_q = unpack_2bit(
            self.weight_packed, self.out_features, self.in_features)

        # Dequantize
        weight_deq = dequantize_weight(
            weight_q, self.scale, self.group_size, self.dtype)

        return weight_deq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with on-the-fly unpacking and dequantization.

        Args:
            x: Input tensor, shape [..., in_features]

        Returns:
            Output tensor, shape [..., out_features]
        """
        # Ensure input is in correct dtype
        x = x.to(self.dtype)

        # Unpack and dequantize weights
        weight = self.unpack_and_dequantize()

        # Linear transformation
        output = F.linear(x, weight, self.bias)

        return output

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias is not None}, '
            f'group_size={self.group_size}, '
            f'bits=2'
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_q: torch.Tensor,
        scale: torch.Tensor,
        group_size: int = 128,
    ):
        """
        Create W2AWQLinear from a standard nn.Linear and quantized weights.

        Args:
            linear: Original nn.Linear layer
            weight_q: Quantized weights (int8, unpacked)
            scale: Quantization scales
            group_size: Group size

        Returns:
            W2AWQLinear module
        """
        has_bias = linear.bias is not None

        quant_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            group_size=group_size,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        # Load weights
        quant_linear.load_weights(
            weight_q=weight_q,
            scale=scale,
            bias=linear.bias.data if has_bias else None,
            packed=False
        )

        return quant_linear


class W2AWQLinearFused(W2AWQLinear):
    """
    Optimized version with cached dequantized weights.

    Trades memory for speed by caching dequantized weights.
    Useful for inference when memory is not a bottleneck.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with cached weights."""
        x = x.to(self.dtype)

        # Use cached weight if available
        if self._weight_cache is None:
            self._weight_cache = self.unpack_and_dequantize()

        output = F.linear(x, self._weight_cache, self.bias)
        return output

    def reset_cache(self):
        """Clear weight cache."""
        self._weight_cache = None


def replace_linear_with_w2awq(
    module: nn.Module,
    quantized_weights: dict,
    group_size: int = 128,
    ignore_modules: Optional[list] = None,
) -> nn.Module:
    """
    Replace all nn.Linear layers in a module with W2AWQLinear.

    Args:
        module: The module to modify
        quantized_weights: Dict mapping layer names to (weight_q, scale) tuples
        group_size: Group size for quantization
        ignore_modules: List of module names to skip

    Returns:
        Modified module
    """
    if ignore_modules is None:
        ignore_modules = ['lm_head']

    for name, child in list(module.named_children()):
        # Check if should ignore
        if name in ignore_modules:
            continue

        # Replace if it's a Linear layer and we have weights for it
        if isinstance(child, nn.Linear) and name in quantized_weights:
            weight_q, scale = quantized_weights[name]

            # Create quantized layer
            quant_layer = W2AWQLinear.from_linear(
                child, weight_q, scale, group_size)

            # Replace
            setattr(module, name, quant_layer)

            print(f"Replaced {name} with W2AWQLinear")
        else:
            # Recursively process children
            replace_linear_with_w2awq(
                child, quantized_weights, group_size, ignore_modules)

    return module


def test_runtime():
    """Test W2AWQLinear module."""
    print("Testing W2AWQLinear runtime module...")

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create original linear layer
    in_features, out_features = 512, 256
    linear = nn.Linear(in_features, out_features, bias=True).to(device)

    # Quantize it
    from .quantize import symmetric_quantize
    weight_q, scale = symmetric_quantize(
        linear.weight.data, n_bits=2, group_size=128)

    # Create quantized layer
    quant_linear = W2AWQLinear.from_linear(
        linear, weight_q, scale, group_size=128)
    quant_linear = quant_linear.to(device)

    print(f"Original Linear: {linear}")
    print(f"Quantized Linear: {quant_linear}")

    # Test forward pass
    x = torch.randn(4, in_features, dtype=torch.float16, device=device)

    # Original output
    with torch.no_grad():
        out_orig = linear(x.to(linear.weight.dtype))
        out_quant = quant_linear(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {out_quant.shape}")
    print(f"Output dtype: {out_quant.dtype}")

    # Compare outputs
    out_orig = out_orig.to(torch.float16)
    error = (out_orig - out_quant).abs().mean()
    rel_error = error / out_orig.abs().mean()

    print(f"\nAbsolute error: {error:.6f}")
    print(f"Relative error: {rel_error:.6f}")

    # Test batched input
    x_batch = torch.randn(8, 16, in_features,
                          dtype=torch.float16, device=device)
    with torch.no_grad():
        out_batch = quant_linear(x_batch)
    print(f"\nBatched input shape: {x_batch.shape}")
    print(f"Batched output shape: {out_batch.shape}")

    # Test memory usage
    import sys
    orig_size = sum(p.numel() * p.element_size() for p in linear.parameters())
    quant_size = sum(p.numel() * p.element_size()
                     for p in quant_linear.parameters())

    print(f"\nOriginal size: {orig_size / 1024:.2f} KB")
    print(f"Quantized size: {quant_size / 1024:.2f} KB")
    print(f"Compression ratio: {orig_size / quant_size:.2f}x")

    # Test fused version
    print("\n--- Testing Fused Version ---")
    quant_linear_fused = W2AWQLinearFused.from_linear(
        linear, weight_q, scale, group_size=128)
    quant_linear_fused = quant_linear_fused.to(device)

    with torch.no_grad():
        out_fused = quant_linear_fused(x)

    error_fused = (out_orig - out_fused).abs().mean()
    print(f"Fused version error: {error_fused:.6f}")

    print("\n✅ All runtime tests passed!")


if __name__ == "__main__":
    test_runtime()
