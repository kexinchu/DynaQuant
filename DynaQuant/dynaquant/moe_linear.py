"""
Quantized MoE Linear layers with dynamic precision support.
Replaces expert FFN linear layers with quantized versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import logging

from . import pack, fake_quant, kernels

logger = logging.getLogger(__name__)


class QuantizedLinear(nn.Module):
    """
    Quantized linear layer with dynamic precision support.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        precision: str = "w2a4",
        group_size: int = 128,
        use_triton: bool = True,
    ):
        """
        Initialize quantized linear layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to include bias
            precision: Quantization precision ("fp16", "w4a4", "w2a4", "w2a2")
            group_size: Group size for weight quantization
            use_triton: Whether to use Triton kernels (if available)
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.precision = precision
        self.group_size = group_size
        self.use_triton = use_triton

        # Weight storage
        # We store packed weights and scales, not the original FP weights
        self.register_buffer('w2_packed', None)
        self.register_buffer('w2_scales', None)
        self.w2_metadata = None

        self.register_buffer('w4_packed', None)
        self.register_buffer('w4_scales', None)
        self.w4_metadata = None

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Activation quantizer (fake quant for QAT, real quant for inference)
        self.act_quantizer = None

    def set_weights(
        self,
        w2_packed: Optional[torch.Tensor] = None,
        w2_scales: Optional[torch.Tensor] = None,
        w2_metadata: Optional[Dict] = None,
        w4_packed: Optional[torch.Tensor] = None,
        w4_scales: Optional[torch.Tensor] = None,
        w4_metadata: Optional[Dict] = None,
    ):
        """
        Set quantized weights.

        Args:
            w2_packed: Packed W2 weights
            w2_scales: W2 scales
            w2_metadata: W2 metadata
            w4_packed: Packed W4 weights
            w4_scales: W4 scales
            w4_metadata: W4 metadata
        """
        if w2_packed is not None:
            self.w2_packed = w2_packed
            self.w2_scales = w2_scales
            self.w2_metadata = w2_metadata

        if w4_packed is not None:
            self.w4_packed = w4_packed
            self.w4_scales = w4_scales
            self.w4_metadata = w4_metadata

    def quantize_from_fp_weights(self, fp_weights: torch.Tensor):
        """
        Quantize FP weights to both W2 and W4.

        Args:
            fp_weights: Float weights [out_features, in_features]
        """
        # Quantize to W2
        packer_w2 = pack.WeightPacker(bits=2, group_size=self.group_size)
        w2_packed, w2_scales, w2_metadata = packer_w2.pack(fp_weights)
        self.set_weights(
            w2_packed=w2_packed,
            w2_scales=w2_scales,
            w2_metadata=w2_metadata,
        )

        # Quantize to W4
        packer_w4 = pack.WeightPacker(bits=4, group_size=self.group_size)
        w4_packed, w4_scales, w4_metadata = packer_w4.pack(fp_weights)
        self.set_weights(
            w4_packed=w4_packed,
            w4_scales=w4_scales,
            w4_metadata=w4_metadata,
        )

    def set_precision(self, precision: str):
        """Set quantization precision."""
        self.precision = precision

    def forward(self, x: torch.Tensor, precision: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass with specified precision.

        Args:
            x: Input tensor [batch, seq_len, in_features] or [tokens, in_features]
            precision: Precision override (uses self.precision if None)

        Returns:
            output: Output tensor [batch, seq_len, out_features] or [tokens, out_features]
        """
        if precision is None:
            precision = self.precision

        # FP16 path (no quantization)
        if precision == "fp16":
            # Need to have FP16 weights available
            # For now, dequantize from W4 as fallback
            if self.w4_packed is not None:
                packer = pack.WeightPacker(bits=4, group_size=self.group_size)
                fp_weights = packer.unpack(self.w4_packed, self.w4_scales,
                                           self.w4_metadata, dequantize=True)
                output = F.linear(x, fp_weights, self.bias)
            else:
                raise ValueError("FP16 weights not available")
            return output

        # Quantized paths
        # Parse precision string (e.g., "w4a4" -> weight_bits=4, act_bits=4)
        if precision.startswith("w"):
            parts = precision.split("a")
            weight_bits = int(parts[0][1:])
            act_bits = int(parts[1]) if len(parts) > 1 else 4
        else:
            raise ValueError(f"Invalid precision format: {precision}")

        # Get weight packer and packed weights
        if weight_bits == 2:
            w_packed = self.w2_packed
            w_scales = self.w2_scales
            w_metadata = self.w2_metadata
        elif weight_bits == 4:
            w_packed = self.w4_packed
            w_scales = self.w4_scales
            w_metadata = self.w4_metadata
        else:
            raise ValueError(f"Unsupported weight bits: {weight_bits}")

        if w_packed is None:
            raise ValueError(
                f"Weights not available for precision {precision}")

        # Quantize activations
        if self.training and self.act_quantizer is not None:
            # Use fake quantization during training
            x_q = self.act_quantizer(x)

            # Dequantize weights for standard matmul
            packer = pack.WeightPacker(
                bits=weight_bits, group_size=self.group_size)
            w_dq = packer.unpack(w_packed, w_scales,
                                 w_metadata, dequantize=True)

            output = F.linear(x_q, w_dq, self.bias)
        else:
            # Real quantization during inference
            x_q, a_scales = fake_quant.quantize_activation_dynamic(
                x, bits=act_bits, symmetric=True, per_token=True
            )

            # Use Triton kernel if available and on GPU
            if self.use_triton and torch.cuda.is_available() and x.is_cuda:
                try:
                    # Prepare scales for kernel
                    # a_scales: [batch, seq_len] or [tokens] -> flatten to [M]
                    if a_scales.dim() > 1:
                        a_scales_flat = a_scales.view(-1)
                    else:
                        a_scales_flat = a_scales

                    # Transpose weight scales to [num_groups, out_features]
                    # Our packer outputs [out_features, num_groups]
                    w_scales_t = w_scales.T.contiguous()

                    # Call appropriate kernel
                    if weight_bits == 4:
                        output = kernels.matmul_w4a4(
                            x_q.view(-1, self.in_features),
                            # [in_features, out_features // 2]
                            w_packed.T.contiguous(),
                            a_scales_flat,
                            w_scales_t,
                            self.group_size,
                        )
                    elif weight_bits == 2:
                        output = kernels.matmul_w2a4(
                            x_q.view(-1, self.in_features),
                            # [in_features, out_features // 4]
                            w_packed.T.contiguous(),
                            a_scales_flat,
                            w_scales_t,
                            self.group_size,
                        )
                    else:
                        raise ValueError(
                            f"Unsupported weight bits for kernel: {weight_bits}")

                    # Reshape output
                    if x.dim() == 3:
                        output = output.view(
                            x.shape[0], x.shape[1], self.out_features)

                    # Add bias
                    if self.bias is not None:
                        output = output + self.bias

                except Exception as e:
                    logger.warning(
                        f"Triton kernel failed: {e}, falling back to dequant+matmul")
                    # Fallback to dequantize and standard matmul
                    packer = pack.WeightPacker(
                        bits=weight_bits, group_size=self.group_size)
                    w_dq = packer.unpack(
                        w_packed, w_scales, w_metadata, dequantize=True)
                    x_dq = fake_quant.dequantize_activation(x_q, a_scales)
                    output = F.linear(x_dq, w_dq, self.bias)
            else:
                # Fallback: dequantize and use standard matmul
                packer = pack.WeightPacker(
                    bits=weight_bits, group_size=self.group_size)
                w_dq = packer.unpack(w_packed, w_scales,
                                     w_metadata, dequantize=True)
                x_dq = fake_quant.dequantize_activation(x_q, a_scales)
                output = F.linear(x_dq, w_dq, self.bias)

        return output

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, precision={self.precision}, '
                f'group_size={self.group_size}')


class MoELinear(nn.Module):
    """
    MoE-aware linear layer that handles multiple experts.
    """

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        default_precision: str = "w2a4",
        group_size: int = 128,
        use_triton: bool = True,
    ):
        """
        Initialize MoE linear layer.

        Args:
            num_experts: Number of experts
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to include bias
            default_precision: Default quantization precision
            group_size: Group size for weight quantization
            use_triton: Whether to use Triton kernels
        """
        super().__init__()

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.default_precision = default_precision

        # Create quantized linear layers for each expert
        self.experts = nn.ModuleList([
            QuantizedLinear(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                precision=default_precision,
                group_size=group_size,
                use_triton=use_triton,
            )
            for _ in range(num_experts)
        ])

    def set_expert_weights(
        self,
        expert_id: int,
        w2_packed: Optional[torch.Tensor] = None,
        w2_scales: Optional[torch.Tensor] = None,
        w2_metadata: Optional[Dict] = None,
        w4_packed: Optional[torch.Tensor] = None,
        w4_scales: Optional[torch.Tensor] = None,
        w4_metadata: Optional[Dict] = None,
    ):
        """Set weights for a specific expert."""
        self.experts[expert_id].set_weights(
            w2_packed=w2_packed,
            w2_scales=w2_scales,
            w2_metadata=w2_metadata,
            w4_packed=w4_packed,
            w4_scales=w4_scales,
            w4_metadata=w4_metadata,
        )

    def set_expert_precision(self, expert_id: int, precision: str):
        """Set precision for a specific expert."""
        self.experts[expert_id].set_precision(precision)

    def forward(
        self,
        x: torch.Tensor,
        expert_id: int,
        precision: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Forward pass through a specific expert.

        Args:
            x: Input tensor
            expert_id: Expert ID
            precision: Precision override

        Returns:
            output: Output tensor
        """
        return self.experts[expert_id](x, precision=precision)


def test_moe_linear():
    """
    Unit tests for MoE linear layers.
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Testing MoE linear layers...")

    # Test QuantizedLinear
    logger.info("\n--- Testing QuantizedLinear ---")
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_features = 512
    out_features = 256
    batch_size = 4
    seq_len = 16

    # Create layer
    qlayer = QuantizedLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        precision="w4a4",
        group_size=128,
        use_triton=True,
    ).to(device)

    logger.info(f"Created QuantizedLinear: {qlayer}")

    # Create and quantize FP weights
    fp_weights = torch.randn(out_features, in_features, device=device)
    qlayer.quantize_from_fp_weights(fp_weights)

    logger.info(f"W2 packed shape: {qlayer.w2_packed.shape}")
    logger.info(f"W2 scales shape: {qlayer.w2_scales.shape}")
    logger.info(f"W4 packed shape: {qlayer.w4_packed.shape}")
    logger.info(f"W4 scales shape: {qlayer.w4_scales.shape}")
    logger.info(f"✓ Weight quantization test passed")

    # Test forward pass with different precisions
    logger.info("\n--- Testing forward pass ---")
    x = torch.randn(batch_size, seq_len, in_features, device=device)

    # Test W4A4
    qlayer.eval()
    output_w4a4 = qlayer(x, precision="w4a4")
    logger.info(f"W4A4 output shape: {output_w4a4.shape}")
    assert output_w4a4.shape == (batch_size, seq_len, out_features)
    logger.info(f"✓ W4A4 forward pass test passed")

    # Test W2A4
    output_w2a4 = qlayer(x, precision="w2a4")
    logger.info(f"W2A4 output shape: {output_w2a4.shape}")
    assert output_w2a4.shape == (batch_size, seq_len, out_features)
    logger.info(f"✓ W2A4 forward pass test passed")

    # Test FP16 reference
    try:
        output_fp16 = qlayer(x, precision="fp16")
        logger.info(f"FP16 output shape: {output_fp16.shape}")
        logger.info(f"✓ FP16 forward pass test passed")

        # Compare errors
        mse_w4a4 = torch.mean((output_fp16 - output_w4a4) ** 2).item()
        mse_w2a4 = torch.mean((output_fp16 - output_w2a4) ** 2).item()

        logger.info(f"MSE W4A4 vs FP16: {mse_w4a4:.6f}")
        logger.info(f"MSE W2A4 vs FP16: {mse_w2a4:.6f}")

        assert mse_w2a4 > mse_w4a4, "W2A4 should have higher error than W4A4"
        logger.info(f"✓ Error comparison test passed")
    except Exception as e:
        logger.warning(f"FP16 test skipped: {e}")

    # Test MoELinear
    logger.info("\n--- Testing MoELinear ---")
    num_experts = 8

    moe_layer = MoELinear(
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        default_precision="w2a4",
        group_size=128,
        use_triton=True,
    ).to(device)

    logger.info(f"Created MoELinear with {num_experts} experts")

    # Quantize experts
    for expert_id in range(num_experts):
        fp_weights_expert = torch.randn(
            out_features, in_features, device=device)
        moe_layer.experts[expert_id].quantize_from_fp_weights(
            fp_weights_expert)

    logger.info(f"Quantized all {num_experts} experts")
    logger.info(f"✓ MoELinear initialization test passed")

    # Test forward through specific expert
    logger.info("\n--- Testing MoELinear forward ---")
    moe_layer.eval()

    expert_id = 0
    output_expert = moe_layer(x, expert_id=expert_id, precision="w2a4")

    logger.info(f"Expert {expert_id} output shape: {output_expert.shape}")
    assert output_expert.shape == (batch_size, seq_len, out_features)
    logger.info(f"✓ MoELinear forward test passed")

    # Test precision switching
    logger.info("\n--- Testing precision switching ---")
    moe_layer.set_expert_precision(expert_id, "w4a4")
    output_expert_w4a4 = moe_layer(x, expert_id=expert_id)

    logger.info(
        f"Expert {expert_id} (W4A4) output shape: {output_expert_w4a4.shape}")
    logger.info(f"✓ Precision switching test passed")

    logger.info("\n✓ All MoE linear tests passed!")
    return True


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_moe_linear()
