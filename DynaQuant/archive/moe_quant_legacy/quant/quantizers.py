"""
W2A2 Quantizer with Activation Distribution Shaping

Implements 2-bit weight and activation quantization with:
- Orthogonal rotation/whitening for activation distribution
- Per-group scaling
- Progressive fallback (A2 -> A3/A4 for problematic channels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import numpy as np


@dataclass
class W2A2Config:
    """Configuration for W2A2 quantization"""
    # Weight quantization
    w_bit: int = 2
    w_group_size: int = 64
    w_symmetric: bool = True

    # Activation quantization
    a_bit: int = 2
    a_group_size: int = 64
    a_symmetric: bool = False

    # Distribution shaping
    use_rotation: bool = True
    use_whitening: bool = True
    rotation_granularity: str = "per_layer"  # "per_layer" or "per_group"

    # Progressive fallback
    enable_fallback: bool = True
    fallback_threshold: float = 0.05  # Top-k flip rate threshold
    fallback_bits: List[int] = None  # [3, 4] for A2->A3->A4

    def __post_init__(self):
        if self.fallback_bits is None:
            self.fallback_bits = [3, 4]


class ActivationShaper:
    """
    Activation distribution shaping via orthogonal transformation

    Applies rotation/whitening to make activations more quantization-friendly,
    then absorbs the inverse transformation into the weight matrix.
    """

    def __init__(self, config: W2A2Config):
        self.config = config
        self.rotation_matrix = None
        self.whitening_scale = None

    def fit(self, X: torch.Tensor) -> None:
        """
        Fit rotation and whitening from calibration data

        Args:
            X: Activations [N, features]
        """
        N, d = X.shape

        # Center data
        X_centered = X - X.mean(dim=0, keepdim=True)

        if self.config.use_whitening:
            # Compute covariance
            cov = (X_centered.T @ X_centered) / N  # [d, d]

            # Eigendecomposition for whitening
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvalues = eigenvalues.clamp(min=1e-8)

            # Whitening transformation
            self.whitening_scale = 1.0 / torch.sqrt(eigenvalues)
            self.rotation_matrix = eigenvectors

        elif self.config.use_rotation:
            # Use Hadamard or random orthogonal matrix
            self.rotation_matrix = self._generate_orthogonal_matrix(
                d, X.device)
            self.whitening_scale = None

    def _generate_orthogonal_matrix(self, d: int, device: torch.device) -> torch.Tensor:
        """Generate orthogonal matrix (random or Hadamard)"""
        # Use random orthogonal matrix via QR decomposition
        A = torch.randn(d, d, device=device)
        Q, _ = torch.linalg.qr(A)
        return Q

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation/whitening to activations

        Args:
            X: Activations [N, features] or [batch, seq, features]

        Returns:
            X_transformed: Shaped activations
        """
        original_shape = X.shape
        if X.dim() > 2:
            X = X.reshape(-1, X.size(-1))

        if self.rotation_matrix is None:
            return X.reshape(original_shape)

        # Apply rotation
        X_rotated = X @ self.rotation_matrix

        # Apply whitening scale
        if self.whitening_scale is not None:
            X_rotated = X_rotated * self.whitening_scale

        return X_rotated.reshape(original_shape)

    def absorb_into_weight(self, W: torch.Tensor) -> torch.Tensor:
        """
        Absorb inverse transformation into weight matrix

        W_new = W @ T^{-1}

        Args:
            W: Original weight [out_features, in_features]

        Returns:
            W_absorbed: Weight with absorbed transformation
        """
        if self.rotation_matrix is None:
            return W

        # Inverse transformation
        if self.whitening_scale is not None:
            # Inverse: scale then rotate back
            inv_scale = 1.0 / self.whitening_scale
            W_absorbed = W * inv_scale.unsqueeze(0)
            W_absorbed = W_absorbed @ self.rotation_matrix.T
        else:
            # Just inverse rotation
            W_absorbed = W @ self.rotation_matrix.T

        return W_absorbed


class W2A2Quantizer:
    """
    W2A2 Quantizer with activation shaping and progressive fallback
    """

    def __init__(self, config: Optional[W2A2Config] = None):
        self.config = config or W2A2Config()
        self.shaper = ActivationShaper(config)
        self.channel_bit_map = {}  # Maps channel_id -> bit_width for fallback

    def quantize_weight(
        self,
        W: torch.Tensor,
        bit_width: int = 2,
        group_size: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize weight matrix per-group

        Args:
            W: Weight [out_features, in_features]
            bit_width: Bit width
            group_size: Group size for quantization

        Returns:
            W_quant: Quantized weight (dequantized to FP)
            scales: Per-group scales [out_features, n_groups]
        """
        out_features, in_features = W.shape
        n_groups = (in_features + group_size - 1) // group_size

        # Pad if necessary
        padding = n_groups * group_size - in_features
        if padding > 0:
            W_padded = F.pad(W, (0, padding))
        else:
            W_padded = W

        # Reshape to groups
        W_grouped = W_padded.reshape(out_features, n_groups, group_size)

        # Compute scales per group
        scales = W_grouped.abs().max(
            dim=-1, keepdim=True)[0]  # [out, n_groups, 1]

        # Quantize
        n_levels = 2 ** (bit_width -
                         1) if self.config.w_symmetric else 2 ** bit_width
        W_normalized = W_grouped / (scales + 1e-8)

        if self.config.w_symmetric:
            W_int = torch.clamp(
                torch.round(W_normalized * (n_levels - 1)),
                -n_levels, n_levels - 1
            )
            W_quant = (W_int / (n_levels - 1)) * scales
        else:
            W_int = torch.clamp(
                torch.round((W_normalized + 1) * (n_levels - 1) / 2),
                0, n_levels - 1
            )
            W_quant = (W_int * 2 / (n_levels - 1) - 1) * scales

        # Reshape back and remove padding
        W_quant = W_quant.reshape(out_features, n_groups * group_size)
        if padding > 0:
            W_quant = W_quant[:, :-padding]

        scales = scales.squeeze(-1)  # [out, n_groups]

        return W_quant, scales

    def quantize_activation(
        self,
        X: torch.Tensor,
        bit_width: int = 2,
        group_size: int = 64,
        channel_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize activation with per-group scaling

        Args:
            X: Activation [batch, seq_len, features] or [N, features]
            bit_width: Bit width
            group_size: Group size
            channel_id: Optional channel ID for tracking fallback

        Returns:
            X_quant: Quantized activation (dequantized)
            scales: Per-group scales
        """
        original_shape = X.shape
        if X.dim() > 2:
            X_flat = X.reshape(-1, X.size(-1))
        else:
            X_flat = X

        N, features = X_flat.shape
        n_groups = (features + group_size - 1) // group_size

        # Pad if necessary
        padding = n_groups * group_size - features
        if padding > 0:
            X_padded = F.pad(X_flat, (0, padding))
        else:
            X_padded = X_flat

        # Reshape to groups
        X_grouped = X_padded.reshape(N, n_groups, group_size)

        # Compute scales per token per group
        if self.config.a_symmetric:
            scales = X_grouped.abs().max(
                dim=-1, keepdim=True)[0]  # [N, n_groups, 1]
        else:
            x_min = X_grouped.min(dim=-1, keepdim=True)[0]
            x_max = X_grouped.max(dim=-1, keepdim=True)[0]
            scales = (x_max - x_min) / 2

        # Quantize
        n_levels = 2 ** (bit_width -
                         1) if self.config.a_symmetric else 2 ** bit_width

        if self.config.a_symmetric:
            X_normalized = X_grouped / (scales + 1e-8)
            X_int = torch.clamp(
                torch.round(X_normalized * (n_levels - 1)),
                -n_levels, n_levels - 1
            )
            X_quant = (X_int / (n_levels - 1)) * scales
        else:
            X_normalized = X_grouped / (scales + 1e-8)
            X_int = torch.clamp(
                torch.round((X_normalized + 1) * (n_levels - 1) / 2),
                0, n_levels - 1
            )
            X_quant = (X_int * 2 / (n_levels - 1) - 1) * scales

        # Reshape back
        X_quant = X_quant.reshape(N, n_groups * group_size)
        if padding > 0:
            X_quant = X_quant[:, :-padding]

        X_quant = X_quant.reshape(original_shape)

        return X_quant, scales

    def check_and_fallback(
        self,
        topk_flip_rate: float,
        layer_id: int
    ) -> int:
        """
        Check if fallback is needed based on top-k flip rate

        Args:
            topk_flip_rate: Measured top-k inconsistency rate
            layer_id: Layer identifier

        Returns:
            bit_width: New bit width (2, 3, or 4)
        """
        if not self.config.enable_fallback:
            return self.config.a_bit

        current_bit = self.channel_bit_map.get(layer_id, self.config.a_bit)

        if topk_flip_rate > self.config.fallback_threshold:
            # Find next higher bit width
            fallback_options = [self.config.a_bit] + self.config.fallback_bits
            current_idx = fallback_options.index(current_bit)
            if current_idx < len(fallback_options) - 1:
                new_bit = fallback_options[current_idx + 1]
                self.channel_bit_map[layer_id] = new_bit
                return new_bit

        return current_bit

    def quantize_linear_layer(
        self,
        layer: nn.Linear,
        X_calib: torch.Tensor,
        layer_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Quantize entire linear layer (weight + activation setup)

        Args:
            layer: Linear layer
            X_calib: Calibration activations
            layer_id: Layer identifier for fallback tracking

        Returns:
            W_quant: Quantized weight
            W_absorbed: Weight with absorbed activation transformation
            stats: Quantization statistics
        """
        # Step 1: Fit activation shaper
        if X_calib.dim() > 2:
            X_flat = X_calib.reshape(-1, X_calib.size(-1))
        else:
            X_flat = X_calib

        self.shaper.fit(X_flat)

        # Step 2: Transform activations
        X_shaped = self.shaper.transform(X_calib)

        # Step 3: Quantize weight
        W = layer.weight.data
        W_quant, w_scales = self.quantize_weight(
            W, self.config.w_bit, self.config.w_group_size)

        # Step 4: Absorb inverse transformation into weight
        W_absorbed = self.shaper.absorb_into_weight(W_quant)

        # Step 5: Test activation quantization
        X_quant, a_scales = self.quantize_activation(
            X_shaped,
            self.config.a_bit,
            self.config.a_group_size,
            channel_id=layer_id
        )

        # Compute statistics
        with torch.no_grad():
            Y_fp = F.linear(X_calib, W, layer.bias)
            Y_quant = F.linear(X_quant, W_absorbed, layer.bias)
            mse = F.mse_loss(Y_fp, Y_quant).item()

            # Compute relative error
            rel_error = ((Y_fp - Y_quant).abs() /
                         (Y_fp.abs() + 1e-8)).mean().item()

        stats = {
            "w_bit": self.config.w_bit,
            "a_bit": self.config.a_bit,
            "mse": mse,
            "relative_error": rel_error,
            "use_rotation": self.config.use_rotation,
            "use_whitening": self.config.use_whitening,
        }

        return W_quant, W_absorbed, stats


class QuantizedLinearW2A2(nn.Module):
    """
    Quantized linear layer with W2A2

    Performs activation shaping -> A2 quantization -> matmul with W2 weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[W2A2Config] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or W2A2Config()

        # Quantized weights (stored as INT2 packed or FP16 dequantized)
        self.register_buffer(
            "weight_quant", torch.zeros(out_features, in_features))
        self.register_buffer("weight_scales", torch.zeros(out_features, 1))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None

        # Activation shaper
        self.shaper = ActivationShaper(config)

        # Quantizer
        self.quantizer = W2A2Quantizer(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with W2A2 quantization

        Args:
            x: Input [batch, seq_len, in_features]

        Returns:
            out: Output [batch, seq_len, out_features]
        """
        # Shape activations
        x_shaped = self.shaper.transform(x)

        # Quantize activations
        x_quant, _ = self.quantizer.quantize_activation(
            x_shaped,
            bit_width=self.config.a_bit,
            group_size=self.config.a_group_size
        )

        # Matrix multiply with quantized weights
        out = F.linear(x_quant, self.weight_quant, self.bias)

        return out

    @staticmethod
    def from_float(
        layer: nn.Linear,
        X_calib: torch.Tensor,
        config: Optional[W2A2Config] = None
    ):
        """
        Create quantized layer from float layer

        Args:
            layer: Float linear layer
            X_calib: Calibration data
            config: W2A2 config

        Returns:
            QuantizedLinearW2A2 instance
        """
        config = config or W2A2Config()
        quantizer = W2A2Quantizer(config)

        # Quantize
        W_quant, W_absorbed, stats = quantizer.quantize_linear_layer(
            layer, X_calib
        )

        # Create quantized layer
        q_layer = QuantizedLinearW2A2(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            config=config
        )

        # Set weights
        q_layer.weight_quant.copy_(W_absorbed)
        if layer.bias is not None:
            q_layer.bias.copy_(layer.bias.data)

        # Copy shaper
        q_layer.shaper = quantizer.shaper

        return q_layer


def create_w2a2_quantizer(
    use_rotation: bool = True,
    use_whitening: bool = True,
    enable_fallback: bool = True,
    w_group_size: int = 64,
    a_group_size: int = 64
) -> W2A2Quantizer:
    """Convenience function to create W2A2 quantizer"""
    config = W2A2Config(
        use_rotation=use_rotation,
        use_whitening=use_whitening,
        enable_fallback=enable_fallback,
        w_group_size=w_group_size,
        a_group_size=a_group_size
    )
    return W2A2Quantizer(config)
