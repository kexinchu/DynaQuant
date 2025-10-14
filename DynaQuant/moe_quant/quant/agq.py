"""
Affinity-Guided Quantization (AGQ)

Quantization that incorporates token-expert affinity (gating scores) into the quantization loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np
from dataclasses import dataclass


@dataclass
class AGQConfig:
    """Configuration for AGQ quantization"""
    bit_width: int = 2
    group_size: int = 64
    symmetric: bool = True
    use_affinity_weighting: bool = True
    use_error_compensation: bool = True
    iterations: int = 10
    damping: float = 0.01


class AGQuantizer:
    """
    Affinity-Guided Quantization

    Quantizes linear layers with gating affinity weighting:
    L = Σ c_i ||W x_i - W_hat x_i||^2
    H = (X * sqrt(c)) (X * sqrt(c))^T

    where c_i is the gating affinity for token i.
    """

    def __init__(self, config: Optional[AGQConfig] = None):
        self.config = config or AGQConfig()

    def collect_activations_and_affinities(
        self,
        layer: nn.Linear,
        inputs: torch.Tensor,
        affinities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect and prepare activations and affinities

        Args:
            layer: Linear layer to quantize
            inputs: Input activations [batch, seq_len, in_features]
            affinities: Gating affinities [batch, seq_len]

        Returns:
            X: Flattened inputs [N, in_features]
            c: Flattened affinities [N]
        """
        # Flatten batch and sequence dimensions
        if inputs.dim() == 3:
            X = inputs.reshape(-1, inputs.size(-1))  # [N, in_features]
            c = affinities.reshape(-1)  # [N]
        else:
            X = inputs
            c = affinities

        # Normalize affinities to sum to N (preserves scale)
        c = c * (c.numel() / (c.sum() + 1e-8))

        return X, c

    def compute_weighted_hessian(
        self,
        X: torch.Tensor,
        c: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute affinity-weighted Hessian approximation

        H = (X * sqrt(c)) (X * sqrt(c))^T

        Args:
            X: Input activations [N, in_features]
            c: Affinities [N]

        Returns:
            H: Hessian matrix [in_features, in_features]
        """
        # Weight inputs by sqrt of affinity
        sqrt_c = torch.sqrt(c.clamp(min=0)).unsqueeze(-1)  # [N, 1]
        X_weighted = X * sqrt_c  # [N, in_features]

        # Compute Hessian
        H = X_weighted.T @ X_weighted  # [in_features, in_features]

        # Add damping for numerical stability
        H += torch.eye(H.size(0), device=H.device) * self.config.damping

        return H

    def quantize_weight_symmetric(
        self,
        W: torch.Tensor,
        bit_width: int,
        group_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Symmetric per-group quantization

        Args:
            W: Weight tensor [out_features, in_features]
            bit_width: Number of bits (2, 4, 8)
            group_size: Group size for quantization

        Returns:
            W_quant: Quantized weights (dequantized to FP)
            scales: Per-group scales
        """
        out_features, in_features = W.shape
        n_groups = (in_features + group_size - 1) // group_size

        # Compute max abs per group
        # [out, n_groups, group_size]
        W_reshaped = W.reshape(out_features, n_groups, -1)
        scales = W_reshaped.abs().max(
            dim=-1, keepdim=True)[0]  # [out, n_groups, 1]

        # Quantize
        n_levels = 2 ** (bit_width - 1)  # 2-bit: 2 levels (for symmetric)
        W_normalized = W_reshaped / (scales + 1e-8)
        W_int = torch.clamp(torch.round(
            W_normalized * (n_levels - 1)), -n_levels, n_levels - 1)
        W_quant = (W_int / (n_levels - 1)) * scales

        # Reshape back
        W_quant = W_quant.reshape(out_features, in_features)
        scales = scales.squeeze(-1)  # [out, n_groups]

        return W_quant, scales

    def quantize_with_error_compensation(
        self,
        W: torch.Tensor,
        H_inv: torch.Tensor,
        bit_width: int,
        group_size: int
    ) -> torch.Tensor:
        """
        Column-wise quantization with error compensation (GPTQ-style)

        Args:
            W: Weight tensor [out_features, in_features]
            H_inv: Inverse Hessian [in_features, in_features]
            bit_width: Number of bits
            group_size: Group size

        Returns:
            W_quant: Quantized weights with error compensation
        """
        out_features, in_features = W.shape
        W_quant = W.clone()

        # Process column by column (or in blocks for efficiency)
        block_size = min(128, in_features)

        for i in range(0, in_features, block_size):
            end_i = min(i + block_size, in_features)
            block_cols = list(range(i, end_i))

            for col in block_cols:
                # Quantize this column
                w_col = W_quant[:, col]

                # Determine group for this column
                group_idx = col // group_size
                group_start = group_idx * group_size
                group_end = min(group_start + group_size, in_features)
                group_cols = W_quant[:, group_start:group_end]

                # Compute scale for this group
                scale = group_cols.abs().max()

                # Quantize
                n_levels = 2 ** (bit_width - 1)
                w_normalized = w_col / (scale + 1e-8)
                w_int = torch.clamp(torch.round(
                    w_normalized * (n_levels - 1)), -n_levels, n_levels - 1)
                w_quant_col = (w_int / (n_levels - 1)) * scale

                # Compute error
                error = w_col - w_quant_col

                # Compensate error to remaining columns
                if col + 1 < in_features:
                    h_inv_col = H_inv[col, col + 1:]
                    compensation = torch.outer(
                        error, h_inv_col) / (H_inv[col, col] + 1e-8)
                    W_quant[:, col + 1:] -= compensation

                # Update quantized weight
                W_quant[:, col] = w_quant_col

        return W_quant

    def quantize_linear(
        self,
        layer: nn.Linear,
        inputs: torch.Tensor,
        affinities: torch.Tensor,
        bit_width: Optional[int] = None,
        group_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Quantize a linear layer with affinity guidance

        Args:
            layer: Linear layer to quantize
            inputs: Input activations
            affinities: Gating affinities
            bit_width: Override bit width
            group_size: Override group size

        Returns:
            W_quant: Quantized weights
            scales: Quantization scales
            stats: Quantization statistics
        """
        bit_width = bit_width or self.config.bit_width
        group_size = group_size or self.config.group_size

        # Collect and prepare data
        X, c = self.collect_activations_and_affinities(
            layer, inputs, affinities)
        W = layer.weight.data  # [out_features, in_features]

        # Compute weighted Hessian
        H = self.compute_weighted_hessian(X, c)

        # Compute quantization
        if self.config.use_error_compensation:
            # Use error compensation (requires Hessian inverse)
            try:
                H_inv = torch.linalg.inv(H)
                W_quant = self.quantize_with_error_compensation(
                    W, H_inv, bit_width, group_size)
                scales = None  # Scales are implicit in error compensation
            except:
                # Fallback to simple quantization
                W_quant, scales = self.quantize_weight_symmetric(
                    W, bit_width, group_size)
        else:
            W_quant, scales = self.quantize_weight_symmetric(
                W, bit_width, group_size)

        # Compute statistics
        mse = F.mse_loss(W, W_quant).item()

        # Compute affinity-weighted output error
        with torch.no_grad():
            Y_fp = F.linear(X, W, layer.bias)
            Y_quant = F.linear(X, W_quant, layer.bias)
            weighted_mse = (c.unsqueeze(-1) * (Y_fp - Y_quant)
                            ** 2).mean().item()

        stats = {
            "mse": mse,
            "weighted_mse": weighted_mse,
            "bit_width": bit_width,
            "group_size": group_size,
            "affinity_mean": c.mean().item(),
            "affinity_std": c.std().item(),
        }

        return W_quant, scales, stats

    def quantize_expert_layers(
        self,
        expert_layers: Dict[int, nn.Linear],
        inputs_per_expert: Dict[int, torch.Tensor],
        affinities_per_expert: Dict[int, torch.Tensor],
        bit_width: Optional[int] = None
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """
        Quantize multiple expert layers

        Args:
            expert_layers: Dictionary of expert_id -> Linear layer
            inputs_per_expert: Dictionary of expert_id -> inputs
            affinities_per_expert: Dictionary of expert_id -> affinities
            bit_width: Override bit width

        Returns:
            Dictionary of expert_id -> (W_quant, scales, stats)
        """
        results = {}

        for expert_id, layer in expert_layers.items():
            if expert_id not in inputs_per_expert:
                continue

            inputs = inputs_per_expert[expert_id]
            affinities = affinities_per_expert[expert_id]

            W_quant, scales, stats = self.quantize_linear(
                layer, inputs, affinities, bit_width
            )

            stats["expert_id"] = expert_id
            results[expert_id] = (W_quant, scales, stats)

        return results


def create_agq_quantizer(
    bit_width: int = 2,
    group_size: int = 64,
    use_error_compensation: bool = True
) -> AGQuantizer:
    """Convenience function to create AGQ quantizer"""
    config = AGQConfig(
        bit_width=bit_width,
        group_size=group_size,
        use_error_compensation=use_error_compensation
    )
    return AGQuantizer(config)
