"""
Mixed Precision Quantizer for MoE Models

Implements different quantization schemes for different layers:
- Router layers: W8A8 (for ranking consistency)
- Non-expert layers (transformer, layernorm): W8A8
- Expert layers: W2A2 or W4A4 (configurable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from pathlib import Path

from ..losses.router_rank_loss import RouterRankLoss
from .quantizers import W2A2Quantizer, W2A2Config
from .agq import AGQuantizer, AGQConfig


logger = logging.getLogger(__name__)


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision quantization"""
    # Router configuration
    router_bits_w: int = 8
    router_bits_a: int = 8
    router_group_size: int = 128

    # Non-expert layers (transformer, layernorm, etc.)
    non_expert_bits_w: int = 8
    non_expert_bits_a: int = 8
    non_expert_group_size: int = 128

    # Expert layers
    expert_bits_w: int = 2  # or 4
    expert_bits_a: int = 2  # or 4
    expert_group_size: int = 64

    # RouterRank configuration
    router_rank_gamma: float = 0.1
    router_rank_lambda: float = 1.0

    # AGQ configuration for experts
    use_agq_for_experts: bool = True
    agq_error_compensation: bool = True

    # W2A2 configuration for experts
    use_activation_shaping: bool = True
    enable_progressive_fallback: bool = True


class MixedPrecisionQuantizer:
    """
    Mixed precision quantizer for MoE models

    Applies different quantization schemes based on layer type:
    - Router: W8A8 with RouterRank loss
    - Non-expert: W8A8 standard quantization
    - Expert: W2A2/W4A4 with AGQ + W2A2
    """

    def __init__(self, config: Optional[MixedPrecisionConfig] = None):
        self.config = config or MixedPrecisionConfig()

        # Initialize quantizers
        self.router_rank_loss = RouterRankLoss(
            gamma=self.config.router_rank_gamma,
            lambda_rank=self.config.router_rank_lambda
        )

        # Expert quantizers
        if self.config.use_agq_for_experts:
            agq_config = AGQConfig(
                bit_width=self.config.expert_bits_w,
                group_size=self.config.expert_group_size,
                use_error_compensation=self.config.agq_error_compensation
            )
            self.expert_agq = AGQuantizer(agq_config)

        if self.config.expert_bits_w == 2:
            w2a2_config = W2A2Config(
                w_bit=2,
                a_bit=self.config.expert_bits_a,
                use_rotation=self.config.use_activation_shaping,
                enable_fallback=self.config.enable_progressive_fallback
            )
            self.expert_w2a2 = W2A2Quantizer(w2a2_config)

        # Standard quantizers for router and non-expert layers
        self.standard_quantizer = StandardQuantizer()

    def identify_layer_type(self, name: str) -> str:
        """
        Identify layer type from module name

        Args:
            name: Module name

        Returns:
            layer_type: "router", "expert", or "non_expert"
        """
        name_lower = name.lower()

        # Router layers
        if any(keyword in name_lower for keyword in ["gate", "router", "gating"]):
            return "router"

        # Expert layers
        if any(keyword in name_lower for keyword in ["expert", "ffn", "mlp"]):
            return "expert"

        # Non-expert layers (transformer, layernorm, embedding, etc.)
        return "non_expert"

    def quantize_router_layer(
        self,
        layer: nn.Linear,
        calibration_data: Dict[str, torch.Tensor],
        top_k: int = 2
    ) -> Tuple[nn.Linear, Dict]:
        """
        Quantize router layer with RouterRank optimization

        Args:
            layer: Router linear layer
            calibration_data: Dict with 'inputs' and 'targets'
            top_k: Number of top experts

        Returns:
            quantized_layer: Quantized router layer
            stats: Quantization statistics
        """
        inputs = calibration_data["inputs"]  # [batch, seq_len, hidden_dim]
        # Optional targets for consistency
        targets = calibration_data.get("targets")

        # Get FP16 logits
        with torch.no_grad():
            logits_fp = layer(inputs)

        # Standard W8A8 quantization
        quantized_layer, q_stats = self.standard_quantizer.quantize_linear(
            layer,
            self.config.router_bits_w,
            self.config.router_bits_a,
            self.config.router_group_size
        )

        # RouterRank optimization
        if targets is not None:
            with torch.no_grad():
                logits_quant = quantized_layer(inputs)

            # Compute RouterRank loss and consistency
            ranking_loss, loss_dict = self.router_rank_loss.compute_total_loss(
                logits_fp, logits_quant, top_k
            )

            consistency = self.router_rank_loss.compute_topk_consistency(
                logits_fp, logits_quant, top_k
            )

            q_stats.update(loss_dict)
            q_stats.update(consistency)

        q_stats["layer_type"] = "router"
        q_stats["quantization_bits"] = f"W{self.config.router_bits_w}A{self.config.router_bits_a}"

        return quantized_layer, q_stats

    def quantize_non_expert_layer(
        self,
        layer: nn.Linear,
        calibration_data: Dict[str, torch.Tensor]
    ) -> Tuple[nn.Linear, Dict]:
        """
        Quantize non-expert layer (transformer, layernorm, etc.) with W8A8

        Args:
            layer: Non-expert linear layer
            calibration_data: Dict with 'inputs'

        Returns:
            quantized_layer: Quantized layer
            stats: Quantization statistics
        """
        inputs = calibration_data["inputs"]

        # Standard W8A8 quantization
        quantized_layer, q_stats = self.standard_quantizer.quantize_linear(
            layer,
            self.config.non_expert_bits_w,
            self.config.non_expert_bits_a,
            self.config.non_expert_group_size
        )

        q_stats["layer_type"] = "non_expert"
        q_stats["quantization_bits"] = f"W{self.config.non_expert_bits_w}A{self.config.non_expert_bits_a}"

        return quantized_layer, q_stats

    def quantize_expert_layer(
        self,
        layer: nn.Linear,
        calibration_data: Dict[str, torch.Tensor],
        expert_id: Optional[int] = None
    ) -> Tuple[nn.Linear, Dict]:
        """
        Quantize expert layer with W2A2/W4A4 + AGQ

        Args:
            layer: Expert linear layer
            calibration_data: Dict with 'inputs' and 'affinities'
            expert_id: Expert ID for tracking

        Returns:
            quantized_layer: Quantized expert layer
            stats: Quantization statistics
        """
        inputs = calibration_data["inputs"]
        affinities = calibration_data.get("affinities")

        stats = {"expert_id": expert_id}

        # Use AGQ + W2A2 for expert layers
        if self.config.expert_bits_w == 2 and self.config.use_agq_for_experts:
            # AGQ quantization
            if affinities is not None:
                W_agq, scales_agq, agq_stats = self.expert_agq.quantize_linear(
                    layer, inputs, affinities
                )
                stats.update(agq_stats)

            # W2A2 quantization with activation shaping
            W_w2a2, W_absorbed, w2a2_stats = self.expert_w2a2.quantize_linear_layer(
                layer, inputs, layer_id=expert_id
            )
            stats.update(w2a2_stats)

            # Create quantized layer
            from .quantizers import QuantizedLinearW2A2
            quantized_layer = QuantizedLinearW2A2.from_float(
                layer, inputs, self.expert_w2a2.config
            )

        else:
            # Standard quantization for W4A4 experts
            quantized_layer, q_stats = self.standard_quantizer.quantize_linear(
                layer,
                self.config.expert_bits_w,
                self.config.expert_bits_a,
                self.config.expert_group_size
            )
            stats.update(q_stats)

        stats["layer_type"] = "expert"
        stats["quantization_bits"] = f"W{self.config.expert_bits_w}A{self.config.expert_bits_a}"

        return quantized_layer, stats

    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Dict[str, Dict[str, torch.Tensor]],
        layer_mapping: Optional[Dict[str, str]] = None
    ) -> Tuple[nn.Module, Dict]:
        """
        Quantize entire MoE model with mixed precision

        Args:
            model: MoE model to quantize
            calibration_data: Dict mapping layer names to calibration data
            layer_mapping: Optional manual layer type mapping

        Returns:
            quantized_model: Quantized model
            quantization_stats: Statistics for all layers
        """
        quantized_model = model
        all_stats = {}

        # Get all linear layers
        linear_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module

        logger.info(f"Found {len(linear_layers)} linear layers to quantize")

        # Quantize each layer based on type
        for layer_name, layer in linear_layers.items():
            if layer_name not in calibration_data:
                logger.warning(f"No calibration data for layer {layer_name}")
                continue

            # Determine layer type
            if layer_mapping and layer_name in layer_mapping:
                layer_type = layer_mapping[layer_name]
            else:
                layer_type = self.identify_layer_type(layer_name)

            logger.info(f"Quantizing {layer_name} as {layer_type}")

            try:
                if layer_type == "router":
                    quantized_layer, stats = self.quantize_router_layer(
                        layer, calibration_data[layer_name]
                    )
                elif layer_type == "expert":
                    expert_id = self._extract_expert_id(layer_name)
                    quantized_layer, stats = self.quantize_expert_layer(
                        layer, calibration_data[layer_name], expert_id
                    )
                else:  # non_expert
                    quantized_layer, stats = self.quantize_non_expert_layer(
                        layer, calibration_data[layer_name]
                    )

                # Replace layer in model
                self._replace_layer_in_model(
                    model, layer_name, quantized_layer)
                all_stats[layer_name] = stats

                logger.info(
                    f"✓ Quantized {layer_name}: {stats.get('quantization_bits', 'unknown')}")

            except Exception as e:
                logger.error(f"Failed to quantize {layer_name}: {e}")
                all_stats[layer_name] = {"error": str(e)}

        return quantized_model, all_stats

    def _extract_expert_id(self, layer_name: str) -> Optional[int]:
        """Extract expert ID from layer name"""
        import re
        match = re.search(r'expert[_\s]*(\d+)', layer_name.lower())
        if match:
            return int(match.group(1))
        return None

    def _replace_layer_in_model(self, model: nn.Module, layer_name: str, new_layer: nn.Module):
        """Replace layer in model by name"""
        parts = layer_name.split('.')
        current = model

        # Navigate to parent module
        for part in parts[:-1]:
            current = getattr(current, part)

        # Replace the layer
        setattr(current, parts[-1], new_layer)


class StandardQuantizer:
    """
    Standard quantization for W8A8 layers
    """

    def quantize_linear(
        self,
        layer: nn.Linear,
        bits_w: int,
        bits_a: int,
        group_size: int
    ) -> Tuple[nn.Linear, Dict]:
        """
        Standard symmetric quantization

        Args:
            layer: Linear layer
            bits_w: Weight bit width
            bits_a: Activation bit width
            group_size: Group size for quantization

        Returns:
            quantized_layer: Quantized layer
            stats: Quantization statistics
        """
        W = layer.weight.data

        # Weight quantization
        W_quant, w_scales = self._quantize_weights(W, bits_w, group_size)

        # Create quantized layer
        quantized_layer = nn.Linear(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None
        )

        quantized_layer.weight.data = W_quant
        if layer.bias is not None:
            quantized_layer.bias.data = layer.bias.data.clone()

        # Statistics
        mse = F.mse_loss(W, W_quant).item()
        stats = {
            "mse": mse,
            "bits_w": bits_w,
            "bits_a": bits_a,
            "group_size": group_size,
        }

        return quantized_layer, stats

    def _quantize_weights(
        self,
        W: torch.Tensor,
        bits: int,
        group_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize weights per-group

        Args:
            W: Weight tensor [out_features, in_features]
            bits: Bit width
            group_size: Group size

        Returns:
            W_quant: Quantized weights
            scales: Per-group scales
        """
        out_features, in_features = W.shape
        n_groups = (in_features + group_size - 1) // group_size

        # Reshape to groups
        W_grouped = W.view(out_features, n_groups, -1)

        # Compute scales per group
        scales = W_grouped.abs().max(dim=-1, keepdim=True)[0]
        scales = scales.clamp(min=1e-8)

        # Quantize
        n_levels = 2 ** (bits - 1)  # Symmetric quantization
        W_normalized = W_grouped / scales
        W_int = torch.clamp(
            torch.round(W_normalized * (n_levels - 1)),
            -n_levels, n_levels - 1
        )
        W_quant = (W_int / (n_levels - 1)) * scales

        # Reshape back
        W_quant = W_quant.view(out_features, in_features)
        scales = scales.squeeze(-1)  # [out_features, n_groups]

        return W_quant, scales


def create_mixed_precision_quantizer(
    expert_bits_w: int = 2,
    expert_bits_a: int = 2,
    router_rank_gamma: float = 0.1,
    use_agq: bool = True
) -> MixedPrecisionQuantizer:
    """
    Convenience function to create mixed precision quantizer

    Args:
        expert_bits_w: Expert weight bits (2 or 4)
        expert_bits_a: Expert activation bits (2 or 4)
        router_rank_gamma: RouterRank guard band
        use_agq: Use AGQ for expert quantization

    Returns:
        MixedPrecisionQuantizer instance
    """
    config = MixedPrecisionConfig(
        expert_bits_w=expert_bits_w,
        expert_bits_a=expert_bits_a,
        router_rank_gamma=router_rank_gamma,
        use_agq_for_experts=use_agq
    )

    return MixedPrecisionQuantizer(config)
