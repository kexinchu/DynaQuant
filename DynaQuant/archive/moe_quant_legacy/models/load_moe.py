"""
MoE Model Loader

Provides unified interface to load and interact with MoE models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import logging


logger = logging.getLogger(__name__)


class MoEModelLoader:
    """
    Unified loader for MoE models

    Supports:
    - Qwen-MoE
    - Mixtral
    - DeepSeek-MoE
    - Custom MoE architectures
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None
        self.config = None
        self.moe_layers = []

    def load(self) -> None:
        """Load model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")

        # Load config
        self.config = AutoConfig.from_pretrained(
            self.model_name, trust_remote_code=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=self.config,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True
        )

        # Detect MoE layers
        self.moe_layers = self._detect_moe_layers()

        logger.info(f"Loaded model with {len(self.moe_layers)} MoE layers")

    def _detect_moe_layers(self) -> List[Tuple[str, nn.Module]]:
        """Detect MoE layers in the model"""
        moe_layers = []

        for name, module in self.model.named_modules():
            # Common MoE layer patterns
            if any(keyword in name.lower() for keyword in ["moe", "expert", "ffn"]):
                # Check if this is a MoE block (has multiple experts)
                if hasattr(module, "experts") or "experts" in name.lower():
                    moe_layers.append((name, module))

        return moe_layers

    def get_moe_layer(self, layer_idx: int) -> Optional[nn.Module]:
        """Get MoE layer by index"""
        if layer_idx < len(self.moe_layers):
            return self.moe_layers[layer_idx][1]
        return None

    def get_router(self, layer_idx: int) -> Optional[nn.Module]:
        """Get router for a specific MoE layer"""
        moe_layer = self.get_moe_layer(layer_idx)
        if moe_layer is None:
            return None

        # Try common router attribute names
        for attr in ["gate", "router", "gating_network"]:
            if hasattr(moe_layer, attr):
                return getattr(moe_layer, attr)

        return None

    def get_experts(self, layer_idx: int) -> Optional[List[nn.Module]]:
        """Get expert modules for a specific MoE layer"""
        moe_layer = self.get_moe_layer(layer_idx)
        if moe_layer is None:
            return None

        # Try common expert container names
        for attr in ["experts", "expert_modules", "ffn_experts"]:
            if hasattr(moe_layer, attr):
                experts = getattr(moe_layer, attr)
                if isinstance(experts, nn.ModuleList):
                    return list(experts)
                elif isinstance(experts, list):
                    return experts

        return None

    def forward_router(
        self,
        x: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward through router and get gating logits + expert IDs

        Args:
            x: Input [batch, seq_len, hidden_dim]
            layer_idx: MoE layer index

        Returns:
            logits: Router logits [batch, seq_len, num_experts]
            expert_ids: Selected expert IDs [batch, seq_len, top_k]
        """
        router = self.get_router(layer_idx)
        if router is None:
            raise ValueError(f"No router found for layer {layer_idx}")

        # Forward through router
        logits = router(x)

        # Get top-k experts
        # Infer top_k from config or use default
        top_k = getattr(self.config, "num_experts_per_tok", 2)
        _, expert_ids = torch.topk(logits, top_k, dim=-1)

        return logits, expert_ids

    def forward_expert(
        self,
        x: torch.Tensor,
        expert_id: int,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Forward through a specific expert

        Args:
            x: Input [batch, seq_len, hidden_dim]
            expert_id: Expert ID
            layer_idx: MoE layer index

        Returns:
            output: Expert output [batch, seq_len, hidden_dim]
        """
        experts = self.get_experts(layer_idx)
        if experts is None or expert_id >= len(experts):
            raise ValueError(
                f"Invalid expert_id {expert_id} for layer {layer_idx}")

        expert = experts[expert_id]
        return expert(x)

    def forward_moe_layer(
        self,
        x: torch.Tensor,
        layer_idx: int,
        return_routing_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward through entire MoE layer

        Args:
            x: Input [batch, seq_len, hidden_dim]
            layer_idx: MoE layer index
            return_routing_info: Whether to return routing information

        Returns:
            output: Layer output
            routing_info: Optional routing information dict
        """
        # Get router logits and expert IDs
        logits, expert_ids = self.forward_router(x, layer_idx)

        # Compute gating weights
        weights = torch.softmax(logits, dim=-1)

        # Gather weights for selected experts
        batch, seq_len, num_experts = logits.shape
        top_k = expert_ids.size(-1)

        # Get weights for selected experts
        selected_weights = weights.gather(
            -1,
            expert_ids
        )  # [batch, seq_len, top_k]

        # Normalize weights
        selected_weights = selected_weights / \
            selected_weights.sum(dim=-1, keepdim=True)

        # Forward through selected experts
        experts = self.get_experts(layer_idx)
        output = torch.zeros_like(x)

        for k in range(top_k):
            # Get expert IDs for this k
            expert_ids_k = expert_ids[:, :, k]  # [batch, seq_len]
            weights_k = selected_weights[:, :, k:k+1]  # [batch, seq_len, 1]

            # Group tokens by expert
            unique_experts = expert_ids_k.unique()

            for expert_id in unique_experts:
                # Mask for this expert
                mask = (expert_ids_k == expert_id)  # [batch, seq_len]

                # Select tokens for this expert
                expert_input = x[mask]  # [num_tokens, hidden_dim]

                if expert_input.numel() == 0:
                    continue

                # Forward through expert
                expert_output = experts[expert_id.item()](expert_input)

                # Apply weights and accumulate
                expert_weights = weights_k[mask]  # [num_tokens, 1]
                weighted_output = expert_output * expert_weights

                # Scatter back
                output[mask] += weighted_output

        routing_info = None
        if return_routing_info:
            routing_info = {
                "logits": logits,
                "expert_ids": expert_ids,
                "weights": selected_weights,
            }

        return output, routing_info

    def get_num_experts(self, layer_idx: int) -> int:
        """Get number of experts in a layer"""
        experts = self.get_experts(layer_idx)
        if experts is None:
            return 0
        return len(experts)

    def get_num_moe_layers(self) -> int:
        """Get total number of MoE layers"""
        return len(self.moe_layers)


def load_moe_model(
    model_name: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16
) -> MoEModelLoader:
    """
    Convenience function to load MoE model

    Args:
        model_name: Model name or path
        device: Device to load on
        torch_dtype: Torch dtype

    Returns:
        MoEModelLoader instance
    """
    loader = MoEModelLoader(model_name, device, torch_dtype)
    loader.load()
    return loader
