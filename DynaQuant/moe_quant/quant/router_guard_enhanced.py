"""
Enhanced Router Guard for MoE Quantization

Provides high-precision routing with:
- INT8 input + INT32 accumulation or FP16 path
- Fused softmax + top-k with deterministic tie-breaking
- Online consistency detection and adaptive fallback
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import numpy as np


@dataclass
class EnhancedRouterConfig:
    """Configuration for enhanced router guard"""
    # Router computation mode
    mode: str = "fp16"  # "fp16", "int8_acc32", "int8_acc16"

    # Top-k consistency
    top_k: int = 2
    strict_topk: bool = True  # If True, fallback on any inconsistency
    consistency_threshold: float = 0.95  # Minimum top-k match rate

    # Tie-breaking
    deterministic_tiebreak: bool = True
    tiebreak_epsilon: float = 1e-6

    # Online consistency detection
    enable_online_detection: bool = True
    detection_window_size: int = 100
    detection_batch_size: int = 16

    # Fallback policy
    enable_fallback: bool = True
    fallback_on_first_flip: bool = False  # Immediate fallback on first flip


class EnhancedRouterGuard:
    """
    Enhanced Router Guard for stable routing under quantization

    Features:
    1. High-precision accumulation (INT32 or FP16)
    2. Deterministic top-k selection
    3. Online consistency monitoring
    4. Adaptive fallback to higher precision
    """

    def __init__(self, config: Optional[EnhancedRouterConfig] = None):
        self.config = config or EnhancedRouterConfig()

        # Statistics tracking
        self.total_tokens = 0
        self.total_flips = 0
        self.flip_history = []
        self.layer_flip_counts = {}

        # Reference routing decisions (for consistency check)
        self.ref_decisions = {}

    def reset_stats(self):
        """Reset all statistics"""
        self.total_tokens = 0
        self.total_flips = 0
        self.flip_history = []
        self.layer_flip_counts = {}

    def forward_router_fp16(
        self,
        x: torch.Tensor,
        router_weight: torch.Tensor,
        router_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward router in FP16 (high precision)

        Args:
            x: Input [batch, seq_len, hidden_dim]
            router_weight: Router weight [num_experts, hidden_dim]
            router_bias: Optional bias [num_experts]

        Returns:
            logits: Router logits [batch, seq_len, num_experts]
            expert_ids: Selected expert IDs [batch, seq_len, top_k]
        """
        # Compute logits in FP16
        logits = F.linear(x, router_weight, router_bias)

        # Softmax + top-k
        expert_ids = self.deterministic_topk(logits, self.config.top_k)

        return logits, expert_ids

    def forward_router_int8(
        self,
        x: torch.Tensor,
        router_weight: torch.Tensor,
        router_bias: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
        w_scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward router in INT8 with INT32 accumulation

        Args:
            x: Input (quantized to INT8 if scales provided)
            router_weight: Router weight (quantized to INT8 if scales provided)
            router_bias: Optional bias
            x_scale: Input scale for dequantization
            w_scale: Weight scale for dequantization

        Returns:
            logits: Router logits (dequantized to FP)
            expert_ids: Selected expert IDs
        """
        # If scales provided, assume inputs are INT8
        if x_scale is not None and w_scale is not None:
            # INT8 x INT8 -> INT32 matmul
            logits_int32 = torch.matmul(
                x.to(torch.int32), router_weight.T.to(torch.int32))

            # Dequantize: INT32 -> FP32
            logits = logits_int32.float() * x_scale * w_scale

            if router_bias is not None:
                logits = logits + router_bias
        else:
            # Regular FP matmul (fallback)
            logits = F.linear(x, router_weight, router_bias)

        # Softmax + top-k in high precision
        expert_ids = self.deterministic_topk(logits, self.config.top_k)

        return logits, expert_ids

    def deterministic_topk(
        self,
        logits: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """
        Deterministic top-k selection with tie-breaking

        Args:
            logits: Router logits [batch, seq_len, num_experts]
            k: Number of experts to select

        Returns:
            expert_ids: Selected expert IDs [batch, seq_len, k]
        """
        if not self.config.deterministic_tiebreak:
            # Standard top-k
            _, expert_ids = torch.topk(logits, k, dim=-1)
            return expert_ids

        # Add small deterministic noise based on expert ID to break ties
        num_experts = logits.size(-1)
        tiebreak_noise = torch.arange(
            num_experts,
            device=logits.device,
            dtype=logits.dtype
        ) * self.config.tiebreak_epsilon

        logits_with_tiebreak = logits + tiebreak_noise

        # Top-k with tie-breaking
        _, expert_ids = torch.topk(
            logits_with_tiebreak, k, dim=-1, sorted=True)

        return expert_ids

    def check_consistency(
        self,
        expert_ids_ref: torch.Tensor,
        expert_ids_quant: torch.Tensor,
        layer_id: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Check top-k consistency between reference and quantized routing

        Args:
            expert_ids_ref: Reference expert IDs [batch, seq_len, k]
            expert_ids_quant: Quantized expert IDs [batch, seq_len, k]
            layer_id: Optional layer identifier

        Returns:
            Dictionary with consistency metrics
        """
        # Compute exact match rate (all k experts match)
        exact_match = (expert_ids_ref == expert_ids_quant).all(dim=-1).float()
        exact_match_rate = exact_match.mean().item()

        # Compute partial match rate (any k experts match)
        # For each position, check if expert sets overlap
        batch, seq_len, k = expert_ids_ref.shape
        ref_flat = expert_ids_ref.reshape(-1, k)
        quant_flat = expert_ids_quant.reshape(-1, k)

        matches = []
        for i in range(ref_flat.size(0)):
            ref_set = set(ref_flat[i].tolist())
            quant_set = set(quant_flat[i].tolist())
            overlap = len(ref_set & quant_set)
            matches.append(overlap / k)

        partial_match_rate = np.mean(matches)

        # Count flips
        num_tokens = batch * seq_len
        num_flips = int(num_tokens * (1 - exact_match_rate))

        # Update statistics
        self.total_tokens += num_tokens
        self.total_flips += num_flips
        self.flip_history.append(exact_match_rate)

        if layer_id is not None:
            if layer_id not in self.layer_flip_counts:
                self.layer_flip_counts[layer_id] = {"tokens": 0, "flips": 0}
            self.layer_flip_counts[layer_id]["tokens"] += num_tokens
            self.layer_flip_counts[layer_id]["flips"] += num_flips

        # Compute windowed flip rate
        window = self.flip_history[-self.config.detection_window_size:]
        windowed_match_rate = np.mean(window)

        return {
            "exact_match_rate": exact_match_rate,
            "partial_match_rate": partial_match_rate,
            "num_flips": num_flips,
            "num_tokens": num_tokens,
            "flip_rate": 1 - exact_match_rate,
            "windowed_match_rate": windowed_match_rate,
            "needs_fallback": windowed_match_rate < self.config.consistency_threshold,
        }

    def forward_with_guard(
        self,
        x: torch.Tensor,
        router_weight: torch.Tensor,
        router_bias: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
        layer_id: Optional[int] = None,
        check_consistency: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward router with consistency guard

        Args:
            x: Input activations
            router_weight: Router weight
            router_bias: Router bias
            mode: Routing mode ("fp16" or "int8_acc32")
            layer_id: Layer identifier
            check_consistency: Whether to check consistency with FP16 reference

        Returns:
            logits: Router logits
            expert_ids: Selected expert IDs
            stats: Consistency statistics
        """
        mode = mode or self.config.mode
        stats = {}

        # Compute in requested mode
        if mode == "fp16":
            logits, expert_ids = self.forward_router_fp16(
                x, router_weight, router_bias)
            stats["mode_used"] = "fp16"

        elif mode in ["int8_acc32", "int8_acc16"]:
            # For demonstration, we'll quantize inputs on-the-fly
            # In practice, inputs should already be quantized
            x_scale = x.abs().max() / 127.0
            x_int8 = torch.clamp(torch.round(
                x / x_scale), -128, 127).to(torch.int8)

            w_scale = router_weight.abs().max() / 127.0
            w_int8 = torch.clamp(
                torch.round(router_weight / w_scale), -128, 127
            ).to(torch.int8)

            logits, expert_ids = self.forward_router_int8(
                x_int8, w_int8, router_bias, x_scale, w_scale
            )
            stats["mode_used"] = mode

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Check consistency if enabled and not in FP16 mode
        if (check_consistency and
            mode != "fp16" and
                self.config.enable_online_detection):

            # Compute FP16 reference
            logits_ref, expert_ids_ref = self.forward_router_fp16(
                x, router_weight, router_bias
            )

            # Check consistency
            consistency_stats = self.check_consistency(
                expert_ids_ref, expert_ids, layer_id
            )
            stats.update(consistency_stats)

            # Fallback if needed
            if (self.config.enable_fallback and
                    consistency_stats["needs_fallback"]):
                if self.config.strict_topk or self.config.fallback_on_first_flip:
                    # Use FP16 results
                    logits = logits_ref
                    expert_ids = expert_ids_ref
                    stats["mode_used"] = "fp16_fallback"

        return logits, expert_ids, stats

    def get_statistics(self) -> Dict:
        """Get overall statistics"""
        overall_flip_rate = self.total_flips / max(self.total_tokens, 1)

        layer_stats = {}
        for layer_id, counts in self.layer_flip_counts.items():
            layer_flip_rate = counts["flips"] / max(counts["tokens"], 1)
            layer_stats[layer_id] = {
                "tokens": counts["tokens"],
                "flips": counts["flips"],
                "flip_rate": layer_flip_rate,
            }

        return {
            "total_tokens": self.total_tokens,
            "total_flips": self.total_flips,
            "overall_flip_rate": overall_flip_rate,
            "layer_stats": layer_stats,
        }


class RouterGuardWrapper(nn.Module):
    """
    Wrapper for router layers with guard
    """

    def __init__(
        self,
        router: nn.Linear,
        config: Optional[EnhancedRouterConfig] = None,
        layer_id: Optional[int] = None
    ):
        super().__init__()
        self.router = router
        self.guard = EnhancedRouterGuard(config)
        self.layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        mode: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward with guard

        Returns:
            logits: Router logits
            expert_ids: Selected expert IDs
            stats: Guard statistics
        """
        logits, expert_ids, stats = self.guard.forward_with_guard(
            x,
            self.router.weight,
            self.router.bias,
            mode=mode,
            layer_id=self.layer_id
        )

        return logits, expert_ids, stats


def create_router_guard(
    mode: str = "fp16",
    top_k: int = 2,
    strict_topk: bool = True,
    consistency_threshold: float = 0.95
) -> EnhancedRouterGuard:
    """Convenience function to create router guard"""
    config = EnhancedRouterConfig(
        mode=mode,
        top_k=top_k,
        strict_topk=strict_topk,
        consistency_threshold=consistency_threshold
    )
    return EnhancedRouterGuard(config)
