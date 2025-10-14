"""
Routing losses for QAT training

Includes:
- Top-k consistency loss
- Margin loss (top1 - top2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


def topk_consistency_loss(
    logits_quant: torch.Tensor,
    logits_fp: torch.Tensor,
    k: int = 2,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Top-k consistency loss between quantized and FP routing

    Encourages quantized routing to match FP routing decisions.

    Args:
        logits_quant: Quantized router logits [batch, seq_len, num_experts]
        logits_fp: FP router logits [batch, seq_len, num_experts]
        k: Number of top experts
        temperature: Temperature for softmax

    Returns:
        loss: Scalar consistency loss
    """
    # Get top-k expert IDs from FP
    _, topk_ids_fp = torch.topk(logits_fp, k, dim=-1)  # [batch, seq, k]

    # Create mask for top-k experts
    batch, seq_len, num_experts = logits_fp.shape
    topk_mask = torch.zeros_like(logits_fp)
    topk_mask.scatter_(-1, topk_ids_fp, 1.0)

    # Compute softmax probabilities
    probs_quant = F.softmax(logits_quant / temperature, dim=-1)
    probs_fp = F.softmax(logits_fp / temperature, dim=-1)

    # Loss: encourage quantized probs to match FP probs on top-k
    # Use KL divergence: KL(p_fp || p_quant)
    kl_loss = F.kl_div(
        probs_quant.log(),
        probs_fp,
        reduction="none"
    )

    # Weight by top-k mask (focus on top-k experts)
    weighted_loss = (kl_loss * topk_mask).sum(dim=-1)

    return weighted_loss.mean()


def margin_loss(
    logits: torch.Tensor,
    k: int = 2,
    margin_target: float = 0.5,
    margin_type: str = "top1_top2"
) -> torch.Tensor:
    """
    Margin loss to increase separation between top experts

    Encourages larger gap between top-1 and top-2 (or top-k and k+1).

    Args:
        logits: Router logits [batch, seq_len, num_experts]
        k: Number of top experts
        margin_target: Target margin
        margin_type: "top1_top2" or "topk_kplus1"

    Returns:
        loss: Scalar margin loss
    """
    if margin_type == "top1_top2":
        # Margin between top-1 and top-2
        top_values, _ = torch.topk(logits, 2, dim=-1)  # [batch, seq, 2]
        margin = top_values[..., 0] - top_values[..., 1]  # [batch, seq]

    elif margin_type == "topk_kplus1":
        # Margin between top-k and (k+1)-th
        top_values, _ = torch.topk(logits, k + 1, dim=-1)  # [batch, seq, k+1]
        margin = top_values[..., k - 1] - top_values[..., k]  # [batch, seq]

    else:
        raise ValueError(f"Unknown margin_type: {margin_type}")

    # Loss: penalize if margin is below target
    # Use hinge loss: max(0, margin_target - margin)
    loss = F.relu(margin_target - margin)

    return loss.mean()


def routing_diversity_loss(
    logits: torch.Tensor,
    expert_counts: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Routing diversity loss to encourage balanced expert usage

    Args:
        logits: Router logits [batch, seq_len, num_experts]
        expert_counts: Optional current expert usage counts [num_experts]

    Returns:
        loss: Scalar diversity loss
    """
    # Compute average expert selection probability
    probs = F.softmax(logits, dim=-1)  # [batch, seq, num_experts]
    avg_probs = probs.mean(dim=[0, 1])  # [num_experts]

    # Target: uniform distribution
    num_experts = logits.size(-1)
    target_prob = 1.0 / num_experts

    # If expert counts provided, weight by inverse frequency
    if expert_counts is not None:
        weights = 1.0 / (expert_counts + 1.0)
        weights = weights / weights.sum()
    else:
        weights = torch.ones_like(avg_probs)

    # L2 loss to target
    loss = ((avg_probs - target_prob) ** 2 * weights).sum()

    return loss


def combined_routing_loss(
    logits_quant: torch.Tensor,
    logits_fp: torch.Tensor,
    k: int = 2,
    lambda_consistency: float = 1.0,
    lambda_margin: float = 0.2,
    lambda_diversity: float = 0.1,
    margin_target: float = 0.5
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined routing loss for QAT

    Args:
        logits_quant: Quantized router logits
        logits_fp: FP router logits
        k: Number of top experts
        lambda_consistency: Weight for consistency loss
        lambda_margin: Weight for margin loss
        lambda_diversity: Weight for diversity loss
        margin_target: Target margin

    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary of individual losses
    """
    # Individual losses
    consistency = topk_consistency_loss(logits_quant, logits_fp, k)
    margin = margin_loss(logits_quant, k, margin_target)
    diversity = routing_diversity_loss(logits_quant)

    # Combined
    total_loss = (
        lambda_consistency * consistency +
        lambda_margin * margin +
        lambda_diversity * diversity
    )

    loss_dict = {
        "consistency": consistency.item(),
        "margin": margin.item(),
        "diversity": diversity.item(),
        "total": total_loss.item(),
    }

    return total_loss, loss_dict


# For backward compatibility
def topk_consistency(z_fp16: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
    """Alias for topk_consistency_loss"""
    return topk_consistency_loss(z_q, z_fp16, k=2)


def margin(z_q: torch.Tensor) -> torch.Tensor:
    """Alias for margin_loss"""
    return margin_loss(z_q, k=2, margin_target=0.5)
