"""
RouterRank: Top-k Ranking Invariance Loss for Router Quantization

Based on RouterRank paper, implements pairwise rank hinge loss to maintain
Top-k ranking invariance for router logits during quantization.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class RouterRankLoss:
    """
    RouterRank loss for maintaining Top-k ranking invariance

    Core idea: Traditional PTQ minimizes MSE but doesn't consider ranking.
    RouterRank directly optimizes for Top-k ranking invariance.

    Loss: L_rank = Σ_{i∈topk(z)} Σ_{j∉topk(z)} max(0, γ - (z_hat_i - z_hat_j))
    where γ is a guard band margin.
    """

    def __init__(self, gamma: float = 0.1, lambda_rank: float = 1.0):
        """
        Initialize RouterRank loss

        Args:
            gamma: Guard band margin (安全间隔)
            lambda_rank: Weight for ranking loss relative to MSE
        """
        self.gamma = gamma
        self.lambda_rank = lambda_rank

    def compute_ranking_loss(
        self,
        logits_fp: torch.Tensor,
        logits_quant: torch.Tensor,
        top_k: int = 2
    ) -> torch.Tensor:
        """
        Compute RouterRank pairwise ranking loss

        Args:
            logits_fp: FP16 router logits [batch, seq_len, num_experts]
            logits_quant: Quantized router logits [batch, seq_len, num_experts]
            top_k: Number of top experts to select

        Returns:
            ranking_loss: Scalar ranking loss
        """
        batch, seq_len, num_experts = logits_fp.shape

        # Get top-k expert indices from FP16 logits
        _, topk_indices = torch.topk(
            logits_fp, top_k, dim=-1)  # [batch, seq_len, top_k]

        # Create masks for selected and unselected experts
        selected_mask = torch.zeros_like(logits_fp, dtype=torch.bool)
        selected_mask.scatter_(-1, topk_indices, True)
        unselected_mask = ~selected_mask

        # Compute pairwise ranking loss
        total_loss = 0.0
        num_pairs = 0

        for b in range(batch):
            for s in range(seq_len):
                # Get selected and unselected expert logits
                selected_logits_fp = logits_fp[b, s][selected_mask[b, s]]
                selected_logits_quant = logits_quant[b, s][selected_mask[b, s]]
                unselected_logits_fp = logits_fp[b, s][unselected_mask[b, s]]
                unselected_logits_quant = logits_quant[b,
                                                       s][unselected_mask[b, s]]

                # Compute pairwise differences for quantized logits
                if len(selected_logits_quant) > 0 and len(unselected_logits_quant) > 0:
                    # Broadcasting: [num_selected, 1] - [1, num_unselected]
                    pairwise_diff = selected_logits_quant.unsqueeze(
                        -1) - unselected_logits_quant.unsqueeze(0)

                    # Hinge loss: max(0, γ - (z_hat_i - z_hat_j))
                    hinge_loss = F.relu(self.gamma - pairwise_diff)

                    total_loss += hinge_loss.sum()
                    num_pairs += pairwise_diff.numel()

        # Average over all pairs
        if num_pairs > 0:
            ranking_loss = total_loss / num_pairs
        else:
            ranking_loss = torch.tensor(0.0, device=logits_fp.device)

        return ranking_loss

    def compute_mse_loss(
        self,
        logits_fp: torch.Tensor,
        logits_quant: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute standard MSE loss

        Args:
            logits_fp: FP16 router logits
            logits_quant: Quantized router logits

        Returns:
            mse_loss: Scalar MSE loss
        """
        return F.mse_loss(logits_quant, logits_fp)

    def compute_total_loss(
        self,
        logits_fp: torch.Tensor,
        logits_quant: torch.Tensor,
        top_k: int = 2
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total RouterRank loss (MSE + λ * Ranking)

        Args:
            logits_fp: FP16 router logits
            logits_quant: Quantized router logits
            top_k: Number of top experts

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        mse_loss = self.compute_mse_loss(logits_fp, logits_quant)
        ranking_loss = self.compute_ranking_loss(
            logits_fp, logits_quant, top_k)

        total_loss = mse_loss + self.lambda_rank * ranking_loss

        loss_dict = {
            "mse": mse_loss.item(),
            "ranking": ranking_loss.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict

    def compute_topk_consistency(
        self,
        logits_fp: torch.Tensor,
        logits_quant: torch.Tensor,
        top_k: int = 2
    ) -> Dict[str, float]:
        """
        Compute Top-k consistency metrics

        Args:
            logits_fp: FP16 router logits
            logits_quant: Quantized router logits
            top_k: Number of top experts

        Returns:
            Dictionary with consistency metrics
        """
        # Get top-k indices
        _, topk_fp = torch.topk(logits_fp, top_k, dim=-1)
        _, topk_quant = torch.topk(logits_quant, top_k, dim=-1)

        # Compute exact match rate (all k experts match)
        exact_match = (topk_fp == topk_quant).all(dim=-1).float()
        exact_match_rate = exact_match.mean().item()

        # Compute partial match rate (any overlap)
        batch, seq_len, _ = logits_fp.shape
        partial_matches = []

        for b in range(batch):
            for s in range(seq_len):
                fp_set = set(topk_fp[b, s].tolist())
                quant_set = set(topk_quant[b, s].tolist())
                overlap = len(fp_set & quant_set)
                partial_matches.append(overlap / top_k)

        partial_match_rate = torch.tensor(partial_matches).mean().item()

        return {
            "exact_match_rate": exact_match_rate,
            "partial_match_rate": partial_match_rate,
            "flip_rate": 1 - exact_match_rate,
        }


def create_router_rank_loss(
    gamma: float = 0.1,
    lambda_rank: float = 1.0
) -> RouterRankLoss:
    """
    Convenience function to create RouterRank loss

    Args:
        gamma: Guard band margin
        lambda_rank: Weight for ranking loss

    Returns:
        RouterRankLoss instance
    """
    return RouterRankLoss(gamma, lambda_rank)


# For backward compatibility and integration
def router_rank_loss(
    logits_fp: torch.Tensor,
    logits_quant: torch.Tensor,
    top_k: int = 2,
    gamma: float = 0.1,
    lambda_rank: float = 1.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Direct function interface for RouterRank loss

    Args:
        logits_fp: FP16 router logits
        logits_quant: Quantized router logits
        top_k: Number of top experts
        gamma: Guard band margin
        lambda_rank: Weight for ranking loss

    Returns:
        total_loss: Combined loss
        loss_dict: Loss breakdown
    """
    loss_fn = RouterRankLoss(gamma, lambda_rank)
    return loss_fn.compute_total_loss(logits_fp, logits_quant, top_k)
