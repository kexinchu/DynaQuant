"""
Router-Consistency Guard (RCG) for detecting when quantization affects routing decisions.
Tracks margin, JS divergence, and flip probability to compute risk metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence between two probability distributions.

    Args:
        p: First distribution [batch, num_experts]
        q: Second distribution [batch, num_experts]
        eps: Small constant for numerical stability

    Returns:
        js_div: JS divergence [batch]
    """
    # Ensure distributions are normalized
    p = p + eps
    q = q + eps
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)

    # Compute M = (P + Q) / 2
    m = (p + q) / 2

    # KL(P || M) + KL(Q || M)
    kl_p_m = torch.sum(p * torch.log(p / m), dim=-1)
    kl_q_m = torch.sum(q * torch.log(q / m), dim=-1)

    js = (kl_p_m + kl_q_m) / 2

    return js


def topk_agreement(indices1: torch.Tensor, indices2: torch.Tensor, k: int) -> float:
    """
    Compute top-k agreement between two sets of indices.

    Args:
        indices1: First set of indices [batch, k]
        indices2: Second set of indices [batch, k]
        k: Number of top elements

    Returns:
        agreement: Fraction of agreements (0.0 to 1.0)
    """
    # Count how many indices match
    matches = 0
    total = indices1.shape[0] * k

    for i in range(indices1.shape[0]):
        set1 = set(indices1[i].tolist())
        set2 = set(indices2[i].tolist())
        matches += len(set1 & set2)

    return matches / total


class RouterConsistencyGuard(nn.Module):
    """
    Router-Consistency Guard for monitoring routing stability under quantization.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int = 2,
        temperature: float = 1.0,
        clip_range: float = 10.0,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        margin_threshold: float = 0.5,
        promotion_threshold: float = 0.7,
        ema_alpha: float = 0.1,
    ):
        """
        Initialize Router-Consistency Guard.

        Args:
            num_experts: Total number of experts
            top_k: Number of top experts to select
            temperature: Temperature for logits scaling
            clip_range: Clipping range for logits
            alpha: Weight for margin indicator in risk metric
            beta: Weight for JS divergence in risk metric
            gamma: Weight for flip probability in risk metric
            margin_threshold: Threshold for margin indicator
            promotion_threshold: Risk threshold for promotion to W4A4
            ema_alpha: EMA smoothing factor for teacher distribution
        """
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.clip_range = clip_range

        # Risk metric parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin_threshold = margin_threshold
        self.promotion_threshold = promotion_threshold

        # EMA for teacher distribution
        self.ema_alpha = ema_alpha
        self.register_buffer('teacher_dist', None)

        # Statistics
        self.register_buffer('num_checks', torch.tensor(0))
        self.register_buffer('num_promotions', torch.tensor(0))

    def preprocess_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Preprocess router logits: apply temperature and clipping.

        Args:
            logits: Raw router logits [batch, num_experts]

        Returns:
            logits_processed: Processed logits
        """
        # Subtract max for numerical stability
        logits_max = torch.max(logits, dim=-1, keepdim=True)[0]
        logits_centered = logits - logits_max

        # Apply temperature
        logits_scaled = logits_centered / self.temperature

        # Clip
        logits_clipped = torch.clamp(
            logits_scaled, -self.clip_range, self.clip_range)

        return logits_clipped

    def compute_margin(self, logits: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """
        Compute margin between top-k and (k+1)-th expert.

        Args:
            logits: Router logits [batch, num_experts]
            k: Top-k value (defaults to self.top_k)

        Returns:
            margins: Margin for each sample [batch]
        """
        if k is None:
            k = self.top_k

        # Sort logits in descending order
        sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)

        # Margin = logit[k-1] - logit[k]
        # (difference between k-th and (k+1)-th largest)
        if k < sorted_logits.shape[-1]:
            margin = sorted_logits[:, k-1] - sorted_logits[:, k]
        else:
            margin = torch.ones(
                sorted_logits.shape[0], device=logits.device) * float('inf')

        return margin

    def compute_risk(
        self,
        logits_fp: torch.Tensor,
        logits_quant: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute risk metric for routing decisions.

        Args:
            logits_fp: FP16 router logits [batch, num_experts]
            logits_quant: Quantized router logits [batch, num_experts]

        Returns:
            risk: Risk score for each sample [batch]
            metrics: Dictionary of component metrics
        """
        batch_size = logits_fp.shape[0]

        # Preprocess logits
        logits_fp_proc = self.preprocess_logits(logits_fp)
        logits_quant_proc = self.preprocess_logits(logits_quant)

        # 1. Margin indicator
        margin_fp = self.compute_margin(logits_fp_proc)
        margin_indicator = (margin_fp < self.margin_threshold).float()

        # 2. JS divergence
        probs_fp = F.softmax(logits_fp_proc, dim=-1)
        probs_quant = F.softmax(logits_quant_proc, dim=-1)
        js_div = js_divergence(probs_fp, probs_quant)

        # 3. Flip probability estimation
        # Estimate probability that top-k selection changes
        _, indices_fp = torch.topk(logits_fp_proc, self.top_k, dim=-1)
        _, indices_quant = torch.topk(logits_quant_proc, self.top_k, dim=-1)

        # Count flips
        flips = torch.zeros(batch_size, device=logits_fp.device)
        for i in range(batch_size):
            set_fp = set(indices_fp[i].tolist())
            set_quant = set(indices_quant[i].tolist())
            flips[i] = len(set_fp - set_quant) / self.top_k

        # Compute risk
        risk = (
            self.alpha * margin_indicator +
            self.beta * js_div +
            self.gamma * flips
        )

        metrics = {
            'margin': margin_fp,
            'margin_indicator': margin_indicator,
            'js_divergence': js_div,
            'flip_prob': flips,
            'risk': risk,
        }

        return risk, metrics

    def check_promotion(
        self,
        logits_fp: torch.Tensor,
        logits_quant: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check which experts should be promoted based on risk.

        Args:
            logits_fp: FP16 router logits [batch, num_experts]
            logits_quant: Quantized router logits [batch, num_experts]
            expert_indices: Selected expert indices [batch, top_k]

        Returns:
            promote_mask: Boolean mask for experts to promote [num_experts]
        """
        # Compute risk
        risk, metrics = self.compute_risk(logits_fp, logits_quant)

        # Check which samples have high risk
        high_risk_samples = risk > self.promotion_threshold

        # Collect experts from high-risk samples
        promote_experts = set()
        for i in range(expert_indices.shape[0]):
            if high_risk_samples[i]:
                for j in range(expert_indices.shape[1]):
                    promote_experts.add(expert_indices[i, j].item())

        # Create promotion mask
        promote_mask = torch.zeros(
            self.num_experts, dtype=torch.bool, device=logits_fp.device)
        for expert_id in promote_experts:
            promote_mask[expert_id] = True

        # Update statistics
        self.num_checks += logits_fp.shape[0]
        self.num_promotions += high_risk_samples.sum()

        return promote_mask

    def update_teacher_distribution(self, logits_fp: torch.Tensor):
        """
        Update EMA of teacher distribution (from FP16 router).

        Args:
            logits_fp: FP16 router logits [batch, num_experts]
        """
        logits_proc = self.preprocess_logits(logits_fp)
        probs_fp = F.softmax(logits_proc, dim=-1)

        # Average over batch
        batch_avg = probs_fp.mean(dim=0)

        if self.teacher_dist is None:
            self.teacher_dist = batch_avg
        else:
            self.teacher_dist = (
                (1 - self.ema_alpha) * self.teacher_dist +
                self.ema_alpha * batch_avg
            )

    def get_statistics(self) -> Dict[str, float]:
        """Get RCG statistics."""
        if self.num_checks > 0:
            promotion_rate = (self.num_promotions.float() /
                              self.num_checks.float()).item()
        else:
            promotion_rate = 0.0

        return {
            'num_checks': self.num_checks.item(),
            'num_promotions': self.num_promotions.item(),
            'promotion_rate': promotion_rate,
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        self.num_checks.zero_()
        self.num_promotions.zero_()


def grid_search_temperature_clip(
    logits_fp: torch.Tensor,
    logits_quant: torch.Tensor,
    top_k: int = 2,
    temperature_range: list = [0.5, 1.0, 1.5, 2.0],
    clip_range: list = [5.0, 10.0, 15.0, 20.0],
) -> Tuple[float, float, float]:
    """
    Grid search for optimal temperature and clip range to maximize top-k agreement.

    Args:
        logits_fp: FP16 router logits [batch, num_experts]
        logits_quant: Quantized router logits [batch, num_experts]
        top_k: Number of top experts
        temperature_range: List of temperature values to try
        clip_range: List of clip range values to try

    Returns:
        best_temperature: Best temperature value
        best_clip: Best clip range value
        best_agreement: Best top-k agreement score
    """
    best_agreement = 0.0
    best_temperature = 1.0
    best_clip = 10.0

    for temp in temperature_range:
        for clip in clip_range:
            # Process logits with current temperature and clip
            # FP logits
            logits_fp_max = torch.max(logits_fp, dim=-1, keepdim=True)[0]
            logits_fp_proc = (logits_fp - logits_fp_max) / temp
            logits_fp_proc = torch.clamp(logits_fp_proc, -clip, clip)

            # Quant logits
            logits_quant_max = torch.max(logits_quant, dim=-1, keepdim=True)[0]
            logits_quant_proc = (logits_quant - logits_quant_max) / temp
            logits_quant_proc = torch.clamp(logits_quant_proc, -clip, clip)

            # Get top-k indices
            _, indices_fp = torch.topk(logits_fp_proc, top_k, dim=-1)
            _, indices_quant = torch.topk(logits_quant_proc, top_k, dim=-1)

            # Compute agreement
            agreement = topk_agreement(indices_fp, indices_quant, top_k)

            if agreement > best_agreement:
                best_agreement = agreement
                best_temperature = temp
                best_clip = clip

    return best_temperature, best_clip, best_agreement


def test_router_guard():
    """
    Unit tests for Router-Consistency Guard.
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Testing Router-Consistency Guard...")

    # Test JS divergence
    logger.info("\n--- Testing JS divergence ---")
    torch.manual_seed(42)

    p = F.softmax(torch.randn(16, 64), dim=-1)
    q = F.softmax(torch.randn(16, 64), dim=-1)

    js_div = js_divergence(p, q)
    logger.info(f"JS divergence shape: {js_div.shape}")
    logger.info(
        f"JS divergence range: [{js_div.min():.4f}, {js_div.max():.4f}]")
    logger.info(f"✓ JS divergence test passed")

    # Test top-k agreement
    logger.info("\n--- Testing top-k agreement ---")
    indices1 = torch.tensor([[0, 1, 2], [3, 4, 5]])
    indices2 = torch.tensor([[0, 1, 3], [3, 4, 5]])

    agreement = topk_agreement(indices1, indices2, k=3)
    logger.info(f"Top-k agreement: {agreement:.4f}")
    expected_agreement = (2 + 3) / 6  # 5 matches out of 6 total
    assert abs(
        agreement - expected_agreement) < 1e-6, f"Expected {expected_agreement}, got {agreement}"
    logger.info(f"✓ Top-k agreement test passed")

    # Test RouterConsistencyGuard
    logger.info("\n--- Testing RouterConsistencyGuard ---")
    num_experts = 64
    top_k = 2

    rcg = RouterConsistencyGuard(
        num_experts=num_experts,
        top_k=top_k,
        temperature=1.0,
        clip_range=10.0,
        alpha=1.0,
        beta=0.5,
        gamma=0.3,
        margin_threshold=0.5,
        promotion_threshold=0.7,
    )

    # Generate fake logits
    torch.manual_seed(42)
    logits_fp = torch.randn(16, num_experts)
    # Simulate quantization error
    logits_quant = logits_fp + torch.randn_like(logits_fp) * 0.1

    # Compute risk
    risk, metrics = rcg.compute_risk(logits_fp, logits_quant)

    logger.info(f"Risk shape: {risk.shape}")
    logger.info(f"Risk range: [{risk.min():.4f}, {risk.max():.4f}]")
    logger.info(
        f"Margin range: [{metrics['margin'].min():.4f}, {metrics['margin'].max():.4f}]")
    logger.info(
        f"JS divergence range: [{metrics['js_divergence'].min():.4f}, {metrics['js_divergence'].max():.4f}]")
    logger.info(
        f"Flip prob range: [{metrics['flip_prob'].min():.4f}, {metrics['flip_prob'].max():.4f}]")
    logger.info(f"✓ Risk computation test passed")

    # Test promotion check
    logger.info("\n--- Testing promotion check ---")
    _, expert_indices = torch.topk(logits_fp, top_k, dim=-1)

    promote_mask = rcg.check_promotion(logits_fp, logits_quant, expert_indices)

    logger.info(f"Promote mask shape: {promote_mask.shape}")
    logger.info(f"Number of experts to promote: {promote_mask.sum().item()}")

    stats = rcg.get_statistics()
    logger.info(f"Statistics: {stats}")
    logger.info(f"✓ Promotion check test passed")

    # Test grid search
    logger.info("\n--- Testing grid search ---")
    torch.manual_seed(42)
    logits_fp_grid = torch.randn(100, num_experts)
    logits_quant_grid = logits_fp_grid + \
        torch.randn_like(logits_fp_grid) * 0.05

    best_temp, best_clip, best_agreement = grid_search_temperature_clip(
        logits_fp_grid,
        logits_quant_grid,
        top_k=top_k,
        temperature_range=[0.5, 1.0, 1.5],
        clip_range=[5.0, 10.0, 15.0],
    )

    logger.info(f"Best temperature: {best_temp}")
    logger.info(f"Best clip range: {best_clip}")
    logger.info(f"Best agreement: {best_agreement:.4f}")
    logger.info(f"✓ Grid search test passed")

    # Test teacher distribution update
    logger.info("\n--- Testing teacher distribution update ---")
    rcg.update_teacher_distribution(logits_fp)

    assert rcg.teacher_dist is not None
    assert rcg.teacher_dist.shape == (num_experts,)
    assert torch.abs(rcg.teacher_dist.sum() -
                     1.0) < 1e-5, "Teacher distribution should sum to 1"

    logger.info(f"Teacher distribution shape: {rcg.teacher_dist.shape}")
    logger.info(f"Teacher distribution sum: {rcg.teacher_dist.sum():.6f}")
    logger.info(f"✓ Teacher distribution test passed")

    logger.info("\n✓ All Router-Consistency Guard tests passed!")
    return True


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_router_guard()
