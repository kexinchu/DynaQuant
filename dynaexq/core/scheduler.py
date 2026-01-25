"""
PrecisionScheduler: computes H_l and emits transition requests.

Implements top-n projection: H_l(t) = Top-n_hi[l] experts ranked by S[l,e](t)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Tier
from .hotness_tracker import HotnessTracker
from .registry import ExpertKey


@dataclass
class TransitionReq:
    """Transition request for an expert."""
    key: ExpertKey
    src: Tier
    dst: Tier
    reason: str  # e.g., "enter_hi_topn" / "leave_hi_topn"
    issued_step: int
    score_gap: float = 0.0  # For prioritization


class PrecisionScheduler:
    """
    Schedules precision transitions based on hotness scores.
    
    Every T_u steps, computes H_l = top_n(S[l,:], n_hi[l]) and emits
    transition requests for experts that need to change tier.
    """
    
    def __init__(
        self,
        num_layers: int,
        experts_per_layer: list[int] | int,
        n_hi: list[int],
        update_period_steps: int = 200,
        rate_limit: Optional[int] = None,
    ):
        """
        Args:
            num_layers: Number of MoE layers
            experts_per_layer: Expert counts per layer
            n_hi: n_hi[l] per layer (fixed by budget init)
            update_period_steps: T_u - update period
            rate_limit: Optional rate limit R per period
        """
        self.num_layers = num_layers
        if isinstance(experts_per_layer, int):
            self.experts_per_layer = [experts_per_layer] * num_layers
        else:
            self.experts_per_layer = list(experts_per_layer)
        
        if len(n_hi) != num_layers:
            raise ValueError(f"n_hi length {len(n_hi)} != num_layers {num_layers}")
        self.n_hi = list(n_hi)
        
        self.update_period_steps = update_period_steps
        self.rate_limit = rate_limit
        
        # Current tier assignments (for comparison)
        self._current_tiers: dict[ExpertKey, Tier] = {}
        self._last_update_step = -1
    
    def should_update(self, step: int) -> bool:
        """Check if scheduler should update at this step."""
        return step % self.update_period_steps == 0 and step != self._last_update_step
    
    def plan(
        self,
        step: int,
        tracker: HotnessTracker,
        current_tiers: Optional[dict[ExpertKey, Tier]] = None,
    ) -> list[TransitionReq]:
        """
        Plan transitions for current step.
        
        Args:
            step: Current step number
            tracker: HotnessTracker with current scores
            current_tiers: Current tier assignments (if None, uses internal state)
        
        Returns:
            List of TransitionReq
        """
        if not self.should_update(step):
            return []
        
        if current_tiers is not None:
            self._current_tiers = current_tiers.copy()
        
        requests: list[TransitionReq] = []
        
        for layer in range(self.num_layers):
            # Get scores for this layer
            scores = tracker.get_layer_scores(layer)
            n_hi_l = self.n_hi[layer]
            
            if len(scores) == 0 or n_hi_l <= 0:
                continue
            
            # Compute top-n_hi[l] experts
            if n_hi_l >= len(scores):
                # All experts should be HI
                top_experts = set(range(len(scores)))
            else:
                # Get top-n_hi[l] with deterministic tie-breaking
                top_indices = np.argsort(-scores, kind='stable')[:n_hi_l]
                top_experts = set(top_indices.tolist())
            
            # Compare with current assignments and generate requests
            for expert_id in range(self.experts_per_layer[layer]):
                key = ExpertKey(layer=layer, expert=expert_id)
                want_hi = expert_id in top_experts
                current_tier = self._current_tiers.get(key, Tier.LO)
                have_hi = (current_tier == Tier.HI)
                
                if want_hi and not have_hi:
                    # Promotion needed
                    score = scores[expert_id] if expert_id < len(scores) else 0.0
                    # Compute score gap for prioritization
                    if len(top_experts) > 0:
                        boundary_score = min(scores[list(top_experts)])
                        score_gap = score - boundary_score
                    else:
                        score_gap = 0.0
                    
                    requests.append(TransitionReq(
                        key=key,
                        src=Tier.LO,
                        dst=Tier.HI,
                        reason="enter_hi_topn",
                        issued_step=step,
                        score_gap=score_gap,
                    ))
                elif not want_hi and have_hi:
                    # Demotion needed
                    score = scores[expert_id] if expert_id < len(scores) else 0.0
                    # Compute score gap
                    if len(top_experts) > 0:
                        boundary_score = min(scores[list(top_experts)])
                        score_gap = boundary_score - score
                    else:
                        score_gap = 0.0
                    
                    requests.append(TransitionReq(
                        key=key,
                        src=Tier.HI,
                        dst=Tier.LO,
                        reason="leave_hi_topn",
                        issued_step=step,
                        score_gap=score_gap,
                    ))
        
        # Apply rate limiting if configured
        if self.rate_limit is not None and len(requests) > self.rate_limit:
            # Prioritize by score gap
            requests.sort(key=lambda r: r.score_gap, reverse=True)
            requests = requests[:self.rate_limit]
        
        # Update internal state
        for req in requests:
            self._current_tiers[req.key] = req.dst
        
        self._last_update_step = step
        
        return requests

