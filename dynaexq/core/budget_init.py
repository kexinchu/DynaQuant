"""
BudgetInitializer: computes n_hi[l] and memory pool sizes.

Implements budget feasibility: Σ_l (n_hi[l]*m_l(HI) + (E_l - n_hi[l])*m_l(LO)) ≤ M_exp
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .config import Tier


@dataclass
class BudgetResult:
    """Result of budget initialization."""
    n_hi: list[int]  # n_hi[l] per layer
    hi_pool_bytes: int
    lo_pool_bytes: int
    transient_bytes: int
    total_expert_bytes: int
    available_memory: int


class BudgetInitializer:
    """
    Initializes budget allocation for expert memory pools.
    
    Computes n_hi[l] such that total expert memory fits within budget:
    Σ_l (n_hi[l]*m_l(HI) + (E_l - n_hi[l])*m_l(LO)) ≤ M_exp
    """

    def __init__(
        self,
        num_layers: int,
        experts_per_layer: list[int] | int,
        memory_footprint_fn: Callable[[int, Tier], int],
        device_mem_bytes: int,
        reserve_kv_bytes: int = 0,
        reserve_act_bytes: int = 0,
        reserve_dense_bytes: int = 0,
        safety_margin_bytes: int = 0,
        max_inflight: int = 4,
    ):
        """
        Args:
            num_layers: Number of MoE layers
            experts_per_layer: Expert counts per layer (or single int if uniform)
            memory_footprint_fn: Function (layer, tier) -> bytes per expert
            device_mem_bytes: Total device memory
            reserve_kv_bytes: Reserved for KV cache
            reserve_act_bytes: Reserved for activations
            reserve_dense_bytes: Reserved for dense weights
            safety_margin_bytes: Safety margin
            max_inflight: Max concurrent transitions (for transient buffer sizing)
        """
        self.num_layers = num_layers
        if isinstance(experts_per_layer, int):
            self.experts_per_layer = [experts_per_layer] * num_layers
        else:
            if len(experts_per_layer) != num_layers:
                raise ValueError(
                    f"experts_per_layer length {len(experts_per_layer)} != num_layers {num_layers}"
                )
            self.experts_per_layer = list(experts_per_layer)
        
        self.memory_footprint_fn = memory_footprint_fn
        self.device_mem_bytes = device_mem_bytes
        self.reserve_kv_bytes = reserve_kv_bytes
        self.reserve_act_bytes = reserve_act_bytes
        self.reserve_dense_bytes = reserve_dense_bytes
        self.safety_margin_bytes = safety_margin_bytes
        self.max_inflight = max_inflight

    def compute(self, strategy: str = "proportional") -> BudgetResult:
        """
        Compute n_hi[l] allocation.
        
        Args:
            strategy: Allocation strategy ("proportional", "uniform", "greedy")
        
        Returns:
            BudgetResult with n_hi[l] and pool sizes
        """
        # Compute available memory for experts
        M_exp = (
            self.device_mem_bytes
            - self.reserve_kv_bytes
            - self.reserve_act_bytes
            - self.reserve_dense_bytes
            - self.safety_margin_bytes
        )
        
        if M_exp <= 0:
            raise ValueError("No memory available for experts after reservations")
        
        # Compute memory footprints
        m_hi = [self.memory_footprint_fn(l, Tier.HI) for l in range(self.num_layers)]
        m_lo = [self.memory_footprint_fn(l, Tier.LO) for l in range(self.num_layers)]
        
        # Compute n_hi[l] based on strategy
        if strategy == "uniform":
            n_hi = self._uniform_allocation(M_exp, m_hi, m_lo)
        elif strategy == "proportional":
            n_hi = self._proportional_allocation(M_exp, m_hi, m_lo)
        elif strategy == "greedy":
            n_hi = self._greedy_allocation(M_exp, m_hi, m_lo)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Verify feasibility
        total_bytes = sum(
            n_hi[l] * m_hi[l] + (self.experts_per_layer[l] - n_hi[l]) * m_lo[l]
            for l in range(self.num_layers)
        )
        
        if total_bytes > M_exp:
            # Reduce allocation greedily if needed
            n_hi = self._greedy_reduce(n_hi, M_exp, m_hi, m_lo)
            total_bytes = sum(
                n_hi[l] * m_hi[l] + (self.experts_per_layer[l] - n_hi[l]) * m_lo[l]
                for l in range(self.num_layers)
            )
        
        # Compute pool sizes
        hi_pool_bytes = sum(n_hi[l] * m_hi[l] for l in range(self.num_layers))
        lo_pool_bytes = sum(
            (self.experts_per_layer[l] - n_hi[l]) * m_lo[l]
            for l in range(self.num_layers)
        )
        
        # Transient buffer: max_inflight transitions, each needs both HI and LO temporarily
        max_transient_per_transition = max(
            m_hi[l] + m_lo[l] for l in range(self.num_layers)
        )
        transient_bytes = self.max_inflight * max_transient_per_transition
        
        return BudgetResult(
            n_hi=n_hi,
            hi_pool_bytes=hi_pool_bytes,
            lo_pool_bytes=lo_pool_bytes,
            transient_bytes=transient_bytes,
            total_expert_bytes=total_bytes,
            available_memory=M_exp,
        )

    def _uniform_allocation(
        self, M_exp: int, m_hi: list[int], m_lo: list[int]
    ) -> list[int]:
        """Uniform allocation: same n_hi for all layers."""
        # Try to allocate equal number of HI experts per layer
        # This is a simplified approach
        avg_experts_per_layer = sum(self.experts_per_layer) / self.num_layers
        n_hi_per_layer = int(avg_experts_per_layer * 0.1)  # Start with 10%
        
        # Verify feasibility
        total = sum(
            n_hi_per_layer * m_hi[l] + (self.experts_per_layer[l] - n_hi_per_layer) * m_lo[l]
            for l in range(self.num_layers)
        )
        
        if total > M_exp:
            # Reduce uniformly
            scale = M_exp / total
            n_hi_per_layer = max(1, int(n_hi_per_layer * scale))
        
        return [min(n_hi_per_layer, self.experts_per_layer[l]) for l in range(self.num_layers)]

    def _proportional_allocation(
        self, M_exp: int, m_hi: list[int], m_lo: list[int]
    ) -> list[int]:
        """Proportional allocation: allocate based on layer size."""
        # Allocate proportionally to expert count
        total_experts = sum(self.experts_per_layer)
        if total_experts == 0:
            return [0] * self.num_layers
        
        # First, compute if all experts in LO tier fit
        total_lo = sum(self.experts_per_layer[l] * m_lo[l] for l in range(self.num_layers))
        if total_lo > M_exp:
            # Even all LO doesn't fit - return zeros (infeasible)
            return [0] * self.num_layers
        
        # Binary search for optimal hi ratio that fits budget
        # We want to maximize n_hi while staying within budget
        min_ratio = 0.0
        max_ratio = 1.0
        best_n_hi = [0] * self.num_layers
        
        for _ in range(20):  # 20 iterations should be enough
            ratio = (min_ratio + max_ratio) / 2.0
            n_hi = [
                max(0, int(self.experts_per_layer[l] * ratio))
                for l in range(self.num_layers)
            ]
            
            total = sum(
                n_hi[l] * m_hi[l] + (self.experts_per_layer[l] - n_hi[l]) * m_lo[l]
                for l in range(self.num_layers)
            )
            
            if total <= M_exp:
                best_n_hi = n_hi
                min_ratio = ratio
            else:
                max_ratio = ratio
        
        # Ensure n_hi[l] <= E_l
        return [
            min(best_n_hi[l], self.experts_per_layer[l]) for l in range(self.num_layers)
        ]

    def _greedy_allocation(
        self, M_exp: int, m_hi: list[int], m_lo: list[int]
    ) -> list[int]:
        """Greedy allocation: maximize HI experts within budget."""
        n_hi = [0] * self.num_layers
        remaining = M_exp
        
        # Compute savings per expert upgrade (HI vs LO)
        savings = [
            m_lo[l] - m_hi[l] for l in range(self.num_layers)
        ]
        
        # Greedily allocate to layers with most savings
        # (This is simplified; a more sophisticated approach would consider
        #  the actual memory cost of each allocation)
        while remaining > 0:
            best_layer = None
            best_cost = float('inf')
            
            for l in range(self.num_layers):
                if n_hi[l] >= self.experts_per_layer[l]:
                    continue
                
                cost = m_hi[l] - m_lo[l]  # Additional cost to upgrade one expert
                if cost <= remaining and cost < best_cost:
                    best_layer = l
                    best_cost = cost
            
            if best_layer is None:
                break
            
            n_hi[best_layer] += 1
            remaining -= best_cost
        
        return n_hi

    def _greedy_reduce(
        self, n_hi: list[int], M_exp: int, m_hi: list[int], m_lo: list[int]
    ) -> list[int]:
        """Reduce allocation greedily to fit budget."""
        n_hi = n_hi.copy()
        total = sum(
            n_hi[l] * m_hi[l] + (self.experts_per_layer[l] - n_hi[l]) * m_lo[l]
            for l in range(self.num_layers)
        )
        
        while total > M_exp:
            # Find layer with smallest savings from downgrading
            best_layer = None
            best_savings = 0
            
            for l in range(self.num_layers):
                if n_hi[l] <= 0:
                    continue
                
                savings = m_hi[l] - m_lo[l]
                if savings > best_savings:
                    best_layer = l
                    best_savings = savings
            
            if best_layer is None:
                break
            
            n_hi[best_layer] -= 1
            total -= best_savings
        
        return n_hi

