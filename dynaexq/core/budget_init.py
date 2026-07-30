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
    resident_budget: int
    runtime_workspace_bytes: int = 0

    @property
    def total_reserved_bytes(self) -> int:
        """Resident expert bytes plus reserved in-flight headroom."""
        return self.total_expert_bytes + self.transient_bytes


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
        reserve_runtime_bytes: int = 0,
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
            reserve_runtime_bytes: Reserved for kernel conversion workspaces
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
        self.reserve_runtime_bytes = reserve_runtime_bytes
        self.safety_margin_bytes = safety_margin_bytes
        self.max_inflight = max_inflight

    def compute(
        self,
        strategy: str = "proportional",
        *,
        high_precision_ratio: float | None = None,
    ) -> BudgetResult:
        """
        Compute n_hi[l] allocation.
        
        Args:
            strategy: Allocation strategy ("proportional", "uniform", "greedy")
            high_precision_ratio: Optional exact experimental quota in
                ``[0, 1]``. Each layer receives
                ``floor(E_l * ratio)`` HI slots. Unlike the automatic
                strategies, an infeasible requested ratio fails rather than
                silently reducing it.
        
        Returns:
            BudgetResult with n_hi[l] and pool sizes
        """
        # Compute available memory for experts
        M_exp = (
            self.device_mem_bytes
            - self.reserve_kv_bytes
            - self.reserve_act_bytes
            - self.reserve_dense_bytes
            - self.reserve_runtime_bytes
            - self.safety_margin_bytes
        )
        
        if M_exp <= 0:
            raise ValueError("No memory available for experts after reservations")
        
        # Compute memory footprints.
        m_hi = [self.memory_footprint_fn(l, Tier.HI) for l in range(self.num_layers)]
        m_lo = [self.memory_footprint_fn(l, Tier.LO) for l in range(self.num_layers)]
        if any(size <= 0 for size in (*m_hi, *m_lo)):
            raise ValueError("expert memory footprints must be positive")
        if any(hi < lo for hi, lo in zip(m_hi, m_lo)):
            raise ValueError("HI precision footprint cannot be smaller than LO")

        # During publish-after-copy the old representation remains live while
        # one destination representation is in flight. Reserve this headroom
        # *before* solving the resident allocation; otherwise a resident set
        # that exactly fills M_exp can still OOM during a transition.
        max_transition_bytes = max(
            max(m_hi[layer], m_lo[layer]) for layer in range(self.num_layers)
        )
        transient_bytes = self.max_inflight * max_transition_bytes
        resident_budget = M_exp - transient_bytes
        if resident_budget <= 0:
            raise ValueError(
                "No resident expert memory remains after transient reservation"
            )

        total_lo = sum(
            self.experts_per_layer[layer] * m_lo[layer]
            for layer in range(self.num_layers)
        )
        if total_lo > resident_budget:
            raise ValueError(
                "All-LO expert footprint exceeds the resident expert budget: "
                f"required={total_lo}, available={resident_budget}"
            )

        # Compute n_hi[l] based on strategy or an exact sensitivity quota.
        if high_precision_ratio is not None:
            if not 0.0 <= high_precision_ratio <= 1.0:
                raise ValueError("high_precision_ratio must be in [0, 1]")
            n_hi = [
                int(experts * high_precision_ratio)
                for experts in self.experts_per_layer
            ]
        elif strategy == "uniform":
            n_hi = self._uniform_allocation(resident_budget, m_hi, m_lo)
        elif strategy == "proportional":
            n_hi = self._proportional_allocation(resident_budget, m_hi, m_lo)
        elif strategy == "greedy":
            n_hi = self._greedy_allocation(resident_budget, m_hi, m_lo)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Verify feasibility
        total_bytes = sum(
            n_hi[l] * m_hi[l] + (self.experts_per_layer[l] - n_hi[l]) * m_lo[l]
            for l in range(self.num_layers)
        )
        
        if total_bytes > resident_budget and high_precision_ratio is not None:
            raise ValueError(
                "requested high-precision ratio exceeds the resident expert "
                f"budget: ratio={high_precision_ratio}, "
                f"required={total_bytes}, available={resident_budget}"
            )
        if total_bytes > resident_budget:
            # Reduce allocation greedily if needed
            n_hi = self._greedy_reduce(n_hi, resident_budget, m_hi, m_lo)
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
        
        if total_bytes + transient_bytes > M_exp:
            raise AssertionError(
                "budget construction violated total HBM invariant: "
                f"resident={total_bytes}, transient={transient_bytes}, cap={M_exp}"
            )
        
        return BudgetResult(
            n_hi=n_hi,
            hi_pool_bytes=hi_pool_bytes,
            lo_pool_bytes=lo_pool_bytes,
            transient_bytes=transient_bytes,
            total_expert_bytes=total_bytes,
            available_memory=M_exp,
            resident_budget=resident_budget,
            runtime_workspace_bytes=self.reserve_runtime_bytes,
        )

    def _uniform_allocation(
        self, M_exp: int, m_hi: list[int], m_lo: list[int]
    ) -> list[int]:
        """Uniform allocation: same n_hi for all layers."""
        # Binary search the largest common per-layer quota. Layers with fewer
        # experts saturate at their own count.
        low = 0
        high = max(self.experts_per_layer)
        best = [0] * self.num_layers
        while low <= high:
            quota = (low + high) // 2
            candidate = [
                min(quota, self.experts_per_layer[layer])
                for layer in range(self.num_layers)
            ]
            total = sum(
                candidate[layer] * m_hi[layer]
                + (self.experts_per_layer[layer] - candidate[layer]) * m_lo[layer]
                for layer in range(self.num_layers)
            )
            if total <= M_exp:
                best = candidate
                low = quota + 1
            else:
                high = quota - 1
        return best

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
        total_lo = sum(
            self.experts_per_layer[layer] * m_lo[layer]
            for layer in range(self.num_layers)
        )
        remaining = M_exp - total_lo
        
        # Greedily allocate to layers with most savings
        # (This is simplified; a more sophisticated approach would consider
        #  the actual memory cost of each allocation)
        while remaining > 0:
            best_layer = None
            best_cost = float('inf')
            
            for l in range(self.num_layers):
                if n_hi[l] >= self.experts_per_layer[l]:
                    continue
                
                cost = m_hi[l] - m_lo[l]  # Additional bytes for one upgrade.
                if cost <= remaining and cost < best_cost:
                    best_layer = l
                    best_cost = cost
            
            if best_layer is None:
                break

            if best_cost == 0:
                # Upgrade every zero-cost expert at once to avoid an
                # unproductive loop.
                for layer in range(self.num_layers):
                    if m_hi[layer] == m_lo[layer]:
                        n_hi[layer] = self.experts_per_layer[layer]
                continue
            n_hi[best_layer] += 1
            remaining -= int(best_cost)
        
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
