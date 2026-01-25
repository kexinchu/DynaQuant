"""
Core modules for DynaExq implementation following the guide specification.

Modules:
    router_observer: Extracts g[l,e] signals from router outputs
    hotness_tracker: Maintains S[l,e] EMA scores
    budget_init: Computes n_hi[l] and memory pool sizes
    scheduler: Top-n projection and transition request generation
    registry: Atomic expert handle management
    memory_pool: Deterministic pool-based allocation
    transition_engine: Non-blocking promotion/demotion pipeline
    config: Configuration management
"""

from .router_observer import RouterObserver
from .hotness_tracker import HotnessTracker
from .budget_init import BudgetInitializer
from .scheduler import PrecisionScheduler, TransitionReq
from .registry import ExpertRegistry, ExpertHandle, ExpertKey
from .memory_pool import MemoryPool, PoolAllocator
from .transition_engine import TransitionEngine
from .weight_store import ModelWeightStore
from .config import DynaExqConfig, Tier

__all__ = [
    "RouterObserver",
    "HotnessTracker",
    "BudgetInitializer",
    "PrecisionScheduler",
    "ExpertRegistry",
    "ExpertHandle",
    "ExpertKey",
    "MemoryPool",
    "PoolAllocator",
    "TransitionEngine",
    "TransitionReq",
    "ModelWeightStore",
    "DynaExqConfig",
    "Tier",
]

