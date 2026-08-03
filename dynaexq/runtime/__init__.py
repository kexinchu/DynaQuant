"""
Runtime components for DynaExQ.

Modules:
    types: Shared dataclasses for expert identity and residency metadata.
    monitor: EWMA-based expert hotness tracking.
    controller: Precision controller with hysteresis and pool limits.
    memmgr: Tiered memory pool manager with Hot/Cold/Transient partitions.
    swap_engine: Asynchronous upgrade/downgrade orchestration.
    prefetch: Lookahead-based prefetch planner.
    telemetry: Lightweight instrumentation helpers.
"""

from .types import ExpertID, Residency, ResidencyLocation, Bitwidth
from .monitor import ExpertMonitor
from .controller import PrecisionController
from .memmgr import MemoryManager
from .swap_engine import SwapConfig, SwapEngine
from .prefetch import PrefetchPlanner
from .telemetry import TelemetryEvent, TelemetrySink
from .weights import (
    DualPrecisionWeights,
    ExpertTensorBundle,
    InMemoryWeightStore,
    ParameterIndex,
    random_precision_selector,
)

__all__ = [
    "Bitwidth",
    "ExpertID",
    "ExpertMonitor",
    "MemoryManager",
    "PrecisionController",
    "PrefetchPlanner",
    "Residency",
    "ResidencyLocation",
    "SwapConfig",
    "SwapEngine",
    "DualPrecisionWeights",
    "ExpertTensorBundle",
    "InMemoryWeightStore",
    "ParameterIndex",
    "random_precision_selector",
    "TelemetryEvent",
    "TelemetrySink",
]
