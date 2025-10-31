"""
Core data structures for DynaExQ runtime
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import time


@dataclass(frozen=True, eq=True)
class ExpertID:
    """Unique identifier for an expert"""
    layer: int
    idx: int  # expert index within layer

    def __hash__(self):
        return hash((self.layer, self.idx))

    def __repr__(self):
        return f"Expert(L{self.layer}E{self.idx})"


@dataclass
class Residency:
    """Expert residency and location information"""
    bitwidth: Literal["W4", "W2"]
    location: Literal["HBM", "DRAM", "SSD"]
    hbm_ptr: Optional[int] = None   # device pointer / handle
    dram_ptr: Optional[int] = None  # host memory pointer
    ssd_offset: Optional[int] = None  # SSD file offset
    bytes: int = 0
    last_used_ts: float = field(default_factory=time.time)

    def update_timestamp(self):
        """Update last used timestamp"""
        self.last_used_ts = time.time()


@dataclass
class SwapTask:
    """Task for swapping expert precision/location"""
    expert: ExpertID
    source_residency: Residency
    target_bitwidth: Literal["W4", "W2"]
    target_location: Literal["HBM", "DRAM", "SSD"]
    priority: int = 0  # Higher priority tasks executed first

    def __lt__(self, other):
        return self.priority > other.priority  # Reverse for priority queue


@dataclass
class TelemetryEvent:
    """Telemetry event for monitoring"""
    timestamp: float
    event_type: str  # "upgrade", "downgrade", "evict", "prefetch", "miss"
    expert: ExpertID
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)
