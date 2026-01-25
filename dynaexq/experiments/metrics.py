"""
Metrics collection for experiments.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LatencyMetrics:
    """Latency metrics for a request."""
    ttft_ms: float  # Time to first token
    tpop_ms: float  # Time per output token
    total_ms: float


class MetricsCollector:
    """
    Collects metrics during experiments.
    
    Tracks:
    - TTFT (Time to First Token)
    - TPOP (Time Per Output Token)
    - Percentiles (P50, P95, P99)
    - Transition statistics
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Args:
            window_size: Size of sliding window for percentiles
        """
        self.window_size = window_size
        self._ttft_samples: deque = deque(maxlen=window_size)
        self._tpop_samples: deque = deque(maxlen=window_size)
        self._total_samples: deque = deque(maxlen=window_size)
        
        self._transition_stats: dict = {
            "promotions": 0,
            "demotions": 0,
            "bytes_moved": 0,
        }
    
    def record_latency(self, metrics: LatencyMetrics) -> None:
        """Record latency metrics for a request."""
        self._ttft_samples.append(metrics.ttft_ms)
        self._tpop_samples.append(metrics.tpop_ms)
        self._total_samples.append(metrics.total_ms)
    
    def record_transition(self, is_promotion: bool, bytes_moved: int) -> None:
        """Record a transition event."""
        if is_promotion:
            self._transition_stats["promotions"] += 1
        else:
            self._transition_stats["demotions"] += 1
        self._transition_stats["bytes_moved"] += bytes_moved
    
    def get_percentiles(self, values: list[float]) -> dict[str, float]:
        """Compute percentiles."""
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        arr = np.array(values)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            "ttft": {
                "mean": float(np.mean(self._ttft_samples)) if self._ttft_samples else 0.0,
                **self.get_percentiles(list(self._ttft_samples)),
            },
            "tpop": {
                "mean": float(np.mean(self._tpop_samples)) if self._tpop_samples else 0.0,
                **self.get_percentiles(list(self._tpop_samples)),
            },
            "total": {
                "mean": float(np.mean(self._total_samples)) if self._total_samples else 0.0,
                **self.get_percentiles(list(self._total_samples)),
            },
            "transitions": self._transition_stats.copy(),
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._ttft_samples.clear()
        self._tpop_samples.clear()
        self._total_samples.clear()
        self._transition_stats = {
            "promotions": 0,
            "demotions": 0,
            "bytes_moved": 0,
        }

