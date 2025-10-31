"""
Telemetry - Metrics collection and monitoring for DynaExQ
"""

import time
import json
import logging
from typing import Dict, List, Optional
from dataclasses import asdict
import threading
from pathlib import Path

from .types import TelemetryEvent

logger = logging.getLogger(__name__)


class TelemetryCollector:
    """
    Collect and export telemetry metrics for DynaExQ runtime.

    Tracks:
    - TTFT (Time to First Token)
    - TPOP (Time per Output Token)
    - Tokens/sec throughput
    - HBM usage
    - Swap latencies
    - Ready-before-use ratio
    - Eviction rate
    - SSD bandwidth
    """

    def __init__(
        self,
        output_file: Optional[str] = None,
        export_format: str = "jsonl",
    ):
        """
        Args:
            output_file: Path to output file (None = no export)
            export_format: "jsonl" or "csv"
        """
        self.output_file = output_file
        self.export_format = export_format

        # Events
        self.events: List[TelemetryEvent] = []
        self.lock = threading.Lock()

        # Aggregated metrics
        self.metrics = {
            "inference": {
                "total_tokens": 0,
                "total_batches": 0,
                "ttft_ms": [],
                "tpop_ms": [],
            },
            "swaps": {
                "upgrades": 0,
                "downgrades": 0,
                "total_latency_ms": 0.0,
                "ready_before_use": 0,
                "misses": 0,
            },
            "memory": {
                "hbm_usage_samples": [],
                "eviction_count": 0,
            },
            "ssd": {
                "reads": 0,
                "writes": 0,
                "total_read_gb": 0.0,
                "total_write_gb": 0.0,
            }
        }

        # File handle
        self.file_handle = None
        if output_file:
            self.file_handle = open(output_file, 'w')
            logger.info(f"Telemetry output: {output_file}")

    def record_event(self, event: TelemetryEvent):
        """Record a telemetry event"""
        with self.lock:
            self.events.append(event)

            # Update aggregated metrics
            if event.event_type == "upgrade":
                self.metrics["swaps"]["upgrades"] += 1
                self.metrics["swaps"]["total_latency_ms"] += event.duration_ms
            elif event.event_type == "downgrade":
                self.metrics["swaps"]["downgrades"] += 1
                self.metrics["swaps"]["total_latency_ms"] += event.duration_ms
            elif event.event_type == "miss":
                self.metrics["swaps"]["misses"] += 1
            elif event.event_type == "prefetch":
                self.metrics["swaps"]["ready_before_use"] += 1

            # Export to file
            if self.file_handle:
                self._export_event(event)

    def record_batch(self, tokens: int, batch_time_ms: float):
        """Record a batch inference"""
        with self.lock:
            self.metrics["inference"]["total_tokens"] += tokens
            self.metrics["inference"]["total_batches"] += 1

            # TPOP (time per output token)
            if tokens > 0:
                tpop = batch_time_ms / tokens
                self.metrics["inference"]["tpop_ms"].append(tpop)

    def record_ttft(self, ttft_ms: float):
        """Record time to first token"""
        with self.lock:
            self.metrics["inference"]["ttft_ms"].append(ttft_ms)

    def record_hbm_usage(self, usage: float):
        """Record HBM usage (0.0 to 1.0)"""
        with self.lock:
            self.metrics["memory"]["hbm_usage_samples"].append(usage)

    def record_eviction(self):
        """Record an eviction event"""
        with self.lock:
            self.metrics["memory"]["eviction_count"] += 1

    def record_ssd_io(self, read_gb: float = 0.0, write_gb: float = 0.0):
        """Record SSD I/O"""
        with self.lock:
            if read_gb > 0:
                self.metrics["ssd"]["reads"] += 1
                self.metrics["ssd"]["total_read_gb"] += read_gb
            if write_gb > 0:
                self.metrics["ssd"]["writes"] += 1
                self.metrics["ssd"]["total_write_gb"] += write_gb

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        with self.lock:
            # Compute aggregates
            total_swaps = (
                self.metrics["swaps"]["upgrades"] +
                self.metrics["swaps"]["downgrades"]
            )

            avg_swap_latency = (
                self.metrics["swaps"]["total_latency_ms"] / total_swaps
                if total_swaps > 0 else 0.0
            )

            ready_ratio = (
                self.metrics["swaps"]["ready_before_use"] / total_swaps
                if total_swaps > 0 else 1.0
            )

            import numpy as np

            return {
                "inference": {
                    "total_tokens": self.metrics["inference"]["total_tokens"],
                    "total_batches": self.metrics["inference"]["total_batches"],
                    "avg_ttft_ms": (
                        np.mean(self.metrics["inference"]["ttft_ms"])
                        if self.metrics["inference"]["ttft_ms"] else 0.0
                    ),
                    "avg_tpop_ms": (
                        np.mean(self.metrics["inference"]["tpop_ms"])
                        if self.metrics["inference"]["tpop_ms"] else 0.0
                    ),
                    "tokens_per_sec": (
                        1000.0 / np.mean(self.metrics["inference"]["tpop_ms"])
                        if self.metrics["inference"]["tpop_ms"] else 0.0
                    ),
                },
                "swaps": {
                    "total_swaps": total_swaps,
                    "upgrades": self.metrics["swaps"]["upgrades"],
                    "downgrades": self.metrics["swaps"]["downgrades"],
                    "avg_latency_ms": avg_swap_latency,
                    "ready_before_use_ratio": ready_ratio,
                    "miss_count": self.metrics["swaps"]["misses"],
                },
                "memory": {
                    "avg_hbm_usage": (
                        np.mean(self.metrics["memory"]["hbm_usage_samples"])
                        if self.metrics["memory"]["hbm_usage_samples"] else 0.0
                    ),
                    "eviction_count": self.metrics["memory"]["eviction_count"],
                },
                "ssd": {
                    "total_reads": self.metrics["ssd"]["reads"],
                    "total_writes": self.metrics["ssd"]["writes"],
                    "total_read_gb": self.metrics["ssd"]["total_read_gb"],
                    "total_write_gb": self.metrics["ssd"]["total_write_gb"],
                },
                "total_events": len(self.events),
            }

    def _export_event(self, event: TelemetryEvent):
        """Export single event to file"""
        if self.export_format == "jsonl":
            line = json.dumps(asdict(event)) + "\n"
            self.file_handle.write(line)
            self.file_handle.flush()

    def export_summary(self, path: str):
        """Export summary to JSON file"""
        summary = self.get_summary()
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Exported telemetry summary to {path}")

    def close(self):
        """Close file handle"""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def __del__(self):
        """Cleanup"""
        self.close()
