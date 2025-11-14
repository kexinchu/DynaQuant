"""
SwapEngine - Async expert swapping with pinned buffers and CUDA streams
"""

import logging
import time
import threading
from typing import Dict, Optional, List, Set, Any
from queue import PriorityQueue, Empty
from dataclasses import dataclass, field
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch not available, SwapEngine will use CPU-only mode")

from .types import ExpertID, Residency, SwapTask, TelemetryEvent
from .memmgr import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class PendingSwap:
    """Track a pending swap operation"""
    expert: ExpertID
    target_bitwidth: str
    target_location: str
    start_time: float = field(default_factory=time.time)
    stage: str = "queued"  # queued, loading, converting, transferring, complete
    event: Optional[threading.Event] = field(default_factory=threading.Event)


class SwapEngine:
    """
    Asynchronously swap experts between precision levels and memory tiers.

    Uses:
    - Pinned host memory for fast DMA
    - Multiple CUDA streams for overlapping transfers
    - Double buffering in transient pool
    - Priority queue for urgent swaps
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        num_h2d_streams: int = 2,
        num_d2h_streams: int = 1,
        enable_cuda: bool = True,
        weight_loader: Optional[Any] = None,
    ):
        """
        Args:
            memory_manager: MemoryManager instance
            num_h2d_streams: Number of host-to-device CUDA streams
            num_d2h_streams: Number of device-to-host CUDA streams
            enable_cuda: Whether to use CUDA streams (False for CPU-only testing)
            weight_loader: Optional ExpertWeightLoader for loading weights from disk
        """
        self.memory_manager = memory_manager
        self.num_h2d_streams = num_h2d_streams
        self.num_d2h_streams = num_d2h_streams
        self.enable_cuda = enable_cuda and TORCH_AVAILABLE
        self.weight_loader = weight_loader

        # Task queue: PriorityQueue of SwapTask
        self.task_queue: PriorityQueue[SwapTask] = PriorityQueue()

        # Track pending swaps
        self.pending_swaps: Dict[ExpertID, PendingSwap] = {}
        self.pending_lock = threading.RLock()

        # Worker thread
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False

        # CUDA streams (if available)
        self.h2d_streams = []
        self.d2h_streams = []
        if self.enable_cuda:
            try:
                for _ in range(num_h2d_streams):
                    self.h2d_streams.append(torch.cuda.Stream())
                for _ in range(num_d2h_streams):
                    self.d2h_streams.append(torch.cuda.Stream())
                logger.info(
                    f"SwapEngine initialized with CUDA: "
                    f"{num_h2d_streams} H2D streams, {num_d2h_streams} D2H streams"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create CUDA streams: {e}, falling back to CPU mode")
                self.enable_cuda = False

        if not self.enable_cuda:
            logger.info("SwapEngine initialized in CPU-only mode")

        # Telemetry
        self.telemetry: List[TelemetryEvent] = []
        self.telemetry_lock = threading.Lock()

        # Statistics
        self.upgrade_count = 0
        self.downgrade_count = 0
        self.ready_before_use = 0
        self.miss_count = 0

        # Pinned memory buffers (simplified simulation)
        self.pinned_buffers = {}

    def start(self):
        """Start the background worker thread"""
        if self.running:
            logger.warning("SwapEngine already running")
            return

        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("SwapEngine worker thread started")

    def stop(self):
        """Stop the background worker thread"""
        if not self.running:
            return

        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("SwapEngine worker thread stopped")

    def upgrade(self, expert: ExpertID, priority: int = 0):
        """
        Upgrade expert from W2 to W4 (bring into HBM hot pool).
        Non-blocking - queues the task and returns immediately.

        Args:
            expert: Expert to upgrade
            priority: Task priority (higher = more urgent)
        """
        with self.pending_lock:
            if expert in self.pending_swaps:
                logger.debug(f"{expert} upgrade already pending")
                return

            residency = self.memory_manager.get_residency(expert)
            if residency and residency.bitwidth == "W4":
                logger.debug(f"{expert} already W4")
                return

            # Create swap task
            task = SwapTask(
                expert=expert,
                source_residency=residency,
                target_bitwidth="W4",
                target_location="HBM",
                priority=priority
            )

            # Track pending
            self.pending_swaps[expert] = PendingSwap(
                expert=expert,
                target_bitwidth="W4",
                target_location="HBM"
            )

            self.task_queue.put(task)
            self.upgrade_count += 1

            logger.debug(f"Queued upgrade for {expert} (priority={priority})")

    def downgrade(self, expert: ExpertID, priority: int = 0):
        """
        Downgrade expert from W4 to W2.
        Non-blocking - queues the task and returns immediately.
        """
        with self.pending_lock:
            if expert in self.pending_swaps:
                logger.debug(f"{expert} downgrade already pending")
                return

            residency = self.memory_manager.get_residency(expert)
            if residency and residency.bitwidth == "W2":
                logger.debug(f"{expert} already W2")
                return

            task = SwapTask(
                expert=expert,
                source_residency=residency,
                target_bitwidth="W2",
                target_location="HBM",  # Can be DRAM/SSD depending on policy
                priority=priority
            )

            self.pending_swaps[expert] = PendingSwap(
                expert=expert,
                target_bitwidth="W2",
                target_location="HBM"
            )

            self.task_queue.put(task)
            self.downgrade_count += 1

            logger.debug(
                f"Queued downgrade for {expert} (priority={priority})")

    def wait_ready(self, expert: ExpertID, timeout: float = 5.0) -> bool:
        """
        Wait for expert to be ready (swap complete).

        Args:
            expert: Expert to wait for
            timeout: Maximum wait time in seconds

        Returns:
            True if ready, False if timeout
        """
        start_time = time.time()

        with self.pending_lock:
            pending = self.pending_swaps.get(expert)
            if not pending:
                # Not pending, check if already in target state
                residency = self.memory_manager.get_residency(expert)
                if residency and residency.location == "HBM":
                    self.ready_before_use += 1
                    return True
                # Not in HBM yet, but not pending - may need to initiate swap
                logger.warning(f"{expert} not in HBM and no swap pending")
                self.miss_count += 1
                return False

        # Wait for completion event
        if pending.event.wait(timeout):
            elapsed = time.time() - start_time
            logger.debug(f"{expert} ready after {elapsed*1000:.1f}ms")
            self.ready_before_use += 1
            return True
        else:
            logger.warning(f"{expert} not ready after {timeout}s timeout")
            self.miss_count += 1
            return False

    def is_ready(self, expert: ExpertID) -> bool:
        """Check if expert is ready (in HBM) without blocking"""
        residency = self.memory_manager.get_residency(expert)
        return residency is not None and residency.location == "HBM"

    def _worker_loop(self):
        """Background worker thread that processes swap tasks"""
        logger.info("SwapEngine worker loop started")

        while self.running:
            try:
                task = self.task_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                self._process_swap(task)
            except Exception as e:
                if self.running:
                    logger.error(
                        f"Error processing swap task: {e}", exc_info=True)

        logger.info("SwapEngine worker loop exited")

    def _process_swap(self, task: SwapTask):
        """Process a single swap task"""
        expert = task.expert
        start_time = time.time()

        with self.pending_lock:
            pending = self.pending_swaps.get(expert)
            if pending:
                pending.stage = "loading"

        logger.debug(f"Processing swap for {expert}: {task.target_bitwidth}")

        try:
            if task.target_bitwidth == "W4":
                self._do_upgrade(task)
            else:
                self._do_downgrade(task)

            # Mark complete
            with self.pending_lock:
                if expert in self.pending_swaps:
                    pending = self.pending_swaps[expert]
                    pending.stage = "complete"
                    pending.event.set()
                    del self.pending_swaps[expert]

            # Record telemetry
            duration_ms = (time.time() - start_time) * 1000
            event = TelemetryEvent(
                timestamp=time.time(),
                event_type="upgrade" if task.target_bitwidth == "W4" else "downgrade",
                expert=expert,
                duration_ms=duration_ms
            )

            with self.telemetry_lock:
                self.telemetry.append(event)

            logger.debug(f"Completed swap for {expert} in {duration_ms:.1f}ms")

        except Exception as e:
            logger.error(f"Failed to swap {expert}: {e}", exc_info=True)
            with self.pending_lock:
                if expert in self.pending_swaps:
                    self.pending_swaps[expert].event.set()
                    del self.pending_swaps[expert]

    def _do_upgrade(self, task: SwapTask):
        """Execute upgrade: W2 -> W4, load into hot pool"""
        expert = task.expert

        # Load W4 weights from disk (if weight_loader is available)
        weight_loaded = False
        if self.weight_loader is not None:
            try:
                weight = self.weight_loader.load_expert_weights(
                    expert.layer, expert.idx, "W4"
                )
                if weight is not None:
                    weight_loaded = True
                    logger.debug(f"Loaded W4 weights for {expert} from disk")
            except Exception as e:
                logger.warning(f"Failed to load W4 weights for {expert}: {e}")

        if not weight_loaded:
            # Simulate loading W4 weights from DRAM/SSD
            # In real implementation: load from storage, dequantize if needed, transfer to GPU
            time.sleep(0.001)  # Simulate I/O latency

        # Reserve space in hot pool
        w4_size = self.memory_manager.w4_size
        if not self.memory_manager.reserve_hot(expert, w4_size):
            logger.warning(f"Failed to reserve hot pool space for {expert}")
            return

        # Simulate CUDA transfer (in real implementation, use pinned memory and streams)
        if self.enable_cuda:
            # Use one of the H2D streams
            stream_idx = expert.layer % len(self.h2d_streams)
            stream = self.h2d_streams[stream_idx]

            # Simulate async transfer
            time.sleep(0.0005)  # Simulate transfer time

        # Update residency
        residency = Residency(
            bitwidth="W4",
            location="HBM",
            bytes=w4_size
        )
        self.memory_manager.place(expert, residency)

    def _do_downgrade(self, task: SwapTask):
        """Execute downgrade: W4 -> W2, move to cold pool or DRAM"""
        expert = task.expert

        # Load W2 weights from disk (if weight_loader is available)
        weight_loaded = False
        if self.weight_loader is not None:
            try:
                weight = self.weight_loader.load_expert_weights(
                    expert.layer, expert.idx, "W2"
                )
                if weight is not None:
                    weight_loaded = True
                    logger.debug(f"Loaded W2 weights for {expert} from disk")
            except Exception as e:
                logger.warning(f"Failed to load W2 weights for {expert}: {e}")

        if not weight_loaded:
            # Simulate quantizing W4 -> W2 (can be done on GPU or CPU)
            time.sleep(0.0005)  # Simulate conversion

        # Reserve space in cold pool or move to DRAM
        w2_size = self.memory_manager.w2_size
        if self.memory_manager.reserve_cold(expert, w2_size):
            location = "HBM"
        else:
            # Cold pool full, move to DRAM
            location = "DRAM"

        residency = Residency(
            bitwidth="W2",
            location=location,
            bytes=w2_size
        )
        self.memory_manager.place(expert, residency)

    def get_statistics(self) -> Dict:
        """Get swap engine statistics"""
        with self.pending_lock:
            pending_count = len(self.pending_swaps)

        total_swaps = self.upgrade_count + self.downgrade_count
        ready_ratio = (
            self.ready_before_use / total_swaps
            if total_swaps > 0 else 1.0
        )

        return {
            "upgrade_count": self.upgrade_count,
            "downgrade_count": self.downgrade_count,
            "pending_swaps": pending_count,
            "ready_before_use": self.ready_before_use,
            "miss_count": self.miss_count,
            "ready_ratio": ready_ratio,
            "telemetry_events": len(self.telemetry),
        }

    def get_telemetry(self) -> List[TelemetryEvent]:
        """Get telemetry events"""
        with self.telemetry_lock:
            return list(self.telemetry)

    def clear_telemetry(self):
        """Clear telemetry history"""
        with self.telemetry_lock:
            self.telemetry.clear()
