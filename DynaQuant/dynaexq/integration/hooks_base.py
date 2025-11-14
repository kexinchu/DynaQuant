"""
Base integration hooks for inference frameworks
"""

import logging
import numpy as np
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

from ..runtime.monitor import ExpertMonitor
from ..runtime.controller import PrecisionController
from ..runtime.memmgr import MemoryManager
from ..runtime.swap_engine import SwapEngine
from ..runtime.prefetch import PrefetchPlanner
from ..runtime.telemetry import TelemetryCollector
from ..runtime.types import ExpertID

logger = logging.getLogger(__name__)


class InferenceHook(ABC):
    """
    Base class for integrating DynaExQ with inference frameworks.

    Subclass this for specific frameworks (SGLang, DeepSpeed, vLLM, etc.)
    """

    def __init__(
        self,
        monitor: ExpertMonitor,
        controller: PrecisionController,
        memory_manager: MemoryManager,
        swap_engine: SwapEngine,
        prefetch_planner: PrefetchPlanner,
        telemetry: Optional[TelemetryCollector] = None,
    ):
        self.monitor = monitor
        self.controller = controller
        self.memory_manager = memory_manager
        self.swap_engine = swap_engine
        self.prefetch_planner = prefetch_planner
        self.telemetry = telemetry

        logger.info(f"InferenceHook initialized: {self.__class__.__name__}")

    @abstractmethod
    def on_forward_start(self, layer_id: int, batch_size: int):
        """
        Called at the start of each MoE layer forward pass.

        Args:
            layer_id: Current layer index
            batch_size: Batch size
        """
        pass

    @abstractmethod
    def on_router_output(
        self,
        layer_id: int,
        topk_indices: np.ndarray,
        logits: Optional[np.ndarray] = None
    ):
        """
        Called after router computes top-k experts.

        Args:
            layer_id: Current layer index
            topk_indices: Array of shape (batch_size, k) with selected expert indices
            logits: Optional router logits/weights
        """
        pass

    @abstractmethod
    def on_forward_end(self, layer_id: int):
        """
        Called at the end of each MoE layer forward pass.

        Args:
            layer_id: Current layer index
        """
        pass

    def get_active_experts(
        self,
        layer_id: int,
        topk_indices: np.ndarray
    ) -> List[ExpertID]:
        """
        Convert top-k indices to ExpertID list.

        Args:
            layer_id: Layer index
            topk_indices: Array of shape (batch_size, k)

        Returns:
            List of unique ExpertIDs
        """
        unique_indices = np.unique(topk_indices.flatten())
        return [ExpertID(layer=layer_id, idx=int(idx)) for idx in unique_indices]

    def plan_and_swap(
        self,
        layer_id: int,
        active_experts: List[ExpertID]
    ) -> Dict[ExpertID, str]:
        """
        Plan precision targets and trigger swaps.

        Args:
            layer_id: Current layer
            active_experts: Active experts for this layer

        Returns:
            Target precision map
        """
        # Get target precision
        targets = self.controller.plan(active_experts, self.monitor)

        # Get current residency
        current = {
            e: self.memory_manager.get_residency(e).bitwidth
            if self.memory_manager.get_residency(e) else "W2"
            for e in active_experts
        }

        # Compute diff
        diff = self.controller.get_diff(targets, current)

        # Launch swaps
        for expert in diff["upgrades"]:
            self.swap_engine.upgrade(expert, priority=5)

        for expert in diff["downgrades"]:
            self.swap_engine.downgrade(expert, priority=1)

        return targets

    def wait_for_experts(self, experts: List[ExpertID], timeout: float = 5.0):
        """
        Ensure all experts are ready before use.

        Args:
            experts: List of experts to wait for
            timeout: Maximum wait time per expert
        """
        for expert in experts:
            ready = self.swap_engine.wait_ready(expert, timeout=timeout)

            if not ready:
                logger.warning(f"{expert} not ready, using fallback")
                # Telemetry
                if self.telemetry:
                    from ..runtime.types import TelemetryEvent
                    import time
                    self.telemetry.record_event(TelemetryEvent(
                        timestamp=time.time(),
                        event_type="miss",
                        expert=expert
                    ))


class DynaExQRuntime:
    """
    Complete DynaExQ runtime orchestrator.

    Example usage:
        runtime = DynaExQRuntime(config)
        runtime.start()

        # In inference loop:
        for layer in layers:
            runtime.on_layer_start(layer.id)
            topk = router(layer)
            runtime.on_router_output(layer.id, topk, logits)
            runtime.ensure_experts_ready(layer.id, topk)
            output = layer(input, topk)
            runtime.on_layer_end(layer.id)

        runtime.stop()
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize runtime with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Create components
        self.monitor = ExpertMonitor(
            ewma_alpha=config.get("ewma_alpha", 0.2),
            epoch_duration=config.get("epoch_duration", 300.0),
            num_layers=config.get("num_layers", 32),
            num_experts_per_layer=config.get("num_experts_per_layer", 64),
        )

        self.controller = PrecisionController(
            tau_h=config.get("tau_h", 0.65),
            tau_c=config.get("tau_c", 0.45),
            max_w4_slots=config.get("max_w4_slots", 16),
            num_layers=config.get("num_layers", 32),
            num_experts_per_layer=config.get("num_experts_per_layer", 64),
        )

        self.memory_manager = MemoryManager(
            hot_pool_gb=config.get("hot_pool_gb", 10.0),
            cold_pool_gb=config.get("cold_pool_gb", 5.0),
            transient_pool_mb=config.get("transient_pool_mb", 2048.0),
        )

        self.swap_engine = SwapEngine(
            memory_manager=self.memory_manager,
            num_h2d_streams=config.get("num_h2d_streams", 2),
            num_d2h_streams=config.get("num_d2h_streams", 1),
            # Optional weight loader
            weight_loader=config.get("weight_loader"),
        )

        self.prefetch_planner = PrefetchPlanner(
            swap_engine=self.swap_engine,
            monitor=self.monitor,
            num_layers=config.get("num_layers", 32),
            lookahead_layers=config.get("lookahead_layers", 1),
            prefetch_top_k=config.get("prefetch_top_k", 8),
        )

        self.telemetry = TelemetryCollector(
            output_file=config.get("telemetry_file"),
        )

        logger.info("DynaExQRuntime initialized")

    def start(self):
        """Start the runtime"""
        self.swap_engine.start()
        logger.info("DynaExQ runtime started")

    def stop(self):
        """Stop the runtime"""
        self.swap_engine.stop()
        logger.info("DynaExQ runtime stopped")

    def on_layer_start(self, layer_id: int):
        """Called at layer start"""
        pass

    def on_router_output(
        self,
        layer_id: int,
        topk_indices: np.ndarray,
        logits: Optional[np.ndarray] = None
    ):
        """Called after router output"""
        # Update monitor
        self.monitor.update_batch(layer_id, topk_indices, logits)

        # Get active experts
        active_experts = [
            ExpertID(layer=layer_id, idx=int(idx))
            for idx in np.unique(topk_indices.flatten())
        ]

        # Plan and swap
        residencies = {
            e: self.memory_manager.get_residency(e) for e in active_experts
        }
        targets = self.controller.plan(active_experts, self.monitor)
        current = {
            e: residencies[e].bitwidth if residencies[e] else "W2"
            for e in active_experts
        }
        diff = self.controller.get_diff(targets, current)

        for expert in diff["upgrades"]:
            self.swap_engine.upgrade(expert, priority=5)
        for expert in diff["downgrades"]:
            self.swap_engine.downgrade(expert, priority=1)

        # Prefetch next layer
        self.prefetch_planner.lookahead(layer_id)
        self.prefetch_planner.update_pattern(layer_id, active_experts)

        # Ensure W2 experts are at least resident in cold pool
        for expert in active_experts:
            if residencies[expert] is None and targets.get(expert, "W2") == "W2":
                self.swap_engine.downgrade(expert, priority=2)

    def ensure_experts_ready(
        self,
        layer_id: int,
        topk_indices: np.ndarray,
        timeout: float = 5.0
    ):
        """Ensure experts are ready before use"""
        active_experts = [
            ExpertID(layer=layer_id, idx=int(idx))
            for idx in np.unique(topk_indices.flatten())
        ]

        for expert in active_experts:
            ready = self.swap_engine.wait_ready(expert, timeout=timeout)
            if not ready:
                logger.warning(f"{expert} not ready")

    def on_layer_end(self, layer_id: int):
        """Called at layer end"""
        # Record HBM usage
        hbm_pressure = self.memory_manager.get_hbm_pressure()
        self.telemetry.record_hbm_usage(hbm_pressure)

    def get_statistics(self) -> Dict:
        """Get all statistics"""
        return {
            "monitor": self.monitor.get_statistics(),
            "controller": self.controller.get_statistics(),
            "memory": self.memory_manager.get_statistics(),
            "swap_engine": self.swap_engine.get_statistics(),
            "prefetch": self.prefetch_planner.get_statistics(),
            "telemetry": self.telemetry.get_summary(),
        }
