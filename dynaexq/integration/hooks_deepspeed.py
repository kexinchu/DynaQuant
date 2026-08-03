"""
DeepSpeed integration hooks for DynaExQ.

The hook object coordinates monitor/controller updates around DeepSpeed's
Mixture-of-Experts pipeline by intercepting router outputs and synchronizing
with the swap engine before executing the expert GEMM kernels.
"""

from __future__ import annotations

from typing import Iterable

from dynaexq.runtime import (
    ExpertID,
    ExpertMonitor,
    PrecisionController,
    SwapEngine,
)


class DeepSpeedDynaExQ:
    """Thin wrapper expected to be registered as a DeepSpeed MoE callback."""

    def __init__(
        self,
        monitor: ExpertMonitor,
        controller: PrecisionController,
        swap_engine: SwapEngine,
    ) -> None:
        self._monitor = monitor
        self._controller = controller
        self._swap_engine = swap_engine

    def on_router_output(
        self,
        layer: int,
        topk_indices,
        logits,
    ) -> None:
        self._monitor.update_batch(layer, topk_indices, logits)

    def before_expert_gemm(self, layer: int, experts: Iterable[ExpertID]) -> None:
        targets = self._controller.plan(experts, self._monitor)
        for expert, bitwidth in targets.items():
            if bitwidth.value == "W4":
                self._swap_engine.upgrade(expert)

        for expert in experts:
            self._swap_engine.wait_ready(expert)


__all__ = ["DeepSpeedDynaExQ"]
