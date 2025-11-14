"""
SGLang integration hooks for DynaExQ.

These hooks provide adapters that surface router top-k indices/logits and
allow the runtime to wait for expert readiness prior to issuing GEMM
launches. The implementation is intentionally lightweight so it can be
customized per deployment.
"""

from __future__ import annotations

from typing import Callable, Iterable

from dynaexq.runtime import (
    ExpertID,
    ExpertMonitor,
    MemoryManager,
    PrecisionController,
    PrefetchPlanner,
    SwapEngine,
)


class SGLangDynaExQ:
    """Glue layer connecting SGLang callbacks with the DynaExQ runtime."""

    def __init__(
        self,
        monitor: ExpertMonitor,
        controller: PrecisionController,
        memory: MemoryManager,
        swap_engine: SwapEngine,
        lookahead: PrefetchPlanner,
    ) -> None:
        self._monitor = monitor
        self._controller = controller
        self._memory = memory
        self._swap_engine = swap_engine
        self._prefetch = lookahead

    def on_layer_router(
        self,
        layer: int,
        topk_indices: Iterable[int],
        logits: Iterable[float],
    ) -> None:
        self._monitor.update_batch(
            layer,
            topk_idx=[list(topk_indices)],
            logits=[list(logits)],
        )

    def pre_dispatch(self, layer: int, experts: Iterable[ExpertID]) -> None:
        targets = self._controller.plan(experts, self._monitor)
        for expert, bitwidth in targets.items():
            if bitwidth.value == "W4":
                self._swap_engine.upgrade(expert)

        for expert in experts:
            self._swap_engine.wait_ready(expert)

    def schedule_next(self, layer: int, next_experts: Iterable[ExpertID]) -> None:
        self._prefetch.lookahead(layer, next_experts)


__all__ = ["SGLangDynaExQ"]

