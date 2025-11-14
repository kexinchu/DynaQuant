from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Set

from .controller import PrecisionController
from .monitor import ExpertMonitor
from .swap_engine import SwapEngine
from .types import Bitwidth, ExpertID


class PrefetchPlanner:
    """Simple lookahead heuristic for per-layer expert prefetching."""

    def __init__(
        self,
        swap_engine: SwapEngine,
        controller: PrecisionController,
        monitor: ExpertMonitor,
    ) -> None:
        self._swap_engine = swap_engine
        self._controller = controller
        self._monitor = monitor
        self._scheduled: Dict[int, Set[ExpertID]] = defaultdict(set)

    def lookahead(self, layer: int, next_active: Iterable[ExpertID]) -> None:
        next_set = list(dict.fromkeys(next_active))
        if not next_set:
            return

        targets = self._controller.plan(next_set, self._monitor)
        for expert in next_set:
            desired = targets.get(expert, Bitwidth.W2)
            if desired is Bitwidth.W4 and expert not in self._scheduled[layer]:
                try:
                    self._swap_engine.upgrade(expert)
                except RuntimeError:
                    # Let runtime fallback handle allocation failures later.
                    continue
                self._scheduled[layer].add(expert)

    def clear_layer(self, layer: int) -> None:
        """Forget any scheduled experts for a layer."""
        self._scheduled.pop(layer, None)


__all__ = ["PrefetchPlanner"]

