from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol

from .memmgr import MemoryManager
from .types import Bitwidth, ExpertID, Residency, ResidencyLocation
from .weights import ExpertTensorBundle


class WeightStore(Protocol):
    """Abstract storage backend for expert weights."""

    def byte_size(self, expert: ExpertID, bitwidth: Bitwidth) -> int:
        ...

    def load(self, expert: ExpertID, bitwidth: Bitwidth) -> ExpertTensorBundle:
        ...

    def persist(self, expert: ExpertID, bitwidth: Bitwidth, payload: ExpertTensorBundle) -> None:
        ...


@dataclass
class SwapConfig:
    max_workers: int = 4
    upgrade_latency_s: float = 0.0
    downgrade_latency_s: float = 0.0


class SwapEngine:
    """Coordinate asynchronous expert upgrades and downgrades."""

    def __init__(
        self,
        memory: MemoryManager,
        store: WeightStore,
        config: Optional[SwapConfig] = None,
        executor_factory: Callable[[int], ThreadPoolExecutor] | None = None,
    ) -> None:
        self._memory = memory
        self._store = store
        self._config = config or SwapConfig()
        self._executor = (
            executor_factory(self._config.max_workers)
            if executor_factory
            else ThreadPoolExecutor(max_workers=self._config.max_workers)
        )
        self._futures: Dict[ExpertID, Future[Residency]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def upgrade(self, expert: ExpertID) -> None:
        """Move an expert into the hot pool at W4 precision."""
        self._submit_swap(expert, Bitwidth.W4,
                          ResidencyLocation.HBM, self._config.upgrade_latency_s)

    def downgrade(self, expert: ExpertID) -> None:
        """Move an expert to cold storage at W2 precision."""
        self._submit_swap(
            expert,
            Bitwidth.W2,
            ResidencyLocation.DRAM,
            self._config.downgrade_latency_s,
        )

    def wait_ready(self, expert: ExpertID, timeout: Optional[float] = None) -> Residency:
        """Block until the expert's in-flight swap completes."""

        future = self._find_future(expert)
        residency = future.result(timeout=timeout)
        if residency.location is ResidencyLocation.HBM:
            self._memory.place(expert, residency)
        return residency

    def close(self) -> None:
        """Shutdown the underlying executor."""
        self._executor.shutdown(wait=True, cancel_futures=True)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _submit_swap(
        self,
        expert: ExpertID,
        target_bitwidth: Bitwidth,
        location: ResidencyLocation,
        latency_s: float,
    ) -> None:
        with self._lock:
            if expert in self._futures and not self._futures[expert].done():
                return

            nbytes = self._store.byte_size(expert, target_bitwidth)
            slot_id = None
            try:
                if location is ResidencyLocation.HBM:
                    success, slot_id = self._memory.reserve_hot(expert, nbytes)
                elif location is ResidencyLocation.DRAM:
                    success, slot_id = self._memory.reserve_cold(
                        expert, nbytes)
                else:
                    success = True
                if not success:
                    raise RuntimeError(
                        f"Unable to reserve memory for {expert} at {location.value}"
                    )
            except Exception:
                raise

            future = self._executor.submit(
                self._execute_swap,
                expert,
                target_bitwidth,
                location,
                nbytes,
                latency_s,
                slot_id,
            )
            self._futures[expert] = future

    def _execute_swap(
        self,
        expert: ExpertID,
        bitwidth: Bitwidth,
        location: ResidencyLocation,
        nbytes: int,
        latency_s: float,
        slot_id: Optional[int],
    ) -> Residency:
        if latency_s > 0:
            time.sleep(latency_s)

        try:
            payload = self._store.load(expert, bitwidth)
        except Exception:
            if location is ResidencyLocation.HBM:
                self._memory.cancel_hot_reservation(expert)
            elif location is ResidencyLocation.DRAM:
                self._memory.cancel_cold_reservation(expert)
            raise

        if payload.nbytes != nbytes:
            if location is ResidencyLocation.HBM:
                self._memory.cancel_hot_reservation(expert)
            elif location is ResidencyLocation.DRAM:
                self._memory.cancel_cold_reservation(expert)
            raise RuntimeError(
                f"Byte-size mismatch for {expert}: expected {nbytes}, got {payload.nbytes}"
            )

        residency = Residency(
            bitwidth=bitwidth,
            location=location,
            bytes=nbytes,
            last_used_ts=time.time(),
            pending=False,
            tensor_bundle=payload,
            tags={"slot_id": slot_id} if slot_id is not None else {},
        )

        if location is not ResidencyLocation.HBM:
            self._store.persist(expert, bitwidth, payload)
            self._memory.place(expert, residency)

        return residency

    def _find_future(self, expert: ExpertID) -> Future[Residency]:
        with self._lock:
            future = self._futures.get(expert)
            if future is None:
                raise KeyError(f"No pending swap for {expert}")
            return future


__all__ = ["SwapConfig", "SwapEngine", "WeightStore"]
