"""
TransitionEngine: asynchronous promotion/demotion pipeline.

Implements asynchronous transitions with stages: fetch, h2d, register, reclaim.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import replace
from typing import Callable, Iterable, Optional

import torch

from .budget_tracker import BudgetTracker, Reservation
from .config import Tier
from .memory_pool import PoolAllocator, PoolBlock
from .quant import PackedTensor, QuantFormat, _INT4PACK_MM_AVAILABLE, _prepare_int4pack_mm
from .registry import ExpertHandle, ExpertKey, ExpertRegistry
from .scheduler import TransitionReq

# WeightStore is an abstract base - ModelWeightStore implements it
from .weight_store import ModelWeightStore

# For backward compatibility, alias WeightStore
WeightStore = ModelWeightStore


class _AdmissionRejected(Exception):
    """Internal sentinel for an all-or-none transition-unit rejection."""


@dataclass
class TransitionStage:
    """Stage timing information."""
    fetch_ms: float = 0.0
    h2d_ms: float = 0.0
    register_ms: float = 0.0
    reclaim_ms: float = 0.0
    total_ms: float = 0.0


# WeightStore is now in weight_store.py


class TransitionEngine:
    """
    Executes expert precision transitions asynchronously.
    
    Pipeline stages:
    1. Fetch: Load weights from storage (SSD/DRAM)
    2. H2D Transfer: Copy to GPU using dedicated CUDA stream
    3. Register: Atomically update ExpertRegistry
    4. Reclaim: Free old block back to pool
    """
    
    def __init__(
        self,
        registry: ExpertRegistry,
        pool_allocator: PoolAllocator,
        weight_store: WeightStore,
        max_workers: int = 4,
        max_inflight: int = 4,
        budget_tracker: Optional[BudgetTracker] = None,
        synchronous: bool = False,
    ):
        """
        Args:
            registry: ExpertRegistry for handle updates
            pool_allocator: PoolAllocator for block allocation
            weight_store: WeightStore for loading weights
            max_workers: Max worker threads
            max_inflight: Max concurrent transitions
            budget_tracker: Optional BudgetTracker. When provided, every
                ``enqueue`` call first attempts ``try_reserve``; failure
                rejects the request (paper §IV.a backpressure). When
                ``None``, the engine runs without HBM accounting — useful
                for unit tests of the leak path itself.
            synchronous: Execute the complete transition inline in
                ``enqueue``. This is the blocking-migration ablation; it also
                disables the dedicated CUDA migration stream so copies and
                layout conversion remain on the caller's critical path.
        """
        self.registry = registry
        self.pool_allocator = pool_allocator
        self.weight_store = weight_store
        self.max_workers = max_workers
        self.max_inflight = max_inflight
        self.budget_tracker = budget_tracker
        self.synchronous = synchronous
        
        # Executor for async execution
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active_transitions: dict[ExpertKey, threading.Event] = {}
        self._transition_lock = threading.Lock()
        
        # CUDA stream for async transfers
        self._uses_cuda = (
            torch.cuda.is_available()
            and pool_allocator.device.type == "cuda"
        )
        self._copy_stream = (
            torch.cuda.Stream(device=pool_allocator.device)
            if self._uses_cuda and not synchronous
            else None
        )
        
        # Statistics
        self._stats_lock = threading.Lock()
        self._total_promotions = 0
        self._total_demotions = 0
        self._failed_transitions = 0
        self._enqueue_attempts = 0
        self._accepted_requests = 0
        self._rejected_inflight_limit = 0
        self._rejected_duplicate = 0
        self._rejected_budget = 0
        self._accepted_bytes = 0
        self._copied_bytes = 0
        self._repatriations = 0
        self._repatriated_bytes = 0
        self._precise_fence_reclaims = 0
        self._global_sync_reclaims = 0
        self._stage_timings: list[TransitionStage] = []
    
    def enqueue(self, req: TransitionReq) -> bool:
        """
        Enqueue a transition request.

        Returns:
            True if enqueued, False if rejected (queue full, already in
            flight, or BudgetTracker reservation refused).
        """
        return self.enqueue_many((req,))

    def enqueue_many(self, requests: Iterable[TransitionReq]) -> bool:
        """Atomically admit one scheduler transition unit.

        A steady-state replacement consists of a demotion followed by its
        matching promotion.  Admitting those requests independently can
        accept only the first half when the executor reaches its in-flight
        limit, leaving a staging-backed handle without the transition that
        creates its destination-tier resident hole.  This method reserves
        executor slots and byte budgets for the complete unit before any
        worker is submitted.
        """
        requests = tuple(requests)
        if not requests:
            raise ValueError("transition unit must not be empty")
        keys = [req.key for req in requests]
        if len(set(keys)) != len(keys):
            raise ValueError("transition unit contains duplicate expert keys")

        with self._stats_lock:
            self._enqueue_attempts += len(requests)

        sizes = []
        for req in requests:
            sizes.append(self.weight_store.get_byte_size(req.key, req.dst))

        with self._transition_lock:
            if (
                len(self._active_transitions) + len(requests)
                > self.max_inflight
            ):
                with self._stats_lock:
                    self._rejected_inflight_limit += len(requests)
                return False
            if any(key in self._active_transitions for key in keys):
                with self._stats_lock:
                    self._rejected_duplicate += len(requests)
                return False  # Already in progress

            for key in keys:
                self._active_transitions[key] = threading.Event()

        # Reserve HBM bytes against the budget BEFORE the worker thread
        # touches the pool. The worker is the only place that allocates
        # blocks; if we let it run without a reservation we'd be racing
        # the budget against the actual cudaMalloc-equivalent. Plan §IV.a
        # backpressure: failed reservation → request is rejected and the
        # caller (Scheduler) defers it until evict-driven release frees
        # bytes.
        reservations: list[Optional[Reservation]] = []
        try:
            for req, nbytes in zip(requests, sizes):
                reservation = None
                if self.budget_tracker is not None:
                    reservation = self.budget_tracker.try_reserve(
                        nbytes,
                        req.dst,
                    )
                    if reservation is None:
                        raise _AdmissionRejected
                reservations.append(reservation)
        except _AdmissionRejected:
            for reservation in reservations:
                if reservation is not None and self.budget_tracker is not None:
                    self.budget_tracker.release(reservation)
            with self._transition_lock:
                for key in keys:
                    self._active_transitions.pop(key, None)
            with self._stats_lock:
                self._rejected_budget += len(requests)
            return False
        except Exception:
            for reservation in reservations:
                if reservation is not None and self.budget_tracker is not None:
                    if not reservation.is_released():
                        self.budget_tracker.release(reservation)
            with self._transition_lock:
                for key in keys:
                    self._active_transitions.pop(key, None)
            raise

        with self._stats_lock:
            self._accepted_requests += len(requests)
            self._accepted_bytes += sum(sizes)

        # The reservation travels with the request without mutating
        # TransitionReq, so the scheduler-facing dataclass stays free of
        # runtime state. The synchronous branch is intentionally the complete
        # opposite of the production path and is used only for the
        # blocking-migration ablation.
        try:
            for req, reservation in zip(requests, reservations):
                if self.synchronous:
                    self._execute_transition(req, reservation)
                else:
                    self._executor.submit(
                        self._execute_transition,
                        req,
                        reservation,
                    )
        except Exception:
            # Executor submission can fail only during an invalid concurrent
            # shutdown.  Preserve fail-closed behavior; ordinary workers own
            # the reservations once submitted and complete their cleanup.
            raise
        return True
    
    def _execute_transition(
        self,
        req: TransitionReq,
        reservation: Optional[Reservation] = None,
    ) -> None:
        """Execute a single transition (runs in background thread).

        Args:
            req: The transition request emitted by the scheduler.
            reservation: BudgetTracker reservation taken in ``enqueue``.
                ``None`` when running without a budget tracker. The
                reservation is committed after Stage 3 (publish) and
                released by Stage 4 of a *future* eviction (it stays
                attached to the new ExpertHandle in the meantime).
        """
        key = req.key
        stage = TransitionStage()
        start_time = time.time()
        committed = False
        block = None
        published = False

        try:
            # Stage 1: Fetch weights
            #
            # weight_store.load_weights returns a PackedTensor directly
            # (Plan A). Tier → format mapping lives in the WeightStore;
            # this method stays format-agnostic and only sees PackedTensor.
            fetch_start = time.time()
            packed = self.weight_store.load_weights(key, req.dst)
            stage.fetch_ms = (time.time() - fetch_start) * 1000

            # Stage 2: Allocate block and transfer
            h2d_start = time.time()
            block = self.pool_allocator.alloc(key.layer, req.dst)
            if block is None:
                raise RuntimeError(f"Failed to allocate block for {key}")

            # Copy packed bytes (qweight + scales) into the block's uint8
            # storage. We concatenate qweight + scales into a single byte
            # stream so a future fused dequant kernel can read both from a
            # contiguous offset. The offsets are recoverable from
            # ``packed`` (out_features, in_features, group_size, fmt).
            payload = self._packed_to_bytes(packed)
            if payload.numel() > block.tensor.numel():
                raise RuntimeError(
                    f"Pool block too small for {key}: payload={payload.numel()} "
                    f"bytes, block={block.tensor.numel()} bytes"
                )
            copy_len = payload.numel()

            if self._copy_stream is not None:
                with torch.cuda.stream(self._copy_stream):
                    block.tensor[:copy_len].copy_(payload, non_blocking=True)
                    copy_done = torch.cuda.Event()
                    copy_done.record(self._copy_stream)
                copy_done.synchronize()
            else:
                block.tensor[:copy_len].copy_(payload)

            # Rebind qweight/scales as typed views into the pool block.
            # Publishing the original host PackedTensor would make the
            # supposedly resident handle execute against CPU storage.
            resident_packed = self._bind_packed_to_block(packed, block.tensor)

            # Stage 2.5 (P4 optimisation): pre-compute the int4pack kernel
            # format in this background thread so the very first fused_linear
            # call after publish does NOT have to run _prepare_int4pack_mm on
            # the critical forward path.  We do this for every PackedTensor in
            # the payload (dict or single) that qualifies for the fast kernel.
            if self._copy_stream is not None:
                with torch.cuda.stream(self._copy_stream):
                    self._materialize_kernel_caches(
                        packed,
                        resident_packed,
                        block.tensor,
                    )
                    layout_done = torch.cuda.Event()
                    layout_done.record(self._copy_stream)
                layout_done.synchronize()
            else:
                self._materialize_kernel_caches(
                    packed,
                    resident_packed,
                    block.tensor,
                )
            stage.h2d_ms = (time.time() - h2d_start) * 1000

            # Stage 3: Register new handle (atomic publish)
            #
            # We snapshot the old handle BEFORE register() so we can free
            # its block in stage 4. The registry's monotonic version field
            # ensures forward-pass readers always see a consistent (block,
            # quant_meta) pair.
            register_start = time.time()
            old_handle = self.registry.get_old_handle(key)

            new_handle = ExpertHandle(
                tier=req.dst,
                block=block,
                quant_meta=resident_packed,
                version=0,  # registry will increment
                reservation=reservation,
            )

            # Convert pending admission into committed ownership immediately
            # before the atomic publication. If publication unexpectedly
            # fails, the exception path releases the committed reservation
            # and the un-published block.
            if reservation is not None and self.budget_tracker is not None:
                self.budget_tracker.commit(reservation)
                committed = True
            self.registry.register(key, new_handle)
            published = True

            stage.register_ms = (time.time() - register_start) * 1000

            # Stage 4: Reclaim old block.
            #
            # Fence ordering (plan §5.3 / Phase 4.3): if the forward path
            # recorded a per-handle event in ``old_handle.last_use_event``
            # after its last read of the block, we wait on THAT event
            # only — it's strictly cheaper than a full
            # ``torch.cuda.synchronize()``. If no fence was recorded we
            # fall back to the conservative global sync (plan §10
            # "退而求其次"). Either way we must fence BEFORE freeing the
            # block so an in-flight forward kernel cannot observe a
            # reused byte range.
            reclaim_start = time.time()
            if old_handle is not None and old_handle.block is not None:
                self._fence_before_reclaim(old_handle)
                self.pool_allocator.free_block(old_handle.block)
                # Release the OLD reservation back to the budget. This
                # is what makes a long promote/demote loop steady-state
                # under a tight cap (the reservation cycle is closed).
                if (
                    old_handle.reservation is not None
                    and self.budget_tracker is not None
                ):
                    self.budget_tracker.release(old_handle.reservation)
                # A prior transition may have published a handle from the
                # global staging pool while this layer's resident pool was
                # full. The block just reclaimed creates a resident hole;
                # move one same-layer/same-tier staging handle into it so
                # transient capacity cannot become permanently occupied.
                if old_handle.block.pool_name != "staging":
                    self._drain_layer_repatriations(key.layer)
            stage.reclaim_ms = (time.time() - reclaim_start) * 1000
            
            stage.total_ms = (time.time() - start_time) * 1000
            
            # Update statistics
            with self._stats_lock:
                if req.dst == Tier.HI:
                    self._total_promotions += 1
                else:
                    self._total_demotions += 1
                self._copied_bytes += copy_len
                self._stage_timings.append(stage)
        
        except Exception as e:
            # Log error (in real implementation)
            print(f"Transition failed for {key}: {e}")
            with self._stats_lock:
                self._failed_transitions += 1
            # Reservation cleanup on the failure path:
            # - If we never committed, the bytes are still pending; release
            #   them so the budget recovers.
            # - If we DID commit (failure happened in stage 4), the new
            #   handle has been published with this reservation already
            #   attached, so we leave it alone — the next eviction of this
            #   expert will release it via the normal Stage 4 path.
            if not published:
                if block is not None and block.in_use:
                    self.pool_allocator.free_block(block)
                if (
                    reservation is not None
                    and self.budget_tracker is not None
                    and not reservation.is_released()
                ):
                    self.budget_tracker.release(reservation)
        finally:
            # Mark transition complete
            with self._transition_lock:
                if key in self._active_transitions:
                    self._active_transitions[key].set()
                    del self._active_transitions[key]
    
    def wait_ready(self, key: ExpertKey, timeout: Optional[float] = None) -> bool:
        """Wait for a transition to complete."""
        with self._transition_lock:
            event = self._active_transitions.get(key)
        
        if event is None:
            return True  # No transition in progress
        
        return event.wait(timeout=timeout)
    
    def _fence_before_reclaim(self, old_handle: ExpertHandle) -> None:
        """
        Block until all in-flight forward reads of ``old_handle.block``
        have drained, then return so the caller may free the block.

        Prefers the per-handle ``last_use_event`` if one was recorded
        via ``ExpertRegistry.mark_used``; otherwise falls back to a
        conservative global ``torch.cuda.synchronize()`` when CUDA is
        available, and is a pure no-op on CPU.

        Separated into its own method so tests can monkey-patch the
        fallback path and assert which branch was taken without having
        to spin up real CUDA events.
        """
        # Publication can race a forward that already acquired the old
        # handle but has not launched its kernels yet. The lease wait closes
        # that gap; the event then fences asynchronous work from the reader.
        self.registry.wait_until_unused(old_handle)
        fences = list(old_handle.last_use_events.values())
        if not fences and old_handle.last_use_event is not None:
            fences = [old_handle.last_use_event]
        if fences:
            # Duck-typed: any object with ``synchronize()`` works, so
            # CPU tests can use a plain mock. ``torch.cuda.Event``
            # satisfies this natively.
            for fence in fences:
                fence.synchronize()
            with self._stats_lock:
                self._precise_fence_reclaims += 1
            return
        # No precise fence was recorded — fall back to the conservative
        # global sync. The CPU path is a no-op.
        self._fallback_global_sync()
        with self._stats_lock:
            self._global_sync_reclaims += 1

    @staticmethod
    def _fallback_global_sync() -> None:
        """Global device sync, the conservative reclaim fence fallback."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _packed_items(packed) -> list[tuple[str, PackedTensor]]:
        if isinstance(packed, dict):
            return list(packed.items())
        return [("weight", packed)]

    @classmethod
    def _materialize_kernel_caches(
        cls,
        source,
        resident,
        storage: torch.Tensor,
    ) -> None:
        """
        Bind kernel-native INT4 layouts into the tail of a pool block.

        A host source has no native cache, so the layout is prepared once and
        copied into pool storage. A staging-backed source already has cache
        metadata; repatriation has copied the entire block, so this method
        recreates typed views without another conversion.
        """
        source_items = cls._packed_items(source)
        resident_items = cls._packed_items(resident)
        if [name for name, _ in source_items] != [
            name for name, _ in resident_items
        ]:
            raise AssertionError("source/resident PackedTensor trees disagree")
        derived_items = [
            item
            for _, item in resident_items
            if item.resident_nbytes != item.nbytes
        ]
        if not derived_items:
            return
        if len(derived_items) != len(resident_items):
            raise RuntimeError(
                "mixed canonical/native layouts in one expert are unsupported"
            )
        # Prepare every matrix before overwriting any canonical bytes. Native
        # INT4 metadata is larger than the canonical scale payload, so writing
        # slot 0 first can overlap slot 1's still-unconverted canonical region
        # in a fused multi-matrix expert.
        prepared = []
        for (_, source_pt), (_, resident_pt) in zip(
            source_items,
            resident_items,
        ):
            if resident_pt.resident_nbytes == resident_pt.nbytes:
                continue
            if resident_pt.fmt != QuantFormat.INT4:
                raise RuntimeError(
                    "derived resident bytes are only implemented for INT4"
                )
            if not (_INT4PACK_MM_AVAILABLE and resident_pt.qweight.is_cuda):
                raise RuntimeError(
                    "INT4 resident footprint requests a kernel cache, but "
                    "the CUDA int4pack kernel is unavailable"
                )

            if source_pt.int4pack_weight is None:
                native_weight, native_scales = _prepare_int4pack_mm(resident_pt)
            else:
                native_weight = source_pt.int4pack_weight
                native_scales = source_pt.int4pack_scales_and_zeros
                if native_scales is None:
                    raise AssertionError("incomplete source INT4 kernel cache")
            prepared.append(
                (
                    source_pt,
                    resident_pt,
                    native_weight,
                    native_scales,
                )
            )

        # Native layouts replace the canonical transfer bytes in-place. The
        # temporary conversion outputs exist only inside this background stage.
        offset = 0
        for source_pt, resident_pt, native_weight, native_scales in prepared:
            weight_bytes = native_weight.numel() * native_weight.element_size()
            weight_end = offset + weight_bytes
            weight_view = storage[offset:weight_end].view(
                native_weight.dtype
            ).view(native_weight.shape)
            if source_pt.int4pack_weight is None:
                weight_view.copy_(native_weight)
            offset = weight_end

            scale_bytes = native_scales.numel() * native_scales.element_size()
            scale_end = offset + scale_bytes
            scales_view = storage[offset:scale_end].view(
                native_scales.dtype
            ).view(native_scales.shape)
            if source_pt.int4pack_weight is None:
                scales_view.copy_(native_scales)
            offset = scale_end
            resident_pt.int4pack_weight = weight_view
            resident_pt.int4pack_scales_and_zeros = scales_view
            resident_pt.canonical_valid = False

        expected = sum(item.resident_nbytes for _, item in resident_items)
        if offset != expected:
            raise AssertionError(
                f"kernel cache consumed {offset} bytes, expected {expected}"
            )
        if offset > storage.numel():
            raise RuntimeError(
                f"pool block cannot hold resident kernel cache: "
                f"required={offset}, block={storage.numel()}"
            )

    def _repatriate_one(self, layer: int, tier: Tier) -> bool:
        """Move one staging-backed handle into a free resident block."""
        candidate_key = None
        candidate_handle = None
        for key, handle in self.registry.handle_snapshot().items():
            if (
                key.layer == layer
                and handle.tier == tier
                and handle.block is not None
                and handle.block.pool_name == "staging"
                and handle.quant_meta is not None
            ):
                candidate_key = key
                candidate_handle = handle
                break
        if candidate_key is None or candidate_handle is None:
            return False

        # Repatriation reads a staging block concurrently with ordinary
        # transitions. Lease the exact snapshotted handle so another worker
        # cannot publish a replacement and recycle that block during the
        # device-to-device copy.
        leased_handle = self.registry.acquire_handle(candidate_key)
        if leased_handle is not candidate_handle:
            if leased_handle is not None:
                self.registry.release_handle(leased_handle)
            return False
        lease_held = True
        resident_block = self.pool_allocator.alloc_resident(layer, tier)
        if resident_block is None:
            self.registry.release_handle(candidate_handle)
            return False
        published = False
        try:
            nbytes = candidate_handle.bytes
            source = candidate_handle.block.tensor[:nbytes]
            if self._copy_stream is not None:
                with torch.cuda.stream(self._copy_stream):
                    resident_block.tensor[:nbytes].copy_(
                        source,
                        non_blocking=True,
                    )
                    copy_done = torch.cuda.Event()
                    copy_done.record(self._copy_stream)
                copy_done.synchronize()
            else:
                resident_block.tensor[:nbytes].copy_(source)
            resident_packed = self._bind_packed_to_block(
                candidate_handle.quant_meta,
                resident_block.tensor,
            )
            self._materialize_kernel_caches(
                candidate_handle.quant_meta,
                resident_packed,
                resident_block.tensor,
            )
            # The copy is synchronized and every source metadata access above
            # is complete. Release before compare-and-swap so the successful
            # path can fence and reclaim the staging block without waiting on
            # its own lease.
            self.registry.release_handle(candidate_handle)
            lease_held = False
            replacement = ExpertHandle(
                tier=tier,
                block=resident_block,
                quant_meta=resident_packed,
                reservation=candidate_handle.reservation,
            )
            published = self.registry.replace_if_current(
                candidate_key,
                candidate_handle,
                replacement,
            )
            if not published:
                return False
            self._fence_before_reclaim(candidate_handle)
            self.pool_allocator.free_block(candidate_handle.block)
            with self._stats_lock:
                self._repatriations += 1
                self._repatriated_bytes += nbytes
            return True
        finally:
            if lease_held:
                self.registry.release_handle(candidate_handle)
            if not published and resident_block.in_use:
                self.pool_allocator.free_block(resident_block)

    def _drain_layer_repatriations(self, layer: int) -> None:
        """Fill every compatible resident hole from the staging pool.

        Concurrent halves of a swap may publish in either order.  Draining
        only the tier of the handle just reclaimed misses the case where the
        peer publishes after that first scan.  The later worker therefore
        scans both tiers and repeats until no compatible move remains.
        """
        while True:
            moved = False
            for tier in (Tier.HI, Tier.LO):
                moved = self._repatriate_one(layer, tier) or moved
            if not moved:
                return

    @staticmethod
    def _single_packed_to_bytes(packed: PackedTensor) -> torch.Tensor:
        """Flatten one PackedTensor into 1-D uint8."""
        if packed.fmt == QuantFormat.FP16:
            payload = packed.qweight.flatten().contiguous().view(torch.uint8)
        else:
            qweight_bytes = packed.qweight.flatten().contiguous()
            assert packed.scales is not None
            scales_bytes = packed.scales.flatten().contiguous().view(torch.uint8)
            payload = torch.cat([qweight_bytes, scales_bytes])

        if payload.numel() != packed.nbytes:
            raise AssertionError(
                f"PackedTensor byte stream length {payload.numel()} does not "
                f"match packed.nbytes {packed.nbytes}; quant.py and "
                f"transition_engine are out of sync"
            )
        return payload

    @classmethod
    def _packed_to_bytes(cls, packed) -> torch.Tensor:
        """
        Flatten a ``PackedTensor`` or ``dict[str, PackedTensor]`` into a
        1-D uint8 byte stream for h2d copy. The dict form is used for
        multi-linear experts (Phi-MoE w1/w2/w3, Qwen3 gate_up_proj/down_proj).
        """
        if isinstance(packed, dict):
            parts = [cls._single_packed_to_bytes(pt) for pt in packed.values()]
            return torch.cat(parts) if parts else torch.empty(0, dtype=torch.uint8)
        return cls._single_packed_to_bytes(packed)

    @staticmethod
    def _bind_one_to_storage(
        packed: PackedTensor,
        storage: torch.Tensor,
        offset: int,
    ) -> tuple[PackedTensor, int]:
        """Create typed qweight/scales views into a uint8 pool allocation."""
        q_bytes = packed.qweight.numel() * packed.qweight.element_size()
        q_end = offset + q_bytes
        qweight = storage[offset:q_end].view(packed.qweight.dtype).view(
            packed.qweight.shape
        )
        offset = q_end
        scales = None
        if packed.scales is not None:
            scale_bytes = packed.scales.numel() * packed.scales.element_size()
            scale_end = offset + scale_bytes
            scales = storage[offset:scale_end].view(packed.scales.dtype).view(
                packed.scales.shape
            )
            offset = scale_end
        resident = replace(
            packed,
            qweight=qweight,
            scales=scales,
            int4pack_weight=None,
            int4pack_scales_and_zeros=None,
        )
        return resident, offset

    @classmethod
    def _bind_packed_to_block(cls, packed, storage: torch.Tensor):
        """Mirror a PackedTensor tree using views backed by ``storage``."""
        offset = 0
        if isinstance(packed, dict):
            resident = {}
            for name, item in packed.items():
                resident[name], offset = cls._bind_one_to_storage(
                    item, storage, offset
                )
        else:
            resident, offset = cls._bind_one_to_storage(packed, storage, offset)
        expected = sum(
            item.nbytes for item in packed.values()
        ) if isinstance(packed, dict) else packed.nbytes
        if offset != expected:
            raise AssertionError(
                f"resident view consumed {offset} bytes, expected {expected}"
            )
        return resident
    
    def get_stats(self, *, include_stage_timings: bool = True) -> dict:
        """Return a point-in-time, JSON-serializable telemetry snapshot.

        Bootstrap can contain tens of thousands of transitions. Callers that
        only need its audited counts may request aggregate timing telemetry
        instead of serializing one dictionary per transition.
        """
        with self._stats_lock:
            stage_timing_summary = {
                "count": len(self._stage_timings),
                "fetch_ms_sum": sum(s.fetch_ms for s in self._stage_timings),
                "h2d_ms_sum": sum(s.h2d_ms for s in self._stage_timings),
                "register_ms_sum": sum(
                    s.register_ms for s in self._stage_timings
                ),
                "reclaim_ms_sum": sum(
                    s.reclaim_ms for s in self._stage_timings
                ),
                "total_ms_sum": sum(s.total_ms for s in self._stage_timings),
            }
            stats = {
                "execution_mode": (
                    "synchronous" if self.synchronous else "asynchronous"
                ),
                "total_promotions": self._total_promotions,
                "total_demotions": self._total_demotions,
                "failed_transitions": self._failed_transitions,
                "enqueue_attempts": self._enqueue_attempts,
                "accepted_requests": self._accepted_requests,
                "rejected_inflight_limit": self._rejected_inflight_limit,
                "rejected_duplicate": self._rejected_duplicate,
                "rejected_budget": self._rejected_budget,
                "accepted_bytes": self._accepted_bytes,
                "copied_bytes": self._copied_bytes,
                "repatriations": self._repatriations,
                "repatriated_bytes": self._repatriated_bytes,
                "precise_fence_reclaims": self._precise_fence_reclaims,
                "global_sync_reclaims": self._global_sync_reclaims,
                "stage_timing_summary": stage_timing_summary,
            }
            if include_stage_timings:
                stats["stage_timings"] = [
                    {
                        "fetch_ms": s.fetch_ms,
                        "h2d_ms": s.h2d_ms,
                        "register_ms": s.register_ms,
                        "reclaim_ms": s.reclaim_ms,
                        "total_ms": s.total_ms,
                    }
                    for s in self._stage_timings
                ]
        with self._transition_lock:
            stats["active_transitions"] = len(self._active_transitions)
        stats["pool"] = self.pool_allocator.snapshot()
        stats["budget"] = (
            self.budget_tracker.snapshot()
            if self.budget_tracker is not None
            else None
        )
        return stats

    def reset_stats(self) -> None:
        """Reset counters after an untimed warm-up/bootstrap phase."""
        with self._transition_lock:
            if self._active_transitions:
                raise RuntimeError("cannot reset statistics with active transitions")
        with self._stats_lock:
            self._total_promotions = 0
            self._total_demotions = 0
            self._failed_transitions = 0
            self._enqueue_attempts = 0
            self._accepted_requests = 0
            self._rejected_inflight_limit = 0
            self._rejected_duplicate = 0
            self._rejected_budget = 0
            self._accepted_bytes = 0
            self._copied_bytes = 0
            self._repatriations = 0
            self._repatriated_bytes = 0
            self._precise_fence_reclaims = 0
            self._global_sync_reclaims = 0
            self._stage_timings.clear()
    
    def shutdown(self) -> None:
        """Shutdown the transition engine."""
        self._executor.shutdown(wait=True)
