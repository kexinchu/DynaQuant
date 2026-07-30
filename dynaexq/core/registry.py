"""
ExpertRegistry: atomic "last stable representation" handle management.

Provides thread-safe access to expert handles used by forward pass.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

import torch

from .config import Tier

from typing import Union

if TYPE_CHECKING:
    # Forward references only — keep registry.py free of import-time
    # dependencies on quant.py / memory_pool.py so it can still be imported
    # by tests that mock both.
    from .budget_tracker import Reservation
    from .memory_pool import PoolBlock
    from .quant import PackedTensor


class _FenceLike:
    """
    Structural protocol for an event fence (Phase 4.3).

    An object is fence-like iff it has a ``synchronize()`` method that
    blocks until all preceding work on its recording stream has
    finished. ``torch.cuda.Event`` satisfies this natively; CPU tests
    can pass a plain object with a ``synchronize`` attribute.

    We do not use ``typing.Protocol`` here on purpose — the whole point
    of the duck-typed signature is that it works without a CUDA-specific
    import on CPU-only machines, and ``Protocol`` runtime checks would
    drag in machinery we do not need.
    """

    def synchronize(self) -> None:  # pragma: no cover - structural only
        ...


@dataclass
class ExpertKey:
    """Unique identifier for an expert."""
    layer: int
    expert: int

    def __hash__(self) -> int:
        return hash((self.layer, self.expert))

    def __eq__(self, other) -> bool:
        if not isinstance(other, ExpertKey):
            return False
        return self.layer == other.layer and self.expert == other.expert


@dataclass
class ExpertHandle:
    """
    Handle to expert representation used by forward pass.

    This is the "last stable registered" representation that forward always
    binds to. Transitions update this atomically.

    Plan B refactor (Phase 4 §5.1): the handle now holds a direct reference
    to the ``PoolBlock`` it occupies (instead of a bare tensor pointer) and
    to its ``PackedTensor`` quantization metadata. This is what unlocks the
    real reclaim path — `TransitionEngine` can recover the block from
    `old_handle.block` and call `pool_allocator.free(...)` instead of leaking.

    Field semantics:
        tier        — HI or LO
        block       — PoolBlock currently backing this handle. None only
                      during the brief "uninitialized" window before the
                      first registration.
        quant_meta  — PackedTensor describing the on-block layout
                      (shape, scales, qweight, fmt, nbytes). For FP16 tier
                      this is still a PackedTensor with ``fmt=FP16`` and
                      ``scales=None``; the runtime never special-cases on
                      tier, only on ``quant_meta.fmt``.
        format      — Legacy string ("fp16"/"int4"/"int2"). Kept as a derived
                      convenience but the source of truth is
                      ``quant_meta.fmt``. Wired up via ``__post_init__``.
        bytes       — On-device byte footprint of this handle (must equal
                      ``quant_meta.resident_nbytes`` including derived
                      kernel layouts). The BudgetTracker (Phase 4
                      §5.2) will reserve / release exactly this many bytes.
        version     — Monotonically incremented by ExpertRegistry on each
                      register call. Forward pass observes the latest
                      published version.
    """

    tier: Tier
    block: Optional["PoolBlock"] = None
    # Single-weight experts pass a ``PackedTensor``; multi-linear experts
    # (Phi-MoE w1/w2/w3, Qwen3 gate_up_proj/down_proj) pass a
    # ``dict[str, PackedTensor]``.  Use ``get_packed(slot)`` to read.
    quant_meta: Optional[Union["PackedTensor", dict[str, "PackedTensor"]]] = None
    format: str = ""
    bytes: int = 0
    version: int = 0
    # BudgetTracker reservation that owns this handle's bytes. Set by
    # TransitionEngine when the handle is published; used by Stage 4
    # reclaim to release the bytes back to the budget. ``None`` is allowed
    # for handles created outside the runtime (tests, no-budget mode).
    reservation: Optional["Reservation"] = None
    # Phase 4.3 event fence: the most recent compute-stream event
    # recorded after a forward kernel finished reading from
    # ``block.tensor``. TransitionEngine Stage 4 waits on this event
    # before freeing the block, which is strictly cheaper than
    # ``torch.cuda.synchronize()``. ``None`` means "no precise fence
    # available, fall back to the conservative global sync".
    last_use_event: Optional[object] = None
    # Latest completion event per compute stream. Keeping one event per
    # stream is bounded by serving concurrency and lets a reclaimer fence
    # multi-stream readers without retaining an event for every invocation.
    last_use_events: dict[int, object] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    # Protected by ExpertRegistry._global_lock. A replacement may be
    # published immediately, but this handle's block cannot be reclaimed
    # while a forward lease is active.
    active_readers: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.quant_meta is not None:
            if isinstance(self.quant_meta, dict):
                # Multi-linear path: ``{"w1": pt1, "w2": pt2, ...}``.
                pts = list(self.quant_meta.values())
                if not pts:
                    raise ValueError("quant_meta dict must not be empty")
                fmts = {pt.fmt.value for pt in pts}
                if len(fmts) > 1:
                    raise ValueError(
                        f"All slots must share the same QuantFormat, got {fmts}"
                    )
                inferred_fmt = pts[0].fmt.value
                total_nbytes = sum(pt.resident_nbytes for pt in pts)
            else:
                # Single PackedTensor path (backward compat).
                inferred_fmt = self.quant_meta.fmt.value
                total_nbytes = self.quant_meta.resident_nbytes

            if self.format and self.format != inferred_fmt:
                raise ValueError(
                    f"ExpertHandle.format={self.format!r} disagrees with "
                    f"quant_meta fmt={inferred_fmt!r}"
                )
            self.format = inferred_fmt
            if self.bytes and self.bytes != total_nbytes:
                raise ValueError(
                    f"ExpertHandle.bytes={self.bytes} disagrees with "
                    f"quant_meta total nbytes={total_nbytes}"
                )
            self.bytes = total_nbytes

    @property
    def device_ptr(self) -> Optional[torch.Tensor]:
        """
        Backwards-compatible accessor: returns the underlying device tensor
        carried by ``block.tensor``. New code should reach for
        ``handle.block.tensor`` directly when possible.
        """
        if self.block is None:
            return None
        return self.block.tensor

    def is_valid(self) -> bool:
        """A handle is usable iff a block is bound to it."""
        return self.block is not None

    def get_packed(self, slot: str = "weight") -> Optional["PackedTensor"]:
        """
        Get the ``PackedTensor`` for *slot*.

        For single-weight experts (default), ``slot="weight"`` returns the
        single ``PackedTensor``. For multi-linear experts (Phi-MoE, Qwen3),
        use the linear's name (``"w1"``, ``"gate_up_proj"``, etc.).
        Returns ``None`` if the slot does not exist or ``quant_meta`` is
        not set.
        """
        if self.quant_meta is None:
            return None
        if isinstance(self.quant_meta, dict):
            return self.quant_meta.get(slot)
        if slot == "weight":
            return self.quant_meta
        return None


class ExpertRegistry:
    """
    Thread-safe registry mapping ExpertKey -> ExpertHandle.

    Transitions publish handles atomically after completion. Model execution
    uses ``acquire_handle``/``release_handle`` leases so an old pool block
    cannot be reclaimed between pointer resolution and kernel launch.

    P2 optimisation — lock-free reads + versioned refresh skipping
    ---------------------------------------------------------------
    ``get_handle`` is a lock-free observational lookup. In CPython the GIL
    guarantees atomicity of a single ``dict.get`` call, so a concurrent
    ``register`` cannot produce a torn read: the forward thread sees either
    the old or the new (fully-constructed) handle, never a partial one.
    Execution paths instead use the short locked lease API to close the
    read-before-kernel-launch reclamation race.

    ``_version`` is incremented under ``_global_lock`` every time any handle
    changes.  ``PhimoeSparseMoeBlock`` caches the last seen version and calls
    ``_refresh_expert_handles`` only when it differs — turning O(num_experts)
    lock acquisitions per step into a single integer comparison for the common
    "no transition this step" path.
    """

    def __init__(self):
        self._handles: dict[ExpertKey, ExpertHandle] = {}
        self._locks: dict[ExpertKey, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._reader_condition = threading.Condition(self._global_lock)
        # Monotonically increasing counter.  Any ``register`` call bumps this
        # under _global_lock so readers can detect staleness with a single
        # attribute read (GIL-safe integer access in CPython).
        self._version: int = 0

    def __len__(self) -> int:
        """Return the number of currently published expert handles."""
        with self._global_lock:
            return len(self._handles)

    def get_handle(self, key: ExpertKey) -> Optional[ExpertHandle]:
        """
        Get current handle for an expert.

        Lock-free in CPython: ``dict.get`` is a single C-level operation
        protected by the GIL.  No additional lock is needed for reads.
        """
        return self._handles.get(key)

    def acquire_handle(self, key: ExpertKey) -> Optional[ExpertHandle]:
        """Acquire a read lease on the currently published handle."""
        with self._global_lock:
            handle = self._handles.get(key)
            if handle is not None:
                handle.active_readers += 1
            return handle

    def release_handle(
        self,
        handle: ExpertHandle,
        event: Optional[object] = None,
        stream_id: Optional[int] = None,
    ) -> None:
        """Release a read lease and optionally attach its completion fence."""
        with self._reader_condition:
            if handle.active_readers <= 0:
                raise RuntimeError("expert handle lease released without acquire")
            if event is not None:
                handle.last_use_event = event
                handle.last_use_events[
                    stream_id if stream_id is not None else id(event)
                ] = event
            handle.active_readers -= 1
            if handle.active_readers == 0:
                self._reader_condition.notify_all()

    def wait_until_unused(self, handle: ExpertHandle) -> None:
        """Block a reclaimer until no forward owns ``handle``."""
        with self._reader_condition:
            self._reader_condition.wait_for(lambda: handle.active_readers == 0)

    def tier_snapshot(self) -> dict[ExpertKey, Tier]:
        """Return a consistent copy of all currently published tiers."""
        with self._global_lock:
            return {key: handle.tier for key, handle in self._handles.items()}

    def handle_snapshot(self) -> dict[ExpertKey, ExpertHandle]:
        """Return a consistent shallow copy of the published handle map."""
        with self._global_lock:
            return dict(self._handles)

    def register(self, key: ExpertKey, handle: ExpertHandle) -> None:
        """
        Register a new handle (atomic update).

        This is called by TransitionEngine after a transition completes.
        The handle becomes visible to forward pass immediately after the
        lock is released.
        """
        with self._global_lock:
            old_handle = self._handles.get(key)
            if old_handle is not None:
                # Increment per-handle version
                handle.version = old_handle.version + 1
            self._handles[key] = handle
            # Bump global version so model layers can detect the change
            # without re-querying every expert.
            self._version += 1

    def replace_if_current(
        self,
        key: ExpertKey,
        expected: ExpertHandle,
        replacement: ExpertHandle,
    ) -> bool:
        """Atomically publish ``replacement`` only if ``expected`` is current."""
        with self._global_lock:
            if self._handles.get(key) is not expected:
                return False
            replacement.version = expected.version + 1
            self._handles[key] = replacement
            self._version += 1
            return True

    def get_old_handle(self, key: ExpertKey) -> Optional[ExpertHandle]:
        """Get the handle that will be replaced (for cleanup)."""
        with self._global_lock:
            return self._handles.get(key)

    def mark_used(self, key: ExpertKey, event: object) -> bool:
        """
        Phase 4.3: record the compute-stream event at which the forward
        path last read from this handle's block.

        Args:
            key: Expert key whose handle was just read by a forward kernel.
            event: A fence-like object with a ``synchronize()`` method,
                typically a ``torch.cuda.Event`` recorded on the compute
                stream immediately after the kernel that consumed
                ``handle.block.tensor``. CPU tests can pass any object
                exposing ``synchronize()``.

        Returns:
            ``True`` if a handle was found for ``key`` and the event was
            stored on it; ``False`` if no handle exists (the expert is
            not currently registered — the caller should skip fence
            recording silently).

        This is a write to the ``last_use_event`` field under the
        registry's lock, so a concurrent ``TransitionEngine`` reclaim
        cannot observe a half-written fence. The fence itself is
        immutable from the registry's point of view — we store the
        reference verbatim and let ``TransitionEngine.synchronize()``
        block on it later.
        """
        with self._global_lock:
            handle = self._handles.get(key)
            if handle is None:
                return False
            handle.last_use_event = event
            return True
    
    def _get_lock(self, key: ExpertKey) -> threading.Lock:
        """Get per-expert lock (for transition engine use)."""
        with self._global_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]
    
    def acquire_lock(self, key: ExpertKey) -> threading.Lock:
        """Acquire per-expert lock (used by TransitionEngine)."""
        return self._get_lock(key)
