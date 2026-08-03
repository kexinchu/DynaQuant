"""
BudgetTracker: atomic HBM envelope reservations (Phase 4 §5.2).

Implements the "never enters a transition without a prior reservation
against the declared HBM envelope" guarantee from plan §III-D / paper
§IV.c. Every byte that the runtime puts on the GPU passes through this
tracker first.

Lifecycle of a reservation
--------------------------
::

    enqueue          ─►  try_reserve(nbytes, tier)  ─►  Reservation token
                                                        (bytes are PENDING)

    H2D copy         ─►  block alloc + memcpy        (no tracker calls)

    register         ─►  commit(reservation)        (PENDING → COMMITTED)

    reclaim old      ─►  release(old_reservation)   (frees old bytes back
                                                     to the budget)

A transition that fails (e.g. weight load throws) calls
``release(reservation)`` directly, which works regardless of whether
the reservation was already committed.

Why split commit and release?
-----------------------------
The window between ``try_reserve`` and ``commit`` is the period during
which the new copy is in flight on the H2D stream. The old block is still
live on the compute stream. Both contribute to peak HBM. The
``staging_cap`` parameter caps how many bytes worth of in-flight copies
the runtime is willing to hold simultaneously, so a burst of promotions
cannot push memory off a cliff while the old blocks are still being
retired.

Atomicity
---------
All state mutations are guarded by a single ``threading.Lock``. The
contended path is short (a handful of integer ops + one append) so the
lock contention is negligible compared to the H2D copy itself.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from .config import Tier


class _State(Enum):
    PENDING = auto()
    COMMITTED = auto()
    RELEASED = auto()


@dataclass
class Reservation:
    """
    Token returned by ``BudgetTracker.try_reserve``.

    Carries enough information for ``commit`` / ``release`` to find the
    counter to update. The ``id`` field is purely a debug aid (helps
    diagnose double-commit / double-release in logs).
    """

    id: int
    tier: Tier
    nbytes: int
    _state: _State = _State.PENDING

    def is_pending(self) -> bool:
        return self._state == _State.PENDING

    def is_committed(self) -> bool:
        return self._state == _State.COMMITTED

    def is_released(self) -> bool:
        return self._state == _State.RELEASED


class BudgetExceeded(RuntimeError):
    """Raised by debug helpers; ``try_reserve`` itself returns ``None``."""


class BudgetTracker:
    """
    Atomic byte-budget tracker for HI / LO tiers and the in-flight staging
    pool.

    Args:
        hi_cap: Maximum bytes that can be live (committed + pending) in
            the HI tier across the entire model.
        lo_cap: Same for the LO tier.
        staging_cap: Maximum bytes that can be PENDING (across both tiers
            together). Caps the in-flight HBM overhead during a burst of
            promotions/demotions. Pass ``None`` to disable the staging
            cap; pass ``0`` to disallow any in-flight transition (useful
            for tests).
        total_cap: Optional cap on all committed plus pending expert bytes.
            This is the publish-before-reclaim peak invariant across tiers.
    """

    def __init__(
        self,
        hi_cap: int,
        lo_cap: int,
        staging_cap: Optional[int] = None,
        total_cap: Optional[int] = None,
    ):
        if hi_cap < 0 or lo_cap < 0:
            raise ValueError(f"caps must be non-negative; got hi={hi_cap}, lo={lo_cap}")
        if staging_cap is not None and staging_cap < 0:
            raise ValueError(f"staging_cap must be non-negative or None; got {staging_cap}")
        if total_cap is not None and total_cap < 0:
            raise ValueError(f"total_cap must be non-negative or None; got {total_cap}")

        self.hi_cap = hi_cap
        self.lo_cap = lo_cap
        self.staging_cap = staging_cap
        self.total_cap = total_cap

        # Bytes that have been registered into the runtime (committed).
        self._hi_committed = 0
        self._lo_committed = 0
        # Bytes that have been reserved but not yet committed.
        self._hi_pending = 0
        self._lo_pending = 0

        self._lock = threading.Lock()
        self._id_gen = itertools.count(1)

    # ------------------------------------------------------------------
    # Reservation lifecycle
    # ------------------------------------------------------------------

    def try_reserve(self, nbytes: int, tier: Tier) -> Optional[Reservation]:
        """
        Atomically attempt to reserve ``nbytes`` against ``tier``.

        Returns a fresh ``Reservation`` (state PENDING) on success or
        ``None`` if the reservation would push either the per-tier cap or
        the global staging cap over the limit. Callers MUST handle the
        ``None`` case (typically by deferring or dropping the transition).
        """
        if nbytes <= 0:
            raise ValueError(f"nbytes must be positive, got {nbytes}")

        with self._lock:
            cap = self._cap_for(tier)
            committed, pending = self._state_for(tier)
            if committed + pending + nbytes > cap:
                return None
            if (
                self.total_cap is not None
                and self._total_live_locked() + nbytes > self.total_cap
            ):
                return None
            if (
                self.staging_cap is not None
                and self._total_pending_locked() + nbytes > self.staging_cap
            ):
                return None

            self._add_pending_locked(tier, nbytes)
            return Reservation(
                id=next(self._id_gen),
                tier=tier,
                nbytes=nbytes,
                _state=_State.PENDING,
            )

    def commit(self, reservation: Reservation) -> None:
        """
        Move a reservation from PENDING to COMMITTED.

        Idempotent for already-committed reservations (no-op). Raises
        ``RuntimeError`` if the reservation has already been released —
        committing released bytes would silently double-count the budget.
        """
        with self._lock:
            if reservation._state == _State.COMMITTED:
                return
            if reservation._state == _State.RELEASED:
                raise RuntimeError(
                    f"cannot commit reservation {reservation.id}: already released"
                )
            self._sub_pending_locked(reservation.tier, reservation.nbytes)
            self._add_committed_locked(reservation.tier, reservation.nbytes)
            reservation._state = _State.COMMITTED

    def release(self, reservation: Reservation) -> None:
        """
        Free the bytes held by ``reservation`` back to the budget.

        Works on both PENDING reservations (transition aborted before
        publish) and COMMITTED reservations (old expert evicted, block
        reclaimed). A double-release is a hard error: it would credit the
        budget more bytes than were actually charged, breaking the HBM
        envelope invariant from plan §III-D.
        """
        with self._lock:
            if reservation._state == _State.RELEASED:
                raise RuntimeError(
                    f"double release of reservation {reservation.id}"
                )
            if reservation._state == _State.PENDING:
                self._sub_pending_locked(reservation.tier, reservation.nbytes)
            else:  # COMMITTED
                self._sub_committed_locked(reservation.tier, reservation.nbytes)
            reservation._state = _State.RELEASED

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, int]:
        """Return a point-in-time copy of the byte counters (for tests / logs)."""
        with self._lock:
            return {
                "hi_committed": self._hi_committed,
                "hi_pending": self._hi_pending,
                "hi_cap": self.hi_cap,
                "lo_committed": self._lo_committed,
                "lo_pending": self._lo_pending,
                "lo_cap": self.lo_cap,
                "staging_used": self._total_pending_locked(),
                "staging_cap": -1 if self.staging_cap is None else self.staging_cap,
                "total_live": self._total_live_locked(),
                "total_cap": -1 if self.total_cap is None else self.total_cap,
            }

    def available(self, tier: Tier) -> int:
        """Bytes that ``try_reserve(_, tier)`` could currently grant."""
        with self._lock:
            cap = self._cap_for(tier)
            committed, pending = self._state_for(tier)
            tier_room = max(0, cap - committed - pending)
            if self.total_cap is not None:
                tier_room = min(
                    tier_room,
                    max(0, self.total_cap - self._total_live_locked()),
                )
            if self.staging_cap is None:
                return tier_room
            staging_room = max(0, self.staging_cap - self._total_pending_locked())
            return min(tier_room, staging_room)

    # ------------------------------------------------------------------
    # Locked helpers (must be called with self._lock held)
    # ------------------------------------------------------------------

    def _cap_for(self, tier: Tier) -> int:
        return self.hi_cap if tier == Tier.HI else self.lo_cap

    def _state_for(self, tier: Tier) -> tuple[int, int]:
        if tier == Tier.HI:
            return self._hi_committed, self._hi_pending
        return self._lo_committed, self._lo_pending

    def _total_pending_locked(self) -> int:
        return self._hi_pending + self._lo_pending

    def _total_live_locked(self) -> int:
        return (
            self._hi_committed
            + self._lo_committed
            + self._hi_pending
            + self._lo_pending
        )

    def _add_pending_locked(self, tier: Tier, nbytes: int) -> None:
        if tier == Tier.HI:
            self._hi_pending += nbytes
        else:
            self._lo_pending += nbytes

    def _sub_pending_locked(self, tier: Tier, nbytes: int) -> None:
        if tier == Tier.HI:
            self._hi_pending -= nbytes
            assert self._hi_pending >= 0, "hi_pending underflow"
        else:
            self._lo_pending -= nbytes
            assert self._lo_pending >= 0, "lo_pending underflow"

    def _add_committed_locked(self, tier: Tier, nbytes: int) -> None:
        if tier == Tier.HI:
            self._hi_committed += nbytes
        else:
            self._lo_committed += nbytes

    def _sub_committed_locked(self, tier: Tier, nbytes: int) -> None:
        if tier == Tier.HI:
            self._hi_committed -= nbytes
            assert self._hi_committed >= 0, "hi_committed underflow"
        else:
            self._lo_committed -= nbytes
            assert self._lo_committed >= 0, "lo_committed underflow"


__all__ = ["BudgetTracker", "Reservation", "BudgetExceeded"]
