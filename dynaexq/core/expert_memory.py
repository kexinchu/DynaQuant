"""Atomic byte claims and extent binding for the shared expert arena."""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field

from .config import Tier
from .registry import ExpertKey
from .shared_arena import ArenaExtent, ExtentState, SharedArenaAllocator


@dataclass(frozen=True, slots=True)
class MemoryRequest:
    key: ExpertKey
    tier: Tier
    purpose: str
    nbytes: int
    # Optional exact decomposition of this one extent. A high representation
    # can simultaneously carry residency bytes (its low-format base cost) and
    # fidelity bytes (the incremental high-format cost). Lookahead owns the
    # complete transient extent until publication or expiry.
    accounting: tuple[int, int, int] | None = None  # fid, res, look

    def __post_init__(self) -> None:
        if self.purpose not in {"fid", "res", "look"}:
            raise ValueError(f"unknown memory purpose {self.purpose}")
        if self.nbytes <= 0:
            raise ValueError("memory request must be positive")
        if self.accounting is not None:
            if any(value < 0 for value in self.accounting):
                raise ValueError("memory accounting bytes must be non-negative")
            if sum(self.accounting) != self.nbytes:
                raise ValueError("memory accounting must sum to request bytes")

    def bytes_by_purpose(self) -> dict[str, int]:
        if self.accounting is None:
            return {
                "fid": self.nbytes if self.purpose == "fid" else 0,
                "res": self.nbytes if self.purpose == "res" else 0,
                "look": self.nbytes if self.purpose == "look" else 0,
            }
        fid, res, look = self.accounting
        return {"fid": fid, "res": res, "look": look}


@dataclass
class MemoryClaim:
    claim_id: int
    request: MemoryRequest
    extent: ArenaExtent
    exchange_id: str


@dataclass
class ExchangeReservation:
    exchange_id: str
    claims: list[MemoryClaim] = field(default_factory=list)
    donor_claim_ids: tuple[int, ...] = ()
    closed: bool = False


class ExpertMemoryManager:
    """Single transaction boundary for the arena and its byte ledger."""

    def __init__(self, allocator: SharedArenaAllocator) -> None:
        self.allocator = allocator
        self._lock = threading.RLock()
        self._claim_ids = itertools.count(1)
        self._claims: dict[int, MemoryClaim] = {}
        self._donor_holds: dict[int, str] = {}
        self._reservation_attempts = 0
        self._reservation_failures_budget = 0
        self._reservation_failures_extent = 0
        self._reservation_rollbacks = 0
        self._peak_reserved_bytes = 0
        self._peak_published_bytes = 0

    @property
    def budget(self) -> int:
        return self.allocator.capacity_bytes

    def try_reserve_exchange(
        self,
        exchange_id: str,
        requests: list[MemoryRequest],
        *,
        donors: list[MemoryClaim] | None = None,
    ) -> ExchangeReservation | None:
        """Hold donors and bind every destination in one transaction.

        A failed call leaves neither arena extents nor donor holds behind.
        Holding a donor prevents another exchange from spending the same
        future release while this exchange is in flight.
        """
        with self._lock:
            self._reservation_attempts += 1
            if not requests:
                return None
            donor_claims = list(donors or ())
            if any(
                self._claims.get(claim.claim_id) is not claim
                for claim in donor_claims
            ):
                return None
            donor_ids = [claim.claim_id for claim in donor_claims]
            if len(donor_ids) != len(set(donor_ids)):
                return None
            if any(claim_id in self._donor_holds for claim_id in donor_ids):
                return None
            for claim_id in donor_ids:
                self._donor_holds[claim_id] = exchange_id

            reservation = ExchangeReservation(
                exchange_id,
                donor_claim_ids=tuple(donor_ids),
            )
            for request in requests:
                extent = self.allocator.allocate(request.nbytes)
                if extent is None:
                    before = self.allocator.snapshot()
                    charged = self.allocator.rounded_bytes(request.nbytes)
                    if before["free_bytes"] >= charged:
                        self._reservation_failures_extent += 1
                    else:
                        self._reservation_failures_budget += 1
                    for claim in reversed(reservation.claims):
                        self.allocator.free(claim.extent)
                    for claim_id in donor_ids:
                        del self._donor_holds[claim_id]
                    if reservation.claims:
                        self._reservation_rollbacks += 1
                    return None
                claim = MemoryClaim(
                    claim_id=next(self._claim_ids),
                    request=request,
                    extent=extent,
                    exchange_id=exchange_id,
                )
                reservation.claims.append(claim)
            for claim in reservation.claims:
                self._claims[claim.claim_id] = claim
            self._update_peaks()
            return reservation

    def publish(self, claim: MemoryClaim) -> None:
        with self._lock:
            self._require_claim(claim)
            self.allocator.mark_published(claim.extent)
            self._update_peaks()

    def mark_reclaim_pending(self, claim: MemoryClaim) -> None:
        with self._lock:
            self._require_claim(claim)
            self.allocator.mark_reclaim_pending(claim.extent)

    def release(self, claim: MemoryClaim) -> None:
        with self._lock:
            self._require_claim(claim)
            if claim.claim_id in self._donor_holds:
                raise RuntimeError("held donor must be released by its exchange")
            self.allocator.free(claim.extent)
            del self._claims[claim.claim_id]

    def reclassify(
        self,
        claim: MemoryClaim,
        *,
        purpose: str,
        accounting: tuple[int, int, int] | None = None,
    ) -> None:
        """Change only ledger attribution; extent ownership is unchanged."""
        with self._lock:
            self._require_claim(claim)
            claim.request = MemoryRequest(
                claim.request.key,
                claim.request.tier,
                purpose,
                claim.request.nbytes,
                accounting,
            )

    def hold_donors(self, exchange_id: str, claims: list[MemoryClaim]) -> bool:
        """Prevent concurrent exchanges from spending the same future bytes."""
        with self._lock:
            if any(self._claims.get(claim.claim_id) is not claim for claim in claims):
                return False
            if any(claim.claim_id in self._donor_holds for claim in claims):
                return False
            for claim in claims:
                self._donor_holds[claim.claim_id] = exchange_id
            return True

    def release_held_donor(self, exchange_id: str, claim: MemoryClaim) -> None:
        with self._lock:
            self._require_claim(claim)
            if self._donor_holds.get(claim.claim_id) != exchange_id:
                raise RuntimeError("claim is not held by this exchange")
            del self._donor_holds[claim.claim_id]
            self.allocator.free(claim.extent)
            del self._claims[claim.claim_id]

    def release_donor_holds(self, exchange_id: str) -> None:
        with self._lock:
            for claim_id in [
                claim_id
                for claim_id, owner in self._donor_holds.items()
                if owner == exchange_id
            ]:
                del self._donor_holds[claim_id]

    def abort(self, reservation: ExchangeReservation) -> None:
        """Release unpublished claims; published claims remain authoritative."""
        with self._lock:
            if reservation.closed:
                return
            for claim in reversed(reservation.claims):
                if claim.extent.state == ExtentState.RESERVED:
                    self.allocator.free(claim.extent)
                    self._claims.pop(claim.claim_id, None)
            self._release_reservation_holds(reservation)
            reservation.closed = True

    def rollback_before_registry_publish(
        self, reservation: ExchangeReservation
    ) -> None:
        """Undo destination publication when no registry handle became visible.

        This narrow failure path is separate from ``abort``: after a handle is
        visible, a published claim is authoritative and must survive cleanup.
        """
        with self._lock:
            if reservation.closed:
                return
            for claim in reversed(reservation.claims):
                if claim.extent.state in {
                    ExtentState.RESERVED,
                    ExtentState.PUBLISHED,
                }:
                    self.allocator.free(claim.extent)
                    self._claims.pop(claim.claim_id, None)
            self._release_reservation_holds(reservation)
            reservation.closed = True

    def finish(self, reservation: ExchangeReservation) -> None:
        with self._lock:
            unreclaimed = [
                claim_id
                for claim_id in reservation.donor_claim_ids
                if self._donor_holds.get(claim_id) == reservation.exchange_id
            ]
            if unreclaimed:
                raise RuntimeError(
                    "exchange finished before reclaiming held donors: "
                    f"{unreclaimed}"
                )
            reservation.closed = True

    def claim_for_extent(self, extent: ArenaExtent) -> MemoryClaim | None:
        with self._lock:
            for claim in self._claims.values():
                if claim.extent is extent:
                    return claim
            return None

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            result = self.allocator.snapshot()
            by_purpose = {"fid": 0, "res": 0, "look": 0}
            for claim in self._claims.values():
                requested = claim.request.nbytes
                charged = claim.extent.capacity_bytes
                parts = claim.request.bytes_by_purpose()
                # Alignment padding is charged to the request's primary use;
                # the semantic components continue to sum to requested bytes.
                parts[claim.request.purpose] += charged - requested
                for purpose, value in parts.items():
                    by_purpose[purpose] += value
            result.update({f"{key}_bytes": value for key, value in by_purpose.items()})
            result["claim_count"] = len(self._claims)
            result["held_donor_count"] = len(self._donor_holds)
            result["reservation_attempts"] = self._reservation_attempts
            result["reservation_failures_budget"] = self._reservation_failures_budget
            result["reservation_failures_extent"] = self._reservation_failures_extent
            result["reservation_rollbacks"] = self._reservation_rollbacks
            result["peak_reserved_bytes"] = self._peak_reserved_bytes
            result["peak_published_bytes"] = self._peak_published_bytes
            return result

    def _require_claim(self, claim: MemoryClaim) -> None:
        if self._claims.get(claim.claim_id) is not claim:
            raise RuntimeError("unknown or released memory claim")

    def _release_reservation_holds(
        self, reservation: ExchangeReservation
    ) -> None:
        for claim_id in reservation.donor_claim_ids:
            if self._donor_holds.get(claim_id) == reservation.exchange_id:
                del self._donor_holds[claim_id]

    def _update_peaks(self) -> None:
        snapshot = self.allocator.snapshot()
        self._peak_reserved_bytes = max(
            self._peak_reserved_bytes, int(snapshot["reserved_bytes"])
        )
        self._peak_published_bytes = max(
            self._peak_published_bytes, int(snapshot["published_bytes"])
        )


__all__ = [
    "ExchangeReservation",
    "ExpertMemoryManager",
    "MemoryClaim",
    "MemoryRequest",
]
