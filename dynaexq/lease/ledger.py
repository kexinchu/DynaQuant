"""Time-bounded byte leases and the live-plus-reserved invariant."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ByteLease:
    """One claim on the expert-memory budget.

    ``t_start`` and ``t_finish`` are a control commitment, not a promise that
    the bytes are occupied for exactly that interval.
    """

    use: str  # "fid", "res", or "look"
    layer: int
    expert: int
    nbytes: int
    t_start: float
    t_finish: float

    def __post_init__(self) -> None:
        if self.use not in {"fid", "res", "look"}:
            raise ValueError(f"unknown lease use: {self.use}")
        if self.nbytes <= 0:
            raise ValueError("lease bytes must be positive")
        if self.t_finish < self.t_start:
            raise ValueError("lease lifetime ends before it starts")


class BudgetViolation(RuntimeError):
    """Raised when live plus reserved bytes would exceed the budget."""


class LeaseLedger:
    """Published bytes plus reserved destinations, checked on every mutation."""

    def __init__(self, budget: int) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        self.budget = budget
        self._live: list[ByteLease] = []
        self._reserved: list[ByteLease] = []

    def live_bytes(self) -> int:
        return sum(lease.nbytes for lease in self._live)

    def reserved_bytes(self) -> int:
        return sum(lease.nbytes for lease in self._reserved)

    def slack(self) -> int:
        return self.budget - self.live_bytes() - self.reserved_bytes()

    def assert_invariant(self) -> None:
        if self.live_bytes() + self.reserved_bytes() > self.budget:
            raise BudgetViolation(
                f"live+reserved {self.live_bytes() + self.reserved_bytes()} > {self.budget}"
            )

    def reserve(self, lease: ByteLease) -> None:
        if lease.nbytes > self.slack():
            raise BudgetViolation(
                f"cannot reserve {lease.nbytes} bytes with slack {self.slack()}"
            )
        self._reserved.append(lease)
        self.assert_invariant()

    def publish(self, lease: ByteLease) -> None:
        """Move a reserved destination into the published set."""
        for index, reserved in enumerate(self._reserved):
            if (
                reserved.use == lease.use
                and reserved.layer == lease.layer
                and reserved.expert == lease.expert
                and reserved.nbytes == lease.nbytes
            ):
                self._reserved.pop(index)
                self._live.append(lease)
                self.assert_invariant()
                return
        raise KeyError("no matching reserved lease to publish")

    def release(self, lease: ByteLease) -> None:
        for store in (self._live, self._reserved):
            for index, current in enumerate(store):
                if current == lease:
                    store.pop(index)
                    self.assert_invariant()
                    return
        raise KeyError("lease is not held by the ledger")

    def peak_of(self, schedule: list[list[ByteLease]]) -> int:
        """Peak of live bytes plus every lease active in each schedule step.

        A step lists the leases that coexist during that interval, including
        an old published block that still has readers and its replacement.
        """
        peak = self.live_bytes() + self.reserved_bytes()
        for step in schedule:
            occupied = self.live_bytes() + self.reserved_bytes() + sum(
                lease.nbytes for lease in step
            )
            peak = max(peak, occupied)
        return peak

    def try_reserve_all(self, leases: list[ByteLease]) -> bool:
        """All-or-nothing reservation. A failure leaves the ledger unchanged."""
        if sum(lease.nbytes for lease in leases) > self.slack():
            return False
        for lease in leases:
            self._reserved.append(lease)
        self.assert_invariant()
        return True
