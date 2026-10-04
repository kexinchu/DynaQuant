"""Executable form of a donor-backed byte exchange."""

from __future__ import annotations

from dataclasses import dataclass

from dynaexq.core.config import Tier
from dynaexq.core.registry import ExpertKey


@dataclass(frozen=True, slots=True)
class LeaseSpec:
    key: ExpertKey
    tier: Tier
    purpose: str
    nbytes: int
    earliest_issue: float
    deadline: float


@dataclass(frozen=True, slots=True)
class DonorSpec:
    claim_id: int
    key: ExpertKey
    purpose: str
    earliest_reclaim: float


@dataclass(frozen=True, slots=True)
class TransitionStep:
    step_id: str
    kind: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExchangePlan:
    exchange_id: str
    requests: tuple[LeaseSpec, ...]
    donors: tuple[DonorSpec, ...]
    schedule: tuple[TransitionStep, ...]
    predicted_gain_s: float

    def __post_init__(self) -> None:
        if not self.exchange_id:
            raise ValueError("exchange id must not be empty")
        if not self.requests:
            raise ValueError("an exchange must contain a request")
        donor_ids = [donor.claim_id for donor in self.donors]
        if len(donor_ids) != len(set(donor_ids)):
            raise ValueError("one donor cannot appear twice in an exchange")
        alternatives = [(item.key, item.purpose) for item in self.requests]
        if len(alternatives) != len(set(alternatives)):
            raise ValueError("duplicate request lease in exchange")
        step_ids = {step.step_id for step in self.schedule}
        if len(step_ids) != len(self.schedule):
            raise ValueError("transition step ids must be unique")
        for step in self.schedule:
            unknown = set(step.dependencies) - step_ids
            if unknown:
                raise ValueError(f"unknown transition dependency: {sorted(unknown)}")


__all__ = ["DonorSpec", "ExchangePlan", "LeaseSpec", "TransitionStep"]
