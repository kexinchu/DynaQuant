from __future__ import annotations

import pytest

from dynaexq.core.config import Tier
from dynaexq.core.registry import ExpertKey
from dynaexq.lease.plan import DonorSpec, ExchangePlan, LeaseSpec, TransitionStep


def _request() -> LeaseSpec:
    return LeaseSpec(ExpertKey(1, 2), Tier.LO, "look", 64, 0.0, 1.0)


def test_exchange_plan_accepts_multiple_distinct_donors() -> None:
    plan = ExchangePlan(
        "x",
        (_request(),),
        (
            DonorSpec(1, ExpertKey(0, 0), "fid", 0.0),
            DonorSpec(2, ExpertKey(0, 1), "res", 0.0),
        ),
        (
            TransitionStep("reclaim", "reclaim"),
            TransitionStep("copy", "copy", ("reclaim",)),
            TransitionStep("publish", "publish", ("copy",)),
        ),
        0.01,
    )
    assert len(plan.donors) == 2


def test_exchange_plan_rejects_duplicate_donor() -> None:
    donor = DonorSpec(1, ExpertKey(0, 0), "res", 0.0)
    with pytest.raises(ValueError, match="donor"):
        ExchangePlan("x", (_request(),), (donor, donor), (), 0.01)


def test_exchange_plan_rejects_unknown_schedule_dependency() -> None:
    with pytest.raises(ValueError, match="dependency"):
        ExchangePlan(
            "x",
            (_request(),),
            (),
            (TransitionStep("copy", "copy", ("missing",)),),
            0.01,
        )
