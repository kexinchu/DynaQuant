from dynaexq.core.registry import ExpertKey
from dynaexq.policy.risk import QualityRiskAccount


def test_quality_risk_uses_routing_weight_times_sensitivity() -> None:
    first = ExpertKey(0, 0)
    second = ExpertKey(0, 1)
    account = QualityRiskAccount(
        {first: 2.0, second: 0.5},
        limit=0.5,
    )
    weights = {first: 0.4, second: 0.2}
    assert account.risk(weights, {first}) == 0.1
    assert account.risk(weights, set()) == 0.9
    assert account.feasible(weights, {first})
    assert not account.feasible(weights, set())


def test_quality_risk_repair_is_minimum_highest_contribution_prefix() -> None:
    keys = [ExpertKey(0, expert) for expert in range(3)]
    account = QualityRiskAccount(
        {keys[0]: 1.0, keys[1]: 2.0, keys[2]: 1.0},
        limit=0.3,
    )
    weights = {keys[0]: 0.4, keys[1]: 0.3, keys[2]: 0.1}
    assert account.repair_order(weights, set(), keys) == [keys[1], keys[0]]
    assert account.snapshot()["repair_plans"] == 1


def test_quality_risk_reports_unrepairable_state() -> None:
    first = ExpertKey(0, 0)
    account = QualityRiskAccount({first: 1.0}, limit=0.0)
    assert account.repair_order({first: 1.0}, set(), []) == []
    assert account.snapshot()["unrepaired_states"] == 1
