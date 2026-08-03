from __future__ import annotations

from types import SimpleNamespace

import pytest

from dynaexq.baselines import moe_infinity as baseline


class _Archer:
    @staticmethod
    def is_tensor_offloaded(tensor_id):
        return tensor_id in {11, 13}


class _Prefetcher:
    def __init__(self):
        self.forwarded = []

    def prefetch_experts_list(self, layer_id, expert_list):
        self.forwarded.append((layer_id, list(expert_list)))


def _runtime():
    prefetcher = _Prefetcher()
    config = SimpleNamespace(
        prefetch=True,
        speculative_prefetch=True,
        speculative_prefetch_overlap=True,
        use_native_engine=False,
    )
    engine = SimpleNamespace(
        expert_tensor_map={(0, 0): 11, (0, 1): 12, (1, 0): 13},
        archer_engine=_Archer(),
        expert_prefetcher=prefetcher,
        archer_config=config,
    )
    return SimpleNamespace(
        engine=engine,
        _configure_hook=lambda input_ids: None,
    )


def test_external_runtime_telemetry_preserves_original_prefetch():
    runtime = _runtime()
    telemetry = baseline.PrefetchTelemetry.install(runtime)
    runtime.engine.expert_prefetcher.prefetch_experts_list(3, [7, 9])
    assert runtime.engine.expert_prefetcher.forwarded == [(3, [7, 9])]
    assert telemetry.snapshot(
        total_expert_tensors=3,
        offloaded_expert_tensors=2,
    ) == {
        "prefetch_calls": 1,
        "prefetch_requested_experts": 2,
        "prefetch_layers_touched": [3],
        "prefetch_unique_experts": [7, 9],
        "total_expert_tensors": 3,
        "offloaded_expert_tensors": 2,
    }
    telemetry.reset()
    assert telemetry.calls == 0
    telemetry.close()
    runtime.engine.expert_prefetcher.prefetch_experts_list(4, [1])
    assert runtime.engine.expert_prefetcher.forwarded[-1] == (4, [1])


def test_offload_state_and_feature_contract_fail_closed():
    runtime = _runtime()
    assert baseline.count_offloaded_expert_tensors(runtime) == (3, 2)
    assert baseline.validate_runtime_configuration(runtime) == {
        "prefetch": True,
        "speculative_prefetch": True,
        "speculative_prefetch_overlap": True,
        "use_native_engine": False,
    }
    runtime.engine.archer_config.speculative_prefetch = False
    with pytest.raises(RuntimeError, match="feature mismatch"):
        baseline.validate_runtime_configuration(runtime)


def test_official_checkout_identity_rejects_wrong_commit(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()

    def fake_git(repo, *args, binary=False):
        if args == ("remote", "get-url", "origin"):
            return baseline.OFFICIAL_REPOSITORY + ".git"
        if args == ("rev-parse", "HEAD"):
            return "wrong"
        raise AssertionError(args)

    monkeypatch.setattr(baseline, "_git", fake_git)
    with pytest.raises(ValueError, match="must be checked out"):
        baseline.verify_official_checkout(tmp_path)
