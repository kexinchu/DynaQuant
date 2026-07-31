from __future__ import annotations

import hashlib
import json

import pytest
import torch

from dynaexq.core import DynaExqConfig
from dynaexq.core import ExpertKey, Tier
from dynaexq.core.scheduler import TransitionReq
from dynaexq.experiments.eval_dynamic import (
    _ablation_paper_metrics,
    _configure_ablation,
    _load_calibration_prompts,
    _load_initial_map,
    _low_expert_set_metadata,
    _quality_average_pct,
    _ranking_sha256,
    _runtime_overhead_paper_metrics,
    _validate_formal_runtime_final_state,
)
from dynaexq.integration.moe_wrapper import MoEWrapper


class _SchedulerMustNotRun:
    def should_update(self, step: int) -> bool:
        raise AssertionError(f"scheduler ran at step {step}")


class _SchedulerRuns:
    def should_update(self, step: int) -> bool:
        return True

    def plan(self, **kwargs) -> list:
        return []


class _Tracker:
    def reset(self) -> None:
        pass


class _Observer:
    pass


class _Registry:
    def tier_snapshot(self) -> dict:
        return {}


def _config() -> DynaExqConfig:
    return DynaExqConfig.from_yaml("dynaexq/configs/qwen30b.yaml")


def test_ablation_switches_are_real_runtime_modes():
    full = _config()
    assert _configure_ablation(full, "full") == (False, True)
    assert full.scheduler.delta_score_margin > 0.0

    static = _config()
    assert _configure_ablation(static, "static") == (False, False)

    blocking = _config()
    assert _configure_ablation(blocking, "blocking") == (True, True)

    no_hysteresis = _config()
    assert _configure_ablation(no_hysteresis, "no_hysteresis") == (
        False,
        True,
    )
    assert no_hysteresis.scheduler.delta_score_margin == 0.0


def test_static_wrapper_does_not_call_scheduler():
    wrapper = MoEWrapper(
        model=torch.nn.Identity(),
        router_observer=_Observer(),  # type: ignore[arg-type]
        hotness_tracker=_Tracker(),  # type: ignore[arg-type]
        scheduler=_SchedulerMustNotRun(),  # type: ignore[arg-type]
        registry=_Registry(),  # type: ignore[arg-type]
        scheduler_enabled=False,
    )
    value = torch.tensor([1.0])
    torch.testing.assert_close(wrapper(value), value)
    assert wrapper.get_stats()["scheduler_enabled"] is False


def test_wrapper_records_raw_scheduler_control_plane_samples():
    wrapper = MoEWrapper(
        model=torch.nn.Identity(),
        router_observer=_Observer(),  # type: ignore[arg-type]
        hotness_tracker=_Tracker(),  # type: ignore[arg-type]
        scheduler=_SchedulerRuns(),  # type: ignore[arg-type]
        registry=_Registry(),  # type: ignore[arg-type]
    )
    wrapper(torch.tensor([1.0]))
    stats = wrapper.get_stats()
    assert stats["scheduler_update_count"] == 1
    assert len(stats["scheduler_update_samples_ms"]) == 1
    assert stats["scheduler_mean_ms"] >= 0.0
    assert stats["scheduler_p99_ms"] == stats["scheduler_mean_ms"]


def test_wrapper_preserves_adjacent_scheduler_swap_as_one_unit():
    requests = [
        TransitionReq(
            ExpertKey(3, 4), Tier.HI, Tier.LO, "leave", 200
        ),
        TransitionReq(
            ExpertKey(3, 9), Tier.LO, Tier.HI, "enter", 200
        ),
        TransitionReq(
            ExpertKey(7, 2), Tier.LO, Tier.HI, "cold_start", 200
        ),
    ]
    assert MoEWrapper._transition_units(requests) == [
        (requests[0], requests[1]),
        (requests[2],),
    ]


def test_ablation_paper_metrics_are_derived_from_raw_results():
    quality = {
        name: {"score": score, "failed": 0}
        for name, score in zip(
            ("mmlu_pro", "gpqa", "aime25", "gsm8k", "humaneval"),
            (0.5, 0.6, 0.7, 0.8, 0.9),
        )
    }
    performance = {
        "metrics": {
            "throughput_tokens_s": {"mean": 123.0},
            "model_e2e_ms": {"p99": 4560.0},
        }
    }
    assert _ablation_paper_metrics(quality, performance) == {
        "average_accuracy_pct": 70.0,
        "throughput_tokens_s": 123.0,
        "p99_s": 4.56,
    }
    assert _quality_average_pct(quality) == 70.0


def test_runtime_overhead_metrics_are_derived_from_raw_telemetry():
    assert _runtime_overhead_paper_metrics(
        {
            "resident_expert_bytes": 2_000_000_000,
            "transient_expert_bytes": 100_000_000,
            "host_cache": {"host_packed_bytes": 3_000_000_000},
        },
        {"scheduler_mean_ms": 1.5, "scheduler_p99_ms": 2.5},
        {
            "total_promotions": 3,
            "total_demotions": 4,
            "copied_bytes": 500_000_000,
        },
        {
            "samples": [
                {"process_hbm_used_peak_bytes": 4_000_000_000},
                {"process_hbm_used_peak_bytes": 4_500_000_000},
            ]
        },
        48_000_000_000,
    ) == {
        "hbm_budget_gb": 48.0,
        "peak_process_hbm_used_gb": 4.5,
        "resident_expert_pool_gb": 2.0,
        "transient_expert_pool_gb": 0.1,
        "migration_count": 7,
        "transferred_gb": 0.5,
        "scheduler_mean_ms": 1.5,
        "scheduler_p99_ms": 2.5,
        "pinned_expert_cache_gb": 3.0,
    }
    with pytest.raises(RuntimeError, match="HBM high-water"):
        _runtime_overhead_paper_metrics(
            {
                "resident_expert_bytes": 2_000_000_000,
                "transient_expert_bytes": 100_000_000,
                "host_cache": {"host_packed_bytes": 3_000_000_000},
            },
            {"scheduler_mean_ms": 1.5, "scheduler_p99_ms": 2.5},
            {
                "total_promotions": 3,
                "total_demotions": 4,
                "copied_bytes": 500_000_000,
            },
            {
                "samples": [
                    {"process_hbm_used_peak_bytes": 48_000_000_001},
                ]
            },
            48_000_000_000,
        )


def test_formal_runtime_final_state_fails_closed():
    wrapper = {
        "scheduler_enabled": True,
        "scheduler_update_samples_ms": [1.0, 2.0],
        "scheduler_update_count": 2,
    }
    stats = {
        "accepted_requests": 2,
        "accepted_bytes": 200,
        "total_promotions": 1,
        "total_demotions": 1,
        "failed_transitions": 0,
        "copied_bytes": 200,
        "precise_fence_reclaims": 2,
        "global_sync_reclaims": 0,
        "active_transitions": 0,
        "budget": {
            "total_cap": 1000,
            "total_live": 500,
            "hi_pending": 0,
            "lo_pending": 0,
            "staging_used": 0,
        },
    }
    _validate_formal_runtime_final_state(
        wrapper,
        stats,
        scheduler_enabled=True,
        require_online_activity=True,
    )

    stats["global_sync_reclaims"] = 1
    with pytest.raises(RuntimeError, match="auditable transition state"):
        _validate_formal_runtime_final_state(
            wrapper,
            stats,
            scheduler_enabled=True,
            require_online_activity=True,
        )
    stats["global_sync_reclaims"] = 0

    stats["budget"]["lo_pending"] = 1
    with pytest.raises(RuntimeError, match="auditable transition state"):
        _validate_formal_runtime_final_state(
            wrapper,
            stats,
            scheduler_enabled=True,
            require_online_activity=True,
        )
    stats["budget"]["lo_pending"] = 0

    stats["accepted_requests"] = 0
    stats["total_promotions"] = 0
    stats["total_demotions"] = 0
    stats["accepted_bytes"] = 0
    stats["copied_bytes"] = 0
    stats["precise_fence_reclaims"] = 0
    with pytest.raises(RuntimeError, match="auditable transition state"):
        _validate_formal_runtime_final_state(
            wrapper,
            stats,
            scheduler_enabled=True,
            require_online_activity=True,
        )
    _validate_formal_runtime_final_state(
        wrapper,
        stats,
        scheduler_enabled=True,
        require_online_activity=False,
    )


def test_calibration_loader_rejects_test_splits_and_hashes_selection(tmp_path):
    source = tmp_path / "calibration.jsonl"
    rows = [
        {
            "dataset": "independent-corpus",
            "split": "train",
            "id": index,
            "prompt": f"prompt {index}",
        }
        for index in range(130)
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    selected, metadata = _load_calibration_prompts(
        str(source),
        seed=42,
        max_prompts=128,
    )
    assert len(selected) == 128
    assert metadata["source_prompt_count"] == 130
    assert metadata["prompt_count"] == 128
    assert metadata["source_sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()

    rows[0]["split"] = "test"
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="train/validation"):
        _load_calibration_prompts(str(source), seed=42, max_prompts=128)


def test_initial_map_binds_checkpoint_model_and_ranking_hash(tmp_path):
    config = _config()
    checkpoint = {"local": False, "revision": "immutable"}
    ranking = {
        str(layer): list(range(config.model.experts_per_layer))
        for layer in range(config.model.layers)
    }
    artifact = {
        "schema_version": 2,
        "artifact_type": "dynaexq_initial_expert_ranking",
        "checkpoint": checkpoint,
        "model_config": config.to_dict()["model"],
        "calibration": {
            "prompt_count": 128,
            "allowed_splits_only": True,
            "precision_policy": "all_low",
            "aggregation": "mean_per_prompt_routing_probability_mass",
            "source_sha256": "source-sha",
            "selected_ids_sha256": "ids-sha",
        },
        "environment": {
            "git": {"commit": "clean-commit", "dirty": False}
        },
        "expert_ranking": ranking,
        "ranking_sha256": _ranking_sha256(ranking),
    }
    path = tmp_path / "initial-map.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    loaded, provenance = _load_initial_map(
        str(path),
        checkpoint,
        config,
    )
    assert loaded[0] == list(range(config.model.experts_per_layer))
    assert provenance["ranking_sha256"] == artifact["ranking_sha256"]

    artifact["expert_ranking"]["0"] = list(
        reversed(artifact["expert_ranking"]["0"])
    )
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="ranking hash mismatch"):
        _load_initial_map(str(path), checkpoint, config)


def test_coldest_prefix_hash_is_deterministic_and_zero_safe():
    ranking = {0: [2, 0, 1], 1: [1, 2, 0]}
    empty, empty_hash = _low_expert_set_metadata(ranking, low_count=0)
    cold, cold_hash = _low_expert_set_metadata(ranking, low_count=1)
    assert empty == {"0": [], "1": []}
    assert cold == {"0": [1], "1": [0]}
    assert len(empty_hash) == len(cold_hash) == 64
    assert empty_hash != cold_hash
