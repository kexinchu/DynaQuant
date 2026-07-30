from __future__ import annotations

import hashlib
import json

from scripts import audit_paper_results as audit
from scripts import compare_quality_artifacts as comparison


def _process_hbm_monitor() -> dict:
    return {
        "backend": "nvml",
        "scope": "current_process_selected_device_used_bytes",
        "pid": 123,
        "poll_interval_ms": 2.0,
        "cuda_device_indices": [0],
        "device_names": ["NVIDIA RTX A6000"],
        "device_uuids": ["GPU-test"],
        "device_total_bytes": [48 * 1024**3],
        "includes_non_pytorch_allocations": True,
        "excludes_other_processes": True,
        "foreign_compute_residency_allowed": True,
        "foreign_compute_activity_policy": (
            "reject_nonzero_nvml_process_utilization"
        ),
        "process_utilization_supported": True,
    }


def _performance_samples(metric_names: tuple[str, ...]) -> list[dict]:
    return [
        {
            **{name: float(index) for name in metric_names},
            "process_hbm_used_baseline_bytes": 0,
            "process_hbm_used_peak_delta_bytes": index,
            "process_hbm_poll_samples": 2,
            "foreign_compute_resident_processes_peak": 1,
            "foreign_hbm_used_peak_bytes": 1024,
            "foreign_compute_activity_samples": 0,
            "foreign_sm_util_max_pct": 0,
            "foreign_mem_util_max_pct": 0,
            "process_util_poll_samples": 2,
        }
        for index in range(1, 101)
    ]


def _calibrated_initial_map(
    checkpoint: dict,
    *,
    layers: int,
    experts: int,
) -> dict:
    ranking = {
        str(layer): list(range(experts))
        for layer in range(layers)
    }
    ranking_hash = hashlib.sha256(
        json.dumps(
            ranking,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "artifact_sha256": "map-artifact-sha",
        "ranking_sha256": ranking_hash,
        "checkpoint": checkpoint,
        "model_config": {
            "layers": layers,
            "experts_per_layer": experts,
        },
        "calibration": {
            "prompt_count": 128,
            "allowed_splits_only": True,
            "precision_policy": "all_low",
            "aggregation": "mean_per_prompt_routing_probability_mass",
            "source_sha256": "source-sha",
            "selected_ids_sha256": "ids-sha",
        },
        "environment": {
            "git": {"commit": "calibration-commit", "dirty": False}
        },
        "expert_ranking": ranking,
    }


def test_accuracy_parser_includes_bold_dynaexq_rows_and_average():
    rows = audit.manuscript_rows()
    assert len(rows) == 9
    assert rows[("Qwen3-MoE-30B", "DynaExQ")] == [
        75.0,
        53.03,
        33.33,
        90.0,
        84.76,
        67.22,
    ]
    assert rows[("Qwen3-Next-80B", "DynaExQ")][-1] == 77.15


def test_ablation_parser_covers_both_models_and_all_runtime_modes():
    rows = audit.manuscript_ablation_rows()
    assert len(rows) == 8
    assert rows[("qwen30b", "blocking")] == (67.12, 790.0, 83.0)
    assert rows[("qwen80b", "no_hysteresis")] == (76.93, 279.0, 198.0)


def test_activation_density_parser_covers_every_table_cell():
    rows = audit.manuscript_activation_density()
    assert len(rows) == 6
    assert rows[("qwen30b", "decode")] == (
        6.3,
        7.9,
        11.8,
        16.1,
        20.8,
        26.0,
    )
    assert rows[("qwen30b", "prefill")] == (
        59.8,
        70.0,
        82.9,
        89.7,
        92.4,
        94.0,
    )
    assert rows[("deepseek_v2_lite", "prefill")][-1] == 96.3


def test_complete_manifest_verifies_all_groups_and_hashes(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "created_at": "2026-01-01T00:00:00+00:00",
                "checkpoint": {
                    "local": False,
                    "revision": "immutable-revision",
                },
                "environment": {
                    "git": {"commit": "abc123", "dirty": False},
                    "process_max_rss_bytes": 1024,
                },
                "seed": 42,
                "benchmark": {
                    "scope": "isolated_model",
                    "warmup_iterations": 5,
                    "measured_iterations": 100,
                    "samples": [{} for _ in range(100)],
                },
                "benchmarks": {"mmlu_pro": {}},
                "runtime_initialization": {},
                "wrapper_stats": {
                    "forward_steps": 1,
                    "router_observations": 2,
                    "attached_layers": 1,
                    "router_layers": 1,
                },
                "transition_stats": {
                    "failed_transitions": 0,
                    "budget": {"total_live": 9, "total_cap": 10},
                },
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    record = {
        "claim_id": "placeholder",
        "path": "artifact.json",
        "sha256": digest,
        "command": "python reproduce.py",
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "groups": {
                    group: [
                        {
                            **record,
                            "claim_id": f"{group}:placeholder",
                        }
                    ]
                    for group in audit.REQUIRED_MANIFEST_GROUPS
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "MANIFEST", manifest)
    monkeypatch.setattr(
        audit,
        "REQUIRED_CLAIMS",
        {
            group: {f"{group}:placeholder"}
            for group in audit.REQUIRED_MANIFEST_GROUPS
        },
    )
    monkeypatch.setattr(
        audit,
        "validate_manifest_artifact",
        lambda *args, **kwargs: [],
    )
    assert audit.validate_complete_manifest() == []

    artifact.write_text('{"schema_version": 3}', encoding="utf-8")
    assert any(
        problem.startswith("MANIFEST HASH MISMATCH")
        for problem in audit.validate_complete_manifest()
    )


def test_model_and_packed_cache_contracts_match_runtime():
    assert audit.validate_implementation_contracts() == []


def test_manifest_artifact_rejects_dirty_or_incomplete_performance_runs():
    artifact = {
        "schema_version": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "checkpoint": {"local": False, "revision": "sha"},
        "environment": {
            "git": {"commit": "abc123", "dirty": True},
            "process_max_rss_bytes": 1024,
        },
        "seed": 42,
        "benchmark": {
            "scope": "isolated_model",
            "warmup_iterations": 5,
            "measured_iterations": 100,
            "samples": [{} for _ in range(99)],
        },
    }
    problems = audit.validate_manifest_artifact(
        "performance",
        "performance[0]",
        artifact,
    )
    assert "DIRTY MANIFEST RUN: performance[0]" in problems
    assert "INCOMPLETE PERFORMANCE SAMPLES: performance[0]" in problems


def test_dynamic_artifact_requires_router_and_budget_evidence():
    problems = audit.validate_dynamic_runtime(
        "Qwen3-Next-80B / DynaExQ",
        {
            "runtime_initialization": {},
            "wrapper_stats": {
                "forward_steps": 5,
                "router_observations": 0,
                "attached_layers": 0,
                "router_layers": 1,
            },
            "transition_stats": {
                "failed_transitions": 1,
                "budget": {"total_live": 11, "total_cap": 10},
            },
        },
    )
    assert any("INCOMPLETE ROUTER OBSERVATION" in item for item in problems)
    assert any("INCOMPLETE MODEL INTEGRATION" in item for item in problems)
    assert any("FAILED DYNAMIC TRANSITIONS" in item for item in problems)
    assert any("INCOMPLETE DYNAMIC BUDGET SNAPSHOT" in item for item in problems)
    assert any("INVALID DYNAMIC SCHEDULER TELEMETRY" in item for item in problems)
    assert any("INCOMPLETE TRANSITION LIFECYCLE" in item for item in problems)


def test_manifest_claim_sets_cover_every_reported_operating_point():
    assert len(audit.REQUIRED_CLAIMS["performance"]) == 42
    assert len(audit.REQUIRED_CLAIMS["quality_significance"]) == 3
    assert len(audit.REQUIRED_CLAIMS["ablation"]) == 8
    assert len(audit.REQUIRED_CLAIMS["runtime_overhead"]) == 2
    assert len(audit.REQUIRED_CLAIMS["budget_sensitivity"]) == 14
    assert len(audit.REQUIRED_CLAIMS["activation_density"]) == 6
    assert len(audit.REQUIRED_CLAIMS["offload_waiting"]) == 3
    assert len(audit.REQUIRED_CLAIMS["routing_hotset"]) == 3
    assert len(audit.REQUIRED_CLAIMS["perplexity_curve"]) == 2
    assert len(audit.REQUIRED_CLAIMS["figure_bundle"]) == 1
    assert sum(map(len, audit.REQUIRED_CLAIMS.values())) == 84
    assert len(audit.EXPECTED_EMPIRICAL_FIGURES) == 17
    assert (
        "performance:qwen30b:moe_infinity:bs32"
        in audit.REQUIRED_CLAIMS["performance"]
    )
    assert (
        "performance:qwen80b:moe_infinity:bs32"
        not in audit.REQUIRED_CLAIMS["performance"]
    )
    assert (
        "budget_sensitivity:qwen30b:ratio20"
        in audit.REQUIRED_CLAIMS["budget_sensitivity"]
    )


def test_quality_significance_claim_recomputes_paired_predictions(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    paper_dir = tmp_path / "results" / "paper"
    paper_dir.mkdir(parents=True)

    def result(correctness):
        return {
            "dataset": {
                "repository": "test/quality",
                "revision": "dataset-sha",
                "config": "default",
                "split": "test",
                "source_rows": len(correctness),
                "fingerprint": "fingerprint",
                "evaluated_rows": len(correctness),
            },
            "total": len(correctness),
            "evaluated": len(correctness),
            "failed": 0,
            "skipped": 0,
            "score": sum(correctness) / len(correctness),
            "details": [
                {
                    "sample_id": f"sample-{index}",
                    "correct": correct,
                }
                for index, correct in enumerate(correctness)
            ],
        }

    def quality(paper_method, correctness):
        return {
            "schema_version": 2,
            "artifact_type": (
                "dynaexq_quality"
                if paper_method == "dynaexq"
                else "quality_evaluation"
            ),
            "created_at": "2026-01-01T00:00:00+00:00",
            "paper_model": "qwen30b",
            "paper_method": paper_method,
            "checkpoint": {
                "local": False,
                "revision": f"{paper_method}-checkpoint",
            },
            "seed": 42,
            "evaluation_protocol": {"name": "tc_main_v2"},
            "environment": {
                "git": {
                    "commit": f"{paper_method}-commit",
                    "dirty": False,
                },
                "process_max_rss_bytes": 1024,
            },
            "benchmarks": {
                name: result(correctness)
                for name in audit.SIGNIFICANCE_BENCHMARKS
            },
        }

    left_path = paper_dir / "qwen30b_int4_quality.json"
    right_path = paper_dir / "qwen30b_dynaexq_quality.json"
    left_path.write_text(
        json.dumps(quality("static_int4", [True, False, False, True])),
        encoding="utf-8",
    )
    right_path.write_text(
        json.dumps(quality("dynaexq", [True, True, False, True])),
        encoding="utf-8",
    )
    artifact = comparison.build_significance_artifact(
        left_path,
        right_path,
        paper_model="qwen30b",
    )
    artifact["environment"] = {
        "git": {"commit": "comparison-commit", "dirty": False},
        "process_max_rss_bytes": 1024,
    }
    assert audit.validate_manifest_artifact(
        "quality_significance",
        "quality_significance[0]",
        artifact,
        "quality_significance:qwen30b:static_ptq_vs_dynaexq",
    ) == []

    artifact["benchmarks"]["mmlu_pro"]["mcnemar_exact_p"] = 0.0
    problems = audit.validate_manifest_artifact(
        "quality_significance",
        "quality_significance[0]",
        artifact,
        "quality_significance:qwen30b:static_ptq_vs_dynaexq",
    )
    assert (
        "INVALID QUALITY SIGNIFICANCE RESULTS: quality_significance[0]"
        in problems
    )


def test_routing_hotset_claim_recomputes_exact_dispatch_top10():
    top_sets = {
        "wikitext": list(range(0, 10)),
        "gsm8k": list(range(10, 20)),
        "humaneval": list(range(20, 30)),
    }
    workloads = {}
    for name, top10 in top_sets.items():
        counts = [0] * 128
        for rank, expert in enumerate(top10):
            counts[expert] = 100 - rank
        summary = (
            {"metric": "perplexity", "windows": 128, "total_tokens": 1000}
            if name == "wikitext"
            else {
                "metric": "accuracy" if name == "gsm8k" else "pass@1",
                "total": 1319 if name == "gsm8k" else 164,
                "evaluated": 1319 if name == "gsm8k" else 164,
                "failed": 0,
            }
        )
        workloads[name] = {
            "expert_counts": counts,
            "total_dispatches": sum(counts),
            "top10": top10,
            "dataset": {
                "revision": f"{name}-revision",
                "fingerprint": f"{name}-fingerprint",
            },
            "request_limit": None,
            "evaluation_summary": summary,
        }
    artifact = {
        "schema_version": 2,
        "artifact_type": "routing_hotset_bundle",
        "created_at": "2026-01-01T00:00:00+00:00",
        "checkpoint": {"local": False, "revision": "checkpoint-sha"},
        "environment": {
            "git": {"commit": "code-sha", "dirty": False},
            "process_max_rss_bytes": 1024,
        },
        "seed": 42,
        "paper_model": "qwen30b",
        "layer": 15,
        "profile_protocol": {
            "name": "tc_routing_hotset_v1",
            "precision_policy": "all_low",
            "scheduler_enabled": False,
            "counter": "selected_token_expert_dispatches",
            "topk": 8,
            "workload_order": ["wikitext", "gsm8k", "humaneval"],
        },
        "wrapper_stats": {
            "scheduler_enabled": False,
            "routing_profile_enabled": True,
            "router_observations": 1,
        },
        "runtime_initialization": {
            "requested_high_precision_ratio": 0.0,
            "realized_high_precision_ratio": 0.0,
            "n_hi": [0] * 48,
        },
        "workloads": workloads,
    }
    assert audit.validate_manifest_artifact(
        "routing_hotset",
        "routing_hotset[0]",
        artifact,
        "routing_hotset:qwen30b:wikitext:layer15",
    ) == []


def test_performance_claim_recomputes_raw_summaries():
    metric_names = (
        "model_ttft_ms",
        "model_tpot_ms",
        "model_e2e_ms",
        "throughput_tokens_s",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "process_hbm_used_peak_bytes",
    )
    samples = _performance_samples(metric_names)
    summary = {
        name: {
            "mean": 50.5,
            "p50": 50.0,
            "p95": 95.0,
            "p99": 99.0,
            "min": 1.0,
            "max": 100.0,
        }
        for name in metric_names
    }
    artifact = {
        "schema_version": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "checkpoint": {"local": False, "revision": "sha"},
        "environment": {
            "git": {"commit": "abc123", "dirty": False},
            "process_max_rss_bytes": 1024,
            "gpus": ["NVIDIA RTX A6000"],
        },
        "seed": 42,
        "paper_model": "qwen30b",
        "paper_method": "static_ptq",
        "method": "quantized_checkpoint",
        "evaluation_protocol": {
            "name": "tc_isolated_performance_v2",
            "seed": 42,
            "process_hbm_high_water": True,
        },
        "benchmark": {
            "scope": "isolated_model",
            "batch_size": 32,
            "input_tokens": 2048,
            "output_tokens_per_sequence": 256,
            "warmup_iterations": 5,
            "measured_iterations": 100,
            "process_hbm_monitor": _process_hbm_monitor(),
            "samples": samples,
            "metrics": summary,
        },
    }
    assert audit.validate_manifest_artifact(
        "performance",
        "performance[0]",
        artifact,
        "performance:qwen30b:static_ptq:bs32",
    ) == []

    artifact["benchmark"]["metrics"]["model_e2e_ms"]["p99"] = 98.0
    problems = audit.validate_manifest_artifact(
        "performance",
        "performance[0]",
        artifact,
        "performance:qwen30b:static_ptq:bs32",
    )
    assert "PERFORMANCE SUMMARY MISMATCH: performance[0] / model_e2e_ms" in problems

    artifact["benchmark"]["metrics"]["model_e2e_ms"]["p99"] = 99.0
    artifact["benchmark"]["samples"][0]["process_hbm_poll_samples"] = 1
    problems = audit.validate_manifest_artifact(
        "performance",
        "performance[0]",
        artifact,
        "performance:qwen30b:static_ptq:bs32",
    )
    assert "INVALID PROCESS HBM SAMPLE: performance[0] / 0" in problems


def test_moe_infinity_claim_requires_pinned_source_and_active_offload():
    metric_names = (
        "model_ttft_ms",
        "model_tpot_ms",
        "model_e2e_ms",
        "throughput_tokens_s",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "process_hbm_used_peak_bytes",
    )
    samples = _performance_samples(metric_names)
    summary = {
        name: {
            "mean": 50.5,
            "p50": 50.0,
            "p95": 95.0,
            "p99": 99.0,
            "min": 1.0,
            "max": 100.0,
        }
        for name in metric_names
    }
    implementation = {
        **audit.MOE_INFINITY_IDENTITY,
        "origin": audit.MOE_INFINITY_IDENTITY["repository"] + ".git",
        "source_hash_algorithm": "sha256(git-ls-tree-r-z)",
        "clean": True,
        "paper_implementation_equivalent": False,
        "variant_note": "official source differs from paper version",
        "imported_module": "moe_infinity/__init__.py",
        "features": {
            "expert_offload": True,
            "activation_aware_cache": True,
            "prefetch": True,
            "speculative_prefetch": True,
            "speculative_prefetch_overlap": True,
            "use_native_engine": False,
        },
    }
    artifact = {
        "schema_version": 2,
        "artifact_type": "moe_infinity_performance",
        "created_at": "2026-01-01T00:00:00+00:00",
        "checkpoint": {"local": False, "revision": "sha"},
        "model_loading": {
            "mode": "pinned_huggingface_snapshot",
            "remote_revision": "sha",
            "snapshot_commit_directory": "sha",
        },
        "environment": {
            "git": {"commit": "abc123", "dirty": False},
            "process_max_rss_bytes": 1024,
            "gpus": ["NVIDIA RTX A6000"],
        },
        "hardware_contract": {
            "device_count": 1,
            "device_name": "NVIDIA RTX A6000",
            "total_memory_bytes": 48 * 1024**3,
        },
        "seed": 42,
        "evaluation_protocol": {
            "name": "tc_isolated_performance_v2",
            "seed": 42,
            "process_hbm_high_water": True,
        },
        "paper_model": "qwen30b",
        "paper_method": "moe_infinity",
        "method": "official_external_offload_runtime",
        "baseline_implementation": implementation,
        "baseline_runtime_stats": {
            "prefetch_calls": 200,
            "prefetch_requested_experts": 400,
            "prefetch_layers_touched": [1, 2],
            "prefetch_unique_experts": [3, 4],
            "total_expert_tensors": 6144,
            "offloaded_expert_tensors": 5000,
        },
        "runtime_config": {
            "offload_path": "/local-ssd/qwen30",
            "device_memory_ratio": 0.7,
            "prefetch": True,
            "speculative_prefetch": True,
            "speculative_prefetch_overlap": True,
            "use_native_engine": False,
        },
        "benchmark": {
            "scope": "isolated_model",
            "batch_size": 32,
            "input_tokens": 2048,
            "output_tokens_per_sequence": 256,
            "warmup_iterations": 5,
            "measured_iterations": 100,
            "process_hbm_monitor": _process_hbm_monitor(),
            "samples": samples,
            "metrics": summary,
        },
    }
    assert audit.validate_manifest_artifact(
        "performance",
        "performance[0]",
        artifact,
        "performance:qwen30b:moe_infinity:bs32",
    ) == []

    artifact["baseline_runtime_stats"]["offloaded_expert_tensors"] = 0
    problems = audit.validate_manifest_artifact(
        "performance",
        "performance[0]",
        artifact,
        "performance:qwen30b:moe_infinity:bs32",
    )
    assert "INACTIVE MOE-INFINITY RUNTIME: performance[0]" in problems


def test_sensitivity_claim_proves_exact_per_layer_quota():
    benchmark_names = (
        "mmlu_pro",
        "gpqa",
        "aime25",
        "gsm8k",
        "humaneval",
    )
    checkpoint = {"local": False, "revision": "sha"}
    artifact = {
        "schema_version": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "checkpoint": checkpoint,
        "environment": {
            "git": {"commit": "abc123", "dirty": False},
            "process_max_rss_bytes": 1024,
        },
        "seed": 42,
        "paper_model": "qwen30b",
        "hi_ratio_pct": 20,
        "evaluation_protocol": {"name": "tc_main_v2"},
        "sensitivity_sequence": list(benchmark_names),
        "benchmarks": {
            name: {"score": 0.5, "failed": 0}
            for name in benchmark_names
        },
        "paper_metrics": {
            "average_accuracy_pct": 50.0,
            "realized_hi_ratio_pct": 19.53125,
            "resident_expert_bytes": 1234,
        },
        "config": {
            "model": {
                "layers": 2,
                "experts_per_layer": 128,
            }
        },
        "initial_map": _calibrated_initial_map(
            checkpoint,
            layers=2,
            experts=128,
        ),
        "runtime_initialization": {
            "n_hi": [25, 25],
            "requested_high_precision_ratio": 0.20,
            "realized_high_precision_ratio": 25 / 128,
            "resident_expert_bytes": 1234,
            "bootstrap_policy": "calibrated_ranking_prefix",
            "bootstrap_hi_experts": {
                "0": list(range(25)),
                "1": list(range(25)),
            },
        },
        "wrapper_stats": {
            "forward_steps": 5,
            "router_observations": 10,
            "attached_layers": 2,
            "router_layers": 2,
            "scheduler_enabled": True,
            "scheduler_update_samples_ms": [1.0],
            "scheduler_update_count": 1,
        },
        "transition_stats": {
            "accepted_requests": 2,
            "accepted_bytes": 200,
            "total_promotions": 1,
            "total_demotions": 1,
            "copied_bytes": 200,
            "precise_fence_reclaims": 2,
            "global_sync_reclaims": 0,
            "active_transitions": 0,
            "failed_transitions": 0,
            "budget": {
                "total_live": 9,
                "total_cap": 10,
                "hi_pending": 0,
                "lo_pending": 0,
                "staging_used": 0,
            },
        },
    }
    assert audit.validate_manifest_artifact(
        "budget_sensitivity",
        "budget_sensitivity[0]",
        artifact,
        "budget_sensitivity:qwen30b:ratio20",
    ) == []

    artifact["transition_stats"]["global_sync_reclaims"] = 1
    problems = audit.validate_manifest_artifact(
        "budget_sensitivity",
        "budget_sensitivity[0]",
        artifact,
        "budget_sensitivity:qwen30b:ratio20",
    )
    assert "INVALID TRANSITION LIFECYCLE: budget_sensitivity[0]" in problems
    artifact["transition_stats"]["global_sync_reclaims"] = 0

    artifact["transition_stats"]["budget"]["hi_pending"] = 1
    problems = audit.validate_manifest_artifact(
        "budget_sensitivity",
        "budget_sensitivity[0]",
        artifact,
        "budget_sensitivity:qwen30b:ratio20",
    )
    assert (
        "INVALID DYNAMIC BUDGET SNAPSHOT: budget_sensitivity[0]"
        in problems
    )
    artifact["transition_stats"]["budget"]["hi_pending"] = 0

    artifact["runtime_initialization"]["n_hi"] = [24, 25]
    problems = audit.validate_manifest_artifact(
        "budget_sensitivity",
        "budget_sensitivity[0]",
        artifact,
        "budget_sensitivity:qwen30b:ratio20",
    )
    assert (
        "SENSITIVITY RUNTIME QUOTA MISMATCH: budget_sensitivity[0]"
        in problems
    )


def test_overhead_claim_recomputes_raw_runtime_telemetry(monkeypatch):
    metric_names = (
        "model_ttft_ms",
        "model_tpot_ms",
        "model_e2e_ms",
        "throughput_tokens_s",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "process_hbm_used_peak_bytes",
    )
    samples = _performance_samples(metric_names)
    summaries = {
        name: {
            "mean": 50.5,
            "p50": 50.0,
            "p95": 95.0,
            "p99": 99.0,
            "min": 1.0,
            "max": 100.0,
        }
        for name in metric_names
    }
    paper_metrics = {
        "hbm_budget_gb": 48.0,
        "peak_process_hbm_used_gb": 100.0 / 1e9,
        "resident_expert_pool_gb": 2.0,
        "transient_expert_pool_gb": 0.1,
        "migration_count": 7,
        "transferred_gb": 0.5,
        "scheduler_mean_ms": 50.5,
        "scheduler_p99_ms": 99.0,
        "pinned_expert_cache_gb": 3.0,
    }
    monkeypatch.setattr(
        audit,
        "manuscript_overhead_rows",
        lambda: {
            key: (value, value)
            for key, value in paper_metrics.items()
        },
    )
    benchmark_names = (
        "mmlu_pro",
        "gpqa",
        "aime25",
        "gsm8k",
        "humaneval",
    )
    checkpoint = {"local": False, "revision": "sha"}
    artifact = {
        "schema_version": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "checkpoint": checkpoint,
        "environment": {
            "git": {"commit": "abc123", "dirty": False},
            "process_max_rss_bytes": 1024,
        },
        "seed": 42,
        "paper_model": "qwen30b",
        "evaluation_protocol": {"name": "tc_main_v2"},
        "overhead_sequence": [*benchmark_names, "performance_bs32"],
        "benchmarks": {
            name: {"score": 0.5, "failed": 0}
            for name in benchmark_names
        },
        "benchmark": {
            "scope": "isolated_model",
            "batch_size": 32,
            "input_tokens": 2048,
            "output_tokens_per_sequence": 256,
            "warmup_iterations": 5,
            "measured_iterations": 100,
            "process_hbm_monitor": _process_hbm_monitor(),
            "samples": samples,
            "metrics": summaries,
        },
        "paper_metrics": paper_metrics,
        "config": {
            "model": {"layers": 1, "experts_per_layer": 2},
            "memory": {"device_mem_bytes": 48_000_000_000},
        },
        "initial_map": _calibrated_initial_map(
            checkpoint,
            layers=1,
            experts=2,
        ),
        "runtime_initialization": {
            "resident_expert_bytes": 2_000_000_000,
            "transient_expert_bytes": 100_000_000,
            "host_cache": {"host_packed_bytes": 3_000_000_000},
            "n_hi": [1],
            "bootstrap_policy": "calibrated_ranking_prefix",
            "bootstrap_hi_experts": {"0": [0]},
        },
        "wrapper_stats": {
            "forward_steps": 100,
            "router_observations": 200,
            "attached_layers": 1,
            "router_layers": 1,
            "scheduler_enabled": True,
            "scheduler_update_samples_ms": [
                float(index) for index in range(1, 101)
            ],
            "scheduler_update_count": 100,
        },
        "transition_stats": {
            "accepted_requests": 7,
            "accepted_bytes": 500_000_000,
            "total_promotions": 3,
            "total_demotions": 4,
            "copied_bytes": 500_000_000,
            "precise_fence_reclaims": 7,
            "global_sync_reclaims": 0,
            "active_transitions": 0,
            "failed_transitions": 0,
            "budget": {
                "total_live": 9,
                "total_cap": 10,
                "hi_pending": 0,
                "lo_pending": 0,
                "staging_used": 0,
            },
        },
    }
    assert audit.validate_manifest_artifact(
        "runtime_overhead",
        "runtime_overhead[0]",
        artifact,
        "runtime_overhead:qwen30b",
    ) == []

    artifact["paper_metrics"]["scheduler_p99_ms"] = 98.0
    problems = audit.validate_manifest_artifact(
        "runtime_overhead",
        "runtime_overhead[0]",
        artifact,
        "runtime_overhead:qwen30b",
    )
    assert "OVERHEAD PAPER METRIC MISMATCH: runtime_overhead[0]" in problems
