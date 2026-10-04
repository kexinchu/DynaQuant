from __future__ import annotations

from scripts.build_background_motivation_data import (
    build_joint_surface,
    build_stage_transfer_pressure,
)
from scripts.plot_background_motivation import (
    render_joint_tradeoff,
    render_transfer_pressure,
)


def _trace() -> dict:
    def trial(name: str, phase: str, layer0: list[int], layer1: list[int]) -> dict:
        return {
            "trial_id": name,
            "phase": phase,
            "layer_active_experts": {"0": layer0, "1": layer1},
        }

    return {
        "schema_version": 2,
        "artifact_type": "routing_active_set_trace",
        "paper_model": "qwen30b",
        "moe_layer_ids": [0, 1],
        "experts_per_layer": 4,
        "expert_bytes_per_layer": [100, 100],
        "points": [
            {
                "input_tokens": 8,
                "trials": [
                    trial("warmup-0", "warmup", [0, 1], [2, 3]),
                    trial("measured-0", "measured", [0, 2], [1, 3]),
                    trial("measured-1", "measured", [0, 3], [0, 3]),
                ],
            }
        ],
    }


def _offload() -> dict:
    return {
        "schema_version": 2,
        "artifact_type": "blocking_offload_waiting",
        "paper_model": "qwen30b",
        "benchmark": {
            "points": [
                {
                    "input_tokens": 8,
                    "samples": [
                        {
                            "transferred_bytes": 400,
                            "device_copy_ms": 2.0,
                        },
                        {
                            "transferred_bytes": 600,
                            "device_copy_ms": 3.0,
                        },
                    ],
                }
            ]
        },
    }


def _dynamic_perf() -> dict:
    return {
        "runtime_initialization": {
            "pool_allocation_bytes": 1600,
            "bootstrap": {
                "pool": {
                    "hi": [{"block_size_bytes": 300}],
                    "lo": [{"block_size_bytes": 100}],
                }
            },
        }
    }


def test_build_transfer_pressure_records_measured_and_derived_terms(tmp_path):
    activation = {
        "schema_version": 2,
        "artifact_type": "activation_density",
        "paper_model": "qwen30b",
        "moe_layer_ids": [0, 1],
        "stages": {
            stage: [
                {
                    "batch_size": batch,
                    "experts_per_layer": 4,
                    "layer_active_counts": [[1, 2]],
                    "ratio_pct": 37.5,
                }
                for batch in (1, 2)
            ]
            for stage in ("prefill", "decode")
        },
    }
    perf = {
        batch: {
            "benchmark": {
                "metrics": {
                    "model_ttft_ms": {"mean": 4.0},
                    "model_tpot_ms": {"mean": 1.0},
                }
            }
        }
        for batch in (1, 2)
    }
    data = build_stage_transfer_pressure(
        activation,
        _offload(),
        perf,
        input_tokens=8,
        low_expert_bytes=100,
    )
    assert data["artifact_type"] == "background_stage_transfer_pressure"
    assert len(data["stages"]["prefill"]) == 2
    assert data["stages"]["prefill"][0]["mean_demanded_bytes_per_layer"] == 150
    assert data["transfer_bandwidth_bytes_s"] == 200_000
    output = tmp_path / "transfer.pdf"
    written = render_transfer_pressure(data, output)
    assert set(written) == {"prefill", "decode", "legend"}
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0
        assert path.parent == tmp_path
    assert written["prefill"].name == "transfer_prefill.pdf"
    assert written["decode"].name == "transfer_decode.pdf"
    assert written["legend"].name == "transfer_legend.pdf"


def test_build_joint_surface_marks_quality_and_memory_feasibility(tmp_path):
    curve = [
        {
            "high_precision_ratio_pct": 0,
            "perplexity": 8.0,
        },
        {
            "high_precision_ratio_pct": 50,
            "perplexity": 6.5,
        },
        {
            "high_precision_ratio_pct": 100,
            "perplexity": 6.0,
        },
    ]
    data = build_joint_surface(
        _trace(),
        _offload(),
        _dynamic_perf(),
        curve,
        input_tokens=8,
        staging_slots=[0, 1, 2],
        max_perplexity_increase=1.0,
    )
    assert data["artifact_type"] == "background_joint_allocation_replay"
    assert data["joint_optimum"] is not None
    assert any(not cell["quality_feasible"] for cell in data["cells"])
    assert any(not cell["memory_feasible"] for cell in data["cells"])
    output = tmp_path / "joint.pdf"
    render_joint_tradeoff(data, output)
    assert output.is_file() and output.stat().st_size > 0
