from __future__ import annotations

from scripts.render_paper_figures import (
    _render_hotsets,
    _render_performance,
    _render_perplexity,
    _render_sensitivity,
    _render_waiting,
)


def _claims() -> dict:
    claims = {}
    for model in ("qwen30b", "qwen80b", "phi35"):
        methods = (
            ("static_ptq", "moe_infinity", "dynaexq")
            if model == "qwen30b"
            else ("static_ptq", "dynaexq")
        )
        for method_index, method in enumerate(
            methods,
            start=1,
        ):
            for batch in (1, 2, 4, 8, 16, 32):
                claims[f"performance:{model}:{method}:bs{batch}"] = {
                    "benchmark": {
                        "metrics": {
                            "model_e2e_ms": {
                                "mean": float(1000 * method_index * batch),
                                "p99": float(1200 * method_index * batch),
                            },
                            "throughput_tokens_s": {
                                "mean": float(batch / method_index),
                            },
                        }
                    }
                }
    for model in ("qwen30b", "qwen80b"):
        for ratio in (0, 5, 10, 15, 20, 25, 30):
            claims[f"budget_sensitivity:{model}:ratio{ratio}"] = {
                "paper_metrics": {
                    "average_accuracy_pct": 50.0 + ratio / 10.0,
                }
            }
    for model in ("qwen30b", "qwen80b", "deepseek_v2_lite"):
        claims[f"offload_waiting:{model}"] = {
            "benchmark": {
                "points": [
                    {
                        "input_tokens": tokens,
                        "mean_waiting_ms": float(tokens),
                    }
                    for tokens in range(32, 545, 32)
                ]
            }
        }
    counts = {
        "wikitext": list(range(128)),
        "gsm8k": list(reversed(range(128))),
        "humaneval": [index % 17 for index in range(128)],
    }
    routing_bundle = {
        "workloads": {
            workload: {
                "expert_counts": values,
                "top10": sorted(
                    range(128),
                    key=lambda expert: (-values[expert], expert),
                )[:10],
            }
            for workload, values in counts.items()
        }
    }
    claims["routing_hotset:qwen30b:wikitext:layer15"] = routing_bundle
    for model in ("qwen30b", "qwen80b"):
        claims[f"perplexity_curve:{model}"] = {
            "points": [
                {
                    "low_ratio_pct": ratio,
                    "perplexity": 5.0 + ratio / 100.0,
                }
                for ratio in (0, 15, 30, 45, 60, 75, 90, 100)
            ]
        }
    return claims


def test_renderer_writes_all_seventeen_empirical_pdfs(tmp_path):
    claims = _claims()
    outputs = [
        *_render_performance(claims, tmp_path),
        *_render_sensitivity(claims, tmp_path),
        *_render_waiting(claims, tmp_path),
        *_render_hotsets(claims, tmp_path),
        *_render_perplexity(claims, tmp_path),
    ]
    assert len(outputs) == 17
    assert len({path.name for path in outputs}) == 17
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
