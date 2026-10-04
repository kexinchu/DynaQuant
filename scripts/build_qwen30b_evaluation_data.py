#!/usr/bin/env python3
"""Build manuscript-ready Qwen3-30B rows from audited raw artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "evaluation" / "qwen30b"
DEFAULT_OUTPUT = ROOT / "results" / "evaluation" / "derived" / "qwen30b"

TRACE_RE = re.compile(
    r"^(?P<policy>[a-z_]+)_trace_(?P<regime>prefill|decode|mixed)_seed"
    r"(?P<seed>\d+)\.json$"
)
QUALITY_RE = re.compile(
    r"^(?P<policy>[a-z_]+)_quality_seed(?P<seed>\d+)\.json$"
)
PERF_RE = re.compile(
    r"^(?P<policy>[a-z_]+)_perf_seed(?P<seed>\d+)_bs"
    r"(?P<batch>\d+)\.json$"
)
ABLATION_RE = re.compile(
    r"^ablation_(?P<variant>[a-z_]+)_seed(?P<seed>\d+)\.json$"
)
SENSITIVITY_RE = re.compile(
    r"^sensitivity_hi(?P<ratio>\d+)_seed(?P<seed>\d+)\.json$"
)
OVERHEAD_RE = re.compile(r"^overhead_seed(?P<seed>\d+)\.json$")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not a JSON object: {path}")
    return value


def _source_hash(artifact: dict[str, Any]) -> str | None:
    return artifact.get("environment", {}).get("git", {}).get(
        "source_tree_sha256"
    )


def _arena(artifact: dict[str, Any]) -> dict[str, Any]:
    return artifact.get("transition_stats", {}).get("arena", {})


def _metric_summary(benchmark: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for metric, summary in benchmark.get("metrics", {}).items():
        if not isinstance(summary, dict):
            continue
        for statistic in ("mean", "p50", "p95", "p99", "min", "max"):
            value = summary.get(statistic)
            if isinstance(value, (int, float)):
                flattened[f"{metric}_{statistic}"] = float(value)
    return flattened


def _runtime_fields(artifact: dict[str, Any]) -> dict[str, Any]:
    arena = _arena(artifact)
    controller = artifact.get("wrapper_stats", {}).get(
        "residency_controller", {}
    )
    transitions = artifact.get("transition_stats", {})
    return {
        "budget_bytes": arena.get("capacity_bytes"),
        "published_bytes": arena.get("published_bytes"),
        "peak_published_bytes": arena.get("peak_published_bytes"),
        "fid_bytes": arena.get("fid_bytes"),
        "res_bytes": arena.get("res_bytes"),
        "look_bytes": arena.get("look_bytes"),
        "external_fragmentation": arena.get("external_fragmentation"),
        "reservation_failures_budget": arena.get(
            "reservation_failures_budget"
        ),
        "reservation_failures_extent": arena.get(
            "reservation_failures_extent"
        ),
        "demand_misses": controller.get("demand_misses"),
        "demand_bytes": controller.get("demand_bytes"),
        "demand_wait_s": controller.get("demand_wait_s"),
        "prefetched_hits": controller.get("prefetched_hits"),
        "prefetch_submitted": controller.get("prefetch_submitted"),
        "counterfactual_candidates": controller.get(
            "counterfactual_candidates"
        ),
        "copied_bytes": transitions.get("copied_bytes"),
        "accepted_requests": transitions.get("accepted_requests"),
        "failed_transitions": transitions.get("failed_transitions"),
    }


def _quality_rows(
    artifact: dict[str, Any], policy: str, seed: int, source: str
) -> Iterable[dict[str, Any]]:
    for benchmark, result in artifact.get("benchmarks", {}).items():
        if not isinstance(result, dict):
            continue
        row = {
            "model": "qwen3-30b-a3b-instruct-2507",
            "policy": policy,
            "seed": seed,
            "benchmark": benchmark,
            "metric": result.get("metric"),
            "score": result.get("score"),
            "perplexity": result.get("perplexity"),
            "evaluated": result.get("evaluated"),
            "failed": result.get("failed"),
            "source_artifact": source,
            "source_tree_sha256": _source_hash(artifact),
        }
        yield row


def _exchange_rows(
    artifact: dict[str, Any], variant: str, seed: int, source: str
) -> Iterable[dict[str, Any]]:
    records = artifact.get("wrapper_stats", {}).get(
        "exchange_planner", {}
    ).get("records", [])
    grouped: dict[tuple[str, str, bool], dict[str, float]] = defaultdict(
        lambda: {
            "count": 0,
            "bytes": 0,
            "predicted_gain_s": 0.0,
            "realized_gain_s": 0.0,
            "reload_bytes": 0,
            "reload_cost_s": 0.0,
        }
    )
    for record in records:
        request = "+".join(record.get("request_uses", [])) or "none"
        donor = "+".join(record.get("donor_uses", [])) or "none"
        key = (request, donor, bool(record.get("accepted")))
        grouped[key]["count"] += 1
        grouped[key]["bytes"] += float(record.get("request_bytes", 0))
        grouped[key]["predicted_gain_s"] += float(
            record.get("predicted_gain_s", 0.0)
        )
        grouped[key]["realized_gain_s"] += float(
            record.get("realized_gain_s", 0.0)
        )
        grouped[key]["reload_bytes"] += float(
            record.get("reload_bytes", 0)
        )
        grouped[key]["reload_cost_s"] += float(
            record.get("reload_cost_s", 0.0)
        )
    for (request, donor, accepted), values in sorted(grouped.items()):
        yield {
            "model": "qwen3-30b-a3b-instruct-2507",
            "variant": variant,
            "seed": seed,
            "request_use": request,
            "donor_use": donor,
            "accepted": accepted,
            **values,
            "source_artifact": source,
            "source_tree_sha256": _source_hash(artifact),
        }


def build(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {
        "latency": [],
        "quality": [],
        "mechanisms": [],
        "sensitivity": [],
        "runtime": [],
    }
    for path in sorted(input_dir.glob("*.json")):
        trace = TRACE_RE.match(path.name)
        quality = QUALITY_RE.match(path.name)
        perf = PERF_RE.match(path.name)
        ablation = ABLATION_RE.match(path.name)
        sensitivity = SENSITIVITY_RE.match(path.name)
        overhead = OVERHEAD_RE.match(path.name)
        if not any((trace, quality, perf, ablation, sensitivity, overhead)):
            continue
        artifact = _read(path)
        common = {
            "model": "qwen3-30b-a3b-instruct-2507",
            "source_artifact": path.relative_to(ROOT).as_posix(),
            "source_tree_sha256": _source_hash(artifact),
            **_runtime_fields(artifact),
        }
        if trace:
            row = {
                **common,
                "policy": trace.group("policy"),
                "seed": int(trace.group("seed")),
                "regime": trace.group("regime"),
                "batch_size": 1,
                **_metric_summary(artifact.get("benchmark", {})),
            }
            rows["latency"].append(row)
            rows["runtime"].append(row)
        elif perf:
            row = {
                **common,
                "policy": perf.group("policy"),
                "seed": int(perf.group("seed")),
                "regime": "fixed_2048_256",
                "batch_size": int(perf.group("batch")),
                **_metric_summary(artifact.get("benchmark", {})),
            }
            rows["latency"].append(row)
            rows["runtime"].append(row)
        elif quality:
            rows["quality"].extend(
                _quality_rows(
                    artifact,
                    quality.group("policy"),
                    int(quality.group("seed")),
                    common["source_artifact"],
                )
            )
        elif ablation:
            variant = ablation.group("variant")
            seed = int(ablation.group("seed"))
            rows["quality"].extend(
                _quality_rows(
                    artifact, variant, seed, common["source_artifact"]
                )
            )
            rows["mechanisms"].extend(
                _exchange_rows(
                    artifact, variant, seed, common["source_artifact"]
                )
            )
            rows["runtime"].append(
                {
                    **common,
                    "policy": variant,
                    "seed": seed,
                    "regime": "ablation_bs32",
                    "batch_size": 32,
                    **_metric_summary(artifact.get("benchmark", {})),
                }
            )
        elif sensitivity:
            rows["sensitivity"].append(
                {
                    **common,
                    "policy": "fidelity",
                    "seed": int(sensitivity.group("seed")),
                    "high_precision_ratio_pct": int(
                        sensitivity.group("ratio")
                    ),
                    **artifact.get("paper_metrics", {}),
                }
            )
        elif overhead:
            rows["runtime"].append(
                {
                    **common,
                    "policy": "joint",
                    "seed": int(overhead.group("seed")),
                    "regime": "overhead_bs32",
                    "batch_size": 32,
                    **_metric_summary(artifact.get("benchmark", {})),
                    **artifact.get("paper_metrics", {}),
                }
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    rows = build(args.input.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in rows.items():
        _write_jsonl(args.output / f"{name}.jsonl", values)
    source_hashes = sorted(
        {
            row["source_tree_sha256"]
            for values in rows.values()
            for row in values
            if row.get("source_tree_sha256")
        }
    )
    manifest = {
        "schema_version": 1,
        "input": str(args.input),
        "row_counts": {name: len(values) for name, values in rows.items()},
        "source_tree_sha256": source_hashes,
        "source_artifacts": sorted(
            {
                row["source_artifact"]
                for values in rows.values()
                for row in values
                if row.get("source_artifact")
            }
        ),
        "failed_artifacts": sorted(path.name for path in args.input.glob("*.failed")),
    }
    if len(source_hashes) > 1 and not args.allow_partial:
        raise SystemExit("raw artifacts contain multiple runtime source hashes")
    manifest_path = args.output / "manifest.json"
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
