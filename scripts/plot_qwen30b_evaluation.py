#!/usr/bin/env python3
"""Render Qwen3-30B evaluation figures from derived JSONL rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "evaluation" / "derived" / "qwen30b"
DEFAULT_OUTPUT = ROOT / "SIGMETRICS_2027" / "figures"
POLICY_ORDER = (
    "uniform_low",
    "static_mixed",
    "residency",
    "lookahead",
    "fidelity",
    "joint",
)
REGIME_ORDER = ("prefill", "mixed", "decode")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 2027


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median_range(values: list[float]) -> tuple[float, float, float]:
    return statistics.median(values), min(values), max(values)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sample")
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(quantile * len(ordered)) - 1),
    )
    return ordered[index]


def _request_samples(
    rows: list[dict[str, Any]], metric: str
) -> dict[tuple[str, str, int], dict[str, float]]:
    """Load request samples keyed by policy, regime, seed, and request ID."""
    samples: dict[tuple[str, str, int], dict[str, float]] = {}
    cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        regime = row.get("regime")
        if regime not in REGIME_ORDER:
            continue
        source = str(row["source_artifact"])
        artifact = cache.get(source)
        if artifact is None:
            artifact = json.loads((ROOT / source).read_text(encoding="utf-8"))
            cache[source] = artifact
        request_values: dict[str, float] = {}
        for sample in artifact.get("benchmark", {}).get("samples", []):
            request_id = sample.get("request_id")
            value = sample.get(metric)
            if request_id is None or not isinstance(value, (int, float)):
                continue
            request_values[str(request_id)] = float(value)
        if request_values:
            samples[(str(row["policy"]), str(regime), int(row["seed"]))] = (
                request_values
            )
    return samples


def _bootstrap_absolute(
    clusters: dict[int, list[float]],
    *,
    rng: random.Random,
    replicates: int,
) -> tuple[float, float, float, dict[str, float]]:
    """Median per-seed p95 with a seed-clustered request bootstrap."""
    seeds = sorted(clusters)
    if not seeds:
        raise ValueError("absolute bootstrap has no seed clusters")
    per_seed = {str(seed): _percentile(clusters[seed], 0.95) for seed in seeds}
    estimate = statistics.median(per_seed.values())
    draws = []
    for _ in range(replicates):
        selected = [rng.choice(seeds) for _ in seeds]
        seed_statistics = []
        for seed in selected:
            values = clusters[seed]
            resampled = [rng.choice(values) for _ in values]
            seed_statistics.append(_percentile(resampled, 0.95))
        draws.append(statistics.median(seed_statistics))
    return (
        estimate,
        _percentile(draws, 0.025),
        _percentile(draws, 0.975),
        per_seed,
    )


def _bootstrap_paired_difference(
    joint: dict[int, dict[str, float]],
    baseline: dict[int, dict[str, float]],
    *,
    rng: random.Random,
    replicates: int,
) -> dict[str, Any]:
    """Bootstrap joint-minus-baseline p95 from paired request samples."""
    paired: dict[int, list[tuple[float, float]]] = {}
    for seed in sorted(set(joint) & set(baseline)):
        ids = sorted(set(joint[seed]) & set(baseline[seed]))
        if ids:
            paired[seed] = [(joint[seed][item], baseline[seed][item]) for item in ids]
    if not paired:
        raise ValueError("paired bootstrap has no shared seed/request samples")
    per_seed = {
        str(seed): _percentile([item[0] for item in values], 0.95)
        - _percentile([item[1] for item in values], 0.95)
        for seed, values in paired.items()
    }
    estimate = statistics.median(per_seed.values())
    seeds = sorted(paired)
    draws = []
    for _ in range(replicates):
        selected = [rng.choice(seeds) for _ in seeds]
        differences = []
        for seed in selected:
            pairs = paired[seed]
            resampled = [rng.choice(pairs) for _ in pairs]
            differences.append(
                _percentile([item[0] for item in resampled], 0.95)
                - _percentile([item[1] for item in resampled], 0.95)
            )
        draws.append(statistics.median(differences))
    return {
        "estimate_ms": estimate,
        "ci95_low_ms": _percentile(draws, 0.025),
        "ci95_high_ms": _percentile(draws, 0.975),
        "per_seed_ms": per_seed,
        "paired_seeds": seeds,
        "paired_requests_per_seed": {
            str(seed): len(paired[seed]) for seed in seeds
        },
    }


def latency_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "estimator": "median of per-seed p95",
        "resampling": "seed clusters and paired request IDs",
        "absolute": [],
        "paired_joint_minus_baseline": [],
    }
    for metric in ("model_ttft_ms", "model_tpot_ms"):
        samples = _request_samples(rows, metric)
        rng = random.Random(f"{BOOTSTRAP_SEED}:{metric}")
        for policy in POLICY_ORDER:
            for regime in REGIME_ORDER:
                clusters = {
                    seed: list(values.values())
                    for (candidate, candidate_regime, seed), values in samples.items()
                    if candidate == policy and candidate_regime == regime
                }
                if not clusters:
                    continue
                estimate, low, high, per_seed = _bootstrap_absolute(
                    clusters,
                    rng=rng,
                    replicates=BOOTSTRAP_REPLICATES,
                )
                result["absolute"].append(
                    {
                        "metric": metric,
                        "policy": policy,
                        "regime": regime,
                        "estimate_ms": estimate,
                        "ci95_low_ms": low,
                        "ci95_high_ms": high,
                        "per_seed_ms": per_seed,
                    }
                )
        for baseline in POLICY_ORDER:
            if baseline == "joint":
                continue
            for regime in REGIME_ORDER:
                joint = {
                    seed: values
                    for (policy, candidate_regime, seed), values in samples.items()
                    if policy == "joint" and candidate_regime == regime
                }
                other = {
                    seed: values
                    for (policy, candidate_regime, seed), values in samples.items()
                    if policy == baseline and candidate_regime == regime
                }
                if not joint or not other:
                    continue
                result["paired_joint_minus_baseline"].append(
                    {
                        "metric": metric,
                        "baseline": baseline,
                        "regime": regime,
                        **_bootstrap_paired_difference(
                            joint,
                            other,
                            rng=rng,
                            replicates=BOOTSTRAP_REPLICATES,
                        ),
                    }
                )
    return result


def _exact_mcnemar_pvalue(reference_only: int, policy_only: int) -> float:
    discordant = reference_only + policy_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(reference_only, policy_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _accuracy_degradation(
    reference: dict[str, Any],
    policy: dict[str, Any],
    *,
    rng: random.Random,
) -> dict[str, Any]:
    ref = {
        str(item["sample_id"]): bool(item["correct"])
        for item in reference.get("details", [])
        if "sample_id" in item and "correct" in item
    }
    candidate = {
        str(item["sample_id"]): bool(item["correct"])
        for item in policy.get("details", [])
        if "sample_id" in item and "correct" in item
    }
    ids = sorted(set(ref) & set(candidate))
    if not ids or set(ref) != set(candidate):
        raise ValueError("quality artifacts do not contain identical sample IDs")
    pairs = [(ref[item], candidate[item]) for item in ids]

    def degradation(values: list[tuple[bool, bool]]) -> float:
        return 100.0 * (
            sum(item[0] for item in values)
            - sum(item[1] for item in values)
        ) / len(values)

    draws = [
        degradation([rng.choice(pairs) for _ in pairs])
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    reference_only = sum(left and not right for left, right in pairs)
    policy_only = sum(right and not left for left, right in pairs)
    return {
        "unit": "percentage_points",
        "degradation": degradation(pairs),
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
        "paired_items": len(pairs),
        "reference_correct_policy_wrong": reference_only,
        "reference_wrong_policy_correct": policy_only,
        "mcnemar_exact_pvalue": _exact_mcnemar_pvalue(
            reference_only, policy_only
        ),
    }


def _perplexity_degradation(
    reference: dict[str, Any],
    policy: dict[str, Any],
    *,
    rng: random.Random,
) -> dict[str, Any]:
    def windows(payload: dict[str, Any]) -> dict[tuple[int, int, int], tuple[float, int]]:
        return {
            (
                int(item["window_index"]),
                int(item["begin_token"]),
                int(item["end_token"]),
            ): (float(item["nll"]), int(item["target_tokens"]))
            for item in payload.get("window_details", [])
        }

    ref = windows(reference)
    candidate = windows(policy)
    keys = sorted(set(ref) & set(candidate))
    if not keys or set(ref) != set(candidate):
        raise ValueError("perplexity artifacts do not contain identical windows")
    pairs = [(ref[key], candidate[key]) for key in keys]

    def perplexity(values: list[tuple[float, int]]) -> float:
        tokens = sum(item[1] for item in values)
        return math.exp(sum(item[0] for item in values) / tokens)

    def degradation(
        values: list[tuple[tuple[float, int], tuple[float, int]]]
    ) -> float:
        return perplexity([item[1] for item in values]) - perplexity(
            [item[0] for item in values]
        )

    draws = [
        degradation([rng.choice(pairs) for _ in pairs])
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    return {
        "unit": "perplexity",
        "degradation": degradation(pairs),
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
        "paired_windows": len(pairs),
    }


def quality_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare every measured quality point with the all-FP16 reference."""
    cache: dict[str, dict[str, Any]] = {}

    def artifact(source: str) -> dict[str, Any]:
        if source not in cache:
            cache[source] = json.loads(
                (ROOT / source).read_text(encoding="utf-8")
            )
        return cache[source]

    reference_rows = {
        str(row["benchmark"]): row
        for row in rows
        if row.get("policy") == "all_high_reference"
    }
    if len(reference_rows) != 6:
        raise ValueError("all-high reference must contain all six quality tasks")
    comparisons = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        policy = str(row.get("policy"))
        benchmark = str(row.get("benchmark"))
        key = (policy, benchmark)
        if policy == "all_high_reference" or key in seen:
            continue
        seen.add(key)
        reference_row = reference_rows.get(benchmark)
        if reference_row is None:
            raise ValueError(f"missing all-high reference for {benchmark}")
        reference_result = artifact(str(reference_row["source_artifact"]))[
            "benchmarks"
        ][benchmark]
        policy_result = artifact(str(row["source_artifact"]))["benchmarks"][
            benchmark
        ]
        rng = random.Random(
            f"{BOOTSTRAP_SEED}:quality:{policy}:{benchmark}"
        )
        if benchmark == "wikitext":
            comparison = _perplexity_degradation(
                reference_result, policy_result, rng=rng
            )
        else:
            comparison = _accuracy_degradation(
                reference_result, policy_result, rng=rng
            )
        comparisons.append(
            {
                "policy": policy,
                "benchmark": benchmark,
                "reference_policy": "all_high_reference",
                "reference_score": reference_result.get("score"),
                "policy_score": policy_result.get("score"),
                **comparison,
            }
        )
    return {
        "schema_version": 1,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "comparisons": comparisons,
    }


def plot_latency(
    rows: list[dict[str, Any]], statistics_payload: dict[str, Any], output: Path
) -> None:
    trace = [row for row in rows if row.get("regime") in REGIME_ORDER]
    fixed = [row for row in rows if row.get("regime") == "fixed_2048_256"]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0))
    for axis, metric, title in (
        (axes[0], "model_ttft_ms_p95", "p95 TTFT"),
        (axes[1], "model_tpot_ms_p95", "p95 TPOT"),
    ):
        raw_metric = metric.removesuffix("_p95")
        grouped = {
            (row["policy"], row["regime"]): row
            for row in statistics_payload["absolute"]
            if row["metric"] == raw_metric
        }
        x = list(range(len(REGIME_ORDER)))
        for policy in POLICY_ORDER:
            medians = []
            lows = []
            highs = []
            positions = []
            for index, regime in enumerate(REGIME_ORDER):
                result = grouped.get((policy, regime))
                if result is None:
                    continue
                median = float(result["estimate_ms"])
                low = float(result["ci95_low_ms"])
                high = float(result["ci95_high_ms"])
                positions.append(index)
                medians.append(median)
                lows.append(median - low)
                highs.append(high - median)
            if positions:
                axis.errorbar(
                    positions,
                    medians,
                    yerr=[lows, highs],
                    marker="o",
                    linewidth=1.2,
                    capsize=2,
                    label=policy.replace("_", " "),
                )
        axis.set_xticks(x, REGIME_ORDER)
        axis.set_ylabel("milliseconds")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)

    grouped_fixed: dict[int, list[float]] = defaultdict(list)
    for row in fixed:
        if row.get("policy") != "joint":
            continue
        value = row.get("model_tpot_ms_p95")
        if isinstance(value, (int, float)):
            grouped_fixed[int(row["batch_size"])].append(float(value))
    batches = sorted(grouped_fixed)
    if batches:
        medians = [_median_range(grouped_fixed[batch])[0] for batch in batches]
        lows = [
            medians[index] - min(grouped_fixed[batch])
            for index, batch in enumerate(batches)
        ]
        highs = [
            max(grouped_fixed[batch]) - medians[index]
            for index, batch in enumerate(batches)
        ]
        axes[2].errorbar(
            batches,
            medians,
            yerr=[lows, highs],
            marker="o",
            capsize=2,
        )
        axes[2].set_xscale("log", base=2)
        axes[2].set_xticks(batches, [str(value) for value in batches])
    axes[2].set_xlabel("batch size")
    axes[2].set_ylabel("p95 TPOT (ms)")
    axes[2].set_title("Joint policy scaling")
    axes[2].grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.subplots_adjust(top=0.76)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_mechanisms(rows: list[dict[str, Any]], output: Path) -> None:
    accepted = [row for row in rows if row.get("accepted") is True]
    variants = sorted({str(row["variant"]) for row in accepted})
    exchange_types = sorted(
        {(str(row["request_use"]), str(row["donor_use"])) for row in accepted}
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    bottoms = [0.0] * len(variants)
    for request, donor in exchange_types:
        values = []
        for variant in variants:
            values.append(
                sum(
                    float(row.get("bytes", 0)) / 1e9
                    for row in accepted
                    if row["variant"] == variant
                    and row["request_use"] == request
                    and row["donor_use"] == donor
                )
            )
        axes[0].bar(
            range(len(variants)),
            values,
            bottom=bottoms,
            label=f"{request} <- {donor}",
        )
        bottoms = [a + b for a, b in zip(bottoms, values)]
    axes[0].set_xticks(
        range(len(variants)),
        [value.replace("_", "\n") for value in variants],
    )
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].set_ylabel("accepted request bytes (GB)")
    axes[0].set_title("Exchange anatomy")
    if exchange_types:
        axes[0].legend(fontsize=7, frameon=False)

    positions = np.arange(len(variants), dtype=float)
    width = 0.25
    predicted = []
    realized = []
    reload = []
    for variant in variants:
        selected = [row for row in accepted if row["variant"] == variant]
        count = max(1, sum(int(row.get("count", 0)) for row in selected))
        predicted.append(
            1000.0
            * sum(float(row.get("predicted_gain_s", 0.0)) for row in selected)
            / count
        )
        realized.append(
            1000.0
            * sum(float(row.get("realized_gain_s", 0.0)) for row in selected)
            / count
        )
        reload.append(
            1000.0
            * sum(float(row.get("reload_cost_s", 0.0)) for row in selected)
            / count
        )
    axes[1].bar(positions - width, predicted, width, label="predicted benefit")
    axes[1].bar(positions, realized, width, label="realized benefit")
    axes[1].bar(positions + width, reload, width, label="reload cost")
    axes[1].set_xticks(
        positions,
        [value.replace("_", "\n") for value in variants],
    )
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].set_ylabel("mean per accepted exchange (ms)")
    axes[1].set_title("Valuation outcome")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(rows: list[dict[str, Any]], output: Path) -> None:
    rows = sorted(rows, key=lambda row: row["high_precision_ratio_pct"])
    fig, axis = plt.subplots(figsize=(3.6, 2.8))
    axis.plot(
        [row["realized_hi_ratio_pct"] for row in rows],
        [row["average_accuracy_pct"] for row in rows],
        marker="o",
    )
    axis.set_xlabel("realized high-precision experts (%)")
    axis.set_ylabel("five-task mean score (%)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    latency = _read_jsonl(args.input / "latency.jsonl")
    mechanisms = _read_jsonl(args.input / "mechanisms.jsonl")
    sensitivity = _read_jsonl(args.input / "sensitivity.jsonl")
    quality = _read_jsonl(args.input / "quality.jsonl")
    if not args.allow_partial and not all((latency, mechanisms, sensitivity)):
        raise SystemExit("latency, mechanism, and sensitivity rows are required")
    args.output.mkdir(parents=True, exist_ok=True)
    if latency:
        statistics_payload = latency_statistics(latency)
        statistics_path = args.input / "latency_statistics.json"
        temporary = statistics_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(statistics_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(statistics_path)
        plot_latency(
            latency,
            statistics_payload,
            args.output / "eval_qwen30b_latency.pdf",
        )
    if quality:
        quality_payload = quality_statistics(quality)
        quality_path = args.input / "quality_statistics.json"
        temporary = quality_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(quality_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(quality_path)
    if mechanisms:
        plot_mechanisms(mechanisms, args.output / "eval_qwen30b_mechanisms.pdf")
    if sensitivity:
        plot_sensitivity(
            sensitivity, args.output / "eval_qwen30b_sensitivity.pdf"
        )
    manifest_path = args.input / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rendered = {}
        for name in (
            "eval_qwen30b_latency.pdf",
            "eval_qwen30b_mechanisms.pdf",
            "eval_qwen30b_sensitivity.pdf",
        ):
            path = args.output / name
            if path.is_file():
                rendered[name] = {
                    "path": path.relative_to(ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
        manifest["rendered_figures"] = rendered
        statistics_path = args.input / "latency_statistics.json"
        if statistics_path.is_file():
            manifest["statistical_outputs"] = {
                "latency_statistics.json": {
                    "path": statistics_path.relative_to(ROOT).as_posix(),
                    "size_bytes": statistics_path.stat().st_size,
                    "sha256": _sha256(statistics_path),
                }
            }
        quality_path = args.input / "quality_statistics.json"
        if quality_path.is_file():
            manifest.setdefault("statistical_outputs", {})[
                "quality_statistics.json"
            ] = {
                "path": quality_path.relative_to(ROOT).as_posix(),
                "size_bytes": quality_path.stat().st_size,
                "sha256": _sha256(quality_path),
            }
        manifest["plotter"] = {
            "script": "scripts/plot_qwen30b_evaluation.py",
            "matplotlib": matplotlib.__version__,
        }
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(manifest_path)


if __name__ == "__main__":
    main()
