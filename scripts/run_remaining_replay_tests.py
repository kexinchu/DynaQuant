#!/usr/bin/env python3
"""CPU tests that the registered replay can still answer.

E4 logs exchanges at the only cell where exposed wait separates: 8 GB/s,
Phi=1.0, 2,048 tokens. E6c scores the frequency predictor on that trace.
The paired bootstrap is exposed wait on the same ten trials, not TPOT.
E5, E1, E1b, E3, and E6 are recorded as blocked when this stack has no input.
"""

from __future__ import annotations

import importlib.util
import json
import random
import statistics
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from dynaexq.lease.exchange import independent_gain
from dynaexq.policy.online import (
    OnlineStats,
    _admit_independent,
    _donor,
    _enforce_quality,
    _fill_slack,
    _predicted_active,
    _rank,
    _settle_prefetches,
    initial_placement,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper/a6000_wikitext_replay/remaining_cpu_tests.json"


def load_replay():
    path = ROOT / "scripts/run_a6000_wikitext_replay.py"
    spec = importlib.util.spec_from_file_location("a6000_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(__import__("math").ceil(quantile * len(ordered))) - 1))
    return ordered[index]


def bootstrap_delta(left: list[float], right: list[float], draws: int = 10000) -> dict:
    rng = random.Random(0)
    means = []
    p95s = []
    n = len(left)
    for _ in range(draws):
        index = [rng.randrange(n) for _ in range(n)]
        sample_left = [left[i] for i in index]
        sample_right = [right[i] for i in index]
        means.append(statistics.fmean(sample_left) - statistics.fmean(sample_right))
        p95s.append(percentile(sample_left, 0.95) - percentile(sample_right, 0.95))
    means.sort()
    p95s.sort()
    mean_lo = means[int(0.025 * (draws - 1))]
    mean_hi = means[int(0.975 * (draws - 1))]
    p95_lo = p95s[int(0.025 * (draws - 1))]
    p95_hi = p95s[int(0.975 * (draws - 1))]
    return {
        "n": n,
        "mean_delta_ms": 1000.0 * (statistics.fmean(left) - statistics.fmean(right)),
        "mean_ci95_ms": [1000.0 * mean_lo, 1000.0 * mean_hi],
        "mean_covers_zero": mean_lo <= 0.0 <= mean_hi,
        "p95_delta_ms": 1000.0 * (percentile(left, 0.95) - percentile(right, 0.95)),
        "p95_delta_ci95_ms": [1000.0 * p95_lo, 1000.0 * p95_hi],
        "p95_covers_zero": p95_lo <= 0.0 <= p95_hi,
        "note": "Paired bootstrap over the ten trials. Exposed wait, not TPOT. The p95 interval resamples p95(left)-p95(right).",
    }


def walk(replay, trials, counts, placements, *, layer_s, bw, lo, hi, budget, horizon, min_high, valuation, width):
    from dynaexq.lease.exchange import admit_exchanges
    from dynaexq.lease.replay import simulate_forward

    rows = []
    exposed = []
    for trial_index, trial in enumerate(trials):
        for placement in placements:
            placement.prefetches.clear()
        _enforce_quality(placements, counts, min_high=min_high, budget=budget, lo_bytes=lo, hi_bytes=hi)
        admitted = []
        for index in range(len(trial)):
            predicted = _predicted_active(counts, width, index, horizon, trial)
            candidates = []
            last = min(len(trial), index + horizon + 1)
            for future in range(index + 1, last):
                resident = placements[future].high | placements[future].low
                missing = [
                    expert
                    for expert in _rank(counts[future])
                    if expert in predicted[future] - resident
                ][:16]
                for expert in missing:
                    donor = _donor(placements[future], counts[future], predicted[future])
                    if donor is None and placements[future].total_bytes(lo, hi) + lo > budget:
                        continue
                    donor_expert, donor_tier = donor if donor is not None else (None, None)
                    from dynaexq.lease.exchange import Exchange

                    candidates.append(
                        Exchange(
                            kind="lookahead",
                            layer=future,
                            expert=expert,
                            tier="lo",
                            nbytes=lo,
                            donor_layer=future if donor is not None else None,
                            donor_expert=donor_expert,
                            donor_tier=donor_tier,
                        )
                    )
            if valuation == "counterfactual":
                chosen = admit_exchanges(
                    predicted, placements, candidates,
                    layer_time_s=layer_s, bandwidth_bytes_s=bw, lo_bytes=lo, hi_bytes=hi,
                    horizon=horizon, budget=budget, limit=8,
                )
            else:
                chosen = _admit_independent(
                    placements, candidates, counts,
                    bandwidth_bytes_s=bw, lo_bytes=lo, hi_bytes=hi, budget=budget, limit=8,
                )
                for exchange in chosen:
                    request_p = counts[exchange.layer][exchange.expert]
                    donor_p = 0
                    if exchange.donor_expert is not None and exchange.donor_layer is not None:
                        donor_p = counts[exchange.donor_layer][exchange.donor_expert]
                    scale = max(1, max(counts[exchange.layer]))
                    object.__setattr__(
                        exchange,
                        "gain_s",
                        independent_gain(
                            exchange,
                            request_probability=request_p / scale,
                            donor_probability=donor_p / scale,
                            bandwidth_bytes_s=bw,
                        ),
                    )
            for exchange in chosen:
                admitted.append(exchange)
        plan = [[] for _ in trial]
        for destination, placement in enumerate(placements):
            for issue, expert, tier in placement.prefetches:
                plan[issue].append((destination, expert, tier))
        result = simulate_forward(
            trial, [item.as_resident() for item in placements], plan,
            layer_time_s=layer_s, bandwidth_bytes_s=bw, lo_bytes=lo, hi_bytes=hi,
        )
        exposed.append(result.exposed_s)
        late_bytes = sum(layer.demand_bytes for layer in result.layers)
        for exchange in admitted:
            hit = exchange.expert in trial[exchange.layer]
            rows.append({
                "trial": trial_index,
                "kind": exchange.kind,
                "layer": exchange.layer,
                "expert": exchange.expert,
                "donor_expert": exchange.donor_expert,
                "nbytes": exchange.nbytes,
                "gain_s": exchange.gain_s,
                "hit": hit,
            })
        _settle_prefetches(placements, trial, OnlineStats())
        for layer, needed in enumerate(trial):
            for expert in needed:
                counts[layer][expert] += 1
        _fill_slack(placements, counts, budget, lo, hi)
        rows.append({
            "trial": trial_index,
            "kind": "_forward",
            "late_or_demand_bytes": late_bytes,
            "exposed_ms": 1000.0 * result.exposed_s,
            "prefetched_hits": sum(layer.prefetched_hits for layer in result.layers),
            "late_prefetches": sum(layer.late_prefetches for layer in result.layers),
        })
    exchanges = [row for row in rows if row["kind"] != "_forward"]
    forwards = [row for row in rows if row["kind"] == "_forward"]
    return exchanges, forwards, exposed


def predictor_scores(counts, trials, width: int, lo_bytes: int) -> dict:
    tp = fp = fn = 0
    for trial in trials:
        for layer, needed in enumerate(trial):
            predicted = set(_rank(counts[layer])[:width])
            tp += len(predicted & needed)
            fp += len(predicted - needed)
            fn += len(needed - predicted)
            for expert in needed:
                counts[layer][expert] += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "width": width,
        "precision": precision,
        "recall": recall,
        "wasted_bytes": fp * lo_bytes,
        "late_bytes": fn * lo_bytes,
        "true_positive_experts": tp,
        "false_positive_experts": fp,
        "false_negative_experts": fn,
    }


def main() -> None:
    replay = load_replay()
    trace = replay.load_json(replay.TRACE)
    offload = replay.load_json(replay.OFFLOAD)
    perf = replay.load_json(replay.PERF)
    curve = replay.perplexity_curve()
    reference = min(point[1] for point in curve)
    n_experts = int(trace["experts_per_layer"])
    n_layers = len(trace["moe_layer_ids"])
    lo = int(perf["runtime_initialization"]["bootstrap"]["pool"]["lo"][0]["block_size_bytes"])
    hi = int(perf["runtime_initialization"]["bootstrap"]["pool"]["hi"][0]["block_size_bytes"])
    low_pool = n_experts * lo
    budget = int(1.0 * low_pool)
    layer_s = float(perf["benchmark"]["metrics"]["model_ttft_ms"]["mean"]) / 1000.0 / n_layers
    bw = 8e9
    warmup, measured = replay.split_trials(trace, 2048)
    counts = replay.frequency(warmup, n_experts)
    min_high = next(
        n for n in range(n_experts + 1)
        if replay.interpolate(n / n_experts, curve) - reference <= 2.1 + 1e-9
    )
    width = max(1, round(statistics.fmean(len(layer) for trial in warmup for layer in trial)))
    choices = sorted({min(n_experts, __import__("math").ceil(fraction * n_experts)) for fraction in replay.HIGH_FRACTIONS} | {min_high})
    joint = replay.search(
        warmup, counts, budget=budget, lo_bytes=lo, hi_bytes=hi, n_experts=n_experts,
        layer_time_s=layer_s, bandwidth_bytes_s=bw, curve=curve, reference=reference,
        epsilon=2.1, high_choices=choices, slot_choices=list(replay.SLOT_GRID),
    )
    static_samples = replay.evaluate(
        measured,
        replay.build(counts, joint["n_high"], joint["n_low"], joint["slots"]),
        layer_time_s=layer_s, bandwidth_bytes_s=bw, lo_bytes=lo, hi_bytes=hi,
    )
    policies = {}
    for valuation in ("counterfactual", "independent"):
        print(f"E4 {valuation}", flush=True)
        placements = initial_placement(counts, n_high=0, n_low=min(n_experts, budget // lo), n_experts=n_experts)
        exchanges, forwards, exposed = walk(
            replay, measured, deepcopy(counts), placements,
            layer_s=layer_s, bw=bw, lo=lo, hi=hi, budget=budget,
            horizon=replay.HORIZON, min_high=min_high, valuation=valuation, width=width,
        )
        hits = sum(1 for row in exchanges if row["hit"])
        policies[valuation] = {
            "exchanges": len(exchanges),
            "hits": hits,
            "misses": len(exchanges) - hits,
            "kinds": {"lookahead": len(exchanges)},
            "with_donor": sum(1 for row in exchanges if row["donor_expert"] is not None),
            "mean_gain_ms": 1000.0 * statistics.fmean(row["gain_s"] for row in exchanges) if exchanges else 0.0,
            "demand_bytes": sum(row["late_or_demand_bytes"] for row in forwards),
            "late_prefetches": sum(row["late_prefetches"] for row in forwards),
            "prefetched_hits": sum(row["prefetched_hits"] for row in forwards),
            "exposed_ms": [1000.0 * value for value in exposed],
        }
    cf_keys = set()
    # Recompute key sets from a second pass stored above is not available.
    # Duplicate credit is an independent admit the counterfactual walk refused.
    # Recover it by repeating the walks' exchange identity from the summary fields
    # is insufficient, so compare the two walks' (trial, layer, expert) lists.
    # The lists were not retained. Run once more into sets.
    key_sets = {}
    for valuation in ("counterfactual", "independent"):
        placements = initial_placement(counts, n_high=0, n_low=min(n_experts, budget // lo), n_experts=n_experts)
        exchanges, _, _ = walk(
            replay, measured, deepcopy(counts), placements,
            layer_s=layer_s, bw=bw, lo=lo, hi=hi, budget=budget,
            horizon=replay.HORIZON, min_high=min_high, valuation=valuation, width=width,
        )
        key_sets[valuation] = {(row["trial"], row["layer"], row["expert"]) for row in exchanges}
    only_independent = key_sets["independent"] - key_sets["counterfactual"]
    only_counterfactual = key_sets["counterfactual"] - key_sets["independent"]
    e4 = {
        "cell": {"tokens": 2048, "phi": 1.0, "bandwidth_gbps": 8.0, "epsilon": 2.1, "layer_time_ms": layer_s * 1000.0},
        "static_joint": {
            "n_high": joint["n_high"],
            "n_low": joint["n_low"],
            "slots": joint["slots"],
            "exposed_ms": [1000.0 * value for value in static_samples],
        },
        "policies": policies,
        "duplicate_credit_candidates": len(only_independent),
        "counterfactual_only": len(only_counterfactual),
        "definition": (
            "An independent lookahead of (trial, layer, expert) that the "
            "counterfactual walk did not admit. All recorded exchanges are lookahead."
        ),
    }
    e6c = predictor_scores(deepcopy(counts), measured, width, lo)
    e6c["note"] = (
        "Width is the mean active-set size on warmup prefill layers, not a token-level top-8. "
        "The trace stores one expert set per layer for the whole block, so this score is not the manuscript recall triple."
    )
    paired = {
        "counterfactual_minus_static": bootstrap_delta(
            [value / 1000.0 for value in policies["counterfactual"]["exposed_ms"]],
            static_samples,
        ),
        "independent_minus_static": bootstrap_delta(
            [value / 1000.0 for value in policies["independent"]["exposed_ms"]],
            static_samples,
        ),
        "independent_minus_counterfactual": bootstrap_delta(
            [value / 1000.0 for value in policies["independent"]["exposed_ms"]],
            [value / 1000.0 for value in policies["counterfactual"]["exposed_ms"]],
        ),
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "remaining_cpu_tests",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "not_end_to_end": True,
        "e4": e4,
        "e6c": e6c,
        "paired_exposed_wait": paired,
        "blocked": {
            "E1": "No 1,000-request TPOT run. The short-window target was withdrawn by E2. This file's paired interval is exposed wait on 10 trials at 8 GB/s.",
            "E1b": "The replay has no arrival process, queue, or concurrency.",
            "E3": "No HOBBIT or DyMoE implementation is in the tree.",
            "E5": "qwen30b_routing_active_set_trace.json contains one WikiText-103 routing stream. No second phase exists to switch.",
            "E6": "The replay admits with the offline perplexity curve, not a per-decision risk score, so it cannot form a confusion matrix.",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "e4_duplicate_credit_candidates": e4["duplicate_credit_candidates"],
        "e4_exchanges": {name: row["exchanges"] for name, row in policies.items()},
        "e6c": e6c,
        "paired": paired,
    }, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
