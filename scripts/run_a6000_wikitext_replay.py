#!/usr/bin/env python3
"""Replay DynaByte and the plan's baselines on one A6000-measured dataset.

The dataset is the WikiText-103 routing trace already collected on an RTX
A6000. Bandwidth, expert-block sizes, the expert-memory budget, and WikiText-2
perplexity are taken from the registered artifacts. This is a deadline replay,
not a new end-to-end kernel measurement. Its validity therefore depends on the
source artifacts rather than the GPU or checkpoint state at replay time.

No test-set quantity is used to choose a static policy. The oracle row is the
only one fit on the measured trials, and it is labeled as such.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from dynaexq.lease.exchange import Placement, exposed_seconds
from dynaexq.policy.online import initial_placement, run_online


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "results/paper/qwen30b_routing_active_set_trace.json"
OFFLOAD = ROOT / "results/paper/qwen30b_offload_waiting.json"
PERF = ROOT / "results/paper/performance/qwen30b_dynaexq_bs1.json"
PPL_DIR = ROOT / "results/perplexity/Qwen3-30B"
OUT = ROOT / "results/paper/a6000_wikitext_replay/summary.json"

# Pre-registered before looking at replay outputs. 0.5 is the plan's bound.
# 2.1 is the bound already used by the background joint-allocation figure.
EPSILONS = (0.5, 2.1)
PHIS = (0.70, 0.85, 1.00, 1.30)
SLOT_GRID = (0, 1, 2, 4, 8, 12, 16)
HIGH_FRACTIONS = (0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0)
PRIMARY_TOKENS = 2048
EXTRA_TOKENS = (128, 512)
HORIZON = 1


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty sample")
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def perplexity_curve() -> list[tuple[float, float]]:
    points = []
    for path in sorted(PPL_DIR.glob("30b-*.txt")):
        low_pct = int(path.stem.split("-")[-1])
        payload = load_json(path)
        points.append((1.0 - low_pct / 100.0, float(payload["perplexity"])))
    points.sort()
    return points


def interpolate(fraction: float, curve: list[tuple[float, float]]) -> float:
    if fraction <= curve[0][0]:
        return curve[0][1]
    if fraction >= curve[-1][0]:
        return curve[-1][1]
    for (left_x, left_y), (right_x, right_y) in zip(curve, curve[1:]):
        if left_x <= fraction <= right_x:
            weight = (fraction - left_x) / (right_x - left_x)
            return left_y + weight * (right_y - left_y)
    raise RuntimeError("perplexity curve is not sorted")


def bandwidth(offload: dict, tokens: int) -> float:
    point = next(item for item in offload["benchmark"]["points"] if int(item["input_tokens"]) == tokens)
    total_bytes = sum(int(sample["transferred_bytes"]) for sample in point["samples"])
    total_ms = sum(float(sample["device_copy_ms"]) for sample in point["samples"])
    return total_bytes / (total_ms / 1000.0)


def split_trials(trace: dict, tokens: int) -> tuple[list[list[set[int]]], list[list[set[int]]]]:
    point = next(item for item in trace["points"] if int(item["input_tokens"]) == tokens)
    layers = [int(layer) for layer in trace["moe_layer_ids"]]

    def convert(trial: dict) -> list[set[int]]:
        return [
            {int(expert) for expert in trial["layer_active_experts"][str(layer)]}
            for layer in layers
        ]

    warmup = [convert(trial) for trial in point["trials"] if trial["phase"] == "warmup"]
    measured = [convert(trial) for trial in point["trials"] if trial["phase"] == "measured"]
    return warmup, measured


def frequency(trials: list[list[set[int]]], n_experts: int) -> list[list[int]]:
    counts = [[0 for _ in range(n_experts)] for _ in range(len(trials[0]))]
    for trial in trials:
        for layer, needed in enumerate(trial):
            for expert in needed:
                counts[layer][expert] += 1
    return counts


def summarize(samples_s: list[float]) -> dict[str, float]:
    ms = [1000.0 * value for value in samples_s]
    return {
        "mean_ms": statistics.fmean(ms),
        "p50_ms": percentile(ms, 0.50),
        "p95_ms": percentile(ms, 0.95),
        "max_ms": max(ms),
    }


def evaluate(
    trials: list[list[set[int]]],
    placements: list[Placement],
    *,
    layer_time_s: float,
    bandwidth_bytes_s: float,
    lo_bytes: int,
    hi_bytes: int,
) -> list[float]:
    return [
        exposed_seconds(
            trial,
            placements,
            layer_time_s=layer_time_s,
            bandwidth_bytes_s=bandwidth_bytes_s,
            lo_bytes=lo_bytes,
            hi_bytes=hi_bytes,
            horizon=HORIZON,
        )
        for trial in trials
    ]


def build(
    counts: list[list[int]],
    n_high: int,
    n_low: int,
    slots: int,
) -> list[Placement]:
    placements = []
    for layer_counts in counts:
        ranking = sorted(range(len(layer_counts)), key=lambda expert: (-layer_counts[expert], expert))
        placements.append(
            Placement(
                high=set(ranking[:n_high]),
                low=set(ranking[n_high : n_high + n_low]),
                staging_slots=slots,
            )
        )
    return placements


def fits(n_high: int, n_low: int, slots: int, budget: int, lo_bytes: int, hi_bytes: int, n_experts: int) -> bool:
    if n_high < 0 or n_low < 0 or slots < 0:
        return False
    if n_high + n_low > n_experts:
        return False
    return n_high * hi_bytes + (n_low + slots) * lo_bytes <= budget


def leftover_low(n_high: int, slots: int, budget: int, lo_bytes: int, hi_bytes: int, n_experts: int) -> int | None:
    remaining = budget - n_high * hi_bytes - slots * lo_bytes
    if remaining < 0:
        return None
    return min(n_experts - n_high, remaining // lo_bytes)


def quality_row(
    n_high: int,
    n_experts: int,
    curve: list[tuple[float, float]],
    reference: float,
    epsilon: float,
) -> tuple[float, float, bool]:
    perplexity = interpolate(n_high / n_experts, curve)
    delta = perplexity - reference
    return perplexity, delta, delta <= epsilon + 1e-9


def search(
    warmup: list[list[set[int]]],
    counts: list[list[int]],
    *,
    budget: int,
    lo_bytes: int,
    hi_bytes: int,
    n_experts: int,
    layer_time_s: float,
    bandwidth_bytes_s: float,
    curve: list[tuple[float, float]],
    reference: float,
    epsilon: float,
    high_choices: list[int],
    slot_choices: list[int],
    lock_slots: int | None = None,
) -> dict | None:
    best: dict | None = None
    for n_high in high_choices:
        _perplexity, delta, ok = quality_row(n_high, n_experts, curve, reference, epsilon)
        if not ok:
            continue
        slots_iter = slot_choices if lock_slots is None else [lock_slots]
        for slots in slots_iter:
            n_low = leftover_low(n_high, slots, budget, lo_bytes, hi_bytes, n_experts)
            if n_low is None:
                continue
            placements = build(counts, n_high, n_low, slots)
            samples = evaluate(
                warmup,
                placements,
                layer_time_s=layer_time_s,
                bandwidth_bytes_s=bandwidth_bytes_s,
                lo_bytes=lo_bytes,
                hi_bytes=hi_bytes,
            )
            mean = statistics.fmean(samples)
            if best is None or mean < best["warmup_exposed_s"]:
                best = {
                    "n_high": n_high,
                    "n_low": n_low,
                    "slots": slots,
                    "delta_ppl": delta,
                    "warmup_exposed_s": mean,
                }
    return best


def coupling_grid(
    warmup: list[list[set[int]]],
    measured: list[list[set[int]]],
    counts: list[list[int]],
    *,
    budget: int,
    lo_bytes: int,
    hi_bytes: int,
    n_experts: int,
    n_layers: int,
    layer_time_s: float,
    bandwidth_bytes_s: float,
    curve: list[tuple[float, float]],
    reference: float,
    high_choices: list[int],
    slot_choices: list[int],
) -> list[dict]:
    """Return every feasible fidelity--lookahead cell, not only the winner.

    Selection remains calibration-only. Both splits are emitted so plotting
    code can show whether the location of the optimum survives held-out replay.
    Byte fields cover the full model and have the same three-use decomposition
    as the paper: every resident has a low representation, fidelity is the
    incremental cost of its high representation, and lookahead is transient.
    """
    rows: list[dict] = []
    for n_high in sorted(set(high_choices)):
        perplexity, delta, _ = quality_row(
            n_high, n_experts, curve, reference, math.inf
        )
        for slots in sorted(set(slot_choices)):
            n_low = leftover_low(
                n_high, slots, budget, lo_bytes, hi_bytes, n_experts
            )
            if n_low is None:
                continue
            placements = build(counts, n_high, n_low, slots)
            common = {
                "n_high": n_high,
                "n_low": n_low,
                "staging_slots": slots,
                "quality_delta": delta,
                "perplexity": perplexity,
                "fid_bytes": n_layers * n_high * (hi_bytes - lo_bytes),
                "res_bytes": n_layers * (n_high + n_low) * lo_bytes,
                "look_bytes": n_layers * slots * lo_bytes,
                "budget_bytes": n_layers * budget,
            }
            for split, trials in (("calibration", warmup), ("test", measured)):
                samples = evaluate(
                    trials,
                    placements,
                    layer_time_s=layer_time_s,
                    bandwidth_bytes_s=bandwidth_bytes_s,
                    lo_bytes=lo_bytes,
                    hi_bytes=hi_bytes,
                )
                rows.append(
                    {
                        **common,
                        "split": split,
                        "exposed_wait_mean_ms": 1000.0 * statistics.fmean(samples),
                        "exposed_wait_p95_ms": 1000.0 * percentile(samples, 0.95),
                    }
                )
    return rows


def record(
    name: str,
    spec: dict,
    trials: list[list[set[int]]],
    counts: list[list[int]],
    *,
    budget: int,
    lo_bytes: int,
    hi_bytes: int,
    n_experts: int,
    layer_time_s: float,
    bandwidth_bytes_s: float,
    curve: list[tuple[float, float]],
    reference: float,
    epsilon: float,
) -> dict:
    perplexity, delta, feasible = quality_row(spec["n_high"], n_experts, curve, reference, epsilon)
    placements = build(counts, spec["n_high"], spec["n_low"], spec["slots"])
    samples = evaluate(
        trials,
        placements,
        layer_time_s=layer_time_s,
        bandwidth_bytes_s=bandwidth_bytes_s,
        lo_bytes=lo_bytes,
        hi_bytes=hi_bytes,
    )
    used = spec["n_high"] * hi_bytes + (spec["n_low"] + spec["slots"]) * lo_bytes
    return {
        "policy": name,
        "quality_feasible": feasible,
        "perplexity": perplexity,
        "delta_ppl": delta,
        "epsilon": epsilon,
        "n_high": spec["n_high"],
        "n_low": spec["n_low"],
        "staging_slots": spec["slots"],
        "bytes_per_layer": used,
        "budget_per_layer": budget,
        "within_budget": used <= budget,
        **summarize(samples),
    }


def run_length(trace: dict, offload: dict, perf: dict, curve: list[tuple[float, float]], tokens: int, reference_window_s: float, online: bool) -> dict:
    n_experts = int(trace["experts_per_layer"])
    n_layers = len(trace["moe_layer_ids"])
    lo_bytes = int(perf["runtime_initialization"]["bootstrap"]["pool"]["lo"][0]["block_size_bytes"])
    hi_bytes = int(perf["runtime_initialization"]["bootstrap"]["pool"]["hi"][0]["block_size_bytes"])
    measured_budget = int(perf["runtime_initialization"]["pool_allocation_bytes"]) // n_layers
    low_pool = n_experts * lo_bytes
    reference = min(point[1] for point in curve)
    warmup, measured = split_trials(trace, tokens)
    counts = frequency(warmup, n_experts)
    copy_bw = bandwidth(offload, tokens)
    if tokens == PRIMARY_TOKENS:
        layer_time_s = float(perf["benchmark"]["metrics"]["model_ttft_ms"]["mean"]) / 1000.0 / n_layers
        window_source = "measured_ttft_over_layers"
    else:
        layer_time_s = reference_window_s * tokens / PRIMARY_TOKENS
        window_source = "scaled_from_2048_ttft_by_token_count"
    high_choices = [min(n_experts, math.ceil(fraction * n_experts)) for fraction in HIGH_FRACTIONS]
    rows = []
    grids = []
    phi_values = list(PHIS) + [measured_budget / low_pool]
    for phi in phi_values:
        budget = int(phi * low_pool) if phi != phi_values[-1] else measured_budget
        if abs(phi - measured_budget / low_pool) < 1e-9:
            budget = measured_budget
        cells = coupling_grid(
            warmup,
            measured,
            counts,
            budget=budget,
            lo_bytes=lo_bytes,
            hi_bytes=hi_bytes,
            n_experts=n_experts,
            n_layers=n_layers,
            layer_time_s=layer_time_s,
            bandwidth_bytes_s=copy_bw,
            curve=curve,
            reference=reference,
            high_choices=[
                min(n_experts, math.ceil(fraction * n_experts))
                for fraction in HIGH_FRACTIONS
            ],
            slot_choices=list(SLOT_GRID),
        )
        for cell in cells:
            cell["phi"] = phi
        grids.extend(cells)
        for epsilon in EPSILONS:
            min_high = next(
                (
                    n_high
                    for n_high in range(n_experts + 1)
                    if interpolate(n_high / n_experts, curve) - reference <= epsilon + 1e-9
                ),
                None,
            )
            high_for_joint = sorted(set(high_choices + ([] if min_high is None else [min_high])))
            joint = search(
                warmup,
                counts,
                budget=budget,
                lo_bytes=lo_bytes,
                hi_bytes=hi_bytes,
                n_experts=n_experts,
                layer_time_s=layer_time_s,
                bandwidth_bytes_s=copy_bw,
                curve=curve,
                reference=reference,
                epsilon=epsilon,
                high_choices=high_for_joint,
                slot_choices=list(SLOT_GRID),
            )
            latency_first = search(
                warmup,
                counts,
                budget=budget,
                lo_bytes=lo_bytes,
                hi_bytes=hi_bytes,
                n_experts=n_experts,
                layer_time_s=layer_time_s,
                bandwidth_bytes_s=copy_bw,
                curve=curve,
                reference=reference,
                epsilon=1e9,
                high_choices=[0],
                slot_choices=list(SLOT_GRID),
            )
            configs: list[tuple[str, dict | None]] = [
                ("uniform-low", {"n_high": 0, "n_low": leftover_low(0, 0, budget, lo_bytes, hi_bytes, n_experts), "slots": 0}),
                ("precision-only", None if min_high is None else {
                    "n_high": min_high,
                    "n_low": leftover_low(min_high, 0, budget, lo_bytes, hi_bytes, n_experts),
                    "slots": 0,
                }),
                ("horizon-only", None if latency_first is None else {
                    "n_high": 0,
                    "n_low": latency_first["n_low"],
                    "slots": latency_first["slots"],
                }),
                ("static-joint-cal", None if joint is None else joint),
                ("seq-pq", None if min_high is None else search(
                    warmup,
                    counts,
                    budget=budget,
                    lo_bytes=lo_bytes,
                    hi_bytes=hi_bytes,
                    n_experts=n_experts,
                    layer_time_s=layer_time_s,
                    bandwidth_bytes_s=copy_bw,
                    curve=curve,
                    reference=reference,
                    epsilon=epsilon,
                    high_choices=[min_high],
                    slot_choices=list(SLOT_GRID),
                )),
            ]
            if latency_first is not None and min_high is not None:
                locked = latency_first["slots"]
                while locked > 0 and leftover_low(min_high, locked, budget, lo_bytes, hi_bytes, n_experts) is None:
                    locked -= 1
                n_low = leftover_low(min_high, locked, budget, lo_bytes, hi_bytes, n_experts)
                configs.append(("seq-qp", None if n_low is None else {"n_high": min_high, "n_low": n_low, "slots": locked}))
            else:
                configs.append(("seq-qp", None))
            fid_bytes = int(0.30 * budget)
            res_bytes = int(0.50 * budget)
            look_bytes = budget - fid_bytes - res_bytes
            n_high = min(n_experts, fid_bytes // max(1, hi_bytes - lo_bytes))
            n_low = min(max(0, n_experts - n_high), max(0, res_bytes // lo_bytes - n_high))
            slots = min(SLOT_GRID[-1], look_bytes // lo_bytes)
            while n_high > 0 and not fits(n_high, n_low, slots, budget, lo_bytes, hi_bytes, n_experts):
                n_high -= 1
            configs.append(("fixed-partition", {"n_high": n_high, "n_low": n_low, "slots": slots}))
            measured_counts = frequency(measured, n_experts)
            oracle = search(
                measured,
                measured_counts,
                budget=budget,
                lo_bytes=lo_bytes,
                hi_bytes=hi_bytes,
                n_experts=n_experts,
                layer_time_s=layer_time_s,
                bandwidth_bytes_s=copy_bw,
                curve=curve,
                reference=reference,
                epsilon=epsilon,
                high_choices=high_for_joint,
                slot_choices=list(SLOT_GRID),
            )
            configs.append(("oracle-grid-test", None if oracle is None else oracle))
            for name, spec in configs:
                if spec is None or spec.get("n_low") is None or not fits(
                    spec["n_high"], spec["n_low"], spec["slots"], budget, lo_bytes, hi_bytes, n_experts
                ):
                    rows.append({
                        "policy": name,
                        "phi": phi,
                        "epsilon": epsilon,
                        "quality_feasible": False,
                        "within_budget": False,
                        "reason": "no memory-feasible placement meets the quality limit",
                    })
                    continue
                row = record(
                    name,
                    spec,
                    measured,
                    measured_counts if name == "oracle-grid-test" else counts,
                    budget=budget,
                    lo_bytes=lo_bytes,
                    hi_bytes=hi_bytes,
                    n_experts=n_experts,
                    layer_time_s=layer_time_s,
                    bandwidth_bytes_s=copy_bw,
                    curve=curve,
                    reference=reference,
                    epsilon=epsilon,
                )
                row["phi"] = phi
                row["selected_on"] = "measured_trials" if name == "oracle-grid-test" else "warmup_trials"
                rows.append(row)
            if online and min_high is not None:
                for valuation in ("counterfactual", "independent"):
                    width = max(1, round(statistics.fmean(len(layer) for trial in warmup for layer in trial)))
                    placements = initial_placement(
                        counts,
                        n_high=0,
                        n_low=min(n_experts, budget // lo_bytes),
                        n_experts=n_experts,
                    )
                    stats = run_online(
                        measured,
                        deepcopy(counts),
                        placements,
                        layer_time_s=layer_time_s,
                        bandwidth_bytes_s=copy_bw,
                        lo_bytes=lo_bytes,
                        hi_bytes=hi_bytes,
                        budget=budget,
                        horizon=HORIZON,
                        min_high=min_high,
                        valuation=valuation,
                        predict_width=width,
                    )
                    mean_high = statistics.fmean(len(item.high) for item in placements)
                    mean_low = statistics.fmean(len(item.low) for item in placements)
                    perplexity, delta, feasible = quality_row(round(mean_high), n_experts, curve, reference, epsilon)
                    kind_counts: dict[str, int] = {}
                    funded = 0
                    for exchange in stats.exchanges:
                        kind_counts[exchange.kind] = kind_counts.get(exchange.kind, 0) + 1
                        if exchange.donor_expert is not None:
                            funded += 1
                    rows.append({
                        "policy": f"dynabyte-{valuation}",
                        "phi": phi,
                        "epsilon": epsilon,
                        "quality_feasible": feasible,
                        "perplexity": perplexity,
                        "delta_ppl": delta,
                        "n_high": mean_high,
                        "n_low": mean_low,
                        "bytes_per_layer": mean_high * hi_bytes + mean_low * lo_bytes,
                        "budget_per_layer": budget,
                        "within_budget": all(
                            item.total_bytes(lo_bytes, hi_bytes) <= budget for item in placements
                        ),
                        "exchanges": len(stats.exchanges),
                        "donor_funded_exchanges": funded,
                        "converted_prefetches": stats.converted_prefetches,
                        "expired_prefetches": stats.expired_prefetches,
                        "exchange_kinds": kind_counts,
                        "selected_on": "online_warmup_initialized",
                        **summarize(stats.exposed_s),
                    })
                    print(
                        f"online {valuation} phi={phi:.3f} eps={epsilon} "
                        f"mean={rows[-1]['mean_ms']:.3f} ms exchanges={len(stats.exchanges)}",
                        flush=True,
                    )
        print(f"tokens={tokens} phi={phi:.3f} done", flush=True)
    return {
        "input_tokens": tokens,
        "layer_time_ms": layer_time_s * 1000.0,
        "layer_time_source": window_source,
        "bandwidth_bytes_s": copy_bw,
        "lo_bytes": lo_bytes,
        "hi_bytes": hi_bytes,
        "low_pool_bytes_per_layer": low_pool,
        "measured_budget_bytes_per_layer": measured_budget,
        "warmup_trials": len(warmup),
        "measured_trials": len(measured),
        "coupling_grid": grids,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-online", action="store_true")
    args = parser.parse_args()
    trace = load_json(TRACE)
    offload = load_json(OFFLOAD)
    perf = load_json(PERF)
    curve = perplexity_curve()
    reference_window = float(perf["benchmark"]["metrics"]["model_ttft_ms"]["mean"]) / 1000.0 / len(trace["moe_layer_ids"])
    lengths = []
    for tokens in (PRIMARY_TOKENS, *EXTRA_TOKENS):
        print(f"replay {tokens} tokens", flush=True)
        lengths.append(
            run_length(
                trace,
                offload,
                perf,
                curve,
                tokens,
                reference_window,
                online=not args.skip_online and tokens == PRIMARY_TOKENS,
            )
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "a6000_wikitext_deadline_replay",
        "dataset": "WikiText-103 disjoint 2048-token blocks, via the registered routing trace",
        "device": "NVIDIA RTX A6000",
        "execution": "deadline replay of measured routing, bandwidth, block sizes, and perplexity",
        "not_end_to_end": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": [str(path.relative_to(ROOT)) for path in (TRACE, OFFLOAD, PERF)],
        "epsilons": list(EPSILONS),
        "perplexity_curve": [{"high_fraction": fraction, "perplexity": value} for fraction, value in curve],
        "lengths": lengths,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
