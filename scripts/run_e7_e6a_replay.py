#!/usr/bin/env python3
"""E7 copy-rate sweep and E6a stale-versus-measured admission replay.

Both runs use the registered WikiText routing trace, block sizes, and
perplexity curve. Layer windows stay on the registered rule unless
--layer-times names an E2 artifact. This does not overwrite summary.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper/a6000_wikitext_replay/e7_e6a.json"
SWEEP_GBPS = (8.0, 16.0, 24.86, 40.0)
STALE_GBPS = 20.0
PHIS = (1.0, 1.3, None)  # None means the registered deployment pool
EPSILON = 2.1
TOKENS = (2048, 128, 512)


def load_replay():
    path = ROOT / "scripts/run_a6000_wikitext_replay.py"
    spec = importlib.util.spec_from_file_location("a6000_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exchange_key(exchange) -> tuple:
    return (
        exchange.kind,
        exchange.layer,
        exchange.expert,
        exchange.tier,
        exchange.donor_layer,
        exchange.donor_expert,
        exchange.donor_tier,
    )


def high_choices(replay, n_experts, min_high):
    import math
    choices = [min(n_experts, math.ceil(fraction * n_experts)) for fraction in replay.HIGH_FRACTIONS]
    if min_high is not None:
        choices.append(min_high)
    return sorted(set(choices))


def policy_rows(replay, measured, warmup_counts, measured_counts, specs, budget, lo, hi, n_experts, layer_s, bw, curve, reference):
    rows = []
    for name, spec in specs:
        if spec is None or spec.get("n_low") is None or not replay.fits(
            spec["n_high"], spec["n_low"], spec["slots"], budget, lo, hi, n_experts
        ):
            rows.append({"policy": name, "quality_feasible": False, "reason": "infeasible"})
            continue
        ranking = measured_counts if name == "oracle-grid-test" else warmup_counts
        row = replay.record(
            name, spec, measured,
            ranking,
            budget=budget, lo_bytes=lo, hi_bytes=hi, n_experts=n_experts,
            layer_time_s=layer_s, bandwidth_bytes_s=bw, curve=curve,
            reference=reference, epsilon=EPSILON,
        )
        rows.append({
            "policy": name,
            "quality_feasible": row["quality_feasible"],
            "n_high": row["n_high"],
            "n_low": row["n_low"],
            "slots": row["staging_slots"],
            "delta_ppl": row["delta_ppl"],
            "p95_ms": row["p95_ms"],
            "mean_ms": row["mean_ms"],
        })
    return rows


def online_row(replay, measured, warmup, counts, budget, lo, hi, n_experts, layer_s, bw, curve, reference, min_high, valuation):
    from dynaexq.policy.online import initial_placement, run_online

    width = max(1, round(statistics.fmean(len(layer) for trial in warmup for layer in trial)))
    placements = initial_placement(
        counts, n_high=0, n_low=min(n_experts, budget // lo), n_experts=n_experts,
    )
    stats = run_online(
        measured, deepcopy(counts), placements,
        layer_time_s=layer_s, bandwidth_bytes_s=bw, lo_bytes=lo, hi_bytes=hi,
        budget=budget, horizon=replay.HORIZON, min_high=min_high, valuation=valuation,
        predict_width=width,
    )
    summary = replay.summarize(stats.exposed_s)
    keys = [exchange_key(item) for item in stats.exchanges]
    return {
        "policy": f"dynabyte-{valuation}",
        "exchanges": len(stats.exchanges),
        "expired_prefetches": stats.expired_prefetches,
        "p95_ms": summary["p95_ms"],
        "mean_ms": summary["mean_ms"],
        "exchange_keys": keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-times", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--token-lengths", type=int, nargs="+", default=list(TOKENS))
    parser.add_argument("--online-tokens", type=int, nargs="+", default=[2048])
    parser.add_argument("--rates", type=float, nargs="+", default=None)
    args = parser.parse_args()
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
    measured_budget = int(perf["runtime_initialization"]["pool_allocation_bytes"]) // n_layers
    low_pool = n_experts * lo
    deployed_phi = measured_budget / low_pool
    measured_windows = {}
    if args.layer_times is not None:
        payload = json.loads(args.layer_times.read_text(encoding="utf-8"))
        measured_windows = {
            int(item["input_tokens"]): float(item["layer_time_ms"]) / 1000.0
            for item in payload["points"]
        }
    reference_window = float(perf["benchmark"]["metrics"]["model_ttft_ms"]["mean"]) / 1000.0 / n_layers
    native_bw = replay.bandwidth(offload, 2048)
    native_gbps = native_bw / 1e9
    rates = args.rates if args.rates is not None else list(SWEEP_GBPS) + [STALE_GBPS, native_gbps]
    online_tokens = set(args.online_tokens)
    cells = []
    for tokens in args.token_lengths:
        warmup, measured = replay.split_trials(trace, tokens)
        counts = replay.frequency(warmup, n_experts)
        if tokens in measured_windows:
            layer_s = measured_windows[tokens]
            window_source = "e2_measured_ttft_over_layers"
        elif tokens == 2048:
            layer_s = reference_window
            window_source = "registered_2048_ttft_over_layers"
        else:
            layer_s = reference_window * tokens / 2048
            window_source = "scaled_from_2048_ttft_by_token_count"
        for gbps in rates:
            bw = gbps * 1e9
            for phi in PHIS:
                budget = measured_budget if phi is None else int(phi * low_pool)
                phi_value = deployed_phi if phi is None else phi
                min_high = next(
                    (
                        n_high
                        for n_high in range(n_experts + 1)
                        if replay.interpolate(n_high / n_experts, curve) - reference <= EPSILON + 1e-9
                    ),
                    None,
                )
                choices = high_choices(replay, n_experts, min_high)
                joint = replay.search(
                    warmup, counts, budget=budget, lo_bytes=lo, hi_bytes=hi, n_experts=n_experts,
                    layer_time_s=layer_s, bandwidth_bytes_s=bw, curve=curve, reference=reference,
                    epsilon=EPSILON, high_choices=choices, slot_choices=list(replay.SLOT_GRID),
                )
                seq = None if min_high is None else replay.search(
                    warmup, counts, budget=budget, lo_bytes=lo, hi_bytes=hi, n_experts=n_experts,
                    layer_time_s=layer_s, bandwidth_bytes_s=bw, curve=curve, reference=reference,
                    epsilon=EPSILON, high_choices=[min_high], slot_choices=list(replay.SLOT_GRID),
                )
                measured_counts = replay.frequency(measured, n_experts)
                oracle = replay.search(
                    measured, measured_counts, budget=budget, lo_bytes=lo, hi_bytes=hi,
                    n_experts=n_experts, layer_time_s=layer_s, bandwidth_bytes_s=bw,
                    curve=curve, reference=reference, epsilon=EPSILON,
                    high_choices=choices, slot_choices=list(replay.SLOT_GRID),
                )
                specs = [
                    ("static-joint-cal", joint),
                    ("seq-pq", seq),
                    ("oracle-grid-test", oracle),
                ]
                static = policy_rows(
                    replay, measured, counts, measured_counts, specs, budget, lo, hi, n_experts,
                    layer_s, bw, curve, reference,
                )
                online = []
                if tokens in online_tokens and min_high is not None:
                    for valuation in ("counterfactual", "independent"):
                        print(
                            f"online tokens={tokens} {gbps:.2f} GB/s phi={phi_value:.3f} {valuation}",
                            flush=True,
                        )
                        online.append(online_row(
                            replay, measured, warmup, counts, budget, lo, hi, n_experts,
                            layer_s, bw, curve, reference, min_high, valuation,
                        ))
                cells.append({
                    "input_tokens": tokens,
                    "layer_time_ms": layer_s * 1000.0,
                    "layer_time_source": window_source,
                    "bandwidth_gbps": gbps,
                    "phi": phi_value,
                    "static": static,
                    "online": [
                        {key: value for key, value in row.items() if key != "exchange_keys"}
                        | {"n_exchange_keys": len(row["exchange_keys"])}
                        for row in online
                    ],
                    "_online_keys": {
                        row["policy"]: row["exchange_keys"] for row in online
                    },
                })
                print(
                    f"tokens={tokens} {gbps:.2f} GB/s phi={phi_value:.3f} "
                    f"static={[ (row['policy'], row.get('slots'), round(row.get('p95_ms', -1), 2)) for row in static ]}",
                    flush=True,
                )
    flips = []
    by_key = {
        (cell["input_tokens"], round(cell["phi"], 6), round(cell["bandwidth_gbps"], 2)): cell
        for cell in cells
    }
    for tokens in args.token_lengths:
        for phi in {cell["phi"] for cell in cells if cell["input_tokens"] == tokens}:
            stale = by_key.get((tokens, round(phi, 6), round(STALE_GBPS, 2)))
            native = by_key.get((tokens, round(phi, 6), round(native_gbps, 2)))
            if stale is None or native is None:
                continue
            for policy in ("dynabyte-counterfactual", "dynabyte-independent"):
                left = set(map(tuple, stale["_online_keys"].get(policy, [])))
                right = set(map(tuple, native["_online_keys"].get(policy, [])))
                flips.append({
                    "input_tokens": tokens,
                    "phi": phi,
                    "policy": policy,
                    "stale_gbps": STALE_GBPS,
                    "native_gbps": native_bw / 1e9,
                    "stale_admits": len(left),
                    "native_admits": len(right),
                    "only_stale": len(left - right),
                    "only_native": len(right - left),
                    "admit_flips": len(left.symmetric_difference(right)),
                })
    public_cells = []
    for cell in cells:
        public = {key: value for key, value in cell.items() if key != "_online_keys"}
        public_cells.append(public)
    payload = {
        "schema_version": 1,
        "artifact_type": "e7_e6a_replay",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "not_end_to_end": True,
        "epsilon": EPSILON,
        "native_bandwidth_bytes_s": native_bw,
        "stale_bandwidth_gbps": STALE_GBPS,
        "sweep_gbps": list(SWEEP_GBPS),
        "layer_times_source": str(args.layer_times) if args.layer_times else None,
        "cells": public_cells,
        "e6a_admit_flips": flips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    for row in flips:
        print(
            f"E6a {row['policy']} phi={row['phi']:.3f} "
            f"flips={row['admit_flips']} stale={row['stale_admits']} native={row['native_admits']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
