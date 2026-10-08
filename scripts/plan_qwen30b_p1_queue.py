#!/usr/bin/env python3
"""Render the optimized Qwen3-30B P1 command queue.

The script does not execute GPU work. It turns the P1 implementation contract
into concrete command lines after P0 has reached the optimized boundary.
The default queue is the staged P1A decision set, not the old full 25-cell
grid.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507"
OUTPUT_ROOT = "results/evaluation/qwen30b"
CALIBRATION = f"{OUTPUT_ROOT}/calibration_wikitext103_256x2048.json"
CONFIG = "dynaexq/configs/qwen30b_joint_a6000.yaml"
TRACE = "ShareGPT_V3_unfiltered_cleaned_split.json"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def eval_dynamic_base(output: str, seed: int, physical_gpu: int) -> list[str]:
    return [
        "python",
        "-m",
        "dynaexq.experiments.eval_dynamic",
        "--model-path",
        MODEL_PATH,
        "--output",
        output,
        "--device",
        "cuda:0",
        "--wait-for-idle-physical-gpu",
        str(physical_gpu),
        "--config",
        CONFIG,
        "--policy",
        "joint",
        "--hash-model-files",
        "--seed",
        str(seed),
        "--initial-map",
        CALIBRATION,
    ]


P1A_GRID_CELLS = (
    (0, 0),
    (40, 0),
    (0, 20),
    (20, 10),
    (20, 0),
    (0, 10),
)


def rq2_grid_commands(*, physical_gpu: int, full_grid: bool = False) -> list[str]:
    commands: list[str] = []
    cells = (
        [(fid, look) for fid in (0, 10, 20, 30, 40) for look in (0, 5, 10, 15, 20)]
        if full_grid
        else list(P1A_GRID_CELLS)
    )
    for fid, look in cells:
        output = f"{OUTPUT_ROOT}/rq2_grid_phi085_fid{fid}_look{look}_seed42.json"
        command = eval_dynamic_base(output, seed=42, physical_gpu=physical_gpu)
        command.extend(
            [
                "--phi-override",
                "0.85",
                "--allocation-policy",
                "dynabyte",
                "--fidelity-share-pct",
                str(fid),
                "--lookahead-share-pct",
                str(look),
                "--residency-share-pct",
                str(100 - fid - look),
                "trace-perf",
                "--trace",
                TRACE,
                "--regime",
                "mixed",
                "--requests",
                "20",
                "--n-warmup",
                "2",
                "--max-input-tokens",
                "2048",
                "--max-output-tokens",
                "256",
            ]
        )
        commands.append(shell_join(command))
    return commands


def switch_commands(*, physical_gpu: int, full_switch: bool = False) -> list[str]:
    commands: list[str] = []
    policies = (
        ("dynabyte", "static_joint", "seq_fr", "seq_rf")
        if full_switch
        else ("dynabyte", "static_joint")
    )
    for policy in policies:
        output = f"{OUTPUT_ROOT}/rq2_switch_{policy}_seed42.json"
        command = eval_dynamic_base(output, seed=42, physical_gpu=physical_gpu)
        command.extend(
            [
                "--allocation-policy",
                policy,
                "--workload-switch-trace",
                "rq2_prefill_decode_mixed_v1",
                "trace-perf",
                "--trace",
                TRACE,
                "--regime",
                "mixed",
                "--requests",
                "20",
                "--n-warmup",
                "2",
                "--max-input-tokens",
                "2048",
                "--max-output-tokens",
                "256",
            ]
        )
        commands.append(shell_join(command))
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physical-gpu",
        type=int,
        default=0,
        help="Physical GPU index to embed in generated commands.",
    )
    parser.add_argument(
        "--section",
        choices=("p1a", "grid", "switch", "full-grid", "full-switch", "full"),
        default="p1a",
    )
    args = parser.parse_args()

    commands: list[str] = []
    if args.section in {"p1a", "grid"}:
        commands.extend(rq2_grid_commands(physical_gpu=args.physical_gpu))
    if args.section in {"p1a", "switch"}:
        commands.extend(switch_commands(physical_gpu=args.physical_gpu))
    if args.section in {"full-grid", "full"}:
        commands.extend(rq2_grid_commands(physical_gpu=args.physical_gpu, full_grid=True))
    if args.section in {"full-switch", "full"}:
        commands.extend(
            switch_commands(physical_gpu=args.physical_gpu, full_switch=True)
        )

    print("# Generated P1 queue. Run only after P0 audit passes.")
    print("# Default section p1a emits the prioritized 8-run decision set.")
    print("# Use full-grid/full-switch only if the sparse decision set is ambiguous.")
    for command in commands:
        print(command)


if __name__ == "__main__":
    main()
