#!/usr/bin/env python3
"""Plot RQ2 only when coupling and adaptation evidence are both present."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def plot(coupling: list[dict], adaptation: list[dict], output: Path, *, allow_incomplete: bool) -> None:
    if not coupling:
        raise ValueError("coupling data are empty")
    if not adaptation and not allow_incomplete:
        raise ValueError("adaptation data are required for the paper figure")

    target = [
        row for row in coupling
        if row.get("model", "").startswith("Qwen3-30B")
        and row.get("input_tokens") == 2048
        and row.get("split") == "test"
        and abs(float(row.get("phi", -1)) - 0.85) < 1e-6
    ]
    if not target:
        raise ValueError("missing Qwen3-30B, 2048-token, test, phi=0.85 grid")
    fid = sorted({round(100 * row["fid_bytes"] / row["budget_bytes"], 2) for row in target})
    look = sorted({round(100 * row["look_bytes"] / row["budget_bytes"], 2) for row in target})
    values = np.full((len(look), len(fid)), np.nan)
    quality = np.full_like(values, np.nan)
    for row in target:
        x = fid.index(round(100 * row["fid_bytes"] / row["budget_bytes"], 2))
        y = look.index(round(100 * row["look_bytes"] / row["budget_bytes"], 2))
        values[y, x] = row["exposed_wait_p95_ms"]
        quality[y, x] = row["quality_delta"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), constrained_layout=True)
    shown = np.ma.masked_where(quality > 0.5, values)
    image = axes[0].imshow(shown, origin="lower", aspect="auto", cmap="viridis_r")
    axes[0].set_xticks(range(len(fid)), [f"{value:g}" for value in fid], rotation=45)
    axes[0].set_yticks(range(len(look)), [f"{value:g}" for value in look])
    axes[0].set_xlabel("Fidelity bytes (% of B)")
    axes[0].set_ylabel("Lookahead bytes (% of B)")
    axes[0].set_title("(a) Held-out deadline replay")
    fig.colorbar(image, ax=axes[0], label="p95 exposed wait (ms)")

    if adaptation:
        ordered = sorted(adaptation, key=lambda row: float(row["time"]))
        time = [float(row["time"]) for row in ordered]
        budget = [float(row["budget_bytes"]) for row in ordered]
        for field, label in (("fid_bytes", "Fidelity"), ("res_bytes", "Residency"), ("look_bytes", "Lookahead")):
            axes[1].plot(time, [100 * float(row[field]) / cap for row, cap in zip(ordered, budget)], label=label)
        axes[1].set_xlabel("Time from workload change")
        axes[1].set_ylabel("Bytes (% of B)")
        axes[1].set_title("(b) Online adaptation")
        axes[1].legend(frameon=False)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "Adaptation measurement pending", ha="center", va="center")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coupling", type=Path, default=ROOT / "results/evaluation/coupling_grid.jsonl")
    parser.add_argument("--adaptation", type=Path, default=ROOT / "results/evaluation/adaptation.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "SIGMETRICS_2027/figures/eval_joint_necessity.pdf")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    plot(read_jsonl(args.coupling), read_jsonl(args.adaptation), args.output, allow_incomplete=args.allow_incomplete)


if __name__ == "__main__":
    main()
