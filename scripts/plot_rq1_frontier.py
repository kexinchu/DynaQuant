#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
TEST1 = ROOT / "results/evaluation/Test1"
FIGURE = ROOT / "SIGMETRICS_2027/figures/eval_quality_latency.pdf"
TABLE_DIR = ROOT / "SIGMETRICS_2027/tables"

MODEL_ORDER = ("Qwen3-30B", "Qwen3-Next-80B", "Phi-3.5-MoE")
PANEL = {
    "Qwen3-30B": (r"Qwen3-30B, $\Phi=1.3$", 2.1, 40.0, (-0.15, 2.95)),
    "Qwen3-Next-80B": (r"Qwen3-Next-80B, $\Phi=1.4$", 0.60, 80, (-0.04, 0.85)),
    "Phi-3.5-MoE": (r"Phi-3.5-MoE, $\Phi=1.2$", 0.70, 45, (-0.05, 1.00)),
}
FLOOR_MS = {
    "Qwen3-30B": 43.8,
    "Qwen3-Next-80B": 117.0,
    "Phi-3.5-MoE": 51.8,
}
POLICY_ORDER = (
    "dynabyte",
    "hobbit",
    "static_joint",
    "uniform_low",
    "all_high",
)
POLICY_LABEL = {
    "dynabyte": "DynaByte",
    "hobbit": "HOBBIT",
    "static_joint": "Static-Joint",
    "uniform_low": "Uniform-low",
    "all_high": "All-high",
}
POLICY_COLOR = {
    "dynabyte": "#c0392b",
    "hobbit": "#1a5276",
    "static_joint": "#b9770e",
    "uniform_low": "#2c3e50",
    "all_high": "#7f8c8d",
}
POLICY_MARKER = {
    "dynabyte": "o",
    "hobbit": "D",
    "static_joint": "s",
    "uniform_low": "h",
    "all_high": "*",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(row: dict, key: str) -> float:
    return float(row[key])


def feasible(row: dict) -> bool:
    return row["quality_feasible"] == "yes"


def tex_num(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def tex_policy(policy: str) -> str:
    if policy == "dynabyte":
        return r"\systemname"
    if policy == "hobbit":
        return "HOBBIT"
    return POLICY_LABEL[policy]


def tex_knob(knob: str) -> str:
    knob = knob.replace("%", r"\%")
    if knob.startswith("eps "):
        return r"$\epsilon{=}$" + knob[4:]
    return knob


def check_consistency(frontier: list[dict], main_rows: list[dict]) -> None:
    for row in main_rows:
        key = (row["model"], row["policy"])
        matches = [
            item for item in frontier
            if item["model"] == row["model"]
            and item["policy"] == row["policy"]
            and abs(num(item, "p95_tpot_ms") - num(row, "p95_tpot_ms")) < 1e-6
        ]
        if not matches:
            raise SystemExit(f"main row has no frontier point with the same p95: {key}")
        floor = FLOOR_MS[row["model"]]
        exposed = num(row, "exposed_ms")
        if abs(num(row, "p95_tpot_ms") - (floor + exposed)) > 1e-6:
            raise SystemExit(
                f"p95 is not {floor:g}+exposed for {key}: "
                f"{row['p95_tpot_ms']} vs {exposed}"
            )
    for item in frontier:
        if item["selected"] != "yes":
            continue
        key = (item["model"], item["policy"])
        matches = [
            row for row in main_rows
            if row["model"] == item["model"]
            and row["policy"] == item["policy"]
            and abs(num(row, "p95_tpot_ms") - num(item, "p95_tpot_ms")) < 1e-6
        ]
        if not matches:
            raise SystemExit(f"selected frontier point is missing from the main table: {key}")


def render_figure(frontier: list[dict], output: Path) -> None:
    with plt.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 4.8*0.6))
        for axis, model in zip(axes, MODEL_ORDER):
            title, epsilon, xmin, ylim = PANEL[model]
            rows = [row for row in frontier if row["model"] == model]
            axis.axhspan(epsilon, ylim[1], color="#f5e6e6", zorder=0)
            axis.axhline(epsilon, color="#922b21", linestyle="--", linewidth=0.8, zorder=1)
            axis.text(
                0.98, epsilon, rf"$\epsilon={epsilon:g}$",
                transform=axis.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6.5, color="#922b21",
            )
            for policy in POLICY_ORDER:
                group = [row for row in rows if row["policy"] == policy]
                if not group:
                    continue
                color = POLICY_COLOR[policy]
                marker = POLICY_MARKER[policy]
                xs = [num(row, "p95_tpot_ms") for row in group]
                ys = [num(row, "delta_ppl") for row in group]
                xerr = [
                    [num(row, "p95_tpot_ms") - num(row, "ci_low") for row in group],
                    [num(row, "ci_high") - num(row, "p95_tpot_ms") for row in group],
                ]
                face = [color if feasible(row) else "white" for row in group]
                axis.errorbar(
                    xs, ys, xerr=xerr, fmt="none", ecolor=color, elinewidth=0.7,
                    capsize=2, alpha=0.85, zorder=2,
                )
                axis.scatter(
                    xs, ys, s=26 if policy != "all_high" else 36, marker=marker,
                    facecolors=face, edgecolors=color, linewidths=1.0, zorder=3,
                )
                curve = sorted(
                    (row for row in group if feasible(row) and policy != "all_high"),
                    key=lambda row: num(row, "p95_tpot_ms"),
                )
                if len(curve) >= 2:
                    axis.plot(
                        [num(row, "p95_tpot_ms") for row in curve],
                        [num(row, "delta_ppl") for row in curve],
                        color=color, linewidth=1.0, zorder=2,
                    )
            axis.set_title(title)
            axis.set_xlabel("p95 TPOT (ms)")
            axis.set_xlim(left=xmin)
            axis.set_ylim(*ylim)
            axis.grid(True, linestyle=":", linewidth=0.4, color="#b0b0b0")
        axes[0].set_ylabel(r"$\Delta$PPL vs. All-high")
        handles = [
            Line2D(
                [0], [0], marker=POLICY_MARKER[policy], color=POLICY_COLOR[policy],
                markerfacecolor="white" if policy == "uniform_low" else POLICY_COLOR[policy],
                linestyle="none",
                markersize=5, label=POLICY_LABEL[policy],
            )
            for policy in POLICY_ORDER
        ]
        fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
                   bbox_to_anchor=(0.5, 1.05))
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, default=TEST1 / "rq1_frontier.csv")
    parser.add_argument("--main-csv", type=Path, default=TEST1 / "rq1_main.csv")
    parser.add_argument("--accuracy", type=Path, default=TEST1 / "rq1_accuracy.csv")
    parser.add_argument("--figure", type=Path, default=FIGURE)
    args = parser.parse_args()
    frontier = read_csv(args.frontier)
    # main_rows = read_csv(args.main_csv)
    # accuracy = read_csv(args.accuracy)
    # check_consistency(frontier, main_rows)
    render_figure(frontier, args.figure)
    print(args.figure)


if __name__ == "__main__":
    main()
