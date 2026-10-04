#!/usr/bin/env python3
"""Render the two replay-backed Background motivation figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "results/paper/background"
DEFAULT_OUTPUT_DIR = ROOT / "SIGMETRICS_2027/figures"


def _load(path: Path, artifact_type: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("artifact_type") != artifact_type:
        raise ValueError(f"unexpected artifact type in {path}")
    return data


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Creator": "DynaByte background motivation renderer",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def transfer_pressure_paths(output: Path) -> dict[str, Path]:
    """Map the shared legend and each stage to its own PDF beside ``output``."""
    paths = {
        stage: output.with_name(f"{output.stem}_{stage}{output.suffix}")
        for stage in ("prefill", "decode")
    }
    paths["legend"] = output.with_name(f"{output.stem}_legend{output.suffix}")
    return paths


def render_transfer_pressure(data: dict, output: Path) -> dict[str, Path]:
    outputs = transfer_pressure_paths(output)
    font = 11
    with plt.rc_context(
        {
            "font.size": font,
            "axes.labelsize": font,
            "axes.titlesize": font,
            "xtick.labelsize": font,
            "ytick.labelsize": font,
            "legend.fontsize": font,
        }
    ):
        for stage in ("prefill", "decode"):
            stage_output = outputs[stage]
            fig, axis = plt.subplots(figsize=(6.4 * 0.6, 4.8 * 0.4))
            records = data["stages"][stage]
            batches = [int(record["batch_size"]) for record in records]
            demand = [
                record["mean_demanded_bytes_per_layer"] / 2**20
                for record in records
            ]
            capacity = [
                record["overlap_capacity_upper_bytes"] / 2**20
                for record in records
            ]
            axis.plot(
                batches,
                demand,
                color="#e45756",
                marker="o",
                linewidth=1.5,
                label="Active expert bytes",
            )
            axis.plot(
                batches,
                capacity,
                color="#54a24b",
                marker="s",
                linewidth=1.3,
                linestyle="--",
                label="Overlap capacity (upper bound)",
            )
            axis.fill_between(
                batches,
                capacity,
                demand,
                where=np.asarray(demand) > np.asarray(capacity),
                color="#e45756",
                alpha=0.16,
            )
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
            axis.yaxis.set_major_formatter(
                LogFormatterMathtext(base=10.0, labelOnlyBase=True)
            )
            axis.yaxis.set_minor_locator(NullLocator())
            axis.set_xticks(batches, batches)
            if len(batches) > 6:
                axis.tick_params(axis="x", labelrotation=30)
            axis.set_xlabel("Batch size")
            axis.set_ylabel("MiB per layer")
            # axis.set_title(stage.capitalize())
            axis.grid(True, alpha=0.20, linewidth=0.5)
            _save(fig, stage_output)
        _save(_transfer_pressure_legend(font), outputs["legend"])
    return outputs


def _transfer_pressure_legend(font: int):
    fig = plt.figure(figsize=(6.4 * 1.0, 0.4))
    handles = [
        plt.Line2D(
            [],
            [],
            color="#e45756",
            marker="o",
            linewidth=1.5,
            label="Active expert bytes",
        ),
        plt.Line2D(
            [],
            [],
            color="#54a24b",
            marker="s",
            linewidth=1.3,
            linestyle="--",
            label="Overlap capacity (upper bound)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=2,
        fontsize=font,
        columnspacing=1.2,
        handlelength=2.4,
    )
    return fig


def render_joint_tradeoff(data: dict, output: Path) -> None:
    ratios = [int(value) for value in data["high_precision_ratios_pct"]]
    slots = [int(value) for value in data["staging_experts_per_layer"]]
    cells_by_ratio = {
        ratio: sorted(
            (
                cell
                for cell in data["cells"]
                if int(cell["high_precision_ratio_pct"]) == ratio
            ),
            key=lambda cell: int(cell["staging_experts_per_layer"]),
        )
        for ratio in ratios
    }

    profiles = {}
    quality_excluded = []
    memory_excluded = []
    for ratio, cells in cells_by_ratio.items():
        quality_ok = any(bool(cell["quality_feasible"]) for cell in cells)
        feasible = [
            cell
            for cell in cells
            if bool(cell["quality_feasible"])
            and bool(cell["memory_feasible"])
            and cell["mean_exposed_transfer_ms"] is not None
        ]
        if feasible:
            profiles[ratio] = feasible
        elif not quality_ok:
            quality_excluded.append(ratio)
        else:
            memory_excluded.append(ratio)

    if not profiles:
        raise ValueError("joint replay contains no quality- and memory-feasible profile")

    fig, axis = plt.subplots(figsize=(6.4, 4.8*0.6))
    colors = ("#4c78a8", "#e45756", "#54a24b", "#b279a2")
    markers = ("o", "s", "^", "D")
    linestyles = ("-", "--", "-.", ":")
    minima = {}
    for index, (ratio, cells) in enumerate(profiles.items()):
        x_values = [int(cell["staging_experts_per_layer"]) for cell in cells]
        y_values = [float(cell["mean_exposed_transfer_ms"]) for cell in cells]
        axis.plot(
            x_values,
            y_values,
            color=colors[index % len(colors)],
            marker=markers[index % len(markers)],
            linestyle=linestyles[index % len(linestyles)],
            linewidth=1.7,
            markersize=4.5,
            label=f"{ratio}% high precision",
        )
        minima[ratio] = min(
            cells,
            key=lambda cell: (
                float(cell["mean_exposed_transfer_ms"]),
                int(cell["staging_experts_per_layer"]),
            ),
        )

    optimum = data.get("joint_optimum")
    optimum_key = None
    if optimum is not None:
        optimum_key = (
            int(optimum["high_precision_ratio_pct"]),
            int(optimum["staging_experts_per_layer"]),
        )

    for index, (ratio, cell) in enumerate(minima.items()):
        x_value = int(cell["staging_experts_per_layer"])
        y_value = float(cell["mean_exposed_transfer_ms"])
        is_joint_optimum = (ratio, x_value) == optimum_key
        axis.scatter(
            [x_value],
            [y_value],
            marker="*" if is_joint_optimum else "D",
            s=105 if is_joint_optimum else 42,
            facecolor="#f2cf5b" if is_joint_optimum else "white",
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )
        offset = (18, 20) if is_joint_optimum else (18, -30)
        suffix = "joint best" if is_joint_optimum else "curve minimum"
        axis.annotate(
            (
                f"{x_value} slots; "
                f"{int(cell['low_precision_residents_per_layer'])} residents\n"
                f"{y_value:.2f} ms ({suffix})"
            ),
            xy=(x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=12,
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.7},
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "#aaaaaa",
                "linewidth": 0.5,
            },
        )

    axis.set_xticks(slots, slots)
    axis.set_xlim(min(slots) - 1, max(slots) + 1)
    axis.set_yscale("log")
    axis.set_ylim(0.3, 200)
    axis.set_xlabel("Lookahead staging slots per layer", fontsize=12)
    axis.set_ylabel("Exposed transfer time (ms/req)", fontsize=12)
    axis.grid(True, alpha=0.22, linewidth=0.5)
    axis.legend(
        frameon=False,
        fontsize=12,
        loc="center left",
        bbox_to_anchor=(0.0, 1.05),
        ncol=2,
        borderaxespad=0.0,
    )
    _save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    transfer = _load(
        args.data_dir / "qwen30b_transfer_pressure.json",
        "background_stage_transfer_pressure",
    )
    joint = _load(
        args.data_dir / "qwen30b_joint_allocation_replay.json",
        "background_joint_allocation_replay",
    )
    transfer_path = args.output_dir / "background_transfer_pressure.pdf"
    joint_path = args.output_dir / "background_joint_allocation.pdf"
    transfer_paths = render_transfer_pressure(transfer, transfer_path)
    render_joint_tradeoff(joint, joint_path)
    print(
        json.dumps(
            {
                "figures": [
                    *(str(path) for path in transfer_paths.values()),
                    str(joint_path),
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
