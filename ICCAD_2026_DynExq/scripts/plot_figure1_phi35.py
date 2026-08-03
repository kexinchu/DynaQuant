#!/usr/bin/env python3
"""Render Figure 1 with Phi-3.5-MoE replacing DeepSeek-V2-Lite.

The Qwen series below are the exact values used by the legacy Figure 1
notebook (``scripts/moe_offload_waiting_latency_plot.ipynb``), rather than
values estimated from raster pixels.  The Phi series is copied from the
registered RTX A6000 artifact ``phi35_offload_waiting.json`` (SHA-256 is
recorded in ``SOURCES``).  The script emits both the PDF and a JSON sidecar so
that every plotted point remains machine-readable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER_DIR / "figures" / "waiting_latency_vs_prompt_length.pdf"
DEFAULT_DATA = PAPER_DIR / "figure_data" / "figure1_waiting_latency.json"

SOURCES = {
    "legacy_figure_pdf_sha256": (
        "ae15f531128e21a454654cf752d5cf441595640cad638dfd7ddbf6acf129dc4a"
    ),
    "legacy_notebook_sha256": (
        "d2e04d716bae7647be47743588fbc61533a40ee6a6ce643f4069ecf1e13277e5"
    ),
    "phi35_registered_artifact_sha256": (
        "87e03ab494c111482e2f860eda45aa87ce7d2997fec20e5c5a9b3ec2d1e7235f"
    ),
}

QWEN_INPUT_TOKENS = [
    1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 288,
    320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 960, 992, 1024,
]

# Exact hard-coded series used to render the previous Figure 1.
QWEN30_WAITING_MS = [
    0.0, 0.0, 0.0, 0.0, 4.09, 75.97, 296.08, 397.64, 887.09,
    1034.41, 1216.57, 1465.46, 1497.02, 1622.94, 1844.58, 1915.50,
    1915.33, 2094.64, 2100.43, 2194.97, 2309.17, 2234.74, 2270.26,
    2272.09, 2403.93, 2374.42, 2467.61, 2448.97, 2389.63, 2470.27,
    2549.53, 2592.10, 2606.82, 2638.91, 2613.00, 2536.34, 2669.89,
    2688.61,
]
QWEN80_WAITING_MS = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.98, 14.35, 82.06, 139.16,
    214.84, 367.77, 427.58, 635.53, 661.10, 951.61, 1038.06,
    1242.11, 1476.44, 1645.19, 1878.45, 1955.26, 2456.87, 2372.86,
    2575.73, 2893.66, 2926.20, 3235.35, 3410.28, 3592.31, 3629.43,
    3769.90, 3846.66, 3750.55, 3990.60, 4082.16, 4125.24, 4419.01,
]

PHI_INPUT_TOKENS = [
    16, 32, 64, 96, 128, 192, 256, 320, 384, 448, 512, 640, 768,
    896, 1024,
]
PHI_WAITING_MS = [
    579.4929608004168,
    673.6261244863272,
    724.3845098768361,
    744.9002452893183,
    758.1549718976021,
    771.4134392794222,
    781.6234677797183,
    784.6710579935461,
    788.3955725003034,
    792.9226394044235,
    796.4910357026383,
    800.5278140190057,
    804.2646969435737,
    807.0002459804527,
    808.2862010924146,
]


def _series() -> dict[str, dict[str, object]]:
    return {
        "Phi-3.5-MoE": {
            "input_tokens": PHI_INPUT_TOKENS,
            "mean_waiting_ms": PHI_WAITING_MS,
            "source": "registered cold-cache routing-trace replay",
        },
        "Qwen3-30B": {
            "input_tokens": QWEN_INPUT_TOKENS,
            "mean_waiting_ms": QWEN30_WAITING_MS,
            "source": "exact series from legacy Figure 1 notebook",
        },
        "Qwen3-Next-80B": {
            "input_tokens": QWEN_INPUT_TOKENS,
            "mean_waiting_ms": QWEN80_WAITING_MS,
            "source": "exact series from legacy Figure 1 notebook",
        },
    }


def render(output: Path, data_output: Path) -> None:
    series = _series()
    assert len(QWEN_INPUT_TOKENS) == len(QWEN30_WAITING_MS)
    assert len(QWEN_INPUT_TOKENS) == len(QWEN80_WAITING_MS)
    assert len(PHI_INPUT_TOKENS) == len(PHI_WAITING_MS)

    styles = {
        "Phi-3.5-MoE": ("#4c78a8", "o", 1),
        "Qwen3-30B": ("#f28e2b", "s", 2),
        "Qwen3-Next-80B": ("#59a14f", "^", 2),
    }
    fig, axis = plt.subplots(figsize=(4.5, 2.65))
    for label, values in series.items():
        color, marker, markevery = styles[label]
        axis.plot(
            values["input_tokens"],
            [value / 1000.0 for value in values["mean_waiting_ms"]],
            color=color,
            marker=marker,
            markevery=markevery,
            linewidth=1.55,
            markersize=3.5,
            label=label,
        )

    axis.set_xlim(0, 1040)
    axis.set_ylim(-0.05, 4.6)
    axis.set_xlabel("Input tokens")
    axis.set_ylabel("Mean waiting time (s)")
    axis.grid(True, alpha=0.28, linewidth=0.6)
    axis.legend(loc="upper left", fontsize=8, framealpha=0.92)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Creator": "DynaExQ Figure 1 renderer",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)

    data_output.parent.mkdir(parents=True, exist_ok=True)
    data_output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "figure": "waiting_latency_vs_prompt_length.pdf",
                "unit": "ms",
                "sources": SOURCES,
                "series": series,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-output", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    render(args.output, args.data_output)
    print(args.output)
    print(args.data_output)


if __name__ == "__main__":
    main()
