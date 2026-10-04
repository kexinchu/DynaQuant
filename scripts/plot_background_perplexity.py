#!/usr/bin/env python3
"""Render the two WikiText-2 perplexity panels in Figure 1.

Points were read from the vector paths of the existing figures
``ICCAD_2026_DynExq/figures/wiki_ppl_qwen30b.pdf`` and
``wiki_ppl_qwen80b.pdf``. The x positions are the labeled ticks
0, 15, 30, 45, 60, 75, 90, and 100. Each y value is the perplexity
that lands on the corresponding path vertex, to three decimal places.
These are the values drawn in those PDFs, not a new measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "SIGMETRICS_2027/figures"
LOW_PRECISION_PERCENT = (0, 15, 30, 45, 60, 75, 90, 100)
CURVES = {
    "qwen30b": (6.413, 6.431, 6.485, 6.593, 6.813, 7.034, 7.170, 7.189),
    "qwen80b": (5.740, 5.760, 5.800, 5.910, 6.120, 6.280, 6.360, 6.400),
}
OUTPUT_NAMES = {
    "qwen30b": "wiki_ppl_qwen30b.pdf",
    "qwen80b": "wiki_ppl_qwen80b.pdf",
}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Creator": "DynaByte background perplexity renderer",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def render_perplexity(output_dir: Path) -> dict[str, Path]:
    outputs = {}
    with plt.rc_context(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    ):
        for model, perplexity in CURVES.items():
            fig, axis = plt.subplots(figsize=(6.4 * 0.6, 4.8 * 0.4))
            axis.plot(
                LOW_PRECISION_PERCENT,
                perplexity,
                color="#4682b4",
                marker="o",
                markersize=8,
                linewidth=2.5,
            )
            axis.set_xticks(LOW_PRECISION_PERCENT)
            axis.set_xlabel("Percent of Low-Precision Experts (%)")
            axis.set_ylabel("Perplexity")
            # axis.set_title("WikiText")
            axis.grid(True, linestyle="--", color="#b0b0b0", linewidth=0.8)
            path = output_dir / OUTPUT_NAMES[model]
            _save(fig, path)
            outputs[model] = path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    outputs = render_perplexity(args.output_dir)
    print(json.dumps({"figures": [str(path) for path in outputs.values()]}, indent=2))


if __name__ == "__main__":
    main()
