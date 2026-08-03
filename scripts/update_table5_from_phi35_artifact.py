#!/usr/bin/env python3
"""Fill the Phi-3.5-MoE column of paper Table V from a formal artifact."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path


PAPER_DIR = Path("/home/kec23008/DynaQuant/ICCAD_2026_DynExq")
TEX_PATH = PAPER_DIR / "05_evaluation.tex"


def nearest_rank_p99(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("scheduler telemetry is empty")
    return ordered[math.ceil(0.99 * len(ordered)) - 1]


def replace_phi_cell(lines: list[str], label: str, value: str) -> None:
    matches = [index for index, line in enumerate(lines) if line.startswith(label)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one Table V row for {label!r}, found {len(matches)}")
    index = matches[0]
    cells = lines[index].split("&")
    if len(cells) != 4 or not cells[-1].rstrip().endswith(r"\\"):
        raise RuntimeError(f"unexpected Table V row format: {lines[index]!r}")
    cells[-1] = f" {value} " + r"\\"
    lines[index] = "&".join(cells)


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count == 0 and new in text:
        return text
    if count != 1:
        raise RuntimeError(f"expected one prose occurrence, found {count}: {old!r}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()

    data = json.loads(args.artifact.read_text(encoding="utf-8"))
    benchmark = data["benchmark"]
    samples = benchmark["samples"]
    scheduler = [float(value) for value in data["wrapper_stats"]["scheduler_update_samples_ms"]]
    transition = data["transition_stats"]

    expected = {
        "paper_model": "phi35",
        "paper_method": "dynaexq",
        "batch_size": 32,
        "input_tokens": 2048,
        "output_tokens_per_sequence": 256,
        "warmup_iterations": 5,
        "measured_iterations": 100,
        "sample_count": 100,
    }
    observed = {
        "paper_model": data.get("paper_model"),
        "paper_method": data.get("paper_method"),
        "batch_size": benchmark.get("batch_size"),
        "input_tokens": benchmark.get("input_tokens"),
        "output_tokens_per_sequence": benchmark.get("output_tokens_per_sequence"),
        "warmup_iterations": benchmark.get("warmup_iterations"),
        "measured_iterations": benchmark.get("measured_iterations"),
        "sample_count": len(samples),
    }
    if observed != expected:
        raise RuntimeError(f"formal protocol mismatch: expected {expected}, observed {observed}")

    metrics = {
        "Peak Process GPU Memory (GB)": f"{max(float(sample['process_hbm_used_peak_bytes']) for sample in samples) / 1e9:.2f}",
        "Migration Count": str(int(transition["total_promotions"]) + int(transition["total_demotions"])),
        "Transferred (GB)": f"{float(transition['copied_bytes']) / 1e9:.2f}",
        "Scheduler Mean (ms)": f"{statistics.fmean(scheduler):.2f}",
        "Scheduler P99 (ms)": f"{nearest_rank_p99(scheduler):.2f}",
    }

    text = TEX_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    for label, value in metrics.items():
        replace_phi_cell(lines, label, value)
    text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    text = replace_once(
        text,
        r"\autoref{tab:overhead} reports completed formal runs for Qwen3-30B (bs\,=\,32) and provisional Qwen3-Next-80B (bs\,=\,8); Phi-3.5-MoE currently reports initialization capacities pending its bs\,=\,32 performance run.",
        r"\autoref{tab:overhead} reports completed formal runs for Qwen3-30B and Phi-3.5-MoE (both bs\,=\,32), alongside the provisional Qwen3-Next-80B run (bs\,=\,8).",
    )
    text = replace_once(
        text,
        r"\caption{Memory utilization and runtime overhead. Q30: bs32; Q80: provisional bs8; Phi: initialization capacities.}",
        r"\caption{Memory utilization and runtime overhead. Q30 and Phi: bs32; Q80: provisional bs8.}",
    )
    text = replace_once(
        text,
        "the table takes the maximum over 100 samples (Qwen3-30B/bs32; provisional Qwen3-Next-80B/bs8).",
        "the table takes the maximum over 100 samples (Qwen3-30B/bs32; provisional Qwen3-Next-80B/bs8; Phi-3.5-MoE/bs32).",
    )
    TEX_PATH.write_text(text, encoding="utf-8")

    summary_path = args.artifact.with_name("phi35_table5_metrics.json")
    summary_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main_sc.tex"],
            cwd=PAPER_DIR,
            check=True,
        )
    print(json.dumps({"table": str(TEX_PATH), "pdf": str(PAPER_DIR / "main_sc.pdf"), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
