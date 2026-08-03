#!/usr/bin/env python3
"""Render every empirical manuscript PDF from registered JSON artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynaexq.experiments.eval_quality import (
    SCHEMA_VERSION,
    environment_metadata,
)


DEFAULT_MANIFEST = ROOT / "results" / "paper" / "manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "ICCAD_2026_DynExq" / "figures"
DEFAULT_PROVENANCE = ROOT / "results" / "paper" / "figure_provenance.json"
FIGURE_GROUPS = (
    "performance",
    "budget_sensitivity",
    "offload_waiting",
    "routing_hotset",
    "perplexity_curve",
)
MODEL_FILES = {
    "qwen30b": "Qwen3-30B",
    "qwen80b": "Qwen3-80B",
    "phi35": "Phi-3.5-MoE",
}
MODEL_LABELS = {
    **MODEL_FILES,
    "qwen80b": "Qwen3-Next-80B",
}
METHOD_LABELS = {
    "static_ptq": "Static PTQ",
    "moe_infinity": "MoE-Infinity",
    "dynaexq": "DynaExQ",
}
METHOD_STYLES = {
    "static_ptq": ("#4c78a8", "o"),
    "moe_infinity": ("#e45756", "s"),
    "dynaexq": ("#54a24b", "^"),
}
MODEL_METHODS = {
    "qwen30b": ("static_ptq", "moe_infinity", "dynaexq"),
    "qwen80b": ("static_ptq", "dynaexq"),
    "phi35": ("static_ptq", "dynaexq"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_claims(
    manifest_path: Path,
    groups: tuple[str, ...] = FIGURE_GROUPS,
) -> tuple[dict[str, dict], dict[str, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", 0)) != 2:
        raise ValueError("figure rendering requires manifest schema 2")
    claims = {}
    hashes = {}
    for group in groups:
        records = manifest.get("groups", {}).get(group)
        if not isinstance(records, list) or not records:
            raise ValueError(f"manifest group is incomplete: {group}")
        for record in records:
            claim_id = record["claim_id"]
            path = ROOT / record["path"]
            observed = sha256(path)
            if observed != record["sha256"]:
                raise ValueError(f"artifact hash mismatch: {claim_id}")
            data = json.loads(path.read_text(encoding="utf-8"))
            if int(data.get("schema_version", 0)) < 2:
                raise ValueError(f"legacy artifact: {claim_id}")
            claims[claim_id] = data
            hashes[claim_id] = observed
    return claims, hashes


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Creator": "DynaExQ registered-artifact renderer",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def _axes(ylabel: str):
    fig, axis = plt.subplots(figsize=(3.15, 2.15))
    axis.set_xlabel("Batch size")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25, linewidth=0.6)
    return fig, axis


def _render_performance(claims: dict[str, dict], output_dir: Path) -> list[Path]:
    outputs = []
    batches = (1, 2, 4, 8, 16, 32)
    metrics = (
        ("mean", "Average latency (s)", "avg_latency_end2end_vs_batch_size"),
        ("p99", "P99 latency (s)", "p99_latency_end2end_vs_batch_size"),
        ("throughput", "Throughput (tokens/s)", "p99_latency_throughput"),
    )
    for model, file_prefix in MODEL_FILES.items():
        for metric, ylabel, suffix in metrics:
            fig, axis = _axes(ylabel)
            for method in MODEL_METHODS[model]:
                artifacts = [
                    claims[
                        f"performance:{model}:{method}:bs{batch}"
                    ]
                    for batch in batches
                ]
                if metric == "throughput":
                    values = [
                        artifact["benchmark"]["metrics"][
                            "throughput_tokens_s"
                        ]["mean"]
                        for artifact in artifacts
                    ]
                else:
                    values = [
                        artifact["benchmark"]["metrics"]["model_e2e_ms"][
                            metric
                        ]
                        / 1000.0
                        for artifact in artifacts
                    ]
                color, marker = METHOD_STYLES[method]
                axis.plot(
                    batches,
                    values,
                    label=METHOD_LABELS[method],
                    color=color,
                    marker=marker,
                    linewidth=1.5,
                    markersize=3.5,
                )
            axis.set_xticks(batches)
            axis.legend(frameon=False, fontsize=7)
            output = output_dir / f"{file_prefix}_{suffix}.pdf"
            _save(fig, output)
            outputs.append(output)
    return outputs


def _render_sensitivity(claims: dict[str, dict], output_dir: Path) -> list[Path]:
    outputs = []
    ratios = (0, 5, 10, 15, 20, 25, 30)
    for model in ("qwen30b", "qwen80b"):
        values = [
            claims[f"budget_sensitivity:{model}:ratio{ratio}"][
                "paper_metrics"
            ]["average_accuracy_pct"]
            for ratio in ratios
        ]
        fig, axis = plt.subplots(figsize=(3.15, 2.15))
        axis.plot(ratios, values, color="#54a24b", marker="o", linewidth=1.5)
        axis.axhline(
            values[0],
            color="#4c78a8",
            linestyle="--",
            linewidth=1.0,
            label="All-low baseline",
        )
        axis.set_xlabel("High-precision experts per layer (%)")
        axis.set_ylabel("Average accuracy (%)")
        axis.grid(True, alpha=0.25, linewidth=0.6)
        axis.legend(frameon=False, fontsize=7)
        output = output_dir / f"budget_sensitivity_{model}.pdf"
        _save(fig, output)
        outputs.append(output)
    return outputs


def _render_waiting(claims: dict[str, dict], output_dir: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(4.5, 2.65))
    for model, color, marker in (
        ("phi35", "#4c78a8", "o"),
        ("qwen30b", "#f2cf5b", "s"),
        ("qwen80b", "#b279a2", "^"),
    ):
        points = claims[f"offload_waiting:{model}"]["benchmark"]["points"]
        axis.plot(
            [point["input_tokens"] for point in points],
            [point["mean_waiting_ms"] / 1000.0 for point in points],
            label=MODEL_LABELS[model],
            color=color,
            marker=marker,
            markersize=2.8,
            linewidth=1.3,
        )
    axis.set_xlabel("Input tokens")
    axis.set_ylabel("Exposed transfer time (s)")
    axis.grid(True, alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False, fontsize=7)
    output = output_dir / "waiting_latency_vs_prompt_length.pdf"
    _save(fig, output)
    return [output]


def _render_hotsets(claims: dict[str, dict], output_dir: Path) -> list[Path]:
    bundle = claims["routing_hotset:qwen30b:wikitext:layer15"]
    outputs = []
    filenames = {
        "wikitext": "wikitext_thinking_on_layer_15.pdf",
        "gsm8k": "gsm8k_thinking_off_layer_15.pdf",
        "humaneval": "humaneval_thinking_on_layer_15.pdf",
    }
    for workload, filename in filenames.items():
        result = bundle["workloads"][workload]
        counts = result["expert_counts"]
        top10 = set(result["top10"])
        colors = ["#e45756" if expert in top10 else "#4c78a8" for expert in range(128)]
        fig, axis = plt.subplots(figsize=(4.25, 2.25))
        axis.bar(range(128), counts, color=colors, width=0.9)
        axis.set_xlabel("Expert ID")
        axis.set_ylabel("Activation count")
        axis.grid(True, axis="y", alpha=0.2, linewidth=0.5)
        output = output_dir / filename
        _save(fig, output)
        outputs.append(output)
    return outputs


def _render_perplexity(claims: dict[str, dict], output_dir: Path) -> list[Path]:
    outputs = []
    for model in ("qwen30b", "qwen80b"):
        points = claims[f"perplexity_curve:{model}"]["points"]
        fig, axis = plt.subplots(figsize=(3.15, 2.15))
        axis.plot(
            [point["low_ratio_pct"] for point in points],
            [point["perplexity"] for point in points],
            color="#54a24b",
            marker="o",
            linewidth=1.5,
            markersize=3.5,
        )
        axis.set_xlabel("Low-precision experts per layer (%)")
        axis.set_ylabel("WikiText-2 perplexity")
        axis.grid(True, alpha=0.25, linewidth=0.6)
        output = output_dir / f"wiki_ppl_{model}.pdf"
        _save(fig, output)
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument(
        "--group",
        action="append",
        choices=FIGURE_GROUPS,
        help="Render only this figure group; repeat for multiple groups.",
    )
    args = parser.parse_args()

    groups = tuple(args.group or FIGURE_GROUPS)
    claims, input_hashes = _load_claims(args.manifest, groups)
    outputs = []
    if "performance" in groups:
        outputs.extend(_render_performance(claims, args.output_dir))
    if "budget_sensitivity" in groups:
        outputs.extend(_render_sensitivity(claims, args.output_dir))
    if "offload_waiting" in groups:
        outputs.extend(_render_waiting(claims, args.output_dir))
    if "routing_hotset" in groups:
        outputs.extend(_render_hotsets(claims, args.output_dir))
    if "perplexity_curve" in groups:
        outputs.extend(_render_perplexity(claims, args.output_dir))
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "paper_figure_bundle",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": dict(sorted(input_hashes.items())),
        "figures": {
            path.resolve().relative_to(ROOT).as_posix(): sha256(path)
            for path in sorted(outputs)
        },
        "command": " ".join([sys.executable, *sys.argv]),
        "environment": environment_metadata(),
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "provenance": str(args.provenance),
                "figures": len(outputs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
