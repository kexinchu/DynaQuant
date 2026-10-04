#!/usr/bin/env python3
"""Export the registered Qwen3-30B replay into Evaluation JSONL contracts.

The exporter never relabels replayed exposed wait as end-to-end latency.  It
also refuses artifacts that do not carry the explicit ``not_end_to_end`` bit,
which prevents this lane from silently populating the headline result table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/paper/a6000_wikitext_replay/summary.json"
DEFAULT_OUTPUT = ROOT / "results/evaluation/coupling_grid.jsonl"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def export(source: Path, output: Path) -> int:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("not_end_to_end") is not True:
        raise ValueError("source must be explicitly marked not_end_to_end")
    if payload.get("artifact_type") != "a6000_wikitext_deadline_replay":
        raise ValueError("unexpected source artifact type")

    try:
        source_name = str(source.relative_to(ROOT))
    except ValueError:
        source_name = str(source)

    rows: list[dict] = []
    for length in payload.get("lengths", []):
        for cell in length.get("coupling_grid", []):
            row = {
                "schema_version": 1,
                "evidence_class": "trace_replay",
                "not_end_to_end": True,
                "model": "Qwen3-30B-A3B-Instruct-2507",
                "device": "NVIDIA RTX A6000",
                "workload": "wikitext103",
                "input_tokens": int(length["input_tokens"]),
                "split": cell["split"],
                "phi": float(cell["phi"]),
                "fid_bytes": int(cell["fid_bytes"]),
                "res_bytes": int(cell["res_bytes"]),
                "look_bytes": int(cell["look_bytes"]),
                "budget_bytes": int(cell["budget_bytes"]),
                "quality_metric": "wikitext2_perplexity_interpolation",
                "quality_delta": float(cell["quality_delta"]),
                "perplexity": float(cell["perplexity"]),
                "exposed_wait_mean_ms": float(cell["exposed_wait_mean_ms"]),
                "exposed_wait_p95_ms": float(cell["exposed_wait_p95_ms"]),
                "n_high": int(cell["n_high"]),
                "n_low": int(cell["n_low"]),
                "staging_slots": int(cell["staging_slots"]),
                "source_artifact": source_name,
            }
            if row["fid_bytes"] + row["res_bytes"] + row["look_bytes"] > row["budget_bytes"]:
                raise ValueError("three-use byte decomposition exceeds budget")
            rows.append(row)
    if not rows:
        raise ValueError("source contains no coupling-grid cells")
    _write_jsonl(output, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    count = export(args.input.resolve(), args.output.resolve())
    print(f"wrote {count} replay rows to {args.output}")


if __name__ == "__main__":
    main()
