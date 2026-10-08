#!/usr/bin/env python3
"""Validate and summarize the promoted Qwen3-30B P1B switch baselines."""

from __future__ import annotations

import json
import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "evaluation" / "qwen30b"

ARTIFACTS = {
    "dynabyte": "rq2_switch_dynabyte_seed42.json",
    "static_joint": "rq2_switch_static_joint_seed42.json",
    "seq_fr": "rq2_switch_seq_fr_seed42.json",
    "seq_rf": "rq2_switch_seq_rf_seed42.json",
}


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def active_outputs() -> dict[str, list[int]]:
    result = run(["ps", "-eo", "pid=,args="])
    outputs: dict[str, list[int]] = {}
    if result.returncode != 0:
        return outputs
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if "dynaexq.experiments.eval_dynamic" not in stripped:
            continue
        if "--output" not in stripped:
            continue
        pid_text, _, args = stripped.partition(" ")
        if not pid_text.isdigit():
            continue
        parts = args.split()
        output = None
        for index, part in enumerate(parts):
            if part == "--output" and index + 1 < len(parts):
                output = parts[index + 1]
                break
            if part.startswith("--output="):
                output = part.split("=", 1)[1]
                break
        if output is None:
            continue
        outputs.setdefault(Path(output).name, []).append(int(pid_text))
    return outputs


def validate(path: Path) -> tuple[bool, str | None]:
    if not path.is_file() or path.stat().st_size == 0:
        return False, "missing"
    result = run(
        [
            sys.executable,
            "scripts/validate_qwen30b_artifact.py",
            str(path),
            "--allow-stale-source",
        ]
    )
    if result.returncode == 0:
        return True, None
    return False, result.stdout.strip() or "invalid"


def metric(metrics: dict[str, Any], family: str, name: str) -> float | None:
    values = metrics.get(family)
    if not isinstance(values, dict):
        return None
    value = values.get(name)
    if value is None:
        return None
    return float(value)


def summarize_artifact(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    benchmark = data.get("benchmark") or {}
    metrics = benchmark.get("metrics") or {}
    transition = data.get("transition_stats") or {}
    wrapper = data.get("wrapper_stats") or {}
    protocol = data.get("allocation_protocol") or {}
    realized = protocol.get("realized") or {}
    arena = realized.get("final_arena") or transition.get("arena") or {}
    return {
        "allocation_policy": protocol.get("allocation_policy"),
        "p95_tpot_ms": metric(metrics, "model_tpot_ms", "p95"),
        "mean_tpot_ms": metric(metrics, "model_tpot_ms", "mean"),
        "p95_ttft_ms": metric(metrics, "model_ttft_ms", "p95"),
        "mean_ttft_ms": metric(metrics, "model_ttft_ms", "mean"),
        "accepted_gb": round(float(transition.get("accepted_bytes") or 0) / 1e9, 3),
        "copied_gb": round(float(transition.get("copied_bytes") or 0) / 1e9, 3),
        "failed_transitions": int(transition.get("failed_transitions") or 0),
        "active_transitions": int(transition.get("active_transitions") or 0),
        "scheduler_p99_ms": wrapper.get("scheduler_p99_ms"),
        "fid_gb": round(float(arena.get("fid_bytes") or 0) / 1e9, 3),
        "res_gb": round(float(arena.get("res_bytes") or 0) / 1e9, 3),
        "look_gb": round(float(arena.get("look_bytes") or 0) / 1e9, 3),
    }


def build_summary() -> dict[str, Any]:
    active = active_outputs()
    rows: dict[str, dict[str, Any]] = {}
    valid_summaries: dict[str, dict[str, Any]] = {}
    for label, name in ARTIFACTS.items():
        path = RESULTS / name
        ok, detail = validate(path)
        pids = active.get(name, [])
        if ok:
            status = "valid"
            summary = summarize_artifact(path)
            valid_summaries[label] = summary
        elif pids:
            status = "running"
            summary = {}
        else:
            status = detail or "missing"
            summary = {}
        rows[label] = {
            "artifact": name,
            "status": status,
            "pids": pids,
            "detail": None if ok or pids else detail,
            **summary,
        }

    valid_latency = {
        label: row["p95_tpot_ms"]
        for label, row in valid_summaries.items()
        if row.get("p95_tpot_ms") is not None
    }
    decision: dict[str, Any] = {
        "ready_to_rescore": all(rows[label]["status"] == "valid" for label in ARTIFACTS),
        "winner_by_p95_tpot": (
            min(valid_latency, key=valid_latency.get) if valid_latency else None
        ),
        "p95_tpot_ms": valid_latency,
    }
    if {"dynabyte", "seq_fr", "seq_rf", "static_joint"}.issubset(valid_latency):
        strongest_baseline = min(
            ("static_joint", "seq_fr", "seq_rf"), key=valid_latency.get
        )
        decision["strongest_deployable_baseline"] = strongest_baseline
        decision["dynabyte_minus_strongest_baseline_ms"] = round(
            valid_latency["dynabyte"] - valid_latency[strongest_baseline], 3
        )
        decision["dynabyte_beats_strongest_baseline"] = (
            valid_latency["dynabyte"] < valid_latency[strongest_baseline]
        )

    return {"artifacts": rows, "decision": decision}


def write_csv(payload: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "policy",
        "artifact",
        "status",
        "p95_tpot_ms",
        "mean_tpot_ms",
        "p95_ttft_ms",
        "mean_ttft_ms",
        "accepted_gb",
        "copied_gb",
        "fid_gb",
        "res_gb",
        "look_gb",
        "scheduler_p99_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy, row in payload["artifacts"].items():
            writer.writerow(
                {
                    field: row.get(field)
                    for field in fieldnames
                    if field != "policy"
                }
                | {"policy": policy}
            )


def _latex_number(value: Any, *, digits: int = 1) -> str:
    if value is None or value == "":
        return "--"
    return f"{float(value):.{digits}f}"


def write_latex(payload: dict[str, Any], path: Path) -> None:
    labels = {
        "dynabyte": r"\systemname",
        "static_joint": "Static-Joint",
        "seq_fr": "Seq-FR",
        "seq_rf": "Seq-RF",
    }
    rows = payload["artifacts"]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{P1B switch-screen validation on Qwen3-30B.}",
        r"  \label{tab:eval-rq2-p1b-switch}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{@{}lrrrr@{}}",
        r"    \toprule",
        r"    Policy & p95 TPOT (ms) & p95 TTFT (ms) & Copy (GB) & Status \\",
        r"    \midrule",
    ]
    for policy in ("dynabyte", "static_joint", "seq_fr", "seq_rf"):
        row = rows[policy]
        lines.append(
            "    "
            + " & ".join(
                [
                    labels[policy],
                    _latex_number(row.get("p95_tpot_ms")),
                    _latex_number(row.get("p95_ttft_ms")),
                    _latex_number(row.get("copied_gb"), digits=3),
                    str(row.get("status", "--")).replace("_", r"\_"),
                ]
            )
            + r" \\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-latex", type=Path)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="fail unless all switch artifacts are valid and ready to rescore",
    )
    args = parser.parse_args(argv)

    payload = build_summary()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_csv is not None:
        write_csv(payload, args.output_csv)
    if args.output_latex is not None:
        write_latex(payload, args.output_latex)

    rows = payload["artifacts"]
    if any(row["status"] not in {"valid", "running"} for row in rows.values()):
        raise SystemExit(1)
    if args.require_ready and not payload["decision"]["ready_to_rescore"]:
        raise SystemExit("P1B switch artifacts are not all valid yet")


if __name__ == "__main__":
    main()
