#!/usr/bin/env python3
"""Summarize the prioritized Qwen3-30B P1A queue state."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "evaluation" / "qwen30b"

P1A_ARTIFACTS = [
    "rq2_grid_phi085_fid0_look0_seed42.json",
    "rq2_grid_phi085_fid40_look0_seed42.json",
    "rq2_grid_phi085_fid0_look20_seed42.json",
    "rq2_grid_phi085_fid20_look10_seed42.json",
    "rq2_grid_phi085_fid20_look0_seed42.json",
    "rq2_grid_phi085_fid0_look10_seed42.json",
    "rq2_switch_dynabyte_seed42.json",
    "rq2_switch_static_joint_seed42.json",
]


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def valid(path: Path) -> tuple[bool, str]:
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
        return True, "valid"
    return False, result.stdout.strip() or "invalid"


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
        try:
            name = str(Path(output).resolve().relative_to(RESULTS))
        except ValueError:
            name = Path(output).name
        outputs.setdefault(name, []).append(int(pid_text))
    return outputs


def main() -> None:
    active = active_outputs()
    rows = []
    for index, name in enumerate(P1A_ARTIFACTS, start=1):
        path = RESULTS / name
        is_valid, detail = valid(path)
        pids = active.get(name, [])
        if is_valid:
            status = "valid"
        elif pids:
            status = "running"
        elif path.with_suffix(path.suffix + ".failed").exists():
            status = "failed"
        elif path.with_suffix(path.suffix + ".invalid").exists():
            status = "invalid"
        else:
            status = "missing"
        rows.append(
            {
                "order": index,
                "artifact": name,
                "status": status,
                "pids": pids,
                "detail": detail if status not in {"valid", "missing"} else None,
            }
        )
    summary = {
        "valid": sum(row["status"] == "valid" for row in rows),
        "running": sum(row["status"] == "running" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "invalid": sum(row["status"] == "invalid" for row in rows),
        "missing": sum(row["status"] == "missing" for row in rows),
        "artifacts": rows,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
