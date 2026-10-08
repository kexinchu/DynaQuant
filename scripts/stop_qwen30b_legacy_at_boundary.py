#!/usr/bin/env python3
"""Stop Qwen3-30B legacy runners only after the optimized boundary is reached."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "evaluation" / "qwen30b"
PARENT_MARKERS = (
    "scripts/run_qwen30b_gpu0_evaluation.sh",
    "scripts/run_qwen30b_gpu1_parallel.sh",
)
EVAL_MARKER = "python -m dynaexq.experiments.eval_dynamic"
BOUNDARY_LOCK_HOLDER_OUTPUTS = (
    "uniform_low_quality_seed42.json.lock",
    "joint_perf_seed45_bs32.json.lock",
)


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    stat: str
    command: str


def process_table() -> list[Process]:
    output = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,stat=,cmd="],
        text=True,
    )
    processes: list[Process] = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) != 4:
            continue
        pid, ppid, stat, command = fields
        try:
            processes.append(Process(int(pid), int(ppid), stat, command))
        except ValueError:
            continue
    return processes


def descendants(processes: list[Process], roots: set[int]) -> set[int]:
    children: dict[int, list[int]] = {}
    for process in processes:
        children.setdefault(process.ppid, []).append(process.pid)
    found: set[int] = set()
    stack = list(roots)
    while stack:
        current = stack.pop()
        for child in children.get(current, []):
            if child in found:
                continue
            found.add(child)
            stack.append(child)
    return found


def boundary_lock_holders(processes: list[Process]) -> list[Process]:
    holders = []
    for process in processes:
        if "flock" not in process.command:
            continue
        if any(name in process.command for name in BOUNDARY_LOCK_HOLDER_OUTPUTS):
            holders.append(process)
    return holders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send SIGTERM to eligible parent runner shells.",
    )
    args = parser.parse_args()

    processes = process_table()
    parents = [
        process
        for process in processes
        if any(marker in process.command for marker in PARENT_MARKERS)
    ]
    parent_pids = {process.pid for process in parents}
    child_pids = descendants(processes, parent_pids)
    eval_children = [
        process
        for process in processes
        if process.pid in child_pids and EVAL_MARKER in process.command
    ]
    lock_holders = boundary_lock_holders(processes)

    required_artifacts = [
        RESULTS / "all_high_reference_quality_seed42.json",
        RESULTS / "joint_perf_seed46_bs2.json",
        RESULTS / "joint_perf_seed46_bs1.json",
    ]
    missing_artifacts = [
        path.relative_to(ROOT).as_posix()
        for path in required_artifacts
        if not path.is_file()
    ]
    safe = (
        len(parents) >= 2
        and not eval_children
        and not missing_artifacts
        and len(lock_holders) >= 2
    )

    report = {
        "safe_to_stop": safe,
        "execute": args.execute,
        "parents": [
            {"pid": process.pid, "ppid": process.ppid, "command": process.command}
            for process in parents
        ],
        "eval_children": [
            {"pid": process.pid, "ppid": process.ppid, "command": process.command}
            for process in eval_children
        ],
        "boundary_lock_holders": [
            {"pid": process.pid, "ppid": process.ppid, "command": process.command}
            for process in lock_holders
        ],
        "missing_artifacts": missing_artifacts,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if not safe:
        raise SystemExit(1)
    if not args.execute:
        return
    for process in parents:
        os.kill(process.pid, signal.SIGTERM)


if __name__ == "__main__":
    main()
