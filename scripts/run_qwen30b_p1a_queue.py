#!/usr/bin/env python3
"""Run the prioritized Qwen3-30B P1A queue on physical GPU 0.

The runner is intentionally conservative:

* it skips artifacts that already pass validation;
* it waits for an already-running evaluator for the same output instead of
  starting a duplicate process;
* it runs one artifact at a time and relies on eval_dynamic's physical-GPU
  idle gate before model placement;
* it stops on the first failed or invalid artifact so the next action can be
  a code/data fix rather than a long invalid queue.
"""

from __future__ import annotations

import datetime as _dt
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results" / "evaluation" / "qwen30b"
MODEL_PATH = Path(
    os.environ.get(
        "QWEN30B_MODEL_PATH",
        "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507",
    )
)
CALIBRATION = OUTPUT_ROOT / "calibration_wikitext103_256x2048.json"
CONFIG = ROOT / "dynaexq" / "configs" / "qwen30b_joint_a6000.yaml"
TRACE = ROOT / "ShareGPT_V3_unfiltered_cleaned_split.json"

P1A_GRID_CELLS = (
    (0, 0),
    (40, 0),
    (0, 20),
    (20, 10),
    (20, 0),
    (0, 10),
)
P1A_SWITCH_POLICIES = ("dynabyte", "static_joint")


def utc_stamp() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def artifact_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    result = run_checked(
        [
            sys.executable,
            "scripts/validate_qwen30b_artifact.py",
            str(path),
            "--allow-stale-source",
        ]
    )
    if result.returncode == 0:
        return True
    print(result.stdout, end="", flush=True)
    return False


def active_eval_pids_for(output: Path) -> list[int]:
    result = run_checked(["ps", "-eo", "pid=,args="])
    if result.returncode != 0:
        return []
    candidates = {str(output)}
    try:
        candidates.add(str(output.resolve().relative_to(ROOT)))
    except ValueError:
        pass
    needles = {f"--output {candidate}" for candidate in candidates}
    eq_needles = {f"--output={candidate}" for candidate in candidates}
    pids: list[int] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, args = stripped.partition(" ")
        if not pid_text.isdigit():
            continue
        if "dynaexq.experiments.eval_dynamic" not in args:
            continue
        if any(needle in args for needle in needles) or any(
            needle in args for needle in eq_needles
        ):
            pids.append(int(pid_text))
    return pids


def wait_for_existing(output: Path) -> bool:
    reported = False
    while True:
        pids = active_eval_pids_for(output)
        if not pids:
            return artifact_valid(output)
        if not reported:
            print(
                f"[resume] waiting for existing evaluator for {output}: "
                f"pid(s)={pids}",
                flush=True,
            )
            reported = True
        time.sleep(300)


def base_command(output: Path, *, physical_gpu: int, seed: int = 42) -> list[str]:
    output_arg = str(output.resolve().relative_to(ROOT))
    return [
        sys.executable,
        "-m",
        "dynaexq.experiments.eval_dynamic",
        "--model-path",
        str(MODEL_PATH),
        "--output",
        output_arg,
        "--device",
        "cuda:0",
        "--wait-for-idle-physical-gpu",
        str(physical_gpu),
        "--config",
        str(CONFIG.relative_to(ROOT)),
        "--policy",
        "joint",
        "--hash-model-files",
        "--seed",
        str(seed),
        "--initial-map",
        str(CALIBRATION.relative_to(ROOT)),
    ]


def grid_command(fid: int, look: int, *, physical_gpu: int) -> tuple[Path, list[str]]:
    output = OUTPUT_ROOT / f"rq2_grid_phi085_fid{fid}_look{look}_seed42.json"
    command = base_command(output, physical_gpu=physical_gpu)
    command.extend(
        [
            "--phi-override",
            "0.85",
            "--allocation-policy",
            "dynabyte",
            "--fidelity-share-pct",
            str(fid),
            "--lookahead-share-pct",
            str(look),
            "--residency-share-pct",
            str(100 - fid - look),
            "trace-perf",
            "--trace",
            str(TRACE.relative_to(ROOT)),
            "--regime",
            "mixed",
            "--requests",
            "20",
            "--n-warmup",
            "2",
            "--max-input-tokens",
            "2048",
            "--max-output-tokens",
            "256",
        ]
    )
    return output, command


def switch_command(policy: str, *, physical_gpu: int) -> tuple[Path, list[str]]:
    output = OUTPUT_ROOT / f"rq2_switch_{policy}_seed42.json"
    command = base_command(output, physical_gpu=physical_gpu)
    command.extend(
        [
            "--allocation-policy",
            policy,
            "--workload-switch-trace",
            "rq2_prefill_decode_mixed_v1",
            "trace-perf",
            "--trace",
            str(TRACE.relative_to(ROOT)),
            "--regime",
            "mixed",
            "--requests",
            "20",
            "--n-warmup",
            "2",
            "--max-input-tokens",
            "2048",
            "--max-output-tokens",
            "256",
        ]
    )
    return output, command


def run_artifact(output: Path, command: list[str], *, physical_gpu: int) -> bool:
    if artifact_valid(output):
        print(f"[skip] validated {output}", flush=True)
        return True
    if active_eval_pids_for(output):
        return wait_for_existing(output)

    log_path = output.with_suffix(
        output.suffix + f".p1a_gpu{physical_gpu}_{utc_stamp()}.log"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["DYNAEXQ_ALLOW_STALE_INITIAL_MAP_SOURCE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[run] {output}", flush=True)
    print(f"[log] {log_path}", flush=True)
    print("[cmd] " + " ".join(shlex.quote(part) for part in command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        status = process.wait()
    if status != 0:
        print(f"[failed] {output} exit_status={status}", flush=True)
        return False
    if not artifact_valid(output):
        print(f"[invalid] {output}", flush=True)
        return False
    print(f"[done] {output}", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physical-gpu",
        type=int,
        default=0,
        help="Physical GPU to use. CUDA_VISIBLE_DEVICES maps it to cuda:0.",
    )
    parser.add_argument(
        "--start-order",
        type=int,
        default=1,
        help="1-based P1A order to start from.",
    )
    parser.add_argument(
        "--stop-order",
        type=int,
        default=8,
        help="1-based P1A order to stop at, inclusive.",
    )
    args = parser.parse_args()
    if args.start_order < 1 or args.stop_order > 8 or args.start_order > args.stop_order:
        raise SystemExit("expected 1 <= --start-order <= --stop-order <= 8")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    full_queue: list[tuple[Path, list[str]]] = []
    full_queue.extend(
        grid_command(fid, look, physical_gpu=args.physical_gpu)
        for fid, look in P1A_GRID_CELLS
    )
    full_queue.extend(
        switch_command(policy, physical_gpu=args.physical_gpu)
        for policy in P1A_SWITCH_POLICIES
    )
    queue = full_queue[args.start_order - 1 : args.stop_order]

    for output, command in queue:
        if not run_artifact(output, command, physical_gpu=args.physical_gpu):
            return 1

    audit = subprocess.run(
        [
            sys.executable,
            "scripts/audit_qwen30b_optimized_manifest.py",
            "--max-priority",
            "P1",
        ],
        cwd=ROOT,
        check=False,
        text=True,
    )
    return audit.returncode


if __name__ == "__main__":
    raise SystemExit(main())
