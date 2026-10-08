#!/usr/bin/env python3
"""Run the minimized Qwen3-30B P2/RQ1 queue.

The runner is intentionally staged.  The default section, ``p2a1``, runs only
the four strongest-baseline screening cells that can change the next paper
claim.  Later sections are admitted by the trigger rules in
``SIGMETRICS_2027/QWEN30B_REQUIREMENTS_ALIGNED_PLAN.md``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
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

P2A1 = (
    (0.85, "mixed", "dynabyte", 42, 20, 2),
    (0.85, "mixed", "seq_rf", 42, 20, 2),
    (1.30, "mixed", "dynabyte", 42, 20, 2),
    (1.30, "mixed", "seq_rf", 42, 20, 2),
)
P2A2 = (
    (0.85, "mixed", "static_joint", 42, 20, 2),
    (1.30, "mixed", "static_joint", 42, 20, 2),
)
P2B_COMPONENTS = (
    (0.85, "mixed", "seq_fr", 42, 20, 2),
    (0.85, "mixed", "fixed_partition", 42, 20, 2),
    (1.30, "mixed", "seq_fr", 42, 20, 2),
    (1.30, "mixed", "fixed_partition", 42, 20, 2),
)
P2_DEFAULT_SHARES = (20, 10, 70)
P2_TIGHT_BUDGET_SHARES = (0, 20, 80)


def shares_for(phi: float, policy: str) -> tuple[int, int, int]:
    """Return fidelity/lookahead/residency shares for a P2 screen.

    At Phi=0.85 the 20/10/70 point is not memory-feasible under the paper
    budget mapping B=Phi*P_lo.  The tight-budget screen therefore uses the
    feasible 0/20/80 boundary point; it is not a headline quality point.
    """
    del policy
    if phi <= 0.85 + 1e-9:
        return P2_TIGHT_BUDGET_SHARES
    return P2_DEFAULT_SHARES


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


def artifact_name(phi: float, regime: str, policy: str, seed: int) -> str:
    phi_tag = f"phi{int(round(phi * 100)):03d}"
    return f"rq1_{phi_tag}_{regime}_{policy}_seed{seed}.json"


def base_command(output: Path, *, physical_gpu: int, seed: int) -> list[str]:
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


def trace_command(
    phi: float,
    regime: str,
    policy: str,
    seed: int,
    requests: int,
    warmups: int,
    *,
    physical_gpu: int,
) -> tuple[Path, list[str]]:
    output = OUTPUT_ROOT / artifact_name(phi, regime, policy, seed)
    command = base_command(output, physical_gpu=physical_gpu, seed=seed)
    fid_share, look_share, res_share = shares_for(phi, policy)
    command.extend(
        [
            "--phi-override",
            f"{phi:.2f}",
            "--allocation-policy",
            policy,
            "--fidelity-share-pct",
            str(fid_share),
            "--lookahead-share-pct",
            str(look_share),
            "--residency-share-pct",
            str(res_share),
            "trace-perf",
            "--trace",
            str(TRACE.relative_to(ROOT)),
            "--regime",
            regime,
            "--requests",
            str(requests),
            "--n-warmup",
            str(warmups),
            "--max-input-tokens",
            "2048",
            "--max-output-tokens",
            "256",
        ]
    )
    return output, command


def queue_for(section: str, *, physical_gpu: int) -> list[tuple[Path, list[str]]]:
    cells: list[tuple[float, str, str, int, int, int]] = []
    if section in {"p2a1", "p2a"}:
        cells.extend(P2A1)
    if section in {"p2a2", "p2a"}:
        cells.extend(P2A2)
    if section == "p2b":
        cells.extend(P2B_COMPONENTS)
    if section == "screen":
        cells.extend(P2A1)
        cells.extend(P2A2)
        cells.extend(P2B_COMPONENTS)
    return [
        trace_command(
            phi,
            regime,
            policy,
            seed,
            requests,
            warmups,
            physical_gpu=physical_gpu,
        )
        for phi, regime, policy, seed, requests, warmups in cells
    ]


def run_artifact(output: Path, command: list[str], *, physical_gpu: int) -> bool:
    if artifact_valid(output):
        print(f"[skip] validated {output}", flush=True)
        return True
    if active_eval_pids_for(output):
        return wait_for_existing(output)

    log_path = output.with_suffix(
        output.suffix + f".p2_gpu{physical_gpu}_{utc_stamp()}.log"
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
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument(
        "--section",
        choices=("p2a1", "p2a2", "p2a", "p2b", "screen"),
        default="p2a1",
    )
    parser.add_argument("--start-order", type=int, default=1)
    parser.add_argument("--stop-order", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    full_queue = queue_for(args.section, physical_gpu=args.physical_gpu)
    stop_order = len(full_queue) if args.stop_order is None else args.stop_order
    if args.start_order < 1 or stop_order > len(full_queue) or args.start_order > stop_order:
        raise SystemExit(
            f"expected 1 <= --start-order <= --stop-order <= {len(full_queue)}"
        )
    queue = full_queue[args.start_order - 1 : stop_order]

    for output, command in queue:
        if args.dry_run:
            print(str(output.relative_to(ROOT)))
            print(" ".join(shlex.quote(part) for part in command))
            continue
        if not run_artifact(output, command, physical_gpu=args.physical_gpu):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
