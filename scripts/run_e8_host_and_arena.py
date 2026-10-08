#!/usr/bin/env python3
"""E8 probe: arena fragmentation on CPU, plus optional checkpoint pack timing.

Pack timing loads the uniform-low runtime and stops after initialization.
Pass --skip-pack to run only the arena stress and the lease invariant tests.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper/a6000_wikitext_replay/e8_host_arena.json"


def arena_stress() -> dict:
    from dynaexq.core.shared_arena import SharedArenaAllocator

    lo = 2_506_752
    hi = 9_437_184
    capacity = 48 * (30 * hi + 15 * lo)
    arena = SharedArenaAllocator(capacity, torch.device("cpu"))
    live = []
    failures = 0
    peak_fragmentation = 0.0
    for step in range(400):
        size = hi if step % 5 == 0 else lo
        extent = arena.allocate(size)
        if extent is None:
            failures += 1
        else:
            arena.mark_published(extent)
            live.append(extent)
        if step % 3 == 2 and live:
            victim = live.pop(0)
            arena.mark_reclaim_pending(victim)
            arena.free(victim)
        peak_fragmentation = max(
            peak_fragmentation, float(arena.snapshot()["external_fragmentation"])
        )
    snap = arena.snapshot()
    snap["allocate_failures"] = failures
    snap["peak_external_fragmentation"] = peak_fragmentation
    snap["capacity_bytes"] = capacity
    return snap


def pack_probe() -> dict:
    from dynaexq.core.config import DynaExqConfig
    from dynaexq.experiments.run_shift import initialize_dynaexq, load_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("pack probe requires CUDA")
    config = DynaExqConfig.from_yaml(ROOT / "dynaexq/configs/qwen30b_uniform_low_a6000.yaml")
    model_path = os.environ.get("E2_MODEL", "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507")
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    model, _tokenizer = load_model(model_path, device)
    loaded = time.perf_counter()
    initialize_dynaexq(config, model, device, high_precision_ratio=0.0)
    packed = time.perf_counter()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "model_path": model_path,
        "load_seconds": loaded - started,
        "initialize_seconds": packed - loaded,
        "rss_kb_before": rss_before,
        "rss_max_kb": rss_after,
        "note": "ru_maxrss is KB on Linux. initialize_seconds includes packing both tiers and the device arena.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pack", action="store_true")
    parser.add_argument(
        "--pack-only",
        action="store_true",
        help="Time the checkpoint pack and merge it into the existing arena artifact",
    )
    args = parser.parse_args()
    if args.pack_only:
        pack = pack_probe()
        sidecar = OUT.with_name("e8_pack_timing.json")
        sidecar.write_text(json.dumps({
            "schema_version": 1,
            "artifact_type": "e8_pack_timing",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pack": pack,
        }, indent=2), encoding="utf-8")
        existing = json.loads(OUT.read_text(encoding="utf-8"))
        existing["pack"] = pack
        existing.pop("pack_blocked", None)
        OUT.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        print("pack", pack)
        print(f"wrote {sidecar}", flush=True)
        return
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "dynaexq/tests/test_lease_invariants.py", "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "e8_host_arena",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lease_invariants": {
            "returncode": tests.returncode,
            "stdout": tests.stdout[-2000:],
            "stderr": tests.stderr[-2000:],
        },
        "arena_stress": arena_stress(),
        "pack": None if args.skip_pack else pack_probe(),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(tests.stdout)
    print("arena", payload["arena_stress"])
    print("pack", payload["pack"])
    print(f"wrote {OUT}", flush=True)
    if tests.returncode != 0:
        raise SystemExit(tests.returncode)


if __name__ == "__main__":
    main()
