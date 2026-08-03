#!/usr/bin/env python3
"""Fail closed when local DynaExQ assets are not yet safe to delete."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import subprocess
import sys
from pathlib import Path

from model_registry import load_registry, verify_remote


ROOT = Path(__file__).resolve().parents[1]
SHAREGPT_NAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT_SHA256 = (
    "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
)

PRIVATE_NOTES = {
    "Paper_Write.md",
    "agent_experience.md",
    "plan.md",
    "ICCAD_2026_DynExq/TC_COVER_LETTER_DRAFT.md",
    "ICCAD_2026_DynExq/TC_READY_VERSION.md",
    "ICCAD_2026_DynExq/agent_experience.md",
    "ICCAD_2026_DynExq/expected_experiment_data.md",
    "ICCAD_2026_DynExq/hpdc_review.md",
    "ICCAD_2026_DynExq/iccad_plan.md",
}

REGENERABLE_PATTERNS = (
    ".pytest_cache/**",
    "DynExq_paper/**",
    "build/**",
    "dynaexq.egg-info/**",
    "**/__pycache__/**",
    "*.log",
    "logs/**",
    "results/logs/**",
    "scripts/results/**",
    "scripts/*.pdf",
    "ICCAD_2026_DynExq/*.aux",
    "ICCAD_2026_DynExq/*.blg",
    "ICCAD_2026_DynExq/*.fdb_latexmk",
    "ICCAD_2026_DynExq/*.fls",
    "ICCAD_2026_DynExq/*.log",
    "ICCAD_2026_DynExq/*.out",
    "ICCAD_2026_DynExq/*.txt",
    "ICCAD_2026_DynExq/figures/*.aux",
    "ICCAD_2026_DynExq/figures/*.log",
    "ICCAD_2026_DynExq/ACM-Reference-Format.bst",
    "ICCAD_2026_DynExq/acmart.cls",
)


def run(*command: str) -> str:
    return subprocess.check_output(
        command,
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def git_preflight() -> list[str]:
    failures: list[str] = []
    status = run("git", "status", "--porcelain", "--untracked-files=all")
    if status:
        failures.append("working tree has non-ignored changes")
    head = run("git", "rev-parse", "HEAD")
    tracking = run("git", "rev-parse", "origin/master")
    remote_line = run("git", "ls-remote", "--heads", "origin", "master")
    remote = remote_line.split()[0] if remote_line else ""
    if len({head, tracking, remote}) != 1:
        failures.append(
            f"master mismatch: HEAD={head}, origin/master={tracking}, remote={remote}"
        )
    else:
        print(f"git\tok\t{head}")
    return failures


def ignored_preflight(allow_private_note_loss: bool) -> list[str]:
    failures: list[str] = []
    output = subprocess.check_output(
        ["git", "ls-files", "-o", "-i", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    paths = [item.decode() for item in output.split(b"\0") if item]
    categories = {"external": [], "private": [], "regenerable": [], "unknown": []}
    for relative in paths:
        if relative == SHAREGPT_NAME:
            categories["external"].append(relative)
        elif relative in PRIVATE_NOTES:
            categories["private"].append(relative)
        elif any(fnmatch.fnmatch(relative, pattern) for pattern in REGENERABLE_PATTERNS):
            categories["regenerable"].append(relative)
        else:
            categories["unknown"].append(relative)
    for category, members in categories.items():
        total = sum((ROOT / member).stat().st_size for member in members)
        print(f"ignored-{category}\t{len(members)} files\t{total} bytes")
    if categories["unknown"]:
        failures.append(
            "unclassified ignored paths: " + ", ".join(categories["unknown"])
        )
    if categories["private"] and not allow_private_note_loss:
        failures.append(
            "private notes would be lost; review them or rerun with "
            "--allow-private-note-loss: " + ", ".join(categories["private"])
        )
    return failures


def sharegpt_preflight() -> list[str]:
    path = ROOT / SHAREGPT_NAME
    if not path.exists():
        print("sharegpt\trecoverable from pinned upstream source")
        return []
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHAREGPT_SHA256:
        return [f"local {SHAREGPT_NAME} has an unexpected SHA-256: {digest}"]
    print(f"sharegpt\tok\t{digest}")
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-private-note-loss",
        action="store_true",
        help="Acknowledge that reviewed local-only notes will not be archived.",
    )
    parser.add_argument(
        "--skip-remote-model-check",
        action="store_true",
        help="Skip Hugging Face checks when intentionally working offline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = git_preflight()
    failures.extend(ignored_preflight(args.allow_private_note_loss))
    failures.extend(sharegpt_preflight())
    if not args.skip_remote_model_check:
        try:
            registry = load_registry(ROOT / "release" / "model_registry.json")
            verify_remote(registry, list(registry["artifacts"]))
        except Exception as error:  # fail closed before destructive cleanup
            failures.append(f"remote model verification failed: {error}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("SAFE: reviewed DynaExQ assets are recoverable from remote storage.")


if __name__ == "__main__":
    main()
