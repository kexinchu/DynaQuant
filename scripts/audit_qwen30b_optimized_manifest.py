#!/usr/bin/env python3
"""Audit the optimized Qwen3-30B/A6000 manifest."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "evaluation" / "qwen30b"
DEFAULT_MANIFEST = RESULTS / "optimized_manifest_v2.json"
PRIORITY_ORDER = ("P0", "P1", "P2", "P3", "P4", "P5", "P6")
sys.path.insert(0, str(ROOT / "scripts"))

from validate_qwen30b_artifact import problems as artifact_problems


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def expand_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    if "artifact" in entry:
        return [entry]
    pattern = entry.get("expand")
    if not isinstance(pattern, str):
        selector = entry.get("selector")
        if isinstance(selector, str):
            return [entry]
        raise ValueError(f"manifest entry has neither artifact nor expand: {entry}")
    keys = [
        key
        for key in entry
        if key not in {"expand", "role", "promotion_rule", "selector"}
    ]
    expanded: list[dict[str, Any]] = []
    for values in itertools.product(*[_as_list(entry[key]) for key in keys]):
        bindings = dict(zip(keys, values, strict=True))
        artifact = pattern.format(**bindings)
        materialized = {
            "artifact": artifact,
            "role": entry.get("role"),
            "expanded_from": pattern,
            "bindings": bindings,
        }
        expanded.append(materialized)
    return expanded


def iter_entries(
    manifest: dict[str, Any],
    bucket: str,
    allowed_priorities: set[str],
) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    for priority in manifest.get("priorities", []):
        priority_name = str(priority.get("priority", "unknown"))
        if priority_name not in allowed_priorities:
            continue
        for entry in priority.get(bucket, []):
            for materialized in expand_entry(entry):
                found.append((priority_name, materialized))
    return found


def allowed_priorities(max_priority: str) -> set[str]:
    if max_priority not in PRIORITY_ORDER:
        raise ValueError(
            f"unknown priority {max_priority!r}; expected one of {PRIORITY_ORDER}"
        )
    limit = PRIORITY_ORDER.index(max_priority)
    return set(PRIORITY_ORDER[: limit + 1])


def artifact_path(name: str) -> Path:
    path = Path(name)
    if path.is_absolute():
        return path
    return RESULTS / name


def validate_existing(path: Path, source_hashes: set[str] | None) -> list[str]:
    return artifact_problems(path, current_source_hash=source_hashes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--max-priority",
        choices=PRIORITY_ORDER,
        default="P6",
        help="audit only priorities up to and including this level",
    )
    parser.add_argument(
        "--include-derived",
        action="store_true",
        help="reserved for the final paper-data audit; raw audit ignores it",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected_priorities = allowed_priorities(args.max_priority)
    source_policy = manifest.get("source_hash_policy", {})
    source_hashes = set()
    source_hash = source_policy.get("current_runtime_source_sha256")
    if isinstance(source_hash, str) and len(source_hash) == 64:
        source_hashes.add(source_hash)
    for source_hash in source_policy.get("accepted_runtime_source_sha256", []):
        if isinstance(source_hash, str) and len(source_hash) == 64:
            source_hashes.add(source_hash)
    source_hash_set = source_hashes or None

    report: dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(args.manifest.relative_to(ROOT)),
        "manifest_status": manifest.get("status"),
        "max_priority": args.max_priority,
        "priorities": {},
        "missing_required": [],
        "invalid_existing": {},
        "present_required": [],
        "present_reused": [],
        "selectors": [],
    }

    for bucket in ("required", "reused_when_valid"):
        for priority_name, entry in iter_entries(
            manifest,
            bucket,
            selected_priorities,
        ):
            priority_report = report["priorities"].setdefault(
                priority_name,
                {
                    "required": {"present": 0, "missing": 0, "invalid": 0},
                    "reused_when_valid": {
                        "present": 0,
                        "missing": 0,
                        "invalid": 0,
                    },
                    "conditional_selectors": 0,
                },
            )
            artifact = entry.get("artifact")
            if not isinstance(artifact, str):
                continue
            path = artifact_path(artifact)
            if not path.is_file():
                priority_report[bucket]["missing"] += 1
                if bucket == "required":
                    report["missing_required"].append(
                        {
                            "priority": priority_name,
                            "artifact": artifact,
                            "role": entry.get("role"),
                        }
                    )
                continue
            problems = validate_existing(path, source_hash_set)
            if problems:
                priority_report[bucket]["invalid"] += 1
                report["invalid_existing"][artifact] = problems
                continue
            priority_report[bucket]["present"] += 1
            key = "present_required" if bucket == "required" else "present_reused"
            report[key].append(
                {
                    "priority": priority_name,
                    "artifact": artifact,
                    "role": entry.get("role"),
                }
            )

    for priority in manifest.get("priorities", []):
        priority_name = str(priority.get("priority", "unknown"))
        if priority_name not in selected_priorities:
            continue
        priority_report = report["priorities"].setdefault(
            priority_name,
            {
                "required": {"present": 0, "missing": 0, "invalid": 0},
                "reused_when_valid": {"present": 0, "missing": 0, "invalid": 0},
                "conditional_selectors": 0,
            },
        )
        selectors = priority.get("conditional", [])
        priority_report["conditional_selectors"] = len(selectors)
        for selector in selectors:
            report["selectors"].append(
                {
                    "priority": priority_name,
                    "selector": selector.get("selector"),
                    "promotion_rule": selector.get("promotion_rule"),
                }
            )

    report["complete_for_current_required_set"] = (
        not report["missing_required"] and not report["invalid_existing"]
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["complete_for_current_required_set"] else 1)


if __name__ == "__main__":
    main()
