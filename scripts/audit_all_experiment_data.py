#!/usr/bin/env python3
"""Audit all 83 non-figure empirical claims after experiment completion."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


BATCHES = (1, 2, 4, 8, 16, 32)
EXPECTED_PAPER_TREE = "050cbda677b664a540d984afd6663f31ea1c4770"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: Path) -> dict[str, object]:
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    commit = subprocess.check_output(
        ["git", "-C", str(probe), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(probe), "status", "--porcelain"], text=True
        ).strip()
    )
    repository_root = subprocess.check_output(
        ["git", "-C", str(probe), "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
    return {
        "root": str(root),
        "repository_root": repository_root,
        "commit": commit,
        "dirty": dirty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--mechanism-root", type=Path, required=True)
    parser.add_argument("--moe-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    formal_root = args.formal_root.resolve()
    sys.path.insert(0, str(formal_root))
    from scripts.audit_paper_results import validate_manifest_artifact

    claims: list[tuple[str, str, Path]] = []

    def add(group: str, claim: str, path: Path) -> None:
        claims.append((group, claim, path))

    native = args.native_root.resolve()
    mechanism = args.mechanism_root.resolve()
    moe = args.moe_root.resolve()

    q30_static = {
        1: native / "qwen30b_static_int4_bs1_isolated_rerun.json",
        2: native / "qwen30b_static_int4_bs2.json",
        4: native / "qwen30b_static_int4_bs4_isolated_tail_rerun.json",
        8: native / "qwen30b_static_int4_bs8_isolated_tail_rerun.json",
        16: native / "qwen30b_static_int4_bs16.json",
        32: native / "qwen30b_static_int4_bs32_isolated_rerun.json",
    }
    q80_static = {
        1: native / "qwen80b_static_int2_bs1_isolated_rerun.json",
        2: native / "qwen80b_static_int2_bs2_isolated_rerun.json",
        4: native / "qwen80b_static_int2_bs4_isolated_rerun.json",
        8: native / "qwen80b_static_int2_bs8.json",
        16: native / "qwen80b_static_int2_bs16.json",
        32: native / "qwen80b_static_int2_bs32.json",
    }
    for batch in BATCHES:
        add(
            "performance",
            f"performance:qwen30b:static_ptq:bs{batch}",
            q30_static[batch],
        )
        add(
            "performance",
            f"performance:qwen30b:dynaexq:bs{batch}",
            native / f"qwen30b_dynaexq_bs{batch}.json",
        )
        add(
            "performance",
            f"performance:qwen30b:moe_infinity:bs{batch}",
            moe / f"qwen30b_moe_infinity_bs{batch}.json",
        )
        add(
            "performance",
            f"performance:qwen80b:static_ptq:bs{batch}",
            q80_static[batch],
        )
        add(
            "performance",
            f"performance:qwen80b:dynaexq:bs{batch}",
            native / f"qwen80b_dynaexq_bs{batch}.json",
        )
        add(
            "performance",
            f"performance:phi35:static_ptq:bs{batch}",
            native / f"phi35_static_int4_bs{batch}.json",
        )
        add(
            "performance",
            f"performance:phi35:dynaexq:bs{batch}",
            native / f"phi35_dynaexq_bs{batch}.json",
        )

    for model in ("qwen30b", "qwen80b", "phi35"):
        add(
            "quality_significance",
            f"quality_significance:{model}:static_ptq_vs_dynaexq",
            mechanism / f"{model}_static_ptq_vs_dynaexq_significance.json",
        )
        for stage in ("decode", "prefill"):
            add(
                "activation_density",
                f"activation_density:{model}:{stage}",
                mechanism / f"{model}_activation_density.json",
            )
        add(
            "offload_waiting",
            f"offload_waiting:{model}",
            mechanism / f"{model}_offload_waiting.json",
        )

    for model in ("qwen30b", "qwen80b"):
        for mode in ("full", "static", "blocking", "no_hysteresis"):
            add(
                "ablation",
                f"ablation:{model}:{mode}",
                mechanism / f"{model}_ablation_{mode}.json",
            )
        add(
            "runtime_overhead",
            f"runtime_overhead:{model}",
            mechanism / f"{model}_runtime_overhead.json",
        )
        for ratio in (0, 5, 10, 15, 20, 25, 30):
            add(
                "budget_sensitivity",
                f"budget_sensitivity:{model}:ratio{ratio}",
                mechanism / f"{model}_budget_ratio{ratio}.json",
            )
        add(
            "perplexity_curve",
            f"perplexity_curve:{model}",
            mechanism / f"{model}_perplexity_curve.json",
        )

    for workload in ("wikitext", "gsm8k", "humaneval"):
        add(
            "routing_hotset",
            f"routing_hotset:qwen30b:{workload}:layer15",
            mechanism / "qwen30b_routing_hotset.json",
        )

    if len(claims) != 83 or len({claim for _, claim, _ in claims}) != 83:
        raise SystemExit("internal claim inventory is not exactly 83 unique items")

    records = []
    data_issues = []
    paper_alignment_issues = []
    cache: dict[Path, dict] = {}
    for group, claim, path in claims:
        problems: list[str]
        if not path.is_file():
            problems = [f"MISSING ARTIFACT: {path}"]
            digest = None
        else:
            digest = sha256(path)
            try:
                data = cache.setdefault(
                    path,
                    json.loads(path.read_text(encoding="utf-8")),
                )
                problems = validate_manifest_artifact(
                    group, str(path), data, claim
                )
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
                problems = [f"UNREADABLE ARTIFACT: {type(error).__name__}: {error}"]
        alignment = [
            problem
            for problem in problems
            if "MANUSCRIPT" in problem.upper()
        ]
        empirical = [problem for problem in problems if problem not in alignment]
        data_issues.extend({"claim": claim, "problem": p} for p in empirical)
        paper_alignment_issues.extend(
            {"claim": claim, "problem": p} for p in alignment
        )
        records.append(
            {
                "group": group,
                "claim": claim,
                "artifact": str(path),
                "sha256": digest,
                "data_problems": empirical,
                "paper_alignment_problems": alignment,
            }
        )

    paper_tree = subprocess.check_output(
        [
            "git",
            "-C",
            str(formal_root),
            "rev-parse",
            "HEAD:ICCAD_2026_DynExq",
        ],
        text=True,
    ).strip()
    paper_tree_unchanged = paper_tree == EXPECTED_PAPER_TREE
    if not paper_tree_unchanged:
        data_issues.append(
            {
                "claim": "paper_tree_immutable_during_experiments",
                "problem": (
                    f"PAPER TREE CHANGED: {paper_tree} != {EXPECTED_PAPER_TREE}"
                ),
            }
        )

    report = {
        "schema_version": 1,
        "artifact_type": "complete_nonfigure_experiment_data_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "required_claim_count": 83,
        "validated_claim_count": sum(not r["data_problems"] for r in records),
        "data_issue_count": len(data_issues),
        "paper_alignment_issue_count": len(paper_alignment_issues),
        "paper_tree": paper_tree,
        "expected_paper_tree": EXPECTED_PAPER_TREE,
        "paper_tree_unchanged": paper_tree_unchanged,
        "figure_bundle_excluded": True,
        "figure_exclusion_reason": (
            "user prohibited manuscript and figure modification during data collection"
        ),
        "worktrees": [git_state(formal_root), git_state(args.moe_root.resolve())],
        "claims": records,
        "data_issues": data_issues,
        "paper_alignment_issues": paper_alignment_issues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in (
        "required_claim_count",
        "validated_claim_count",
        "data_issue_count",
        "paper_alignment_issue_count",
    )}, indent=2))
    if data_issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
