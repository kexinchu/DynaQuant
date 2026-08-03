#!/usr/bin/env python3
"""Audit manuscript accuracy values against machine-readable result files.

This is deliberately strict.  A journal table should not be populated from
expected, interpolated, or failed runs.  The script exits non-zero when a
checked manuscript cell differs from its source, when a source reports skipped
samples, or when a manuscript row has no registered source.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "ICCAD_2026_DynExq" / "05_evaluation.tex"
IMPLEMENTATION = ROOT / "ICCAD_2026_DynExq" / "04_implement.tex"
QWEN3_NEXT_CONFIG = ROOT / "dynaexq" / "configs" / "qwen3_next_80b.yaml"
PHI35_CONFIG = ROOT / "dynaexq" / "configs" / "phi35_moe.yaml"
RUN_SHIFT = ROOT / "dynaexq" / "experiments" / "run_shift.py"
QWEN3_NEXT_ADAPTER = (
    ROOT / "dynaexq" / "integration" / "qwen3_next_adapter.py"
)
MANIFEST = ROOT / "results" / "paper" / "manifest.json"
REQUIRED_MANIFEST_GROUPS = {
    "quality_significance",
    "performance",
    "ablation",
    "runtime_overhead",
    "budget_sensitivity",
    "activation_density",
    "offload_waiting",
    "routing_hotset",
    "perplexity_curve",
    "figure_bundle",
}
PAPER_MODELS = ("qwen30b", "qwen80b", "phi35")
PERFORMANCE_METHODS = {
    "qwen30b": ("static_ptq", "moe_infinity", "dynaexq"),
    "qwen80b": ("static_ptq", "dynaexq"),
    "phi35": ("static_ptq", "dynaexq"),
}
PERFORMANCE_BATCHES = (1, 2, 4, 8, 16, 32)
MOE_INFINITY_IDENTITY = {
    "repository": "https://github.com/EfficientMoE/MoE-Infinity",
    "commit": "ba5651897a80d9c9b7a1500cef2c68adaa63db0f",
    "tree": "6c463a9ab298f352b0c1e855961b82ce2c545a64",
    "source_hash": (
        "c9f83ea65a2ed83c3454af861560d666a5ada14134e4e8bcd6d389e8231db30b"
    ),
}
ABLATION_CONFIGS = ("full", "static", "blocking", "no_hysteresis")
BUDGET_RATIOS = (0, 5, 10, 15, 20, 25, 30)
OFFLOAD_INPUT_LENGTHS = (
    16,
    32,
    64,
    96,
    128,
    192,
    256,
    320,
    384,
    448,
    512,
    640,
    768,
    896,
    1024,
    1536,
    2048,
)
OFFLOAD_MODEL_CONTRACTS = {
    "qwen30b": (48, 128, 8),
    "qwen80b": (48, 512, 10),
    "phi35": (32, 16, 2),
}
REQUIRED_CLAIMS = {
    "quality_significance": {
        f"quality_significance:{model}:static_ptq_vs_dynaexq"
        for model in PAPER_MODELS
    },
    "performance": {
        f"performance:{model}:{method}:bs{batch}"
        for model in PAPER_MODELS
        for method in PERFORMANCE_METHODS[model]
        for batch in PERFORMANCE_BATCHES
    },
    "ablation": {
        f"ablation:{model}:{config}"
        for model in ("qwen30b", "qwen80b")
        for config in ABLATION_CONFIGS
    },
    "runtime_overhead": {
        f"runtime_overhead:{model}"
        for model in ("qwen30b", "qwen80b")
    },
    "budget_sensitivity": {
        f"budget_sensitivity:{model}:ratio{ratio}"
        for model in ("qwen30b", "qwen80b")
        for ratio in BUDGET_RATIOS
    },
    "activation_density": {
        f"activation_density:{model}:{stage}"
        for model in ("qwen30b", "qwen80b", "phi35")
        for stage in ("decode", "prefill")
    },
    "offload_waiting": {
        f"offload_waiting:{model}"
        for model in ("qwen30b", "qwen80b", "phi35")
    },
    "routing_hotset": {
        f"routing_hotset:qwen30b:{workload}:layer15"
        for workload in ("wikitext", "gsm8k", "humaneval")
    },
    "perplexity_curve": {
        f"perplexity_curve:{model}"
        for model in ("qwen30b", "qwen80b")
    },
    "figure_bundle": {"figure_bundle:main"},
}
FIGURE_INPUT_GROUPS = {
    "performance",
    "budget_sensitivity",
    "offload_waiting",
    "routing_hotset",
    "perplexity_curve",
}
EXPECTED_EMPIRICAL_FIGURES = {
    *{
        f"ICCAD_2026_DynExq/figures/{prefix}_{suffix}.pdf"
        for prefix in ("Qwen3-30B", "Qwen3-80B", "Phi-3.5-MoE")
        for suffix in (
            "avg_latency_end2end_vs_batch_size",
            "p99_latency_end2end_vs_batch_size",
            "p99_latency_throughput",
        )
    },
    "ICCAD_2026_DynExq/figures/budget_sensitivity_qwen30b.pdf",
    "ICCAD_2026_DynExq/figures/budget_sensitivity_qwen80b.pdf",
    "ICCAD_2026_DynExq/figures/waiting_latency_vs_prompt_length.pdf",
    "ICCAD_2026_DynExq/figures/wikitext_thinking_on_layer_15.pdf",
    "ICCAD_2026_DynExq/figures/gsm8k_thinking_off_layer_15.pdf",
    "ICCAD_2026_DynExq/figures/humaneval_thinking_on_layer_15.pdf",
    "ICCAD_2026_DynExq/figures/wiki_ppl_qwen30b.pdf",
    "ICCAD_2026_DynExq/figures/wiki_ppl_qwen80b.pdf",
}


@dataclass(frozen=True)
class Source:
    path: Path
    result_key: str | None = None


SOURCES = {
    ("Qwen3-MoE-30B", "FP16"): Source(
        ROOT / "results" / "paper" / "qwen30b_fp16_quality.json"
    ),
    ("Qwen3-MoE-30B", "INT4"): Source(
        ROOT / "results" / "paper" / "qwen30b_int4_quality.json"
    ),
    ("Phi-3.5-MoE", "FP16"): Source(
        ROOT / "results" / "paper" / "phi35_fp16_quality.json"
    ),
    ("Phi-3.5-MoE", "INT4"): Source(
        ROOT / "results" / "paper" / "phi35_int4_quality.json"
    ),
    ("Qwen3-MoE-30B", "DynaExQ"): Source(
        ROOT / "results" / "paper" / "qwen30b_dynaexq_quality.json"
    ),
    ("Qwen3-Next-80B", "INT4"): Source(
        ROOT / "results" / "paper" / "qwen80b_int4_quality.json"
    ),
    ("Qwen3-Next-80B", "INT2"): Source(
        ROOT / "results" / "paper" / "qwen80b_int2_quality.json"
    ),
    ("Qwen3-Next-80B", "DynaExQ"): Source(
        ROOT / "results" / "paper" / "qwen80b_dynaexq_quality.json"
    ),
    ("Phi-3.5-MoE", "DynaExQ"): Source(
        ROOT / "results" / "paper" / "phi35_dynaexq_quality.json"
    ),
}
QUALITY_SIGNIFICANCE_SOURCES = {
    "qwen30b": (
        Path("results/paper/qwen30b_int4_quality.json"),
        Path("results/paper/qwen30b_dynaexq_quality.json"),
        "static_int4",
    ),
    "qwen80b": (
        Path("results/paper/qwen80b_int2_quality.json"),
        Path("results/paper/qwen80b_dynaexq_quality.json"),
        "static_int2",
    ),
    "phi35": (
        Path("results/paper/phi35_int4_quality.json"),
        Path("results/paper/phi35_dynaexq_quality.json"),
        "static_int4",
    ),
}

BENCHMARK_KEYS = ("mmlu_pro", "gpqa", "aime", "gsm8k", "humaneval")
SIGNIFICANCE_BENCHMARKS = (
    "mmlu_pro",
    "gpqa",
    "aime25",
    "gsm8k",
    "humaneval",
)
BENCHMARK_ALIASES = {"aime": ("aime", "aime25")}
EXPECTED_BENCHMARK_TOTALS = {
    "mmlu_pro": None,
    "gpqa": 198,
    "aime": 30,
    "gsm8k": None,
    "humaneval": 164,
}
ROW_RE = re.compile(
    r"^(?P<model>Qwen3-MoE-30B|Qwen3-Next-80B|Phi-3\.5-MoE)?\s*&\s*"
    r"(?P<method>FP16|INT4|INT2|\\systemname)\s*&\s*"
    r"(?P<values>.*?)\\\\\s*$",
    re.MULTILINE,
)


def manuscript_ablation_rows() -> dict[tuple[str, str], tuple[float, ...]]:
    """Parse the six displayed values for every Table-IV configuration."""
    text = PAPER.read_text(encoding="utf-8")
    try:
        table = text.split(r"\label{tab:ablation}", 1)[1].split(
            r"\end{table}",
            1,
        )[0]
    except IndexError:
        return {}
    labels = {
        "Full": "full",
        "Static precision": "static",
        "Blocking migration": "blocking",
        "w/o Hysteresis": "no_hysteresis",
    }
    rows: dict[tuple[str, str], tuple[float, ...]] = {}
    for display_label, config in labels.items():
        match = re.search(
            rf"^{re.escape(display_label)}(?:\s+\\systemname)?\s*&"
            rf"(?P<values>.*?)\\\\",
            table,
            flags=re.MULTILINE,
        )
        if match is None:
            continue
        values = tuple(
            float(value)
            for value in re.findall(
                r"(?<![A-Za-z])\d+(?:\.\d+)?",
                match.group("values"),
            )
        )
        if len(values) == 6:
            rows[("qwen30b", config)] = values[:3]
            rows[("qwen80b", config)] = values[3:]
    return rows


def manuscript_overhead_rows() -> dict[str, tuple[float, float]]:
    """Parse the two displayed values for every runtime-overhead metric."""
    text = PAPER.read_text(encoding="utf-8")
    try:
        table = text.split(r"\label{tab:overhead}", 1)[1].split(
            r"\end{table}",
            1,
        )[0]
    except IndexError:
        return {}
    labels = {
        "HBM Budget (GB)": "hbm_budget_gb",
        "Peak Process HBM Used (GB)": "peak_process_hbm_used_gb",
        "Resident Expert Pools (GB)": "resident_expert_pool_gb",
        "Transient Pool (GB)": "transient_expert_pool_gb",
        "Migration Count": "migration_count",
        "Transferred (GB)": "transferred_gb",
        "Scheduler Mean (ms)": "scheduler_mean_ms",
        "Scheduler P99 (ms)": "scheduler_p99_ms",
        "Pinned Expert Cache (GB)": "pinned_expert_cache_gb",
    }
    rows = {}
    for display_label, key in labels.items():
        match = re.search(
            rf"^{re.escape(display_label)}\s*&(?P<values>.*?)\\\\",
            table,
            flags=re.MULTILINE,
        )
        if match is None:
            continue
        values = tuple(
            float(value)
            for value in re.findall(
                r"(?<![A-Za-z])\d+(?:\.\d+)?",
                match.group("values"),
            )
        )
        if len(values) >= 2:
            # The manuscript may include an additional Phi-3.5-MoE column;
            # the registered overhead audit currently covers the two Qwen
            # claims and intentionally selects those first two values.
            rows[key] = values[:2]
    return rows


def manuscript_activation_density() -> dict[tuple[str, str], tuple[float, ...]]:
    """Parse all 36 activation-density cells in the motivation table."""
    text = PAPER.parent.joinpath("02_background.tex").read_text(
        encoding="utf-8"
    )
    try:
        table = text.split(r"\label{tab:expert_activation_ratio}", 1)[1].split(
            r"\end{table}",
            1,
        )[0]
    except IndexError:
        return {}
    model_aliases = {
        "Qwen3-30B": "qwen30b",
        "Qwen3-Next-80B": "qwen80b",
        "Phi-3.5-MoE": "phi35",
    }
    rows = {}
    current_model = None
    for line in table.splitlines():
        if "&" not in line or r"\\" not in line:
            continue
        for display, model in model_aliases.items():
            if display in line:
                current_model = model
                break
        stage_match = re.search(r"&\s*(Decode|Prefill)\s*&", line)
        if current_model is None or stage_match is None:
            continue
        values = tuple(
            float(value)
            for value in re.findall(
                r"(?<![A-Za-z])\d+(?:\.\d+)?",
                line.split(stage_match.group(0), 1)[1],
            )
        )
        if len(values) == 6:
            rows[(current_model, stage_match.group(1).lower())] = values
    return rows


def manuscript_rows() -> dict[tuple[str, str], list[float]]:
    text = PAPER.read_text(encoding="utf-8")
    rows: dict[tuple[str, str], list[float]] = {}
    current_model: str | None = None
    for match in ROW_RE.finditer(text):
        current_model = match.group("model") or current_model
        if current_model is None:
            continue
        method = match.group("method").replace("\\systemname", "DynaExQ")
        numeric = re.findall(r"-?\d+(?:\.\d+)?", match.group("values"))
        rows[(current_model, method)] = [float(value) for value in numeric]
    return rows


def source_benchmarks(source: Source, data: dict) -> dict:
    if source.result_key is not None:
        return data["results"][source.result_key]["benchmarks"]
    return data["benchmarks"]


def validate_artifact(key: tuple[str, str], data: dict) -> list[str]:
    """Reject legacy or non-reproducible JSON even when its values match."""
    label = f"{key[0]} / {key[1]}"
    problems = []
    if int(data.get("schema_version", 0)) < 2:
        problems.append(f"LEGACY SCHEMA: {label}")
        return problems
    identity = {
        ("Qwen3-MoE-30B", "FP16"): (
            "qwen30b",
            "reference_fp16",
            "quality_evaluation",
        ),
        ("Qwen3-MoE-30B", "INT4"): (
            "qwen30b",
            "static_int4",
            "quality_evaluation",
        ),
        ("Qwen3-MoE-30B", "DynaExQ"): (
            "qwen30b",
            "dynaexq",
            "dynaexq_quality",
        ),
        ("Qwen3-Next-80B", "INT4"): (
            "qwen80b",
            "static_int4",
            "quality_evaluation",
        ),
        ("Qwen3-Next-80B", "INT2"): (
            "qwen80b",
            "static_int2",
            "quality_evaluation",
        ),
        ("Qwen3-Next-80B", "DynaExQ"): (
            "qwen80b",
            "dynaexq",
            "dynaexq_quality",
        ),
        ("Phi-3.5-MoE", "FP16"): (
            "phi35",
            "reference_fp16",
            "quality_evaluation",
        ),
        ("Phi-3.5-MoE", "INT4"): (
            "phi35",
            "static_int4",
            "quality_evaluation",
        ),
        ("Phi-3.5-MoE", "DynaExQ"): (
            "phi35",
            "dynaexq",
            "dynaexq_quality",
        ),
    }.get(key)
    if identity is None:
        problems.append(f"UNKNOWN QUALITY IDENTITY: {label}")
    else:
        paper_model, paper_method, artifact_type = identity
        if (
            data.get("paper_model") != paper_model
            or data.get("paper_method") != paper_method
            or data.get("artifact_type") != artifact_type
        ):
            problems.append(f"QUALITY IDENTITY MISMATCH: {label}")
    for field in ("created_at", "checkpoint", "environment", "benchmarks"):
        if field not in data:
            problems.append(f"MISSING PROVENANCE FIELD: {label} / {field}")
    environment = data.get("environment")
    git = environment.get("git", {}) if isinstance(environment, dict) else {}
    if (
        not git.get("commit")
        or git.get("dirty") is not False
        or int(
            environment.get("process_max_rss_bytes", 0)
            if isinstance(environment, dict)
            else 0
        )
        <= 0
    ):
        problems.append(f"DIRTY OR INCOMPLETE QUALITY ENVIRONMENT: {label}")
    checkpoint = data.get("checkpoint", {})
    if checkpoint.get("local") is True:
        if not checkpoint.get("weight_hashes_included"):
            problems.append(f"UNHASHED LOCAL CHECKPOINT: {label}")
    elif not checkpoint.get("revision"):
        problems.append(f"UNPINNED REMOTE CHECKPOINT: {label}")
    protocol = data.get("evaluation_protocol")
    if not isinstance(protocol, dict) or protocol.get("name") != "tc_main_v2":
        problems.append(f"INVALID EVALUATION PROTOCOL: {label}")
    if key[1] == "DynaExQ":
        problems.extend(validate_dynamic_runtime(label, data))
    return problems


def validate_dynamic_runtime(label: str, data: dict) -> list[str]:
    """Prove that a DynaExQ artifact observed and consumed runtime state."""
    problems: list[str] = []
    if not isinstance(data.get("runtime_initialization"), dict):
        problems.append(f"MISSING DYNAMIC INITIALIZATION: {label}")
    wrapper = data.get("wrapper_stats")
    if not isinstance(wrapper, dict):
        problems.append(f"MISSING WRAPPER STATS: {label}")
    else:
        steps = int(wrapper.get("forward_steps", 0))
        observations = int(wrapper.get("router_observations", 0))
        attached = int(wrapper.get("attached_layers", 0))
        routers = int(wrapper.get("router_layers", 0))
        if steps <= 0 or observations < steps:
            problems.append(f"INCOMPLETE ROUTER OBSERVATION: {label}")
        if attached <= 0 or routers <= 0 or attached != routers:
            problems.append(f"INCOMPLETE MODEL INTEGRATION: {label}")
        scheduler_enabled = wrapper.get("scheduler_enabled")
        scheduler_samples = wrapper.get("scheduler_update_samples_ms")
        scheduler_count = wrapper.get("scheduler_update_count")
        if (
            not isinstance(scheduler_enabled, bool)
            or not isinstance(scheduler_samples, list)
            or isinstance(scheduler_count, bool)
            or not isinstance(scheduler_count, int)
            or scheduler_count != len(scheduler_samples)
        ):
            problems.append(f"INVALID DYNAMIC SCHEDULER TELEMETRY: {label}")
        elif (
            (scheduler_enabled and scheduler_count <= 0)
            or (not scheduler_enabled and scheduler_count != 0)
        ):
            problems.append(f"INACTIVE DYNAMIC SCHEDULER: {label}")

    stats = data.get("transition_stats", data.get("final_transition_stats"))
    if not isinstance(stats, dict):
        problems.append(f"MISSING DYNAMIC TRANSITION STATS: {label}")
    else:
        if int(stats.get("failed_transitions", 0)) != 0:
            problems.append(f"FAILED DYNAMIC TRANSITIONS: {label}")
        budget = stats.get("budget")
        if not isinstance(budget, dict):
            problems.append(f"MISSING DYNAMIC BUDGET SNAPSHOT: {label}")
        else:
            try:
                cap = int(budget["total_cap"])
                live = int(budget["total_live"])
                hi_pending = int(budget["hi_pending"])
                lo_pending = int(budget["lo_pending"])
                staging_used = int(budget["staging_used"])
            except (KeyError, TypeError, ValueError):
                problems.append(
                    f"INCOMPLETE DYNAMIC BUDGET SNAPSHOT: {label}"
                )
            else:
                if (
                    cap < 0
                    or not 0 <= live <= cap
                    or hi_pending != 0
                    or lo_pending != 0
                    or staging_used != 0
                ):
                    problems.append(
                        f"INVALID DYNAMIC BUDGET SNAPSHOT: {label}"
                    )

        try:
            accepted = int(stats["accepted_requests"])
            promotions = int(stats["total_promotions"])
            demotions = int(stats["total_demotions"])
            copied_bytes = int(stats["copied_bytes"])
            accepted_bytes = int(stats["accepted_bytes"])
            precise_reclaims = int(stats["precise_fence_reclaims"])
            global_reclaims = int(stats["global_sync_reclaims"])
            active_transitions = int(stats["active_transitions"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"INCOMPLETE TRANSITION LIFECYCLE: {label}")
        else:
            if (
                min(
                    accepted,
                    promotions,
                    demotions,
                    copied_bytes,
                    accepted_bytes,
                    precise_reclaims,
                    global_reclaims,
                    active_transitions,
                )
                < 0
                or accepted != promotions + demotions
                or active_transitions != 0
                or global_reclaims != 0
                or (
                    accepted > 0
                    and (
                        copied_bytes <= 0
                        or accepted_bytes <= 0
                        or precise_reclaims <= 0
                    )
                )
            ):
                problems.append(f"INVALID TRANSITION LIFECYCLE: {label}")

            scheduler_enabled = (
                wrapper.get("scheduler_enabled")
                if isinstance(wrapper, dict)
                else None
            )
            zero_budget_sensitivity = (
                data.get("hi_ratio_pct") == 0
                and str(data.get("artifact_type", "")).endswith(
                    "sensitivity"
                )
            )
            if (
                scheduler_enabled is True
                and not zero_budget_sensitivity
                and accepted <= 0
            ):
                problems.append(f"INACTIVE DYNAMIC TRANSITIONS: {label}")
    problems.extend(validate_initial_map(label, data))
    return problems


def validate_initial_map(label: str, data: dict) -> list[str]:
    """Prove formal runs used one clean, independent calibrated ranking."""
    problems: list[str] = []
    initialization = data.get("runtime_initialization")
    initial_map = data.get("initial_map")
    if (
        not isinstance(initialization, dict)
        or initialization.get("bootstrap_policy")
        != "calibrated_ranking_prefix"
    ):
        return [f"UNCALIBRATED DYNAMIC BOOTSTRAP: {label}"]
    if not isinstance(initial_map, dict):
        return [f"MISSING INITIAL EXPERT MAP: {label}"]
    calibration = initial_map.get("calibration")
    environment = initial_map.get("environment")
    git = environment.get("git", {}) if isinstance(environment, dict) else {}
    if (
        not isinstance(calibration, dict)
        or int(calibration.get("prompt_count", 0)) < 128
        or calibration.get("allowed_splits_only") is not True
        or calibration.get("precision_policy") != "all_low"
        or calibration.get("aggregation")
        != "mean_per_prompt_routing_probability_mass"
        or not calibration.get("source_sha256")
        or not calibration.get("selected_ids_sha256")
    ):
        problems.append(f"INVALID CALIBRATION TRACE: {label}")
    if not git.get("commit") or git.get("dirty") is not False:
        problems.append(f"DIRTY OR UNPINNED CALIBRATION CODE: {label}")
    if initial_map.get("checkpoint") != data.get("checkpoint"):
        problems.append(f"CALIBRATION CHECKPOINT MISMATCH: {label}")
    config = data.get("config")
    model_config = config.get("model") if isinstance(config, dict) else None
    if initial_map.get("model_config") != model_config:
        problems.append(f"CALIBRATION MODEL MISMATCH: {label}")
    ranking = initial_map.get("expert_ranking")
    if not isinstance(ranking, dict) or not isinstance(model_config, dict):
        problems.append(f"INVALID INITIAL EXPERT RANKING: {label}")
        return problems
    try:
        layers = int(model_config["layers"])
        experts = int(model_config["experts_per_layer"])
        canonical_ranking = {
            str(layer): [int(value) for value in ranking[str(layer)]]
            for layer in range(layers)
        }
        ranking_hash = hashlib.sha256(
            json.dumps(
                canonical_ranking,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        n_hi = [int(value) for value in initialization["n_hi"]]
        actual_hi = initialization["bootstrap_hi_experts"]
    except (KeyError, TypeError, ValueError):
        problems.append(f"INVALID INITIAL EXPERT RANKING: {label}")
        return problems
    expected_ids = set(range(experts))
    if (
        len(canonical_ranking) != layers
        or any(
            len(values) != experts or set(values) != expected_ids
            for values in canonical_ranking.values()
        )
        or len(n_hi) != layers
    ):
        problems.append(f"INCOMPLETE INITIAL EXPERT RANKING: {label}")
    if initial_map.get("ranking_sha256") != ranking_hash:
        problems.append(f"INITIAL EXPERT RANKING HASH MISMATCH: {label}")
    try:
        if any(
            sorted(canonical_ranking[str(layer)][: n_hi[layer]])
            != [int(value) for value in actual_hi[str(layer)]]
            for layer in range(layers)
        ):
            problems.append(f"INITIAL EXPERT PREFIX MISMATCH: {label}")
    except (KeyError, TypeError, ValueError):
        problems.append(f"INVALID BOOTSTRAP EXPERT SET: {label}")
    return problems


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if left_only < 0 or right_only < 0:
        raise ValueError("negative discordant count")
    if discordant == 0:
        return 1.0
    tail_count = 1
    term = 1
    for index in range(1, min(left_only, right_only) + 1):
        term = term * (discordant - index + 1) // index
        tail_count += term
    tail = tail_count / (2**discordant)
    return min(1.0, 2.0 * tail)


def _quality_correctness(result: dict) -> dict[str, bool]:
    """Validate one complete task result and return sample correctness."""
    dataset = result.get("dataset")
    details = result.get("details")
    identity_fields = {
        "repository",
        "revision",
        "config",
        "split",
        "source_rows",
        "fingerprint",
        "evaluated_rows",
    }
    if (
        not isinstance(dataset, dict)
        or not identity_fields.issubset(dataset)
        or not isinstance(details, list)
        or not details
    ):
        raise ValueError("incomplete paired result")
    correctness = {}
    for item in details:
        sample_id = item.get("sample_id") if isinstance(item, dict) else None
        correct = item.get("correct") if isinstance(item, dict) else None
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or not isinstance(correct, bool)
            or sample_id in correctness
        ):
            raise ValueError("invalid paired detail")
        correctness[sample_id] = correct
    count = len(correctness)
    if (
        int(result.get("total", -1)) != count
        or int(result.get("evaluated", -1)) != count
        or int(result.get("failed", result.get("skipped", -1))) != 0
        or int(result.get("skipped", result.get("failed", -1))) != 0
        or int(dataset.get("evaluated_rows", -1)) != count
        or abs(float(result.get("score", -1.0)) - sum(correctness.values()) / count)
        > 1e-12
    ):
        raise ValueError("paired result summary")
    return correctness


def _load_perplexity_source_points(
    data: dict,
    *,
    model: str,
    expected_ratios: tuple[int, ...],
) -> dict[int, dict]:
    """Verify and load the immutable point artifacts behind a PPL curve."""
    sources = data.get("source_points")
    if not isinstance(sources, list) or len(sources) != len(expected_ratios):
        raise ValueError("source point count")
    source_ratios = tuple(int(source["low_ratio_pct"]) for source in sources)
    if source_ratios != expected_ratios:
        raise ValueError("source ratio grid")
    loaded: dict[int, dict] = {}
    for source in sources:
        relative = Path(str(source["path"]))
        if relative.is_absolute():
            raise ValueError("absolute source path")
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise ValueError("missing or escaped source point")
        if _sha256(path) != source["sha256"]:
            raise ValueError("source point hash")
        point = json.loads(path.read_text(encoding="utf-8"))
        ratio = int(source["low_ratio_pct"])
        git = point.get("environment", {}).get("git", {})
        if (
            int(point.get("schema_version", 0)) < 2
            or point.get("artifact_type") != "dynaexq_perplexity_point"
            or point.get("paper_model") != model
            or point.get("evaluation_protocol", {}).get("name")
            != "tc_main_v2"
            or point.get("selection_policy")
            != "calibrated_coldest_prefix"
            or int(point.get("low_ratio_pct", -1)) != ratio
            or point.get("wrapper_stats", {}).get("scheduler_enabled")
            is not False
            or not git.get("commit")
            or git.get("dirty") is not False
        ):
            raise ValueError("source point identity")
        if (
            point.get("checkpoint") != data.get("checkpoint")
            or point.get("config") != data.get("config")
            or point.get("initial_map", {}).get("ranking_sha256")
            != data.get("ranking_sha256")
            or point.get("initial_map", {}).get("expert_ranking")
            != data.get("expert_ranking")
        ):
            raise ValueError("source point provenance")
        loaded[ratio] = point
    return loaded


def _nearest_rank(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = max(1, -(-int(quantile * 100) * len(ordered) // 100))
    return ordered[rank - 1]


def _validate_performance_benchmark(label: str, benchmark: object) -> list[str]:
    """Validate raw samples and recompute every plotted summary."""
    problems: list[str] = []
    if not isinstance(benchmark, dict):
        return [f"MISSING RAW PERFORMANCE BENCHMARK: {label}"]
    measured = benchmark.get("measured_iterations")
    samples = benchmark.get("samples")
    if measured != 100 or not isinstance(samples, list):
        problems.append(f"INVALID PERFORMANCE SAMPLE COUNT: {label}")
        return problems
    if len(samples) != measured:
        problems.append(f"INCOMPLETE PERFORMANCE SAMPLES: {label}")
        return problems
    if benchmark.get("warmup_iterations") != 5:
        problems.append(f"INVALID PERFORMANCE WARMUP: {label}")
    if benchmark.get("scope") != "isolated_model":
        problems.append(f"INVALID PERFORMANCE SCOPE: {label}")

    metrics = benchmark.get("metrics")
    required_metrics = (
        "model_ttft_ms",
        "model_tpot_ms",
        "model_e2e_ms",
        "throughput_tokens_s",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "process_hbm_used_peak_bytes",
    )
    monitor = benchmark.get("process_hbm_monitor")
    try:
        monitor_valid = (
            isinstance(monitor, dict)
            and monitor.get("backend") == "nvml"
            and monitor.get("scope")
            == "current_process_selected_device_used_bytes"
            and 0.0 < float(monitor.get("poll_interval_ms", 0.0)) <= 5.0
            and monitor.get("includes_non_pytorch_allocations") is True
            and monitor.get("excludes_other_processes") is True
            and monitor.get("foreign_compute_residency_allowed") is True
            and monitor.get("foreign_compute_activity_policy")
            == "reject_nonzero_nvml_process_utilization"
            and monitor.get("process_utilization_supported") is True
            and len(monitor["cuda_device_indices"]) == 1
            and len(monitor["device_names"]) == 1
            and len(monitor["device_uuids"]) == 1
            and len(monitor["device_total_bytes"]) == 1
            and "A6000" in str(monitor["device_names"][0])
            and int(monitor["device_total_bytes"][0]) >= 47 * 1024**3
        )
    except (KeyError, TypeError, ValueError):
        monitor_valid = False
    if not monitor_valid:
        problems.append(f"INVALID PROCESS HBM MONITOR: {label}")

    for index, sample in enumerate(samples):
        try:
            baseline = int(sample["process_hbm_used_baseline_bytes"])
            peak = int(sample["process_hbm_used_peak_bytes"])
            delta = int(sample["process_hbm_used_peak_delta_bytes"])
            polls = int(sample["process_hbm_poll_samples"])
            foreign_resident = int(
                sample["foreign_compute_resident_processes_peak"]
            )
            foreign_hbm = int(sample["foreign_hbm_used_peak_bytes"])
            foreign_activity = int(
                sample["foreign_compute_activity_samples"]
            )
            foreign_sm = int(sample["foreign_sm_util_max_pct"])
            foreign_mem = int(sample["foreign_mem_util_max_pct"])
            utilization_polls = int(sample["process_util_poll_samples"])
        except (KeyError, TypeError, ValueError):
            problems.append(
                f"INCOMPLETE PROCESS HBM SAMPLE: {label} / {index}"
            )
            continue
        if (
            baseline < 0
            or peak < baseline
            or delta != peak - baseline
            or polls < 2
            or foreign_resident < 0
            or foreign_hbm < 0
            or foreign_activity != 0
            or foreign_sm != 0
            or foreign_mem != 0
            or utilization_polls < 2
            or (
                monitor_valid
                and peak > int(monitor["device_total_bytes"][0])
            )
        ):
            problems.append(
                f"INVALID PROCESS HBM SAMPLE: {label} / {index}"
            )
    if not isinstance(metrics, dict):
        problems.append(f"MISSING PERFORMANCE SUMMARIES: {label}")
        return problems
    for metric in required_metrics:
        try:
            values = [float(sample[metric]) for sample in samples]
            summary = metrics[metric]
            expected = {
                "mean": sum(values) / len(values),
                "p50": _nearest_rank(values, 0.50),
                "p95": _nearest_rank(values, 0.95),
                "p99": _nearest_rank(values, 0.99),
                "min": min(values),
                "max": max(values),
            }
        except (KeyError, TypeError, ValueError):
            problems.append(f"INCOMPLETE PERFORMANCE METRIC: {label} / {metric}")
            continue
        if not isinstance(summary, dict) or any(
            abs(float(summary.get(key, float("inf"))) - value)
            > max(1e-6, abs(value) * 1e-9)
            for key, value in expected.items()
        ):
            problems.append(f"PERFORMANCE SUMMARY MISMATCH: {label} / {metric}")
    return problems


def validate_manifest_artifact(
    group: str,
    label: str,
    data: dict,
    claim_id: str | None = None,
) -> list[str]:
    """Validate the raw JSON behind a non-accuracy table or figure."""
    problems: list[str] = []
    if int(data.get("schema_version", 0)) < 2:
        problems.append(f"LEGACY MANIFEST ARTIFACT: {label}")
    if group == "figure_bundle":
        provenance_fields = ("created_at", "environment")
    elif group == "quality_significance":
        provenance_fields = ("created_at", "environment", "seed")
    else:
        provenance_fields = (
            "created_at",
            "checkpoint",
            "environment",
            "seed",
        )
    for field in provenance_fields:
        if field not in data:
            problems.append(f"MISSING MANIFEST PROVENANCE: {label} / {field}")

    if group not in {"figure_bundle", "quality_significance"}:
        checkpoint = data.get("checkpoint", {})
        if checkpoint.get("local") is True:
            if not checkpoint.get("weight_hashes_included"):
                problems.append(f"UNHASHED MANIFEST CHECKPOINT: {label}")
        elif not checkpoint.get("revision"):
            problems.append(f"UNPINNED MANIFEST CHECKPOINT: {label}")

    environment = data.get("environment", {})
    git = environment.get("git", {}) if isinstance(environment, dict) else {}
    if not git.get("commit"):
        problems.append(f"MISSING MANIFEST GIT COMMIT: {label}")
    if git.get("dirty") is not False:
        problems.append(f"DIRTY MANIFEST RUN: {label}")
    if (
        not isinstance(environment, dict)
        or int(environment.get("process_max_rss_bytes", 0)) <= 0
    ):
        problems.append(f"MISSING PROCESS PEAK RSS: {label}")

    if group == "quality_significance":
        if data.get("artifact_type") != "quality_significance":
            problems.append(f"INVALID QUALITY SIGNIFICANCE TYPE: {label}")
        model = None
        if claim_id is not None:
            _, model, comparison_id = claim_id.split(":")
            if comparison_id != "static_ptq_vs_dynaexq":
                problems.append(
                    f"INVALID QUALITY COMPARISON ID: {label}"
                )
        if model not in QUALITY_SIGNIFICANCE_SOURCES:
            problems.append(f"UNKNOWN QUALITY COMPARISON MODEL: {label}")
        else:
            left_path_expected, right_path_expected, left_method = (
                QUALITY_SIGNIFICANCE_SOURCES[model]
            )
            protocol = data.get("evaluation_protocol")
            comparison = data.get("comparison")
            if (
                data.get("paper_model") != model
                or data.get("seed") != 42
                or not isinstance(protocol, dict)
                or protocol.get("name") != "tc_paired_quality_v1"
                or protocol.get("test")
                != "paired_exact_mcnemar_two_sided"
                or protocol.get("multiple_testing") != "holm"
                or protocol.get("family")
                != list(SIGNIFICANCE_BENCHMARKS)
                or protocol.get("alpha") != 0.05
                or comparison
                != {
                    "left_paper_method": left_method,
                    "right_paper_method": "dynaexq",
                }
            ):
                problems.append(
                    f"INVALID QUALITY SIGNIFICANCE PROTOCOL: {label}"
                )
            loaded_sources = {}
            sources = data.get("sources")
            try:
                if not isinstance(sources, dict) or set(sources) != {
                    "left",
                    "right",
                }:
                    raise ValueError("source records")
                for side, expected_path, expected_method in (
                    ("left", left_path_expected, left_method),
                    ("right", right_path_expected, "dynaexq"),
                ):
                    source = sources[side]
                    relative = Path(str(source["path"]))
                    if relative != expected_path or relative.is_absolute():
                        raise ValueError("source path")
                    path = (ROOT / relative).resolve()
                    if (
                        not path.is_relative_to(ROOT.resolve())
                        or not path.is_file()
                        or _sha256(path) != source["sha256"]
                    ):
                        raise ValueError("source path or hash")
                    quality = json.loads(path.read_text(encoding="utf-8"))
                    source_git = quality["environment"]["git"]
                    expected_type = (
                        "dynaexq_quality"
                        if side == "right"
                        else "quality_evaluation"
                    )
                    if (
                        int(quality.get("schema_version", 0)) < 2
                        or quality.get("artifact_type") != expected_type
                        or not quality.get("created_at")
                        or quality.get("paper_model") != model
                        or quality.get("paper_method") != expected_method
                        or quality.get("seed") != 42
                        or quality.get("evaluation_protocol", {}).get("name")
                        != "tc_main_v2"
                        or not source_git.get("commit")
                        or source_git.get("dirty") is not False
                        or int(
                            quality.get("environment", {}).get(
                                "process_max_rss_bytes",
                                0,
                            )
                        )
                        <= 0
                        or quality.get("checkpoint")
                        != source.get("checkpoint")
                    ):
                        raise ValueError("source identity")
                    checkpoint = quality["checkpoint"]
                    if checkpoint.get("local") is True:
                        if not checkpoint.get("weight_hashes_included"):
                            raise ValueError("unhashed checkpoint")
                    elif not checkpoint.get("revision"):
                        raise ValueError("unpinned checkpoint")
                    loaded_sources[side] = quality
            except (
                json.JSONDecodeError,
                KeyError,
                OSError,
                TypeError,
                ValueError,
            ):
                problems.append(
                    f"INVALID QUALITY SIGNIFICANCE SOURCES: {label}"
                )
                loaded_sources = {}

            try:
                reported = data["benchmarks"]
                if (
                    set(loaded_sources) != {"left", "right"}
                    or not isinstance(reported, dict)
                    or set(reported) != set(SIGNIFICANCE_BENCHMARKS)
                ):
                    raise ValueError("benchmark family")
                expected_results = {}
                dataset_fields = (
                    "repository",
                    "revision",
                    "config",
                    "split",
                    "source_rows",
                    "fingerprint",
                    "evaluated_rows",
                )
                for benchmark in SIGNIFICANCE_BENCHMARKS:
                    left_result = loaded_sources["left"]["benchmarks"][
                        benchmark
                    ]
                    right_result = loaded_sources["right"]["benchmarks"][
                        benchmark
                    ]
                    if any(
                        left_result["dataset"].get(field)
                        != right_result["dataset"].get(field)
                        for field in dataset_fields
                    ):
                        raise ValueError("dataset identity")
                    left_correct = _quality_correctness(left_result)
                    right_correct = _quality_correctness(right_result)
                    if left_correct.keys() != right_correct.keys():
                        raise ValueError("sample identity")
                    left_only = sum(
                        left_correct[key] and not right_correct[key]
                        for key in left_correct
                    )
                    right_only = sum(
                        right_correct[key] and not left_correct[key]
                        for key in left_correct
                    )
                    both_correct = sum(
                        left_correct[key] and right_correct[key]
                        for key in left_correct
                    )
                    total = len(left_correct)
                    both_wrong = (
                        total - left_only - right_only - both_correct
                    )
                    left_accuracy = (both_correct + left_only) / total
                    right_accuracy = (both_correct + right_only) / total
                    expected_results[benchmark] = {
                        "total": total,
                        "both_correct": both_correct,
                        "both_wrong": both_wrong,
                        "left_only_correct": left_only,
                        "right_only_correct": right_only,
                        "left_accuracy": left_accuracy,
                        "right_accuracy": right_accuracy,
                        "delta_percentage_points": (
                            right_accuracy - left_accuracy
                        )
                        * 100.0,
                        "mcnemar_exact_p": _exact_mcnemar_p(
                            left_only,
                            right_only,
                        ),
                    }
                ordered = sorted(
                    SIGNIFICANCE_BENCHMARKS,
                    key=lambda name: expected_results[name][
                        "mcnemar_exact_p"
                    ],
                )
                running = 0.0
                for rank, benchmark in enumerate(ordered):
                    running = max(
                        running,
                        min(
                            1.0,
                            (len(ordered) - rank)
                            * expected_results[benchmark][
                                "mcnemar_exact_p"
                            ],
                        ),
                    )
                    expected_results[benchmark][
                        "holm_adjusted_p"
                    ] = running
                    expected_results[benchmark][
                        "significant_at_0_05"
                    ] = running < 0.05
                for benchmark, expected in expected_results.items():
                    observed = reported[benchmark]
                    for field, value in expected.items():
                        observed_value = observed[field]
                        if isinstance(value, bool):
                            if observed_value is not value:
                                raise ValueError("significance flag")
                        elif abs(float(observed_value) - float(value)) > 1e-12:
                            raise ValueError("significance summary")
            except (
                KeyError,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ):
                problems.append(
                    f"INVALID QUALITY SIGNIFICANCE RESULTS: {label}"
                )
    elif group == "performance":
        benchmark = data.get("benchmark")
        problems.extend(_validate_performance_benchmark(label, benchmark))
        if claim_id is not None:
            _, model, method, batch_text = claim_id.split(":")
            batch = int(batch_text.removeprefix("bs"))
            if data.get("paper_model") != model:
                problems.append(f"PERFORMANCE MODEL MISMATCH: {label}")
            if data.get("paper_method") != method:
                problems.append(f"PERFORMANCE METHOD MISMATCH: {label}")
            environment = data.get("environment")
            gpu_names = (
                environment.get("gpus", [])
                if isinstance(environment, dict)
                else []
            )
            if (
                len(gpu_names) != 1
                or "A6000" not in str(gpu_names[0])
            ):
                problems.append(
                    f"PERFORMANCE HARDWARE MISMATCH: {label}"
                )
            protocol = data.get("evaluation_protocol")
            if (
                not isinstance(protocol, dict)
                or protocol.get("name") != "tc_isolated_performance_v2"
                or protocol.get("seed") != 42
                or protocol.get("process_hbm_high_water") is not True
            ):
                problems.append(
                    f"INVALID PERFORMANCE PROTOCOL: {label}"
                )
            if method not in PERFORMANCE_METHODS.get(model, ()):
                problems.append(
                    f"UNSUPPORTED PERFORMANCE METHOD/MODEL: {label}"
                )
            if isinstance(benchmark, dict):
                if int(benchmark.get("batch_size", -1)) != batch:
                    problems.append(f"PERFORMANCE BATCH MISMATCH: {label}")
                if benchmark.get("input_tokens") != 2048:
                    problems.append(f"PERFORMANCE INPUT LENGTH MISMATCH: {label}")
                if benchmark.get("output_tokens_per_sequence") != 256:
                    problems.append(f"PERFORMANCE OUTPUT LENGTH MISMATCH: {label}")
            if method == "dynaexq":
                problems.extend(validate_dynamic_runtime(label, data))
            elif method == "static_ptq":
                if data.get("method") != "quantized_checkpoint":
                    problems.append(f"STATIC PTQ METHOD NOT ACTIVATED: {label}")
            elif method == "moe_infinity":
                implementation = data.get("baseline_implementation")
                if not isinstance(implementation, dict):
                    problems.append(
                        f"MISSING MOE-INFINITY IMPLEMENTATION: {label}"
                    )
                else:
                    identity_mismatch = any(
                        implementation.get(key) != value
                        for key, value in MOE_INFINITY_IDENTITY.items()
                    )
                    features = implementation.get("features")
                    required_features = {
                        "expert_offload": True,
                        "activation_aware_cache": True,
                        "prefetch": True,
                        "speculative_prefetch": True,
                        "speculative_prefetch_overlap": True,
                        "use_native_engine": False,
                    }
                    if (
                        identity_mismatch
                        or implementation.get("clean") is not True
                        or implementation.get(
                            "paper_implementation_equivalent"
                        )
                        is not False
                        or implementation.get("source_hash_algorithm")
                        != "sha256(git-ls-tree-r-z)"
                        or not implementation.get("variant_note")
                        or implementation.get("imported_module")
                        != "moe_infinity/__init__.py"
                        or features != required_features
                    ):
                        problems.append(
                            f"INVALID MOE-INFINITY IMPLEMENTATION: {label}"
                        )
                stats = data.get("baseline_runtime_stats")
                try:
                    total_tensors = int(stats["total_expert_tensors"])
                    offloaded_tensors = int(
                        stats["offloaded_expert_tensors"]
                    )
                    prefetch_calls = int(stats["prefetch_calls"])
                    requested = int(
                        stats["prefetch_requested_experts"]
                    )
                    layers = [int(value) for value in stats[
                        "prefetch_layers_touched"
                    ]]
                    experts = [int(value) for value in stats[
                        "prefetch_unique_experts"
                    ]]
                except (KeyError, TypeError, ValueError):
                    problems.append(
                        f"INCOMPLETE MOE-INFINITY RUNTIME STATS: {label}"
                    )
                else:
                    if (
                        total_tensors <= 0
                        or not 0 < offloaded_tensors <= total_tensors
                        or prefetch_calls <= 0
                        or requested < prefetch_calls
                        or not layers
                        or not experts
                        or len(layers) != len(set(layers))
                        or len(experts) != len(set(experts))
                    ):
                        problems.append(
                            f"INACTIVE MOE-INFINITY RUNTIME: {label}"
                        )
                config = data.get("runtime_config")
                try:
                    device_ratio = float(config["device_memory_ratio"])
                except (KeyError, TypeError, ValueError):
                    device_ratio = -1.0
                if (
                    not isinstance(config, dict)
                    or not config.get("offload_path")
                    or not 0.1 <= device_ratio <= 0.85
                    or any(
                        config.get(key) is not value
                        for key, value in {
                            "prefetch": True,
                            "speculative_prefetch": True,
                            "speculative_prefetch_overlap": True,
                            "use_native_engine": False,
                        }.items()
                    )
                ):
                    problems.append(
                        f"INVALID MOE-INFINITY CONFIGURATION: {label}"
                    )
                hardware = data.get("hardware_contract")
                environment = data.get("environment")
                gpu_names = (
                    environment.get("gpus", [])
                    if isinstance(environment, dict)
                    else []
                )
                if (
                    not isinstance(hardware, dict)
                    or hardware.get("device_count") != 1
                    or "A6000" not in str(hardware.get("device_name", ""))
                    or int(hardware.get("total_memory_bytes", 0))
                    < 47 * 1024**3
                    or len(gpu_names) != 1
                    or "A6000" not in str(gpu_names[0])
                ):
                    problems.append(
                        f"MOE-INFINITY HARDWARE MISMATCH: {label}"
                    )
                loading = data.get("model_loading")
                checkpoint = data.get("checkpoint")
                if (
                    not isinstance(loading, dict)
                    or not isinstance(checkpoint, dict)
                ):
                    problems.append(
                        f"UNPINNED MOE-INFINITY MODEL LOAD: {label}"
                    )
                elif checkpoint.get("local") is True:
                    if loading.get("mode") != "hashed_local_checkpoint":
                        problems.append(
                            f"UNPINNED MOE-INFINITY MODEL LOAD: {label}"
                        )
                elif (
                    loading.get("mode") != "pinned_huggingface_snapshot"
                    or loading.get("remote_revision")
                    != checkpoint.get("revision")
                    or loading.get("snapshot_commit_directory")
                    != checkpoint.get("revision")
                ):
                    problems.append(
                        f"UNPINNED MOE-INFINITY MODEL LOAD: {label}"
                    )
                if (
                    data.get("artifact_type")
                    != "moe_infinity_performance"
                    or data.get("method")
                    != "official_external_offload_runtime"
                ):
                    problems.append(
                        f"MOE-INFINITY METHOD NOT ACTIVATED: {label}"
                    )
    elif group == "runtime_overhead":
        initialization = data.get("runtime_initialization")
        stats = data.get("transition_stats", data.get("final_transition_stats"))
        if not isinstance(initialization, dict):
            problems.append(f"MISSING RUNTIME INITIALIZATION: {label}")
        if not isinstance(stats, dict):
            problems.append(f"MISSING FINAL TRANSITION STATS: {label}")
        else:
            if int(stats.get("failed_transitions", 0)) != 0:
                problems.append(f"FAILED TRANSITIONS IN ARTIFACT: {label}")
            budget = stats.get("budget")
            if isinstance(budget, dict):
                total_cap = int(budget.get("total_cap", -1))
                total_live = int(budget.get("total_live", -1))
                if total_cap < 0 or not 0 <= total_live <= total_cap:
                    problems.append(f"INVALID FINAL BUDGET SNAPSHOT: {label}")
        problems.extend(validate_dynamic_runtime(label, data))
        benchmarks = data.get("benchmarks")
        if (
            not isinstance(benchmarks, dict)
            or set(benchmarks)
            != {"mmlu_pro", "gpqa", "aime25", "gsm8k", "humaneval"}
        ):
            problems.append(f"INVALID OVERHEAD BENCHMARK SET: {label}")
        elif any(
            not isinstance(result, dict)
            or "score" not in result
            or int(result.get("failed", 0)) != 0
            for result in benchmarks.values()
        ):
            problems.append(f"INVALID OVERHEAD QUALITY TRACE: {label}")
        performance = data.get("benchmark")
        problems.extend(_validate_performance_benchmark(label, performance))
        if isinstance(performance, dict) and (
            performance.get("batch_size") != 32
            or performance.get("input_tokens") != 2048
            or performance.get("output_tokens_per_sequence") != 256
        ):
            problems.append(f"INVALID OVERHEAD PERFORMANCE SHAPE: {label}")
        if data.get("overhead_sequence") != [
            "mmlu_pro",
            "gpqa",
            "aime25",
            "gsm8k",
            "humaneval",
            "performance_bs32",
        ]:
            problems.append(f"INVALID OVERHEAD SEQUENCE: {label}")
        protocol = data.get("evaluation_protocol")
        if not isinstance(protocol, dict) or protocol.get("name") != "tc_main_v2":
            problems.append(f"INVALID OVERHEAD EVALUATION PROTOCOL: {label}")
        metrics = data.get("paper_metrics")
        wrapper = data.get("wrapper_stats")
        if (
            not isinstance(metrics, dict)
            or not isinstance(initialization, dict)
            or not isinstance(stats, dict)
            or not isinstance(wrapper, dict)
            or not isinstance(performance, dict)
        ):
            problems.append(f"MISSING OVERHEAD PAPER METRICS: {label}")
        else:
            try:
                scheduler_samples = [
                    float(value)
                    for value in wrapper["scheduler_update_samples_ms"]
                ]
                scheduler_ordered = sorted(scheduler_samples)
                scheduler_p99 = scheduler_ordered[
                    max(
                        0,
                        -(-99 * len(scheduler_ordered) // 100) - 1,
                    )
                ]
                expected = {
                    "hbm_budget_gb": (
                        float(data["config"]["memory"]["device_mem_bytes"])
                        / 1e9
                    ),
                    "peak_process_hbm_used_gb": (
                        max(
                            float(
                                sample["process_hbm_used_peak_bytes"]
                            )
                            for sample in performance["samples"]
                        )
                        / 1e9
                    ),
                    "resident_expert_pool_gb": (
                        float(initialization["resident_expert_bytes"]) / 1e9
                    ),
                    "transient_expert_pool_gb": (
                        float(initialization["transient_expert_bytes"]) / 1e9
                    ),
                    "migration_count": (
                        int(stats["total_promotions"])
                        + int(stats["total_demotions"])
                    ),
                    "transferred_gb": float(stats["copied_bytes"]) / 1e9,
                    "scheduler_mean_ms": (
                        sum(scheduler_samples) / len(scheduler_samples)
                    ),
                    "scheduler_p99_ms": scheduler_p99,
                    "pinned_expert_cache_gb": (
                        float(
                            initialization["host_cache"][
                                "host_packed_bytes"
                            ]
                        )
                        / 1e9
                    ),
                }
            except (
                IndexError,
                KeyError,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ):
                problems.append(f"INVALID OVERHEAD RAW TELEMETRY: {label}")
            else:
                if any(
                    abs(float(metrics.get(key, float("inf"))) - value)
                    > max(1e-6, abs(value) * 1e-9)
                    for key, value in expected.items()
                ):
                    problems.append(f"OVERHEAD PAPER METRIC MISMATCH: {label}")
                if (
                    expected["peak_process_hbm_used_gb"]
                    > expected["hbm_budget_gb"]
                ):
                    problems.append(
                        f"PROCESS HBM BUDGET EXCEEDED: {label}"
                    )
        if claim_id is not None:
            _, model = claim_id.split(":")
            if data.get("paper_model") != model:
                problems.append(f"RUNTIME OVERHEAD MODEL MISMATCH: {label}")
            displayed_column = 0 if model == "qwen30b" else 1
            displayed = manuscript_overhead_rows()
            tolerances = {
                "hbm_budget_gb": 0.0501,
                "peak_process_hbm_used_gb": 0.0501,
                "resident_expert_pool_gb": 0.0501,
                "transient_expert_pool_gb": 0.0051,
                "migration_count": 0.5001,
                "transferred_gb": 0.0501,
                "scheduler_mean_ms": 0.0051,
                "scheduler_p99_ms": 0.0051,
                "pinned_expert_cache_gb": 0.0501,
            }
            for metric, tolerance in tolerances.items():
                row = displayed.get(metric)
                if row is None:
                    problems.append(
                        f"MISSING MANUSCRIPT OVERHEAD ROW: {label} / {metric}"
                    )
                elif isinstance(metrics, dict) and abs(
                    row[displayed_column] - float(metrics.get(metric, float("inf")))
                ) > tolerance:
                    problems.append(
                        f"MANUSCRIPT OVERHEAD MISMATCH: {label} / {metric}"
                    )
    elif group == "ablation":
        benchmarks = data.get("benchmarks")
        if not isinstance(benchmarks, dict):
            problems.append(f"MISSING MANIFEST BENCHMARKS: {label}")
        elif set(benchmarks) != {
            "mmlu_pro",
            "gpqa",
            "aime25",
            "gsm8k",
            "humaneval",
        }:
            problems.append(f"INVALID ABLATION BENCHMARK SET: {label}")
        else:
            for benchmark_name, result in benchmarks.items():
                if (
                    not isinstance(result, dict)
                    or "score" not in result
                    or int(result.get("failed", 0)) != 0
                ):
                    problems.append(
                        f"INVALID ABLATION QUALITY RESULT: "
                        f"{label} / {benchmark_name}"
                    )
        performance = data.get("benchmark")
        problems.extend(_validate_performance_benchmark(label, performance))
        if isinstance(performance, dict):
            if (
                performance.get("batch_size") != 32
                or performance.get("input_tokens") != 2048
                or performance.get("output_tokens_per_sequence") != 256
            ):
                problems.append(f"INVALID ABLATION PERFORMANCE SHAPE: {label}")
        metrics = data.get("paper_metrics")
        if not isinstance(metrics, dict):
            problems.append(f"MISSING ABLATION PAPER METRICS: {label}")
        elif isinstance(benchmarks, dict) and isinstance(performance, dict):
            try:
                expected_metrics = {
                    "average_accuracy_pct": (
                        100.0
                        * sum(
                            float(benchmarks[name]["score"])
                            for name in (
                                "mmlu_pro",
                                "gpqa",
                                "aime25",
                                "gsm8k",
                                "humaneval",
                            )
                        )
                        / 5.0
                    ),
                    "throughput_tokens_s": float(
                        performance["metrics"]["throughput_tokens_s"]["mean"]
                    ),
                    "p99_s": float(
                        performance["metrics"]["model_e2e_ms"]["p99"]
                    )
                    / 1000.0,
                }
            except (KeyError, TypeError, ValueError):
                problems.append(f"INVALID ABLATION PAPER METRICS: {label}")
            else:
                if any(
                    abs(float(metrics.get(key, float("inf"))) - expected)
                    > max(1e-6, abs(expected) * 1e-9)
                    for key, expected in expected_metrics.items()
                ):
                    problems.append(f"ABLATION PAPER METRIC MISMATCH: {label}")
        protocol = data.get("evaluation_protocol")
        if not isinstance(protocol, dict) or protocol.get("name") != "tc_main_v2":
            problems.append(f"INVALID ABLATION EVALUATION PROTOCOL: {label}")
        expected_sequence = [
            "mmlu_pro",
            "gpqa",
            "aime25",
            "gsm8k",
            "humaneval",
            "performance_bs32",
        ]
        if data.get("ablation_sequence") != expected_sequence:
            problems.append(f"INVALID ABLATION SEQUENCE: {label}")
        if claim_id is not None:
            _, model, config = claim_id.split(":")
            if data.get("paper_model") != model:
                problems.append(f"ABLATION MODEL MISMATCH: {label}")
            if data.get("ablation_config") != config:
                problems.append(f"ABLATION CONFIG MISMATCH: {label}")
            wrapper = data.get("wrapper_stats", {})
            stats = data.get(
                "transition_stats",
                data.get("final_transition_stats", {}),
            )
            initialization = data.get("runtime_initialization", {})
            expected_scheduler = config != "static"
            expected_execution = (
                "synchronous" if config == "blocking" else "asynchronous"
            )
            if (
                not isinstance(wrapper, dict)
                or wrapper.get("scheduler_enabled") is not expected_scheduler
            ):
                problems.append(f"ABLATION SCHEDULER MODE MISMATCH: {label}")
            if (
                not isinstance(stats, dict)
                or stats.get("execution_mode") != expected_execution
            ):
                problems.append(f"ABLATION EXECUTION MODE MISMATCH: {label}")
            if (
                not isinstance(initialization, dict)
                or initialization.get("transition_execution_mode")
                != expected_execution
            ):
                problems.append(
                    f"ABLATION INITIALIZATION MODE MISMATCH: {label}"
                )
            configured_margin = (
                data.get("config", {})
                .get("scheduler", {})
                .get("delta_score_margin")
            )
            if config == "no_hysteresis":
                if configured_margin != 0.0:
                    problems.append(
                        f"ABLATION HYSTERESIS STILL ACTIVE: {label}"
                    )
            elif config in {"full", "static", "blocking"}:
                if configured_margin is None or float(configured_margin) <= 0.0:
                    problems.append(
                        f"ABLATION FULL HYSTERESIS NOT ACTIVE: {label}"
                    )
            if (
                config == "static"
                and isinstance(stats, dict)
                and int(stats.get("enqueue_attempts", -1)) != 0
            ):
                problems.append(f"STATIC ABLATION TRANSITIONED: {label}")
            displayed = manuscript_ablation_rows().get((model, config))
            if displayed is None:
                problems.append(f"MISSING MANUSCRIPT ABLATION ROW: {label}")
            elif isinstance(metrics, dict):
                expected = (
                    float(metrics.get("average_accuracy_pct", float("inf"))),
                    float(metrics.get("throughput_tokens_s", float("inf"))),
                    float(metrics.get("p99_s", float("inf"))),
                )
                tolerances = (0.0051, 0.5001, 0.5001)
                if any(
                    abs(shown - source) > tolerance
                    for shown, source, tolerance in zip(
                        displayed,
                        expected,
                        tolerances,
                    )
                ):
                    problems.append(f"MANUSCRIPT ABLATION MISMATCH: {label}")
    elif group == "budget_sensitivity":
        benchmarks = data.get("benchmarks")
        expected_benchmarks = {
            "mmlu_pro",
            "gpqa",
            "aime25",
            "gsm8k",
            "humaneval",
        }
        if not isinstance(benchmarks, dict):
            problems.append(f"MISSING MANIFEST BENCHMARKS: {label}")
        elif set(benchmarks) != expected_benchmarks:
            problems.append(f"INVALID SENSITIVITY BENCHMARK SET: {label}")
        else:
            for benchmark_name, result in benchmarks.items():
                if (
                    not isinstance(result, dict)
                    or "score" not in result
                    or int(result.get("failed", 0)) != 0
                ):
                    problems.append(
                        f"INVALID SENSITIVITY QUALITY RESULT: "
                        f"{label} / {benchmark_name}"
                    )
        metrics = data.get("paper_metrics")
        if not isinstance(metrics, dict):
            problems.append(f"MISSING SENSITIVITY PAPER METRICS: {label}")
        elif isinstance(benchmarks, dict):
            try:
                expected_average = (
                    100.0
                    * sum(
                        float(benchmarks[name]["score"])
                        for name in expected_benchmarks
                    )
                    / len(expected_benchmarks)
                )
                reported_average = float(metrics["average_accuracy_pct"])
            except (KeyError, TypeError, ValueError):
                problems.append(f"INVALID SENSITIVITY PAPER METRICS: {label}")
            else:
                if abs(reported_average - expected_average) > max(
                    1e-6,
                    abs(expected_average) * 1e-9,
                ):
                    problems.append(
                        f"SENSITIVITY PAPER METRIC MISMATCH: {label}"
                    )
        protocol = data.get("evaluation_protocol")
        if not isinstance(protocol, dict) or protocol.get("name") != "tc_main_v2":
            problems.append(
                f"INVALID SENSITIVITY EVALUATION PROTOCOL: {label}"
            )
        if data.get("sensitivity_sequence") != [
            "mmlu_pro",
            "gpqa",
            "aime25",
            "gsm8k",
            "humaneval",
        ]:
            problems.append(f"INVALID SENSITIVITY SEQUENCE: {label}")
        problems.extend(validate_dynamic_runtime(label, data))
        if claim_id is not None:
            _, model, ratio_text = claim_id.split(":")
            ratio = int(ratio_text.removeprefix("ratio"))
            if data.get("paper_model") != model:
                problems.append(f"SENSITIVITY MODEL MISMATCH: {label}")
            if int(data.get("hi_ratio_pct", -1)) != ratio:
                problems.append(f"SENSITIVITY RATIO MISMATCH: {label}")
            initialization = data.get("runtime_initialization", {})
            config = data.get("config", {})
            model_config = (
                config.get("model", {})
                if isinstance(config, dict)
                else {}
            )
            try:
                layers = int(model_config["layers"])
                experts = int(model_config["experts_per_layer"])
                n_hi = initialization["n_hi"]
                expected_per_layer = int(experts * ratio / 100.0)
                realized_pct = (
                    100.0 * sum(int(value) for value in n_hi)
                    / (layers * experts)
                )
                requested = initialization[
                    "requested_high_precision_ratio"
                ]
                recorded_realized = initialization[
                    "realized_high_precision_ratio"
                ]
                metric_realized = metrics["realized_hi_ratio_pct"]
                metric_resident = metrics["resident_expert_bytes"]
                resident = initialization["resident_expert_bytes"]
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                problems.append(
                    f"INVALID SENSITIVITY RUNTIME QUOTA: {label}"
                )
            else:
                if (
                    len(n_hi) != layers
                    or any(
                        int(value) != expected_per_layer for value in n_hi
                    )
                    or abs(float(requested) - ratio / 100.0) > 1e-12
                    or abs(
                        100.0 * float(recorded_realized) - realized_pct
                    )
                    > 1e-9
                    or abs(float(metric_realized) - realized_pct) > 1e-9
                    or int(metric_resident) != int(resident)
                ):
                    problems.append(
                        f"SENSITIVITY RUNTIME QUOTA MISMATCH: {label}"
                    )
    elif group == "activation_density":
        if data.get("artifact_type") != "activation_density":
            problems.append(f"INVALID ACTIVATION ARTIFACT TYPE: {label}")
        if claim_id is not None:
            _, model, stage = claim_id.split(":")
            if data.get("paper_model") != model:
                problems.append(f"ACTIVATION MODEL MISMATCH: {label}")
            contracts = {
                "qwen30b": (128, 8, 48),
                "qwen80b": (512, 10, 48),
                "phi35": (16, 2, 32),
            }
            experts_expected, topk_expected, layers_expected = contracts[
                model
            ]
            protocol = data.get("protocol")
            layer_ids = data.get("moe_layer_ids")
            protocol_valid = (
                isinstance(protocol, dict)
                and protocol.get("name") == "tc_activation_density_v1"
                and protocol.get("batch_sizes") == [1, 2, 4, 8, 16, 32]
                and protocol.get("repeats") == 5
                and protocol.get("max_input_tokens") == 2048
                and protocol.get("padding_side") == "left"
                and protocol.get("prefill_scope")
                == "all_nonpadding_prompt_tokens"
                and protocol.get("decode_scope")
                == "first_single_token_step_after_prefill"
                and protocol.get("decode_token_selection")
                == "greedy_argmax_last_position"
                and protocol.get("causal_lm_logits_scope")
                == "last_position_only"
                and protocol.get("aggregation")
                == "mean_unique_experts_across_moe_layers_and_repeats"
                and protocol.get("experts_per_layer") == experts_expected
                and protocol.get("topk") == topk_expected
                and isinstance(layer_ids, list)
                and len(layer_ids) == layers_expected
                and [int(value) for value in layer_ids]
                == sorted(set(int(value) for value in layer_ids))
            )
            prompt_source = (
                protocol.get("prompt_source")
                if isinstance(protocol, dict)
                else None
            )
            if not protocol_valid:
                problems.append(f"INVALID ACTIVATION PROTOCOL: {label}")
            if not isinstance(prompt_source, dict):
                problems.append(f"MISSING ACTIVATION PROMPT SOURCE: {label}")
            else:
                try:
                    relative = Path(str(prompt_source["path"]))
                    prompt_path = (ROOT / relative).resolve()
                    prompt_valid = (
                        not relative.is_absolute()
                        and prompt_path.is_relative_to(ROOT.resolve())
                        and prompt_path.is_file()
                        and _sha256(prompt_path)
                        == prompt_source["source_sha256"]
                        and int(prompt_source["selected_prompt_count"]) == 160
                        and bool(prompt_source["selected_ids_sha256"])
                        and prompt_source["selection"]
                        == "ordered_blocks_of_32_nested_by_batch_size"
                    )
                except (KeyError, OSError, TypeError, ValueError):
                    prompt_valid = False
                if not prompt_valid:
                    problems.append(
                        f"INVALID ACTIVATION PROMPT SOURCE: {label}"
                    )
            stage_points = data.get("stages", {}).get(stage)
            if not isinstance(stage_points, list):
                problems.append(f"MISSING ACTIVATION STAGE: {label}")
            else:
                expected_batches = (1, 2, 4, 8, 16, 32)
                observed_values = []
                try:
                    if tuple(
                        int(point["batch_size"]) for point in stage_points
                    ) != expected_batches:
                        raise ValueError("batch grid")
                    for point in stage_points:
                        experts = int(point["experts_per_layer"])
                        raw = point["layer_active_counts"]
                        if (
                            not isinstance(raw, list)
                            or len(raw) != 5
                            or any(
                                not isinstance(sample, list)
                                or len(sample) != layers_expected
                                for sample in raw
                            )
                        ):
                            raise ValueError("sample grid")
                        counts = [
                            int(count)
                            for sample in raw
                            for count in sample
                        ]
                        if (
                            experts <= 0
                            or not counts
                            or any(
                                count < 0 or count > experts
                                for count in counts
                            )
                        ):
                            raise ValueError("raw counts")
                        ratio = 100.0 * sum(counts) / (len(counts) * experts)
                        if abs(float(point["ratio_pct"]) - ratio) > 1e-9:
                            raise ValueError("summary")
                        observed_values.append(ratio)
                except (KeyError, TypeError, ValueError):
                    problems.append(f"INVALID ACTIVATION RAW COUNTS: {label}")
                else:
                    displayed = manuscript_activation_density().get(
                        (model, stage)
                    )
                    if displayed is None:
                        problems.append(
                            f"MISSING MANUSCRIPT ACTIVATION ROW: {label}"
                        )
                    elif any(
                        abs(shown - source) > 0.0501
                        for shown, source in zip(displayed, observed_values)
                    ):
                        problems.append(
                            f"MANUSCRIPT ACTIVATION MISMATCH: {label}"
                        )
    elif group == "offload_waiting":
        if data.get("artifact_type") != "blocking_offload_waiting":
            problems.append(f"INVALID OFFLOAD WAITING TYPE: {label}")
        model = None
        if claim_id is not None:
            _, model = claim_id.split(":")
            if data.get("paper_model") != model:
                problems.append(f"OFFLOAD WAITING MODEL MISMATCH: {label}")
        if data.get("offload_method") != "blocking_on_demand":
            problems.append(f"MISLABELED OFFLOAD WAITING METHOD: {label}")
        benchmark_device = data.get("benchmark_device")
        if (
            not isinstance(benchmark_device, dict)
            or benchmark_device.get("type") != "cuda"
            or not isinstance(benchmark_device.get("index"), int)
            or "RTX A6000" not in str(benchmark_device.get("name", ""))
        ):
            problems.append(f"OFFLOAD WAITING HARDWARE MISMATCH: {label}")
        if model not in OFFLOAD_MODEL_CONTRACTS:
            problems.append(f"UNKNOWN OFFLOAD WAITING MODEL: {label}")
        else:
            (
                layers_expected,
                experts_expected,
                topk_expected,
            ) = OFFLOAD_MODEL_CONTRACTS[model]
            benchmark = data.get("benchmark")
            protocol = (
                benchmark.get("protocol")
                if isinstance(benchmark, dict)
                else None
            )
            points = (
                benchmark.get("points")
                if isinstance(benchmark, dict)
                else None
            )
            protocol_valid = (
                isinstance(protocol, dict)
                and protocol.get("name") == "tc_blocking_offload_v1"
                and protocol.get("cache_start") == "cold_per_trial"
                and protocol.get("transfer") == "pinned_host_to_device"
                and protocol.get("execution")
                == "serial_blocking_on_demand"
                and protocol.get("payload")
                == "measured_packed_expert_bytes"
                and protocol.get("warmup_trials") == 2
                and protocol.get("measured_trials") == 10
                and protocol.get("input_lengths")
                == list(OFFLOAD_INPUT_LENGTHS)
            )
            if not protocol_valid:
                problems.append(f"INVALID OFFLOAD WAITING PROTOCOL: {label}")
            layer_ids = data.get("moe_layer_ids")
            expert_bytes = data.get("expert_bytes_per_layer")
            try:
                layer_ids = [int(value) for value in layer_ids]
                expert_bytes = [int(value) for value in expert_bytes]
                if (
                    len(layer_ids) != layers_expected
                    or layer_ids != sorted(set(layer_ids))
                    or len(expert_bytes) != layers_expected
                    or any(value <= 0 for value in expert_bytes)
                    or int(data["experts_per_layer"]) != experts_expected
                ):
                    raise ValueError("model contract")
            except (KeyError, TypeError, ValueError):
                problems.append(f"INVALID OFFLOAD PAYLOAD CONTRACT: {label}")
                layer_ids = []
                expert_bytes = []

            source_trace = data.get("source_trace")
            trace = None
            try:
                if not isinstance(source_trace, dict):
                    raise ValueError("source record")
                relative = Path(str(source_trace["path"]))
                trace_path = (ROOT / relative).resolve()
                if (
                    relative.is_absolute()
                    or not trace_path.is_relative_to(ROOT.resolve())
                    or not trace_path.is_file()
                    or _sha256(trace_path) != source_trace["sha256"]
                ):
                    raise ValueError("source path or hash")
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                trace_git = trace["environment"]["git"]
                trace_checkpoint = trace["checkpoint"]
                trace_protocol = trace["protocol"]
                prompt_source = trace_protocol["prompt_source"]
                if (
                    int(trace.get("schema_version", 0)) < 2
                    or trace.get("artifact_type")
                    != "routing_active_set_trace"
                    or not trace.get("created_at")
                    or trace.get("paper_model") != model
                    or trace.get("seed") != data.get("seed")
                    or trace_checkpoint != data.get("checkpoint")
                    or not trace_git.get("commit")
                    or trace_git.get("dirty") is not False
                    or int(
                        trace.get("environment", {}).get(
                            "process_max_rss_bytes",
                            0,
                        )
                    )
                    <= 0
                    or trace.get("moe_layer_ids") != layer_ids
                    or int(trace.get("experts_per_layer", 0))
                    != experts_expected
                    or trace.get("expert_bytes_per_layer") != expert_bytes
                    or trace_protocol.get("name")
                    != "tc_routing_active_set_v1"
                    or trace_protocol.get("input_lengths")
                    != list(OFFLOAD_INPUT_LENGTHS)
                    or trace_protocol.get("warmup_trials") != 2
                    or trace_protocol.get("measured_trials") != 10
                    or trace_protocol.get("batch_size") != 1
                    or trace_protocol.get("padding") != "none"
                    or trace_protocol.get("prefix_policy")
                    != "nested_prefix_per_disjoint_source_window"
                    or trace_protocol.get("router_metric")
                    != "unique_selected_experts_per_layer"
                    or trace_protocol.get("topk") != topk_expected
                    or trace_protocol.get("causal_lm_logits_scope")
                    != "last_position_only"
                    or trace_protocol.get("expert_payload_measurement")
                    != "stored_routed_expert_parameter_and_buffer_bytes"
                ):
                    raise ValueError("source provenance")
                prompt_relative = Path(str(prompt_source["path"]))
                prompt_path = (ROOT / prompt_relative).resolve()
                if (
                    prompt_relative.is_absolute()
                    or not prompt_path.is_relative_to(ROOT.resolve())
                    or not prompt_path.is_file()
                    or _sha256(prompt_path)
                    != prompt_source["source_sha256"]
                    or prompt_source.get("selected_token_count")
                    != 12 * 2048
                    or prompt_source.get("selection")
                    != (
                        "concatenated_eos_separated_disjoint_"
                        "2048_token_blocks"
                    )
                ):
                    raise ValueError("prompt source")
                prompt_ids = []
                for line in prompt_path.read_text(
                    encoding="utf-8"
                ).splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    prompt_ids.append(str(row["id"]))
                selected_rows = int(prompt_source["selected_row_count"])
                selected_ids = prompt_ids[:selected_rows]
                selected_hash = hashlib.sha256(
                    json.dumps(
                        selected_ids,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                if (
                    selected_rows <= 0
                    or len(selected_ids) != selected_rows
                    or len(set(prompt_ids)) != len(prompt_ids)
                    or selected_hash
                    != prompt_source["selected_ids_sha256"]
                ):
                    raise ValueError("prompt selection")
                storage = trace["expert_storage_tensors"]
                if set(storage) != {str(layer) for layer in layer_ids}:
                    raise ValueError("expert storage layers")
                seen_tensor_names = set()
                for offset, layer in enumerate(layer_ids):
                    records = storage[str(layer)]
                    if not isinstance(records, list) or not records:
                        raise ValueError("expert storage records")
                    total_bytes = 0
                    for record in records:
                        name = str(record["name"])
                        shape = [int(value) for value in record["shape"]]
                        numel = int(record["numel"])
                        element_size = int(record["element_size"])
                        size_bytes = int(record["size_bytes"])
                        layer_match = re.search(
                            r"(?:^|\.)layers\.(\d+)(?:\.|$)",
                            name,
                        )
                        if (
                            name in seen_tensor_names
                            or layer_match is None
                            or int(layer_match.group(1)) != layer
                            or "experts" not in name.split(".")
                            or not str(record["dtype"])
                            or any(value < 0 for value in shape)
                            or math.prod(shape) != numel
                            or element_size <= 0
                            or size_bytes != numel * element_size
                        ):
                            raise ValueError("expert storage tensor")
                        seen_tensor_names.add(name)
                        total_bytes += size_bytes
                    if (
                        total_bytes <= 0
                        or total_bytes % experts_expected
                        or total_bytes // experts_expected
                        != expert_bytes[offset]
                    ):
                        raise ValueError("expert storage summary")
                if trace_checkpoint.get("local") is True:
                    if not trace_checkpoint.get("weight_hashes_included"):
                        raise ValueError("unhashed source checkpoint")
                elif not trace_checkpoint.get("revision"):
                    raise ValueError("unpinned source checkpoint")
            except (
                json.JSONDecodeError,
                KeyError,
                OSError,
                TypeError,
                ValueError,
            ):
                problems.append(f"INVALID OFFLOAD SOURCE TRACE: {label}")
                trace = None

            try:
                if not isinstance(points, list) or not isinstance(trace, dict):
                    raise ValueError("curve or trace")
                trace_points = trace["points"]
                if (
                    tuple(int(point["input_tokens"]) for point in points)
                    != OFFLOAD_INPUT_LENGTHS
                    or tuple(
                        int(point["input_tokens"]) for point in trace_points
                    )
                    != OFFLOAD_INPUT_LENGTHS
                ):
                    raise ValueError("input grid")
                for point, trace_point in zip(points, trace_points):
                    if (
                        int(point["warmup_trials"]) != 2
                        or int(point["measured_trials"]) != 10
                    ):
                        raise ValueError("trial protocol")
                    trace_trials = trace_point["trials"]
                    samples = point["samples"]
                    waiting = [float(value) for value in point["waiting_ms"]]
                    if (
                        not isinstance(trace_trials, list)
                        or len(trace_trials) != 12
                        or [
                            trial.get("phase") for trial in trace_trials
                        ]
                        != ["warmup"] * 2 + ["measured"] * 10
                        or not isinstance(samples, list)
                        or len(samples) != 10
                        or len(waiting) != 10
                    ):
                        raise ValueError("trial count")
                    measured_trace = trace_trials[2:]
                    if [sample["trial_id"] for sample in samples] != [
                        trial["trial_id"] for trial in measured_trace
                    ]:
                        raise ValueError("trial identity")
                    for sample, trial, waiting_value in zip(
                        samples,
                        measured_trace,
                        waiting,
                    ):
                        active = trial["layer_active_experts"]
                        if set(active) != {
                            str(layer) for layer in layer_ids
                        }:
                            raise ValueError("trace layers")
                        misses = 0
                        transferred = 0
                        for offset, layer in enumerate(layer_ids):
                            selected = [
                                int(value) for value in active[str(layer)]
                            ]
                            if (
                                not selected
                                or selected != sorted(set(selected))
                                or any(
                                    value < 0
                                    or value >= experts_expected
                                    for value in selected
                                )
                                or len(selected)
                                > min(
                                    experts_expected,
                                    int(point["input_tokens"])
                                    * topk_expected,
                                )
                            ):
                                raise ValueError("active expert set")
                            misses += len(selected)
                            transferred += (
                                len(selected) * expert_bytes[offset]
                            )
                        wall = float(sample["waiting_ms"])
                        device_copy = float(sample["device_copy_ms"])
                        if (
                            not math.isfinite(wall)
                            or not math.isfinite(device_copy)
                            or wall < 0
                            or device_copy < 0
                            or wall != waiting_value
                            or int(sample["cache_misses"]) != misses
                            or int(sample["transferred_bytes"])
                            != transferred
                        ):
                            raise ValueError("raw sample")
                    expected_mean = sum(waiting) / len(waiting)
                    if (
                        abs(float(point["mean_waiting_ms"]) - expected_mean)
                        > max(1e-6, abs(expected_mean) * 1e-9)
                    ):
                        raise ValueError("summary")
            except (
                IndexError,
                KeyError,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ):
                problems.append(f"INVALID OFFLOAD WAITING SAMPLES: {label}")
    elif group == "routing_hotset":
        if data.get("artifact_type") != "routing_hotset_bundle":
            problems.append(f"INVALID ROUTING HOTSET TYPE: {label}")
        if claim_id is not None:
            _, model, workload, layer_text = claim_id.split(":")
            layer = int(layer_text.removeprefix("layer"))
            if data.get("paper_model") != model or data.get("layer") != layer:
                problems.append(f"ROUTING HOTSET IDENTITY MISMATCH: {label}")
        profile_protocol = data.get("profile_protocol")
        wrapper_stats = data.get("wrapper_stats")
        initialization = data.get("runtime_initialization")
        if (
            not isinstance(profile_protocol, dict)
            or profile_protocol.get("name") != "tc_routing_hotset_v1"
            or profile_protocol.get("precision_policy") != "all_low"
            or profile_protocol.get("scheduler_enabled") is not False
            or profile_protocol.get("counter")
            != "selected_token_expert_dispatches"
            or profile_protocol.get("workload_order")
            != ["wikitext", "gsm8k", "humaneval"]
            or int(profile_protocol.get("topk", 0)) != 8
        ):
            problems.append(f"INVALID ROUTING HOTSET PROTOCOL: {label}")
        if (
            not isinstance(wrapper_stats, dict)
            or wrapper_stats.get("scheduler_enabled") is not False
            or wrapper_stats.get("routing_profile_enabled") is not True
            or int(wrapper_stats.get("router_observations", 0)) <= 0
        ):
            problems.append(f"INVALID ROUTING PROFILER STATE: {label}")
        if (
            not isinstance(initialization, dict)
            or initialization.get("requested_high_precision_ratio") != 0.0
            or initialization.get("realized_high_precision_ratio") != 0.0
            or not isinstance(initialization.get("n_hi"), list)
            or len(initialization["n_hi"]) != 48
            or any(int(value) != 0 for value in initialization["n_hi"])
        ):
            problems.append(f"ROUTING PROFILE NOT ALL-LOW: {label}")
        workloads = data.get("workloads")
        required_workloads = {"wikitext", "gsm8k", "humaneval"}
        derived_top10 = {}
        if (
            not isinstance(workloads, dict)
            or set(workloads) != required_workloads
        ):
            problems.append(f"INCOMPLETE ROUTING HOTSET BUNDLE: {label}")
        else:
            try:
                for name, result in workloads.items():
                    counts = [int(value) for value in result["expert_counts"]]
                    if len(counts) != 128 or any(value < 0 for value in counts):
                        raise ValueError("counts")
                    if int(result["total_dispatches"]) != sum(counts):
                        raise ValueError("total")
                    top10 = sorted(
                        range(128),
                        key=lambda expert: (-counts[expert], expert),
                    )[:10]
                    if [int(value) for value in result["top10"]] != top10:
                        raise ValueError("top10")
                    dataset = result["dataset"]
                    summary = result["evaluation_summary"]
                    if (
                        not isinstance(dataset, dict)
                        or not dataset.get("revision")
                        or not dataset.get("fingerprint")
                        or not isinstance(summary, dict)
                    ):
                        raise ValueError("workload provenance")
                    if name == "wikitext":
                        if (
                            int(summary.get("windows", 0)) != 128
                            or int(summary.get("total_tokens", 0)) <= 0
                        ):
                            raise ValueError("wikitext scope")
                    elif (
                        int(summary.get("total", 0))
                        != {"gsm8k": 1319, "humaneval": 164}[name]
                        or int(summary.get("evaluated", -1))
                        != int(summary["total"])
                        or int(summary.get("failed", 1)) != 0
                    ):
                        raise ValueError("generation scope")
                    derived_top10[name] = set(top10)
            except (KeyError, TypeError, ValueError):
                problems.append(f"INVALID ROUTING HOTSET COUNTS: {label}")
            else:
                if any(
                    derived_top10[left] & derived_top10[right]
                    for left, right in (
                        ("wikitext", "gsm8k"),
                        ("wikitext", "humaneval"),
                        ("gsm8k", "humaneval"),
                    )
                ):
                    problems.append(f"ROUTING TOP10 NOT DISJOINT: {label}")
                if claim_id is not None and workload not in derived_top10:
                    problems.append(f"MISSING ROUTING WORKLOAD: {label}")
    elif group == "perplexity_curve":
        model = None
        if claim_id is not None:
            _, model = claim_id.split(":")
            if data.get("paper_model") != model:
                problems.append(f"PERPLEXITY CURVE MODEL MISMATCH: {label}")
        points = data.get("points")
        expected_ratios = (0, 15, 30, 45, 60, 75, 90, 100)
        if not isinstance(points, list):
            problems.append(f"MISSING PERPLEXITY CURVE: {label}")
        else:
            try:
                source_points = _load_perplexity_source_points(
                    data,
                    model=model,
                    expected_ratios=expected_ratios,
                )
                layers = 48
                experts = {"qwen30b": 128, "qwen80b": 512}[model]
                ranking = {
                    str(layer): [
                        int(value)
                        for value in data["expert_ranking"][str(layer)]
                    ]
                    for layer in range(layers)
                }
                if any(
                    len(values) != experts
                    or set(values) != set(range(experts))
                    for values in ranking.values()
                ):
                    raise ValueError("ranking permutation")
                computed_ranking_hash = hashlib.sha256(
                    json.dumps(
                        ranking,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                if tuple(
                    int(point["low_ratio_pct"]) for point in points
                ) != expected_ratios:
                    raise ValueError("ratio grid")
                ranking_hash = data["ranking_sha256"]
                if ranking_hash != computed_ranking_hash:
                    raise ValueError("ranking")
                for point in points:
                    ratio = int(point["low_ratio_pct"])
                    low_count = int(experts * ratio / 100.0)
                    low_sets = {
                        str(layer): sorted(
                            ranking[str(layer)][-low_count:]
                            if low_count
                            else []
                        )
                        for layer in range(layers)
                    }
                    low_set_hash = hashlib.sha256(
                        json.dumps(
                            low_sets,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    if (
                        int(point["low_experts_per_layer"]) != low_count
                        or point["low_experts_sha256"] != low_set_hash
                        or point.get("selection_policy")
                        != "calibrated_coldest_prefix"
                    ):
                        raise ValueError("low set")
                    source = source_points[ratio]
                    source_result = source["benchmarks"]["wikitext"]
                    if (
                        source.get("low_experts") != low_sets
                        or source.get("low_experts_sha256") != low_set_hash
                        or source.get("low_experts_per_layer") != low_count
                        or {
                            "low_ratio_pct": ratio,
                            "low_experts_per_layer": low_count,
                            "selection_policy": source["selection_policy"],
                            "low_experts_sha256": source[
                                "low_experts_sha256"
                            ],
                            "perplexity": source_result["score"],
                            "total_nll": source_result["total_nll"],
                            "total_tokens": source_result["total_tokens"],
                            "windows": source_result["windows"],
                            "window_tokens": source_result["window_tokens"],
                            "stride_tokens": source_result["stride_tokens"],
                            "window_details": source_result["window_details"],
                            "dataset": source_result["dataset"],
                        }
                        != point
                    ):
                        raise ValueError("source point content")
                    total_nll = float(point["total_nll"])
                    total_tokens = int(point["total_tokens"])
                    windows = int(point["windows"])
                    window_details = point["window_details"]
                    if (
                        total_nll < 0
                        or total_tokens <= 0
                        or not 0 < windows <= 128
                        or int(point["window_tokens"]) != 2048
                        or not isinstance(window_details, list)
                        or len(window_details) != windows
                    ):
                        raise ValueError("protocol")
                    raw_tokens = sum(
                        int(window["target_tokens"])
                        for window in window_details
                    )
                    raw_nll = sum(
                        float(window["nll"])
                        for window in window_details
                    )
                    if (
                        raw_tokens != total_tokens
                        or abs(raw_nll - total_nll)
                        > max(1e-6, abs(total_nll) * 1e-9)
                        or any(
                            int(window["window_index"]) != index
                            or int(window["begin_token"])
                            >= int(window["end_token"])
                            or int(window["target_tokens"]) <= 0
                            or float(window["mean_loss"]) < 0
                            or abs(
                                float(window["mean_loss"])
                                * int(window["target_tokens"])
                                - float(window["nll"])
                            )
                            > max(
                                1e-6,
                                abs(float(window["nll"])) * 1e-9,
                            )
                            for index, window in enumerate(window_details)
                        )
                    ):
                        raise ValueError("raw windows")
                    expected_ppl = math.exp(total_nll / total_tokens)
                    if abs(float(point["perplexity"]) - expected_ppl) > max(
                        1e-6,
                        abs(expected_ppl) * 1e-9,
                    ):
                        raise ValueError("perplexity")
                    dataset = point["dataset"]
                    if (
                        not isinstance(dataset, dict)
                        or not dataset.get("revision")
                        or not dataset.get("fingerprint")
                    ):
                        raise ValueError("dataset")
            except (
                KeyError,
                OverflowError,
                TypeError,
                ValueError,
            ):
                problems.append(f"INVALID PERPLEXITY CURVE EVIDENCE: {label}")
    elif group == "figure_bundle":
        if data.get("artifact_type") != "paper_figure_bundle":
            problems.append(f"INVALID FIGURE BUNDLE TYPE: {label}")
        inputs = data.get("inputs")
        expected_inputs = set().union(
            *(REQUIRED_CLAIMS[group] for group in FIGURE_INPUT_GROUPS)
        )
        if not isinstance(inputs, dict) or set(inputs) != expected_inputs:
            problems.append(f"INCOMPLETE FIGURE INPUT SET: {label}")
        figures = data.get("figures")
        if (
            not isinstance(figures, dict)
            or set(figures) != EXPECTED_EMPIRICAL_FIGURES
        ):
            problems.append(f"INCOMPLETE EMPIRICAL FIGURE SET: {label}")
        else:
            for relative, expected_hash in figures.items():
                path = ROOT / relative
                if not path.is_file():
                    problems.append(
                        f"MISSING RENDERED FIGURE: {label} / {relative}"
                    )
                elif _sha256(path) != expected_hash:
                    problems.append(
                        f"RENDERED FIGURE HASH MISMATCH: {label} / {relative}"
                    )
    return problems


def validate_complete_manifest() -> list[str]:
    """Verify non-accuracy tables/figures are tied to immutable artifacts."""
    if not MANIFEST.exists():
        return [f"MISSING COMPLETE RESULT MANIFEST: {MANIFEST}"]
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"INVALID RESULT MANIFEST: {type(error).__name__}: {error}"]
    problems = []
    registered_hashes: dict[str, str] = {}
    figure_inputs: dict[str, str] | None = None
    if int(manifest.get("schema_version", 0)) != 2:
        problems.append("INVALID RESULT MANIFEST SCHEMA")
    groups = manifest.get("groups", {})
    for group in sorted(REQUIRED_MANIFEST_GROUPS):
        records = groups.get(group)
        if not isinstance(records, list) or not records:
            problems.append(f"MISSING MANIFEST GROUP: {group}")
            continue
        for index, record in enumerate(records):
            label = f"{group}[{index}]"
            if not isinstance(record, dict):
                problems.append(f"INVALID MANIFEST RECORD: {label}")
                continue
            claim_id = record.get("claim_id")
            relative = record.get("path")
            expected_hash = record.get("sha256")
            command = record.get("command")
            if not claim_id or not relative or not expected_hash or not command:
                problems.append(f"INCOMPLETE MANIFEST RECORD: {label}")
                continue
            if claim_id not in REQUIRED_CLAIMS[group]:
                problems.append(f"UNEXPECTED MANIFEST CLAIM: {label} / {claim_id}")
                continue
            path = ROOT / str(relative)
            if not path.exists():
                problems.append(f"MISSING MANIFEST ARTIFACT: {label} / {path}")
                continue
            observed_hash = _sha256(path)
            if observed_hash != expected_hash:
                problems.append(
                    f"MANIFEST HASH MISMATCH: {label}: "
                    f"expected={expected_hash}, observed={observed_hash}"
                )
                continue
            registered_hashes[claim_id] = observed_hash
            try:
                artifact = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                problems.append(
                    f"INVALID MANIFEST JSON: {label}: "
                    f"{type(error).__name__}: {error}"
                )
                continue
            if not isinstance(artifact, dict):
                problems.append(f"INVALID MANIFEST ARTIFACT ROOT: {label}")
                continue
            if group == "figure_bundle" and isinstance(
                artifact.get("inputs"),
                dict,
            ):
                figure_inputs = artifact["inputs"]
            problems.extend(
                validate_manifest_artifact(group, label, artifact, claim_id)
            )
        claim_ids = [
            record.get("claim_id")
            for record in records
            if isinstance(record, dict)
        ]
        duplicates = {
            claim_id for claim_id in claim_ids if claim_ids.count(claim_id) > 1
        }
        for claim_id in sorted(duplicates):
            problems.append(f"DUPLICATE MANIFEST CLAIM: {group} / {claim_id}")
        for claim_id in sorted(REQUIRED_CLAIMS[group] - set(claim_ids)):
            problems.append(f"MISSING MANIFEST CLAIM: {claim_id}")
    expected_figure_inputs = {
        claim_id: registered_hashes.get(claim_id)
        for group in FIGURE_INPUT_GROUPS
        for claim_id in REQUIRED_CLAIMS[group]
    }
    if figure_inputs is not None and figure_inputs != expected_figure_inputs:
        problems.append("FIGURE BUNDLE INPUT HASH MISMATCH")
    return problems


def _packed_bytes(
    shapes: tuple[tuple[int, int], ...],
    experts: int,
    layers: int,
    bits: int,
    group_size: int,
) -> int:
    total = 0
    for out_features, in_features in shapes:
        if bits == 16:
            per_matrix = out_features * in_features * 2
        else:
            if in_features % group_size:
                raise AssertionError("paper model shape is not group aligned")
            payload = out_features * in_features * bits // 8
            scales = out_features * (in_features // group_size) * 2
            per_matrix = payload + scales
        total += per_matrix
    return total * experts * layers


def validate_implementation_contracts() -> list[str]:
    """Keep model claims, packed-byte arithmetic, and runtime support aligned."""
    problems: list[str] = []
    try:
        import yaml

        config = yaml.safe_load(QWEN3_NEXT_CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"INVALID QWEN3-NEXT CONFIG: {type(error).__name__}: {error}"]
    expected_model = {
        "name": "qwen3-next-80b-a3b-instruct",
        "layers": 48,
        "experts_per_layer": 512,
        "topk": 10,
    }
    if config.get("model") != expected_model:
        problems.append(
            f"QWEN3-NEXT CONFIG DRIFT: {config.get('model')!r}"
        )
    if config.get("precision", {}).get("hi") != "int4":
        problems.append("QWEN3-NEXT HI TIER IS NOT INT4")
    if config.get("precision", {}).get("lo") != "int2":
        problems.append("QWEN3-NEXT LO TIER IS NOT INT2")
    try:
        phi_config = yaml.safe_load(PHI35_CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        problems.append(
            f"INVALID PHI-3.5 CONFIG: {type(error).__name__}: {error}"
        )
    else:
        expected_phi = {
            "name": "phi-3.5-moe-instruct",
            "layers": 32,
            "experts_per_layer": 16,
            "topk": 2,
        }
        if phi_config.get("model") != expected_phi:
            problems.append(f"PHI-3.5 CONFIG DRIFT: {phi_config.get('model')!r}")
        if phi_config.get("precision", {}).get("hi") != "fp16":
            problems.append("PHI-3.5 HI TIER IS NOT FP16")
        if phi_config.get("precision", {}).get("lo") != "int4":
            problems.append("PHI-3.5 LO TIER IS NOT INT4")

    paper = PAPER.read_text(encoding="utf-8")
    implementation = IMPLEMENTATION.read_text(encoding="utf-8")
    required_fragments = (
        (paper, "Qwen/Qwen3-Next-80B-A3B-Instruct"),
        (paper, "512 (+1 shared)"),
        (paper, "Qwen3-Next shared expert remains fixed"),
        (implementation, "Qwen3-Next's shared expert remains on the fixed dense path"),
    )
    for text, fragment in required_fragments:
        if fragment not in text:
            problems.append(f"MISSING MANUSCRIPT CONTRACT: {fragment}")

    run_shift = RUN_SHIFT.read_text(encoding="utf-8")
    adapter = QWEN3_NEXT_ADAPTER.read_text(encoding="utf-8")
    if 'discovered.model_type == "qwen3_next"' not in run_shift:
        problems.append("QWEN3-NEXT LOAD PATH MISSING")
    if "Qwen3NextExperts" not in adapter or "acquire_handle" not in adapter:
        problems.append("QWEN3-NEXT HANDLE ADAPTER MISSING")

    qwen30_cache = _packed_bytes(
        ((1536, 2048), (2048, 768)),
        128,
        48,
        16,
        1,
    ) + _packed_bytes(
        ((1536, 2048), (2048, 768)),
        128,
        48,
        4,
        128,
    )
    qwen80_cache = _packed_bytes(
        ((1024, 2048), (2048, 512)),
        512,
        48,
        4,
        128,
    ) + _packed_bytes(
        ((1024, 2048), (2048, 512)),
        512,
        48,
        2,
        64,
    )
    cache_row = re.search(
        r"Pinned Expert Cache \(GB\)\s*&\s*"
        r"(?P<q30>\d+(?:\.\d+)?)\s*&\s*"
        r"(?P<q80>\d+(?:\.\d+)?)",
        paper,
    )
    if cache_row is None:
        problems.append("UNPARSED PINNED EXPERT CACHE ROW")
    else:
        expected = (qwen30_cache / 1e9, qwen80_cache / 1e9)
        observed = (
            float(cache_row.group("q30")),
            float(cache_row.group("q80")),
        )
        for model, paper_value, computed in zip(
            ("Qwen3-30B", "Qwen3-Next-80B"),
            observed,
            expected,
        ):
            if abs(paper_value - computed) > 0.05:
                problems.append(
                    f"PACKED CACHE MISMATCH: {model}: "
                    f"paper={paper_value:.1f}, computed={computed:.3f}"
                )
    return problems


def main() -> int:
    rows = manuscript_rows()
    problems: list[str] = validate_complete_manifest()
    problems.extend(validate_implementation_contracts())
    expected_rows = set(SOURCES)
    missing_rows = expected_rows - set(rows)
    unexpected_rows = set(rows) - expected_rows
    for key in sorted(missing_rows):
        problems.append(f"UNPARSED MANUSCRIPT ROW: {key[0]} / {key[1]}")
    for key in sorted(unexpected_rows):
        problems.append(f"UNREGISTERED MANUSCRIPT ROW: {key[0]} / {key[1]}")

    for key, values in sorted(rows.items()):
        source = SOURCES.get(key)
        if source is None:
            problems.append(f"MISSING SOURCE: {key[0]} / {key[1]}")
            continue
        if not source.path.exists():
            problems.append(f"MISSING FILE: {source.path}")
            continue

        data = json.loads(source.path.read_text(encoding="utf-8"))
        problems.extend(validate_artifact(key, data))
        benchmarks = source_benchmarks(source, data)
        observed: list[float] = []
        for benchmark in BENCHMARK_KEYS:
            result = None
            for candidate in BENCHMARK_ALIASES.get(benchmark, (benchmark,)):
                result = benchmarks.get(candidate)
                if result is not None:
                    break
            if result is None:
                problems.append(f"MISSING BENCHMARK: {key} / {benchmark}")
                continue
            if int(data.get("schema_version", 0)) >= 2:
                dataset = result.get("dataset")
                required_dataset_fields = {
                    "repository",
                    "revision",
                    "split",
                    "source_rows",
                    "fingerprint",
                    "evaluated_rows",
                }
                if not isinstance(dataset, dict) or not required_dataset_fields.issubset(
                    dataset
                ):
                    problems.append(
                        f"MISSING DATASET PROVENANCE: {key} / {benchmark}"
                    )
                interval = result.get("confidence_interval")
                if (
                    not isinstance(interval, dict)
                    or interval.get("method") != "wilson"
                    or not 0.0 <= float(interval.get("low", -1.0))
                    <= float(result.get("score", -1.0))
                    <= float(interval.get("high", 2.0))
                    <= 1.0
                ):
                    problems.append(
                        f"INVALID CONFIDENCE INTERVAL: {key} / {benchmark}"
                    )
            skipped = int(result.get("skipped", 0))
            if skipped:
                problems.append(
                    f"SKIPPED SAMPLES: {key} / {benchmark}: {skipped}"
                )
            total = result.get("total")
            evaluated = result.get("evaluated")
            if int(data.get("schema_version", 0)) >= 2:
                expected_total = EXPECTED_BENCHMARK_TOTALS[benchmark]
                dataset_source_rows = (
                    result.get("dataset", {}).get("source_rows")
                    if isinstance(result.get("dataset"), dict)
                    else None
                )
                expected_total = (
                    dataset_source_rows
                    if expected_total is None
                    else expected_total
                )
                if (
                    total is None
                    or expected_total is None
                    or int(total) != int(expected_total)
                ):
                    expected_label = (
                        "full_source_split"
                        if EXPECTED_BENCHMARK_TOTALS[benchmark] is None
                        else str(expected_total)
                    )
                    problems.append(
                        f"INVALID BENCHMARK SAMPLE COUNT: {key} / {benchmark}: "
                        f"expected={expected_label}, observed={total}"
                    )
            if (
                total is not None
                and evaluated is not None
                and int(evaluated) != int(total)
            ):
                problems.append(
                    f"INCOMPLETE EVALUATION: {key} / {benchmark}: "
                    f"evaluated={evaluated}, total={total}"
                )
            if "accuracy_pct" in result:
                observed.append(float(result["accuracy_pct"]))
            elif "score" in result:
                observed.append(float(result["score"]) * 100.0)
            else:
                problems.append(f"MISSING SCORE: {key} / {benchmark}")

        if len(observed) != 5 or len(values) != 6:
            problems.append(
                f"BAD TABLE WIDTH: {key}: paper={len(values)}, observed={len(observed)}"
            )
            continue
        observed.append(sum(observed) / len(observed))
        for benchmark, paper_value, raw_value in zip(
            (*BENCHMARK_KEYS, "average"), values, observed
        ):
            if abs(paper_value - raw_value) > 0.01:
                problems.append(
                    f"MISMATCH: {key} / {benchmark}: "
                    f"paper={paper_value:.2f}, raw={raw_value:.2f}"
                )

    if problems:
        print("Paper-result provenance audit FAILED")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Paper-result provenance audit passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
