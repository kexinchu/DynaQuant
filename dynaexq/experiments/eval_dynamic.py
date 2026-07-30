"""Evaluate the validated DynaExq runtime and emit provenance-rich JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..core import DynaExqConfig
from ..integration.moe_wrapper import MoEWrapper
from .eval_perf import measure_latency
from .eval_quality import (
    PAPER_PROTOCOL,
    SCHEMA_VERSION,
    checkpoint_metadata,
    environment_metadata,
    evaluate,
)
from .run_shift import initialize_dynaexq, load_model


ABLATION_CONFIGS = ("full", "static", "blocking", "no_hysteresis")
ABLATION_BENCHMARKS = (
    "mmlu_pro",
    "gpqa",
    "aime25",
    "gsm8k",
    "humaneval",
)
SENSITIVITY_RATIOS_PCT = (0, 5, 10, 15, 20, 25, 30)
PERPLEXITY_LOW_RATIOS_PCT = (0, 15, 30, 45, 60, 75, 90, 100)
ROUTING_HOTSET_WORKLOADS = ("wikitext", "gsm8k", "humaneval")
ROUTING_HOTSET_LAYER = 15
CALIBRATION_SPLITS = {"train", "validation", "dev", "calibration"}


def _paper_model_key(config: DynaExqConfig) -> str | None:
    return {
        "qwen3-moe-30b-a3b": "qwen30b",
        "qwen3-next-80b-a3b-instruct": "qwen80b",
        "phi-3.5-moe-instruct": "phi35",
    }.get(config.model.name)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ranking_sha256(ranking: dict[str, list[int]]) -> str:
    canonical = json.dumps(
        ranking,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _low_expert_set_metadata(
    ranking: dict[int, list[int]],
    *,
    low_count: int,
) -> tuple[dict[str, list[int]], str]:
    low_sets = {
        str(layer): sorted(values[-low_count:] if low_count else [])
        for layer, values in sorted(ranking.items())
    }
    digest = _sha256_bytes(
        json.dumps(
            low_sets,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return low_sets, digest


def _load_initial_map(
    path_text: str,
    checkpoint: dict,
    config: DynaExqConfig,
) -> tuple[dict[int, list[int]], dict]:
    """Load and validate a calibration-derived full expert ranking."""
    path = Path(path_text)
    try:
        payload_bytes = path.read_bytes()
        data = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot load initial map {path}: {type(error).__name__}: {error}"
        ) from error
    if (
        int(data.get("schema_version", 0)) < 2
        or data.get("artifact_type") != "dynaexq_initial_expert_ranking"
    ):
        raise ValueError("initial map is not a schema-v2 ranking artifact")
    if data.get("checkpoint") != checkpoint:
        raise ValueError("initial map checkpoint does not match this run")
    if data.get("model_config") != config.to_dict()["model"]:
        raise ValueError("initial map model contract does not match this run")
    calibration = data.get("calibration")
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
        raise ValueError("initial map lacks a valid independent calibration trace")
    environment = data.get("environment")
    git = environment.get("git", {}) if isinstance(environment, dict) else {}
    if not git.get("commit") or git.get("dirty") is not False:
        raise ValueError("initial map must be generated from a clean Git commit")
    raw_ranking = data.get("expert_ranking")
    if not isinstance(raw_ranking, dict):
        raise ValueError("initial map has no expert ranking")
    expected_layers = set(range(config.model.layers))
    try:
        ranking = {
            int(layer): [int(expert) for expert in experts]
            for layer, experts in raw_ranking.items()
        }
    except (TypeError, ValueError) as error:
        raise ValueError("initial map ranking is not integer-valued") from error
    if set(ranking) != expected_layers:
        raise ValueError("initial map does not rank every model layer")
    expected_experts = set(range(config.model.experts_per_layer))
    if any(
        len(experts) != config.model.experts_per_layer
        or set(experts) != expected_experts
        for experts in ranking.values()
    ):
        raise ValueError(
            "each initial-map layer must be a permutation of all experts"
        )
    string_ranking = {
        str(layer): ranking[layer] for layer in sorted(ranking)
    }
    ranking_hash = _ranking_sha256(string_ranking)
    if data.get("ranking_sha256") != ranking_hash:
        raise ValueError("initial map ranking hash mismatch")
    provenance = {
        "artifact_sha256": _sha256_bytes(payload_bytes),
        "ranking_sha256": ranking_hash,
        "checkpoint": data["checkpoint"],
        "model_config": data["model_config"],
        "calibration": calibration,
        "environment": environment,
        "expert_ranking": string_ranking,
    }
    return ranking, provenance


def _load_calibration_prompts(
    path_text: str,
    *,
    seed: int,
    max_prompts: int,
) -> tuple[list[tuple[str, str]], dict]:
    """Load a deterministic, non-test calibration subset with stable IDs."""
    path = Path(path_text)
    if max_prompts < 128:
        raise ValueError("calibration requires at least 128 prompts")
    records = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid calibration JSONL at line {line_number}"
                ) from error
            dataset = item.get("dataset")
            split = str(item.get("split", "")).lower()
            sample_id = item.get("id")
            prompt = item.get("prompt")
            if (
                not isinstance(dataset, str)
                or split not in CALIBRATION_SPLITS
                or sample_id is None
                or not isinstance(prompt, str)
                or not prompt.strip()
            ):
                raise ValueError(
                    "each calibration row needs dataset, a train/validation/"
                    "dev/calibration split, id, and non-empty prompt"
                )
            stable_id = f"{dataset}:{split}:{sample_id}"
            if stable_id in seen_ids:
                raise ValueError(f"duplicate calibration id: {stable_id}")
            seen_ids.add(stable_id)
            records.append((stable_id, prompt))
    if len(records) < 128:
        raise ValueError("calibration source contains fewer than 128 prompts")
    rng = random.Random(seed)
    selected_indices = sorted(
        rng.sample(range(len(records)), min(max_prompts, len(records)))
    )
    selected = [records[index] for index in selected_indices]
    selected_ids = [sample_id for sample_id, _ in selected]
    metadata = {
        "source_path": str(path.resolve()),
        "source_sha256": _sha256_file(path),
        "source_prompt_count": len(records),
        "prompt_count": len(selected),
        "selected_ids": selected_ids,
        "selected_ids_sha256": _sha256_bytes(
            json.dumps(
                selected_ids,
                separators=(",", ":"),
            ).encode("utf-8")
        ),
        "selection_seed": seed,
        "allowed_splits_only": True,
    }
    return selected, metadata


def _configure_ablation(
    config: DynaExqConfig,
    ablation_config: str | None,
) -> tuple[bool, bool]:
    """Apply one executable ablation and return runtime mode switches."""
    if ablation_config is None or ablation_config == "full":
        return False, True
    if ablation_config == "static":
        return False, False
    if ablation_config == "blocking":
        return True, True
    if ablation_config == "no_hysteresis":
        config.scheduler.delta_score_margin = 0.0
        return False, True
    raise ValueError(f"unknown ablation configuration: {ablation_config}")


def _ablation_paper_metrics(
    benchmarks: dict,
    benchmark: dict,
) -> dict[str, float]:
    """Derive every Table-IV value from the raw combined artifact."""
    scores = []
    for name in ABLATION_BENCHMARKS:
        result = benchmarks.get(name)
        if not isinstance(result, dict) or "score" not in result:
            raise RuntimeError(f"missing ablation accuracy result: {name}")
        if int(result.get("failed", 0)) != 0:
            raise RuntimeError(
                f"ablation benchmark has infrastructure failures: {name}"
            )
        scores.append(float(result["score"]))
    metrics = benchmark["metrics"]
    return {
        "average_accuracy_pct": 100.0 * sum(scores) / len(scores),
        "throughput_tokens_s": float(
            metrics["throughput_tokens_s"]["mean"]
        ),
        "p99_s": float(metrics["model_e2e_ms"]["p99"]) / 1000.0,
    }


def _quality_average_pct(benchmarks: dict) -> float:
    """Return the manuscript's unweighted five-task accuracy average."""
    scores = []
    for name in ABLATION_BENCHMARKS:
        result = benchmarks.get(name)
        if not isinstance(result, dict) or "score" not in result:
            raise RuntimeError(f"missing paper accuracy result: {name}")
        if int(result.get("failed", 0)) != 0:
            raise RuntimeError(
                f"paper benchmark has infrastructure failures: {name}"
            )
        scores.append(float(result["score"]))
    return 100.0 * sum(scores) / len(scores)


def _runtime_overhead_paper_metrics(
    initialization: dict,
    wrapper_stats: dict,
    transition_stats: dict,
    benchmark: dict,
    device_budget_bytes: int,
) -> dict[str, float | int]:
    """Derive the runtime-overhead table from raw telemetry."""
    samples = benchmark.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("runtime overhead requires raw performance samples")
    peak_process_hbm = max(
        float(sample["process_hbm_used_peak_bytes"]) for sample in samples
    )
    if peak_process_hbm > device_budget_bytes:
        raise RuntimeError(
            "measured whole-process HBM high-water exceeds the declared budget"
        )
    return {
        "hbm_budget_gb": device_budget_bytes / 1e9,
        "peak_process_hbm_used_gb": peak_process_hbm / 1e9,
        "resident_expert_pool_gb": (
            float(initialization["resident_expert_bytes"]) / 1e9
        ),
        "transient_expert_pool_gb": (
            float(initialization["transient_expert_bytes"]) / 1e9
        ),
        "migration_count": (
            int(transition_stats["total_promotions"])
            + int(transition_stats["total_demotions"])
        ),
        "transferred_gb": float(transition_stats["copied_bytes"]) / 1e9,
        "scheduler_mean_ms": float(wrapper_stats["scheduler_mean_ms"]),
        "scheduler_p99_ms": float(wrapper_stats["scheduler_p99_ms"]),
        "pinned_expert_cache_gb": (
            float(initialization["host_cache"]["host_packed_bytes"]) / 1e9
        ),
    }


def _validate_formal_runtime_final_state(
    wrapper_stats: dict,
    transition_stats: dict,
    *,
    scheduler_enabled: bool,
    require_online_activity: bool,
) -> None:
    """Fail before serialization when a formal runtime trace is incomplete."""
    samples = wrapper_stats.get("scheduler_update_samples_ms")
    count = wrapper_stats.get("scheduler_update_count")
    if (
        wrapper_stats.get("scheduler_enabled") is not scheduler_enabled
        or not isinstance(samples, list)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count != len(samples)
        or (scheduler_enabled and count <= 0)
        or (not scheduler_enabled and count != 0)
    ):
        raise RuntimeError("formal run has invalid scheduler telemetry")

    try:
        accepted = int(transition_stats["accepted_requests"])
        promotions = int(transition_stats["total_promotions"])
        demotions = int(transition_stats["total_demotions"])
        failed = int(transition_stats["failed_transitions"])
        copied_bytes = int(transition_stats["copied_bytes"])
        accepted_bytes = int(transition_stats["accepted_bytes"])
        precise_reclaims = int(transition_stats["precise_fence_reclaims"])
        global_reclaims = int(transition_stats["global_sync_reclaims"])
        active = int(transition_stats["active_transitions"])
        budget = transition_stats["budget"]
        cap = int(budget["total_cap"])
        live = int(budget["total_live"])
        hi_pending = int(budget["hi_pending"])
        lo_pending = int(budget["lo_pending"])
        staging_used = int(budget["staging_used"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "formal run has incomplete transition lifecycle telemetry"
        ) from error

    if (
        min(
            accepted,
            promotions,
            demotions,
            failed,
            copied_bytes,
            accepted_bytes,
            precise_reclaims,
            global_reclaims,
            active,
            live,
            hi_pending,
            lo_pending,
            staging_used,
        )
        < 0
        or cap < 0
        or live > cap
        or failed != 0
        or accepted != promotions + demotions
        or active != 0
        or hi_pending != 0
        or lo_pending != 0
        or staging_used != 0
        or global_reclaims != 0
        or (
            accepted > 0
            and (
                copied_bytes <= 0
                or accepted_bytes <= 0
                or precise_reclaims <= 0
            )
        )
        or (not scheduler_enabled and accepted != 0)
        or (require_online_activity and accepted <= 0)
    ):
        raise RuntimeError(
            "formal run did not finish in an auditable transition state"
        )


def _collect_calibration_ranking(
    wrapper: MoEWrapper,
    tokenizer,
    tracker,
    prompts: list[tuple[str, str]],
    config: DynaExqConfig,
    *,
    max_input_tokens: int,
) -> tuple[dict[str, list[int]], dict[str, float | int]]:
    """Observe routing on independent prompts and rank every layer."""
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")
    if not prompts:
        raise ValueError("calibration prompts must not be empty")
    try:
        device = next(wrapper.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    wrapper.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for prompt_index, (_, prompt) in enumerate(prompts, start=1):
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            )
            wrapper(
                input_ids=encoded.input_ids.to(device),
                attention_mask=encoded.attention_mask.to(device),
                use_cache=False,
            )
            if prompt_index % 8 == 0 or prompt_index == len(prompts):
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "stage": "calibration_forward",
                            "completed_prompts": prompt_index,
                            "total_prompts": len(prompts),
                            "elapsed_seconds": elapsed,
                            "mean_seconds_per_prompt": elapsed / prompt_index,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    ranking = {}
    for layer in range(config.model.layers):
        scores = tracker.get_cumulative_layer_scores(layer)
        if len(scores) != config.model.experts_per_layer:
            raise RuntimeError(
                f"calibration score width mismatch at layer {layer}"
            )
        if float(scores.sum()) <= 0.0:
            raise RuntimeError(
                f"calibration produced no routing mass at layer {layer}"
            )
        ranking[str(layer)] = sorted(
            range(config.model.experts_per_layer),
            key=lambda expert: (-float(scores[expert]), expert),
        )
    elapsed = time.perf_counter() - started
    return ranking, {
        "completed_prompt_count": len(prompts),
        "forward_elapsed_seconds": elapsed,
        "mean_seconds_per_prompt": elapsed / len(prompts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Exactly one execution device (for example cuda:0)",
    )
    parser.add_argument("--hash-model-files", action="store_true")
    parser.add_argument(
        "--initial-map",
        help=(
            "Schema-v2 calibration ranking artifact. Required by formal "
            "dynamic paper runs."
        ),
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    quality = subparsers.add_parser("quality")
    quality.add_argument("--benchmarks", required=True)
    quality.add_argument("--n-samples", type=int, default=None)
    quality.add_argument(
        "--paper-protocol",
        action="store_true",
        help="Use the exact per-benchmark sampling protocol reported in the paper",
    )
    quality.add_argument("--allow-code-execution", action="store_true")

    perf = subparsers.add_parser("perf")
    perf.add_argument("--batch-size", type=int, default=1)
    perf.add_argument("--input-length", type=int, default=128)
    perf.add_argument("--output-length", type=int, default=64)
    perf.add_argument("--n-warmup", type=int, default=5)
    perf.add_argument("--n-repeats", type=int, default=100)
    perf.add_argument(
        "--paper-protocol",
        action="store_true",
        help=(
            "Enforce the exact TC latency grid, calibrated initial map, and "
            "whole-process NVML HBM high-water monitoring"
        ),
    )
    ablation = subparsers.add_parser(
        "ablation",
        help=(
            "Run the journal ablation trace: five full quality benchmarks "
            "followed by the bs=32 performance protocol"
        ),
    )
    ablation.add_argument(
        "--ablation-config",
        required=True,
        choices=ABLATION_CONFIGS,
    )
    ablation.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Required opt-in for official HumanEval execution",
    )
    sensitivity = subparsers.add_parser(
        "sensitivity",
        help="Run one exact high-precision-ratio point for Figure 8",
    )
    sensitivity.add_argument(
        "--hi-ratio-pct",
        required=True,
        type=int,
        choices=SENSITIVITY_RATIOS_PCT,
    )
    sensitivity.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Required opt-in for official HumanEval execution",
    )
    overhead = subparsers.add_parser(
        "overhead",
        help=(
            "Run the pinned shift trace followed by the bs=32 performance "
            "protocol and derive the runtime-overhead table"
        ),
    )
    overhead.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Required opt-in for official HumanEval execution",
    )
    calibrate = subparsers.add_parser(
        "calibrate",
        help="Create a full expert ranking from independent calibration JSONL",
    )
    calibrate.add_argument("--prompts", required=True)
    calibrate.add_argument("--max-prompts", type=int, default=256)
    calibrate.add_argument("--max-input-tokens", type=int, default=2048)
    perplexity_point = subparsers.add_parser(
        "perplexity-point",
        help="Run one frozen coldest-prefix WikiText perplexity point",
    )
    perplexity_point.add_argument(
        "--low-ratio-pct",
        required=True,
        type=int,
        choices=PERPLEXITY_LOW_RATIOS_PCT,
    )
    routing_hotset = subparsers.add_parser(
        "routing-hotset",
        help=(
            "Collect exact layer-15 selected-expert counts for the three "
            "motivation workloads under a frozen all-low map"
        ),
    )
    routing_hotset.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Required opt-in for official HumanEval execution",
    )
    args = parser.parse_args()
    if (
        args.mode == "quality"
        and args.paper_protocol
        and args.n_samples is not None
    ):
        parser.error("--paper-protocol and --n-samples are mutually exclusive")
    if (
        args.mode == "quality"
        and args.paper_protocol
        and args.seed != PAPER_PROTOCOL["seed"]
    ):
        parser.error(
            f"--paper-protocol requires --seed={PAPER_PROTOCOL['seed']}"
        )
    if args.mode == "perf" and args.paper_protocol:
        expected = {
            "input_length": 2048,
            "output_length": 256,
            "n_warmup": 5,
            "n_repeats": 100,
            "seed": PAPER_PROTOCOL["seed"],
        }
        if args.batch_size not in (1, 2, 4, 8, 16, 32):
            parser.error(
                "perf --paper-protocol requires batch size in "
                "1,2,4,8,16,32"
            )
        mismatches = [
            f"--{name.replace('_', '-')}={getattr(args, name)}"
            for name, value in expected.items()
            if getattr(args, name) != value
        ]
        if mismatches:
            parser.error(
                "perf --paper-protocol has incompatible arguments: "
                + ", ".join(mismatches)
            )
    if (
        args.mode in {
            "ablation",
            "sensitivity",
            "overhead",
            "routing-hotset",
        }
        and args.seed != PAPER_PROTOCOL["seed"]
    ):
        parser.error(
            f"{args.mode} mode requires --seed={PAPER_PROTOCOL['seed']}"
        )
    if (
        args.mode in {
            "ablation",
            "sensitivity",
            "overhead",
            "routing-hotset",
        }
        and not args.allow_code_execution
    ):
        parser.error(
            f"{args.mode} mode requires --allow-code-execution for HumanEval"
        )
    if args.mode == "calibrate" and args.seed != PAPER_PROTOCOL["seed"]:
        parser.error(
            f"calibrate mode requires --seed={PAPER_PROTOCOL['seed']}"
        )
    if (
        args.mode == "perplexity-point"
        and args.seed != PAPER_PROTOCOL["seed"]
    ):
        parser.error(
            f"perplexity-point mode requires --seed={PAPER_PROTOCOL['seed']}"
        )

    torch.manual_seed(args.seed)
    config = DynaExqConfig.from_yaml(args.config)
    ablation_config = (
        args.ablation_config if args.mode == "ablation" else None
    )
    transition_synchronous, scheduler_enabled = _configure_ablation(
        config,
        ablation_config,
    )
    if args.mode == "calibrate":
        scheduler_enabled = False
    if args.mode == "perplexity-point":
        scheduler_enabled = False
    if args.mode == "routing-hotset":
        scheduler_enabled = False
    high_precision_ratio = None
    if args.mode == "sensitivity":
        high_precision_ratio = args.hi_ratio_pct / 100.0
    elif args.mode in {"calibrate", "routing-hotset"}:
        high_precision_ratio = 0.0
    elif args.mode == "perplexity-point":
        high_precision_ratio = (
            (
                config.model.experts_per_layer
                - int(
                    config.model.experts_per_layer
                    * args.low_ratio_pct
                    / 100.0
                )
            )
            / config.model.experts_per_layer
        )
    paper_model = _paper_model_key(config)
    if args.mode == "routing-hotset" and (
        paper_model != "qwen30b"
        or config.model.experts_per_layer != 128
        or config.model.layers <= ROUTING_HOTSET_LAYER
    ):
        parser.error(
            "routing-hotset is the pinned Qwen3-30B layer-15 experiment"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(f"CUDA device requested but CUDA is unavailable: {args.device}")
    checkpoint = checkpoint_metadata(
        args.model_path,
        hash_weight_files=args.hash_model_files,
    )
    revision = checkpoint.get("revision")
    if checkpoint.get("local") is False and not revision:
        parser.error(
            "remote checkpoint revision could not be resolved; refusing an "
            "unpinned paper run"
        )
    if (
        (
            args.mode in {
                "ablation",
                "sensitivity",
                "overhead",
                "calibrate",
                "perplexity-point",
                "routing-hotset",
            }
            or (args.mode == "quality" and args.paper_protocol)
            or (args.mode == "perf" and args.paper_protocol)
        )
        and checkpoint.get("local") is True
        and not checkpoint.get("weight_hashes_included")
    ):
        parser.error(
            "paper protocol runs require --hash-model-files for a local checkpoint"
        )
    formal_result_run = (
        args.mode in {
            "ablation",
            "sensitivity",
            "overhead",
            "perplexity-point",
            "routing-hotset",
        }
        or (args.mode == "quality" and args.paper_protocol)
        or (args.mode == "perf" and args.paper_protocol)
    )
    requires_initial_map = formal_result_run and args.mode != "routing-hotset"
    if requires_initial_map and not args.initial_map:
        parser.error(
            "formal dynamic paper runs require --initial-map from an "
            "independent calibration trace"
        )
    initial_ranking = None
    initial_map_provenance = None
    if args.initial_map:
        try:
            initial_ranking, initial_map_provenance = _load_initial_map(
                args.initial_map,
                checkpoint,
                config,
            )
        except ValueError as error:
            parser.error(str(error))
    model, tokenizer = load_model(
        args.model_path,
        device,
        revision=revision,
    )
    (
        observer,
        tracker,
        scheduler,
        registry,
        transition_engine,
        _,
        initialization,
    ) = initialize_dynaexq(
        config,
        model,
        device,
        transition_synchronous=transition_synchronous,
        high_precision_ratio=high_precision_ratio,
        initial_expert_ranking=initial_ranking,
    )
    wrapper = MoEWrapper(
        model=model,
        router_observer=observer,
        hotness_tracker=tracker,
        scheduler=scheduler,
        registry=registry,
        transition_engine=transition_engine,
        num_layers=config.model.layers,
        experts_per_layer=config.model.experts_per_layer,
        topk=config.model.topk,
        scheduler_enabled=scheduler_enabled,
        routing_profile_enabled=args.mode == "routing-hotset",
    )

    engine_shutdown = False
    try:
        wrapper.validate_integration()
        if args.mode == "calibrate":
            try:
                calibration_prompts, calibration_metadata = (
                    _load_calibration_prompts(
                        args.prompts,
                        seed=args.seed,
                        max_prompts=args.max_prompts,
                    )
                )
            except (OSError, ValueError) as error:
                parser.error(str(error))
            calibration_metadata["max_input_tokens"] = args.max_input_tokens
            calibration_metadata["precision_policy"] = "all_low"
            calibration_metadata["aggregation"] = (
                "mean_per_prompt_routing_probability_mass"
            )
            expert_ranking, calibration_runtime = _collect_calibration_ranking(
                wrapper,
                tokenizer,
                tracker,
                calibration_prompts,
                config,
                max_input_tokens=args.max_input_tokens,
            )
            calibration_metadata["runtime"] = calibration_runtime
            result_payload = {
                "artifact_type": "dynaexq_initial_expert_ranking",
                "model_config": config.to_dict()["model"],
                "calibration": calibration_metadata,
                "expert_ranking": expert_ranking,
                "ranking_sha256": _ranking_sha256(expert_ranking),
            }
        elif args.mode == "perplexity-point":
            result = evaluate(
                wrapper,
                tokenizer,
                ["wikitext"],
                sample_limits=PAPER_PROTOCOL["sample_limits"],
                wikitext_max_windows=PAPER_PROTOCOL[
                    "wikitext_max_windows"
                ],
                allow_code_execution=False,
            )
            result_payload = {"benchmarks": result}
        elif args.mode == "routing-hotset":
            workloads = {}
            for workload in ROUTING_HOTSET_WORKLOADS:
                wrapper.reset_routing_profile()
                result = evaluate(
                    wrapper,
                    tokenizer,
                    [workload],
                    sample_limits=PAPER_PROTOCOL["sample_limits"],
                    wikitext_max_windows=PAPER_PROTOCOL[
                        "wikitext_max_windows"
                    ],
                    allow_code_execution=True,
                )[workload]
                profile = wrapper.get_routing_profile()
                if ROUTING_HOTSET_LAYER not in profile:
                    raise RuntimeError(
                        f"no routing profile for layer {ROUTING_HOTSET_LAYER}"
                    )
                counts = profile[ROUTING_HOTSET_LAYER]
                if (
                    len(counts) != config.model.experts_per_layer
                    or sum(counts) <= 0
                ):
                    raise RuntimeError(
                        f"invalid routing counts for {workload}"
                    )
                top10 = sorted(
                    range(config.model.experts_per_layer),
                    key=lambda expert: (-counts[expert], expert),
                )[:10]
                workloads[workload] = {
                    "expert_counts": counts,
                    "total_dispatches": sum(counts),
                    "top10": top10,
                    "dataset": result["dataset"],
                    "request_limit": result["request_limit"],
                    "evaluation_summary": {
                        key: result[key]
                        for key in (
                            "metric",
                            "total",
                            "evaluated",
                            "failed",
                            "windows",
                            "total_tokens",
                        )
                        if key in result
                    },
                }
            result_payload = {
                "artifact_type": "routing_hotset_bundle",
                "layer": ROUTING_HOTSET_LAYER,
                "profile_protocol": {
                    "name": "tc_routing_hotset_v1",
                    "precision_policy": "all_low",
                    "scheduler_enabled": False,
                    "counter": "selected_token_expert_dispatches",
                    "topk": config.model.topk,
                    "workload_order": list(ROUTING_HOTSET_WORKLOADS),
                },
                "workloads": workloads,
            }
        elif args.mode == "quality":
            benchmarks = [
                name.strip()
                for name in args.benchmarks.split(",")
                if name.strip()
            ]
            result = evaluate(
                wrapper,
                tokenizer,
                benchmarks,
                n_samples=args.n_samples,
                sample_limits=(
                    PAPER_PROTOCOL["sample_limits"]
                    if args.paper_protocol
                    else None
                ),
                wikitext_max_windows=(
                    PAPER_PROTOCOL["wikitext_max_windows"]
                    if args.paper_protocol
                    else 128
                ),
                allow_code_execution=args.allow_code_execution,
            )
            result_payload = {"benchmarks": result}
        elif args.mode == "perf":
            result = measure_latency(
                wrapper,
                tokenizer,
                batch_size=args.batch_size,
                input_length=args.input_length,
                output_length=args.output_length,
                n_warmup=args.n_warmup,
                n_repeats=args.n_repeats,
                require_process_hbm_monitor=args.paper_protocol,
            )
            result_payload = {"benchmark": result}
        elif args.mode == "ablation":
            quality_result = evaluate(
                wrapper,
                tokenizer,
                list(ABLATION_BENCHMARKS),
                sample_limits=PAPER_PROTOCOL["sample_limits"],
                wikitext_max_windows=PAPER_PROTOCOL[
                    "wikitext_max_windows"
                ],
                allow_code_execution=True,
            )
            performance_result = measure_latency(
                wrapper,
                tokenizer,
                batch_size=32,
                input_length=2048,
                output_length=256,
                n_warmup=5,
                n_repeats=100,
                require_process_hbm_monitor=True,
            )
            result_payload = {
                "benchmarks": quality_result,
                "benchmark": performance_result,
                "paper_metrics": _ablation_paper_metrics(
                    quality_result,
                    performance_result,
                ),
                "ablation_sequence": [
                    *ABLATION_BENCHMARKS,
                    "performance_bs32",
                ],
            }
        elif args.mode == "sensitivity":
            quality_result = evaluate(
                wrapper,
                tokenizer,
                list(ABLATION_BENCHMARKS),
                sample_limits=PAPER_PROTOCOL["sample_limits"],
                wikitext_max_windows=PAPER_PROTOCOL[
                    "wikitext_max_windows"
                ],
                allow_code_execution=True,
            )
            result_payload = {
                "benchmarks": quality_result,
                "paper_metrics": {
                    "average_accuracy_pct": _quality_average_pct(
                        quality_result
                    ),
                    "realized_hi_ratio_pct": (
                        100.0
                        * initialization["realized_high_precision_ratio"]
                    ),
                    "resident_expert_bytes": initialization[
                        "resident_expert_bytes"
                    ],
                },
                "sensitivity_sequence": list(ABLATION_BENCHMARKS),
            }
        else:
            quality_result = evaluate(
                wrapper,
                tokenizer,
                list(ABLATION_BENCHMARKS),
                sample_limits=PAPER_PROTOCOL["sample_limits"],
                wikitext_max_windows=PAPER_PROTOCOL[
                    "wikitext_max_windows"
                ],
                allow_code_execution=True,
            )
            performance_result = measure_latency(
                wrapper,
                tokenizer,
                batch_size=32,
                input_length=2048,
                output_length=256,
                n_warmup=5,
                n_repeats=100,
                require_process_hbm_monitor=True,
            )
            result_payload = {
                "benchmarks": quality_result,
                "benchmark": performance_result,
                "overhead_sequence": [
                    *ABLATION_BENCHMARKS,
                    "performance_bs32",
                ],
            }

        # No artifact may snapshot a still-mutating transition executor.
        # Draining here captures late failures and final pool/budget counters.
        transition_engine.shutdown()
        engine_shutdown = True
        final_transition_stats = transition_engine.get_stats()
        wrapper_stats = wrapper.get_stats()
        if formal_result_run:
            _validate_formal_runtime_final_state(
                wrapper_stats,
                final_transition_stats,
                scheduler_enabled=scheduler_enabled,
                require_online_activity=(
                    scheduler_enabled
                    and not (
                        args.mode == "sensitivity"
                        and args.hi_ratio_pct == 0
                    )
                ),
            )
        if args.mode == "overhead":
            result_payload["paper_metrics"] = (
                _runtime_overhead_paper_metrics(
                    initialization,
                    wrapper_stats,
                    final_transition_stats,
                    result_payload["benchmark"],
                    config.memory.device_mem_bytes,
                )
            )
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": f"dynaexq_{args.mode}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "method": "dynaexq",
            "paper_model": paper_model,
            "paper_method": "dynaexq",
            "device": str(device),
            "model": args.model_path,
            "checkpoint": checkpoint,
            "config": config.to_dict(),
            "seed": args.seed,
            "evaluation_protocol": (
                {"name": "independent_calibration_v1"}
                if args.mode == "calibrate"
                else {
                    "name": "tc_isolated_performance_v2",
                    "seed": PAPER_PROTOCOL["seed"],
                    "process_hbm_high_water": True,
                }
                if args.mode == "perf" and args.paper_protocol
                else PAPER_PROTOCOL
                if formal_result_run
                else {"name": "custom"}
            ),
            "runtime_initialization": initialization,
            **result_payload,
            "wrapper_stats": wrapper_stats,
            "transition_stats": final_transition_stats,
            "environment": environment_metadata(),
        }
        if initial_map_provenance is not None:
            artifact["initial_map"] = initial_map_provenance
        if args.mode == "calibrate":
            artifact["method"] = "dynaexq_calibration"
            artifact["paper_method"] = None
        if ablation_config is not None:
            artifact["ablation_config"] = ablation_config
        if args.mode == "sensitivity":
            artifact["hi_ratio_pct"] = args.hi_ratio_pct
        if args.mode == "perplexity-point":
            if initial_ranking is None:
                raise AssertionError("formal perplexity run lacks initial map")
            low_count = int(
                config.model.experts_per_layer
                * args.low_ratio_pct
                / 100.0
            )
            low_sets, low_set_hash = _low_expert_set_metadata(
                initial_ranking,
                low_count=low_count,
            )
            artifact.update(
                {
                    "artifact_type": "dynaexq_perplexity_point",
                    "low_ratio_pct": args.low_ratio_pct,
                    "low_experts_per_layer": low_count,
                    "selection_policy": "calibrated_coldest_prefix",
                    "low_experts_sha256": low_set_hash,
                    "low_experts": low_sets,
                }
            )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(output),
                    "method": "dynaexq",
                    "mode": args.mode,
                    "transition_stats": {
                        key: value
                        for key, value in artifact["transition_stats"].items()
                        if key != "stage_timings"
                    },
                },
                indent=2,
            )
        )
    finally:
        wrapper.remove_hooks()
        if not engine_shutdown:
            transition_engine.shutdown()


if __name__ == "__main__":
    main()
