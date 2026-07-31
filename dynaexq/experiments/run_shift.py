"""
Run shift benchmark experiment.

Implements workload shift experiment from the guide:
- Phase A: WikiText-like prompts
- Phase B: GSM8K + AIME-style math prompts
- Phase C: HumanEval-style code prompts
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..core import (
    BudgetInitializer,
    BudgetTracker,
    DynaExqConfig,
    ExpertKey,
    ExpertRegistry,
    HotnessTracker,
    ModelWeightStore,
    PrecisionScheduler,
    RouterObserver,
    Tier,
    TransitionReq,
    TransitionEngine,
)
from ..core.quant import budget_safe_dispatch_available
from ..integration.generation_utils import (
    last_logit_only_kwargs,
    prepare_one_token_decode,
)
from ..integration.moe_wrapper import MoEWrapper
from .metrics import MetricsCollector
from .metrics import LatencyMetrics
from .workloads import PhaseConfig, WorkloadStream
from .eval_quality import checkpoint_metadata, environment_metadata


def load_model(
    model_path: str,
    device: torch.device,
    *,
    revision: str | None = None,
):
    """Load one immutable checkpoint through a DynaExq-capable adapter."""
    try:
        from transformers import AutoConfig, AutoTokenizer
        
        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=False,
        )
        discovered = AutoConfig.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=False,
        )
        if discovered.model_type == "qwen3_moe":
            from ..models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

            model_config = Qwen3MoeConfig.from_pretrained(
                model_path,
                revision=revision,
            )
            model_class = Qwen3MoeForCausalLM
        elif discovered.model_type == "qwen3_next":
            try:
                from transformers import (
                    Qwen3NextConfig,
                    Qwen3NextForCausalLM,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "this Qwen3-Next checkpoint requires a Transformers "
                    "release that exports Qwen3NextConfig and "
                    "Qwen3NextForCausalLM"
                ) from exc

            model_config = Qwen3NextConfig.from_pretrained(
                model_path,
                revision=revision,
            )
            model_class = Qwen3NextForCausalLM
        elif discovered.model_type in {"phimoe", "phi_moe"}:
            from ..models.phimoe import PhimoeConfig, PhimoeForCausalLM

            model_config = PhimoeConfig.from_pretrained(
                model_path,
                revision=revision,
            )
            model_class = PhimoeForCausalLM
        else:
            raise ValueError(
                f"unsupported model_type={discovered.model_type!r}; "
                "a DynaExq handle adapter is required"
            )
        model = model_class.from_pretrained(
            model_path,
            config=model_config,
            revision=revision,
            dtype=torch.float16,
            device_map={"": "cpu"},
        )
        model.eval()
        print("Model loaded successfully")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def _validate_model_contract(
    config: DynaExqConfig,
    model: torch.nn.Module,
) -> None:
    """Reject a runtime YAML that disagrees with checkpoint architecture."""
    model_config = getattr(model, "config", None)
    if model_config is None:
        return
    discovered = {
        "layers": getattr(model_config, "num_hidden_layers", None),
        "experts_per_layer": getattr(
            model_config,
            "num_experts",
            getattr(model_config, "num_local_experts", None),
        ),
        "topk": getattr(model_config, "num_experts_per_tok", None),
    }
    declared = {
        "layers": config.model.layers,
        "experts_per_layer": config.model.experts_per_layer,
        "topk": config.model.topk,
    }
    mismatches = {
        field: (declared[field], observed)
        for field, observed in discovered.items()
        if observed is not None and int(observed) != declared[field]
    }
    if mismatches:
        detail = ", ".join(
            f"{field}: yaml={yaml_value}, checkpoint={checkpoint_value}"
            for field, (yaml_value, checkpoint_value) in mismatches.items()
        )
        raise ValueError(f"runtime model contract mismatch ({detail})")


def initialize_dynaexq(
    config: DynaExqConfig,
    model: torch.nn.Module,
    device: torch.device,
    *,
    transition_synchronous: bool = False,
    high_precision_ratio: float | None = None,
    initial_expert_ranking: dict[int, list[int]] | None = None,
) -> tuple[
    RouterObserver,
    HotnessTracker,
    PrecisionScheduler,
    ExpertRegistry,
    TransitionEngine,
    BudgetInitializer,
    dict,
]:
    """Initialize DynaExq components."""
    print("Initializing DynaExq components...")
    _validate_model_contract(config, model)
    
    # RouterObserver
    observer = RouterObserver(use_probabilities=True)
    
    # HotnessTracker
    tracker = HotnessTracker(
        num_layers=config.model.layers,
        experts_per_layer=config.model.experts_per_layer,
        alpha=config.scheduler.alpha,
    )
    
    # Derive byte-accurate footprints from the real expert projections.
    weight_store = ModelWeightStore(
        model=model,
        hi_format=config.precision.hi,
        lo_format=config.precision.lo,
        backend=config.precision.backend,
        pin_memory=torch.cuda.is_available(),
        enable_int4_kernel_cache=device.type == "cuda",
    )
    if device.type == "cuda":
        for fmt in {weight_store.hi_fmt, weight_store.lo_fmt}:
            if not budget_safe_dispatch_available(fmt, device):
                raise RuntimeError(
                    f"{fmt.value} has no budget-safe CUDA dispatch kernel; "
                    "refusing to run a paper experiment that would "
                    "materialize full dequantized expert weights"
                )
    host_cache = weight_store.preload_and_release_all(
        config.model.layers,
        config.model.experts_per_layer,
    )
    released_source_bytes = host_cache["released_native_expert_bytes"]
    dense_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    dense_buffer_bytes = sum(
        buffer.numel() * buffer.element_size()
        for buffer in model.buffers()
    )
    actual_dense_bytes = dense_parameter_bytes + dense_buffer_bytes
    if device.type == "cuda":
        model.to(device)
        nonempty_devices = {
            parameter.device
            for parameter in model.parameters()
            if parameter.numel()
        }
        if len(nonempty_devices) != 1 or next(iter(nonempty_devices)).type != "cuda":
            raise RuntimeError(
                f"model is not resident on exactly one CUDA device: "
                f"{sorted(map(str, nonempty_devices))}"
            )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def memory_footprint_fn(layer: int, tier: Tier) -> int:
        return weight_store.get_byte_size(ExpertKey(layer, 0), tier)

    # CUDA's INT4 conversion creates a temporary kernel-native tensor before
    # copying it into the pool-backed block. Reserve a deliberately
    # conservative two-full-expert workspace per admitted transition. This
    # upper-bounds the simultaneously live nibble-swapped input, native
    # output, and scale/zero construction tensors before pool rebinding.
    int4_tiers = [
        tier
        for tier, fmt in (
            (Tier.HI, weight_store.hi_fmt),
            (Tier.LO, weight_store.lo_fmt),
        )
        if fmt.value == "int4"
    ]
    uses_int4_native_layout = device.type == "cuda" and bool(int4_tiers)
    automatic_kernel_workspace_bytes = (
        2
        * config.memory.max_inflight
        * max(
            memory_footprint_fn(layer, tier)
            for layer in range(config.model.layers)
            for tier in int4_tiers
        )
        if uses_int4_native_layout
        else 0
    )
    effective_kernel_workspace_bytes = max(
        config.memory.reserve_kernel_workspace_bytes,
        automatic_kernel_workspace_bytes,
    )
    
    budget_init = BudgetInitializer(
        num_layers=config.model.layers,
        experts_per_layer=config.model.experts_per_layer,
        memory_footprint_fn=memory_footprint_fn,
        device_mem_bytes=config.memory.device_mem_bytes,
        reserve_kv_bytes=config.memory.reserve_kv_bytes,
        reserve_act_bytes=config.memory.reserve_act_bytes,
        reserve_dense_bytes=max(
            config.memory.reserve_dense_bytes,
            actual_dense_bytes,
        ),
        reserve_runtime_bytes=effective_kernel_workspace_bytes,
        safety_margin_bytes=config.memory.safety_margin_bytes,
        max_inflight=config.memory.max_inflight,
    )
    
    budget_result = budget_init.compute(
        strategy="proportional",
        high_precision_ratio=high_precision_ratio,
    )
    print(f"Budget initialized: n_hi[0]={budget_result.n_hi[0]}, "
          f"total_expert_bytes={budget_result.total_expert_bytes / 1024**3:.2f}GB")

    if initial_expert_ranking is not None:
        expected_layers = set(range(config.model.layers))
        if set(initial_expert_ranking) != expected_layers:
            raise ValueError(
                "initial expert ranking must contain exactly every model layer"
            )
        expected_experts = set(range(config.model.experts_per_layer))
        for layer, ranking in initial_expert_ranking.items():
            if (
                len(ranking) != config.model.experts_per_layer
                or set(ranking) != expected_experts
            ):
                raise ValueError(
                    f"initial expert ranking for layer {layer} is not a "
                    "permutation of all expert ids"
                )
    
    # PrecisionScheduler
    scheduler = PrecisionScheduler(
        num_layers=config.model.layers,
        experts_per_layer=config.model.experts_per_layer,
        n_hi=budget_result.n_hi,
        update_period_steps=config.scheduler.update_period_steps,
        rate_limit=config.scheduler.rate_limit,
        delta_score_margin=config.scheduler.delta_score_margin,
    )
    
    # ExpertRegistry
    registry = ExpertRegistry()
    
    # Pools use one exact expert-sized block per resident slot.
    from ..core.memory_pool import PoolAllocator
    hi_block_sizes = [
        memory_footprint_fn(layer, Tier.HI)
        for layer in range(config.model.layers)
    ]
    lo_block_sizes = [
        memory_footprint_fn(layer, Tier.LO)
        for layer in range(config.model.layers)
    ]
    hi_pool_sizes = [
        budget_result.n_hi[layer] * hi_block_sizes[layer]
        for layer in range(config.model.layers)
    ]
    lo_pool_sizes = [
        (config.model.experts_per_layer - budget_result.n_hi[layer])
        * lo_block_sizes[layer]
        for layer in range(config.model.layers)
    ]
    pool_allocation_bytes = (
        sum(hi_pool_sizes)
        + sum(lo_pool_sizes)
        + budget_result.transient_bytes
    )
    cuda_free_before_pools = None
    cuda_total_bytes = None
    if device.type == "cuda":
        cuda_free_before_pools, cuda_total_bytes = torch.cuda.mem_get_info(device)
        required_free = (
            pool_allocation_bytes
            + config.memory.reserve_kv_bytes
            + config.memory.reserve_act_bytes
            + effective_kernel_workspace_bytes
            + config.memory.safety_margin_bytes
        )
        if cuda_free_before_pools < required_free:
            raise RuntimeError(
                "insufficient single-GPU free memory for preallocated expert "
                "pools plus declared runtime reserves: "
                f"free={cuda_free_before_pools}, required={required_free}"
            )
    pool_allocator = PoolAllocator(
        num_layers=config.model.layers,
        hi_pool_sizes=hi_pool_sizes,
        lo_pool_sizes=lo_pool_sizes,
        device=device,
        hi_block_sizes=hi_block_sizes,
        lo_block_sizes=lo_block_sizes,
        staging_pool_size_bytes=budget_result.transient_bytes,
        staging_block_size_bytes=max((*hi_block_sizes, *lo_block_sizes)),
    )

    budget_tracker = BudgetTracker(
        hi_cap=sum(hi_pool_sizes) + budget_result.transient_bytes,
        lo_cap=sum(lo_pool_sizes) + budget_result.transient_bytes,
        staging_cap=budget_result.transient_bytes,
        total_cap=budget_result.total_reserved_bytes,
    )
    
    # TransitionEngine
    transition_engine = TransitionEngine(
        registry=registry,
        pool_allocator=pool_allocator,
        weight_store=weight_store,
        max_workers=4,
        max_inflight=config.memory.max_inflight,
        budget_tracker=budget_tracker,
        synchronous=transition_synchronous,
    )

    # Deterministic, fully materialized bootstrap. A paper run supplies a
    # calibration-derived full ranking and takes the quota-sized prefix. The
    # first-n fallback remains for unit tests and exploratory runs but is
    # explicitly identified as uncalibrated in runtime metadata.
    bootstrap_hi_experts: dict[str, list[int]] = {}
    for layer in range(config.model.layers):
        hi_experts = set(
            (
                initial_expert_ranking[layer]
                if initial_expert_ranking is not None
                else list(range(config.model.experts_per_layer))
            )[: budget_result.n_hi[layer]]
        )
        bootstrap_hi_experts[str(layer)] = sorted(hi_experts)
        for expert in range(config.model.experts_per_layer):
            tier = Tier.HI if expert in hi_experts else Tier.LO
            key = ExpertKey(layer, expert)
            req = TransitionReq(
                key=key,
                src=tier,
                dst=tier,
                reason="bootstrap",
                issued_step=0,
            )
            if not transition_engine.enqueue(req):
                raise RuntimeError(f"bootstrap admission rejected for {key} {tier}")
            if not transition_engine.wait_ready(key, timeout=300):
                raise TimeoutError(f"bootstrap timed out for {key} {tier}")
            handle = registry.get_handle(key)
            if handle is None or handle.tier != tier:
                raise RuntimeError(f"bootstrap failed to publish {key} {tier}")

    bootstrap_stats = transition_engine.get_stats(include_stage_timings=False)
    if bootstrap_stats["failed_transitions"]:
        raise RuntimeError("one or more bootstrap transitions failed")
    transition_engine.reset_stats()
    runtime_metadata = {
        "host_cache": host_cache,
        "released_native_expert_bytes": released_source_bytes,
        "dense_parameter_bytes": dense_parameter_bytes,
        "dense_buffer_bytes": dense_buffer_bytes,
        "configured_dense_reservation_bytes": config.memory.reserve_dense_bytes,
        "effective_dense_reservation_bytes": max(
            config.memory.reserve_dense_bytes,
            actual_dense_bytes,
        ),
        "pool_allocation_bytes": pool_allocation_bytes,
        "resident_expert_bytes": budget_result.total_expert_bytes,
        "transient_expert_bytes": budget_result.transient_bytes,
        "automatic_kernel_workspace_bytes": automatic_kernel_workspace_bytes,
        "configured_kernel_workspace_bytes": (
            config.memory.reserve_kernel_workspace_bytes
        ),
        "effective_kernel_workspace_bytes": effective_kernel_workspace_bytes,
        "cuda_free_before_pools": cuda_free_before_pools,
        "cuda_total_bytes": cuda_total_bytes,
        "bootstrap": bootstrap_stats,
        "bootstrap_policy": (
            "calibrated_ranking_prefix"
            if initial_expert_ranking is not None
            else "uncalibrated_expert_id_prefix"
        ),
        "bootstrap_hi_experts": bootstrap_hi_experts,
        "transition_execution_mode": (
            "synchronous"
            if transition_synchronous
            else "asynchronous"
        ),
        "requested_high_precision_ratio": high_precision_ratio,
        "n_hi": budget_result.n_hi,
        "realized_high_precision_ratio": (
            sum(budget_result.n_hi)
            / (
                config.model.layers
                * config.model.experts_per_layer
            )
        ),
    }
    
    print("DynaExq components initialized")
    
    return (
        observer,
        tracker,
        scheduler,
        registry,
        transition_engine,
        budget_init,
        runtime_metadata,
    )


def run_shift_experiment(
    config: DynaExqConfig,
    model_path: str,
    output_dir: Path,
    device_name: str | None = None,
    hash_model_files: bool = False,
):
    """Run the shift benchmark experiment."""
    device_name = device_name or (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device_name}")
    
    checkpoint = checkpoint_metadata(
        model_path,
        hash_weight_files=hash_model_files,
    )
    revision = checkpoint.get("revision")
    if checkpoint.get("local") is False and not revision:
        raise RuntimeError(
            "remote checkpoint revision could not be resolved; refusing an "
            "unpinned paper run"
        )

    # Load exactly the checkpoint revision recorded in the artifact.
    model, tokenizer = load_model(
        model_path,
        device,
        revision=revision,
    )
    
    # Initialize DynaExq
    (
        observer,
        tracker,
        scheduler,
        registry,
        transition_engine,
        budget_init,
        runtime_metadata,
    ) = initialize_dynaexq(config, model, device)
    
    # Create MoE wrapper
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
    )
    wrapper.validate_integration()
    
    # Create workload stream
    calibration_dir = Path("calibration_datasets/requests")
    phases = [
        PhaseConfig(
            name="wikitext",
            dataset_path=str(calibration_dir / "wikitext2_128x2048.jsonl"),
            duration_s=config.experiments.phase_duration_s,
            concurrency=config.experiments.concurrency,
        ),
        PhaseConfig(
            name="gsm8k",
            dataset_path=str(calibration_dir / "gsm8k_200.jsonl"),
            duration_s=config.experiments.phase_duration_s,
            concurrency=config.experiments.concurrency,
        ),
        PhaseConfig(
            name="humaneval",
            dataset_path=str(calibration_dir / "humaneval_200.jsonl"),
            duration_s=config.experiments.phase_duration_s,
            concurrency=config.experiments.concurrency,
        ),
    ]
    
    workload = WorkloadStream(phases=phases, cycles=config.experiments.cycles)
    
    # Run experiment
    print("\n" + "=" * 60)
    print("Starting shift benchmark experiment")
    print("=" * 60)
    
    results = []
    phase_start_time = time.time()

    def request_prompt(request: dict) -> str:
        for key in ("prompt", "question", "problem", "text"):
            value = request.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise ValueError("request contains no prompt/question/problem/text")

    def generate_one(prompt: str) -> LatencyMetrics:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.experiments.input_tokens,
        )
        input_device = next(model.parameters()).device
        input_ids = encoded.input_ids.to(input_device)
        attention_mask = encoded.attention_mask.to(input_device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        last_logit_kwargs = last_logit_only_kwargs(wrapper)
        with torch.inference_mode():
            outputs = wrapper.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                **last_logit_kwargs,
            )
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat((input_ids, next_token), dim=1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_token)), dim=1
            )
            past = outputs.past_key_values
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            first_token_at = time.perf_counter()

            for _ in range(config.experiments.output_tokens - 1):
                prepared = prepare_one_token_decode(
                    model,
                    generated=generated,
                    next_token=next_token,
                    past_key_values=past,
                    attention_mask=attention_mask,
                )
                prepared.update(last_logit_kwargs)
                outputs = wrapper.forward(**prepared)
                next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
                past = outputs.past_key_values
                generated = torch.cat((generated, next_token), dim=1)
                attention_mask = torch.cat(
                    (attention_mask, torch.ones_like(next_token)), dim=1
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        finished = time.perf_counter()
        ttft_ms = (first_token_at - started) * 1000.0
        total_ms = (finished - started) * 1000.0
        tpot_ms = (
            (total_ms - ttft_ms) / (config.experiments.output_tokens - 1)
            if config.experiments.output_tokens > 1
            else 0.0
        )
        return LatencyMetrics(
            ttft_ms=ttft_ms,
            tpop_ms=tpot_ms,
            total_ms=total_ms,
        )
    
    for phase_name, requests in workload.generate_phases():
        print(f"\nPhase: {phase_name}")
        print(f"Requests: {len(requests)}")
        
        phase_metrics = {
            "phase": phase_name,
            "start_time": phase_start_time,
            "requests_processed": 0,
        }
        metrics = MetricsCollector()
        
        for i, req in enumerate(requests):
            if time.time() - phase_start_time >= config.experiments.phase_duration_s:
                break
            if i % 10 == 0:
                print(f"  Processing request {i + 1}/{len(requests)}")
            metrics.record_latency(generate_one(request_prompt(req)))
            phase_metrics["requests_processed"] += 1
        
        phase_metrics["end_time"] = time.time()
        phase_metrics["duration_s"] = phase_metrics["end_time"] - phase_metrics["start_time"]
        phase_metrics["metrics"] = metrics.get_summary()
        phase_metrics["transition_stats"] = transition_engine.get_stats()
        
        results.append(phase_metrics)
        phase_start_time = phase_metrics["end_time"]
        
        print(f"  Phase complete: {phase_metrics['duration_s']:.2f}s")
    
    # Drain late phase-boundary work before freezing the artifact, so final
    # failures and pool/budget counters cannot change after serialization.
    transition_engine.shutdown()
    final_transition_stats = transition_engine.get_stats()

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "shift_results.json"
    artifact = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_path,
        "device": str(device),
        "checkpoint": checkpoint,
        "config": config.to_dict(),
        "phases": results,
        "environment": environment_metadata(),
        "runtime_initialization": runtime_metadata,
        "wrapper_stats": wrapper.get_stats(),
        "final_transition_stats": final_transition_stats,
    }
    with open(results_file, "w") as f:
        json.dump(artifact, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print("=" * 60)
    
def main():
    parser = argparse.ArgumentParser(description="Run shift benchmark experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="dynaexq/configs/qwen30b.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/shift_benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Exactly one execution device (for example cuda:0)",
    )
    parser.add_argument(
        "--hash-model-files",
        action="store_true",
        help="Hash every local weight shard for submission provenance",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = DynaExqConfig.from_yaml(args.config)
    
    # Run experiment
    run_shift_experiment(
        config=config,
        model_path=args.model_path,
        output_dir=Path(args.output_dir),
        device_name=args.device,
        hash_model_files=args.hash_model_files,
    )


if __name__ == "__main__":
    main()
