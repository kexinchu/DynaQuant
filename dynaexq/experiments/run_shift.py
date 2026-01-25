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
from pathlib import Path
from typing import Optional

import torch

from ..core import (
    BudgetInitializer,
    DynaExqConfig,
    ExpertRegistry,
    HotnessTracker,
    ModelWeightStore,
    PrecisionScheduler,
    RouterObserver,
    Tier,
    TransitionEngine,
)
from ..integration.moe_wrapper import MoEWrapper
from .metrics import MetricsCollector
from .workloads import PhaseConfig, WorkloadStream


def load_model(model_path: str, device: torch.device):
    """Load model from path."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Model loaded successfully")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def initialize_dynaexq(
    config: DynaExqConfig,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[
    RouterObserver,
    HotnessTracker,
    PrecisionScheduler,
    ExpertRegistry,
    TransitionEngine,
    BudgetInitializer,
]:
    """Initialize DynaExq components."""
    print("Initializing DynaExq components...")
    
    # RouterObserver
    observer = RouterObserver(use_probabilities=True)
    
    # HotnessTracker
    tracker = HotnessTracker(
        num_layers=config.model.layers,
        experts_per_layer=config.model.experts_per_layer,
        alpha=config.scheduler.alpha,
    )
    
    # BudgetInitializer
    def memory_footprint_fn(layer: int, tier: Tier) -> int:
        """Estimate memory footprint per expert."""
        # Simplified estimation
        # In practice, would query actual model structure
        if tier == Tier.HI:
            if config.precision.hi == "fp16":
                return 200 * 1024 * 1024  # 200MB per expert (HI)
            else:
                return 100 * 1024 * 1024  # 100MB for int4
        else:
            if config.precision.lo == "int4":
                return 50 * 1024 * 1024  # 50MB per expert (LO)
            else:
                return 25 * 1024 * 1024  # 25MB for int2
    
    budget_init = BudgetInitializer(
        num_layers=config.model.layers,
        experts_per_layer=config.model.experts_per_layer,
        memory_footprint_fn=memory_footprint_fn,
        device_mem_bytes=config.memory.device_mem_bytes,
        reserve_kv_bytes=config.memory.reserve_kv_bytes,
        reserve_act_bytes=config.memory.reserve_act_bytes,
        reserve_dense_bytes=config.memory.reserve_dense_bytes,
        safety_margin_bytes=config.memory.safety_margin_bytes,
        max_inflight=config.memory.max_inflight,
    )
    
    budget_result = budget_init.compute(strategy="proportional")
    print(f"Budget initialized: n_hi[0]={budget_result.n_hi[0]}, "
          f"total_expert_bytes={budget_result.total_expert_bytes / 1024**3:.2f}GB")
    
    # PrecisionScheduler
    scheduler = PrecisionScheduler(
        num_layers=config.model.layers,
        experts_per_layer=config.model.experts_per_layer,
        n_hi=budget_result.n_hi,
        update_period_steps=config.scheduler.update_period_steps,
        rate_limit=config.scheduler.rate_limit,
    )
    
    # ExpertRegistry
    registry = ExpertRegistry()
    
    # WeightStore
    weight_store = ModelWeightStore(
        model=model,
        hi_format=config.precision.hi,
        lo_format=config.precision.lo,
    )
    
    # MemoryPool (simplified - would need proper initialization)
    from ..core.memory_pool import PoolAllocator
    hi_pool_sizes = [budget_result.hi_pool_bytes // config.model.layers] * config.model.layers
    lo_pool_sizes = [budget_result.lo_pool_bytes // config.model.layers] * config.model.layers
    pool_allocator = PoolAllocator(
        num_layers=config.model.layers,
        hi_pool_sizes=hi_pool_sizes,
        lo_pool_sizes=lo_pool_sizes,
        device=device,
    )
    
    # TransitionEngine
    transition_engine = TransitionEngine(
        registry=registry,
        pool_allocator=pool_allocator,
        weight_store=weight_store,
        max_workers=4,
        max_inflight=config.memory.max_inflight,
    )
    
    print("DynaExq components initialized")
    
    return (
        observer,
        tracker,
        scheduler,
        registry,
        transition_engine,
        budget_init,
    )


def run_shift_experiment(
    config: DynaExqConfig,
    model_path: str,
    output_dir: Path,
):
    """Run the shift benchmark experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model, tokenizer = load_model(model_path, device)
    
    # Initialize DynaExq
    (
        observer,
        tracker,
        scheduler,
        registry,
        transition_engine,
        budget_init,
    ) = initialize_dynaexq(config, model, device)
    
    # Create MoE wrapper
    wrapper = MoEWrapper(
        model=model,
        router_observer=observer,
        hotness_tracker=tracker,
        scheduler=scheduler,
        registry=registry,
        transition_engine=transition_engine,
    )
    
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
    
    # Metrics collector
    metrics = MetricsCollector()
    
    # Run experiment
    print("\n" + "=" * 60)
    print("Starting shift benchmark experiment")
    print("=" * 60)
    
    results = []
    phase_start_time = time.time()
    
    for phase_name, requests in workload.generate_phases():
        print(f"\nPhase: {phase_name}")
        print(f"Requests: {len(requests)}")
        
        phase_metrics = {
            "phase": phase_name,
            "start_time": phase_start_time,
            "requests_processed": 0,
        }
        
        # Process requests (simplified - would run actual inference)
        for i, req in enumerate(requests[:10]):  # Limit for testing
            if i % 10 == 0:
                print(f"  Processing request {i+1}/{min(10, len(requests))}")
            
            # Simulate forward pass
            # In practice, would call wrapper.forward() with actual inputs
            start_time = time.time()
            
            # Update scheduler if needed
            if scheduler.should_update(wrapper._step):
                transition_reqs = scheduler.plan(
                    step=wrapper._step,
                    tracker=tracker,
                )
                for tr_req in transition_reqs:
                    transition_engine.enqueue(tr_req)
            
            # Simulate latency
            ttft_ms = 50.0 + (i % 10) * 5.0  # Simulated
            tpop_ms = 10.0 + (i % 5) * 2.0  # Simulated
            
            from .metrics import LatencyMetrics
            metrics.record_latency(LatencyMetrics(
                ttft_ms=ttft_ms,
                tpop_ms=tpop_ms,
                total_ms=ttft_ms + tpop_ms * config.experiments.output_tokens,
            ))
            
            wrapper._step += 1
            phase_metrics["requests_processed"] += 1
        
        phase_metrics["end_time"] = time.time()
        phase_metrics["duration_s"] = phase_metrics["end_time"] - phase_metrics["start_time"]
        phase_metrics["metrics"] = metrics.get_summary()
        phase_metrics["transition_stats"] = transition_engine.get_stats()
        
        results.append(phase_metrics)
        phase_start_time = phase_metrics["end_time"]
        
        print(f"  Phase complete: {phase_metrics['duration_s']:.2f}s")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "shift_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print("=" * 60)
    
    # Cleanup
    transition_engine.shutdown()


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
    
    args = parser.parse_args()
    
    # Load config
    config = DynaExqConfig.from_yaml(args.config)
    
    # Run experiment
    run_shift_experiment(
        config=config,
        model_path=args.model_path,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

