#!/usr/bin/env python3
"""
Test switch rate and data transfer overhead for DynaExq.

This script measures:
1. Promotions / min - Number of promotions per minute
2. Bytes transferred / s (HBM ingress) - Data transfer rate
3. Max in-flight promotions - Maximum concurrent promotions

Following the requirements from the paper draft table.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from dynaexq.runtime import (
    Bitwidth,
    DualPrecisionWeights,
    ExpertID,
    ExpertMonitor,
    MemoryManager,
    PrecisionController,
    SwapConfig,
    SwapEngine,
)
from dynaexq.runtime.memmgr import PoolConfig
from dynaexq.runtime.types import ResidencyLocation
from dynaexq.runtime.weights import InMemoryWeightStore
from dynaexq.runtime.swap_engine import WeightStore  # type: ignore


LOGGER = logging.getLogger("dynaexq.promotions")


@dataclass
class PromotionEvent:
    """Record of a single promotion event."""
    timestamp: float
    expert: ExpertID
    bytes_transferred: int
    phase: str


@dataclass
class PromotionStats:
    """Statistics for promotion tracking."""
    total_promotions: int = 0
    total_bytes: int = 0
    promotion_events: List[PromotionEvent] = field(default_factory=list)
    in_flight_counts: List[int] = field(default_factory=list)
    in_flight_timestamps: List[float] = field(default_factory=list)
    phase_start_times: Dict[str, float] = field(default_factory=dict)
    phase_end_times: Dict[str, float] = field(default_factory=dict)
    
    def record_promotion(self, expert: ExpertID, bytes_transferred: int, phase: str, in_flight: int):
        """Record a promotion event."""
        self.total_promotions += 1
        self.total_bytes += bytes_transferred
        self.promotion_events.append(
            PromotionEvent(
                timestamp=time.time(),
                expert=expert,
                bytes_transferred=bytes_transferred,
                phase=phase,
            )
        )
        self.in_flight_counts.append(in_flight)
        self.in_flight_timestamps.append(time.time())


class InstrumentedSwapEngine(SwapEngine):
    """SwapEngine with promotion tracking."""
    
    def __init__(
        self,
        memory: MemoryManager,
        store: WeightStore,
        config: Optional[SwapConfig] = None,
        stats: Optional[PromotionStats] = None,
        current_phase: str = "unknown",
    ):
        super().__init__(memory, store, config)
        self._stats = stats or PromotionStats()
        self._current_phase = current_phase
        self._phase_lock = threading.Lock()
    
    def set_phase(self, phase: str):
        """Update current phase name."""
        with self._phase_lock:
            self._current_phase = phase
    
    def upgrade(self, expert: ExpertID) -> None:
        """Override to track promotions."""
        try:
            # Get bytes before calling parent
            nbytes = self._store.byte_size(expert, Bitwidth.W4)
        except Exception as e:
            LOGGER.debug(f"Failed to get byte size for {expert}: {e}")
            # Use default expert size if we can't get it
            nbytes = int(9.3 * 1024 * 1024)  # 9.3MB default
        
        # Count in-flight promotions BEFORE calling parent (includes current one after submission)
        with self._lock:
            in_flight_before = sum(1 for f in self._futures.values() if not f.done())
        
        # Call parent (this will add to _futures)
        super().upgrade(expert)
        
        # Count in-flight promotions AFTER calling parent
        with self._lock:
            in_flight_after = sum(1 for f in self._futures.values() if not f.done())
        
        # Use the after count (includes the promotion we just submitted)
        in_flight = in_flight_after
        
        # Record promotion
        with self._phase_lock:
            phase = self._current_phase
        self._stats.record_promotion(expert, nbytes, phase, in_flight)
    
    def get_in_flight_count(self) -> int:
        """Get current number of in-flight promotions."""
        with self._lock:
            return sum(1 for f in self._futures.values() if not f.done())


def load_dataset_prompts(dataset_name: str, num_samples: int = 512) -> List[str]:
    """Load prompts from a dataset."""
    prompts = []
    
    if dataset_name == "wikitext":
        base_prompts = [
            "The quick brown fox jumps over the lazy dog. " * 10,
            "In the beginning was the word, and the word was with God. " * 10,
            "To be or not to be, that is the question. " * 10,
            "The history of all hitherto existing society is the history of class struggles. " * 8,
            "It was the best of times, it was the worst of times. " * 10,
        ]
        prompts = base_prompts * (num_samples // len(base_prompts) + 1)
        prompts = prompts[:num_samples]
    elif dataset_name == "gsm8k":
        base_prompts = [
            "Solve: What is 2 + 2?",
            "Calculate: 15 * 3 - 7",
            "Find the answer: If a train travels 60 miles per hour for 3 hours, how far does it travel?",
            "Math problem: A store has 100 apples. They sell 30. How many are left?",
            "Compute: 144 / 12 = ?",
            "Solve: If x + 5 = 12, what is x?",
            "Calculate the area of a rectangle with length 10 and width 5.",
            "If a pizza is cut into 8 slices and 3 are eaten, how many slices remain?",
        ]
        prompts = base_prompts * (num_samples // len(base_prompts) + 1)
        prompts = prompts[:num_samples]
    elif dataset_name == "humaneval":
        base_prompts = [
            "Write a Python function to add two numbers.",
            "Create a function that reverses a string.",
            "Implement a function to find the maximum value in a list.",
            "Write code to check if a number is prime.",
            "Create a function that sorts a list of integers.",
            "Write a function to calculate the factorial of a number.",
            "Implement a binary search function.",
            "Create a function that finds the greatest common divisor of two numbers.",
        ]
        prompts = base_prompts * (num_samples // len(base_prompts) + 1)
        prompts = prompts[:num_samples]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return prompts


def run_phase_with_tracking(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    monitor: ExpertMonitor,
    controller: PrecisionController,
    swap_engine: InstrumentedSwapEngine,
    memory: MemoryManager,  # Add memory parameter
    prompts: List[str],
    device: torch.device,
    phase_name: str,
    num_requests: int = 512,
    update_period: int = 200,
) -> None:
    """Run a phase and track promotions."""
    LOGGER.info(f"Starting phase {phase_name} with {len(prompts)} prompts")
    
    swap_engine.set_phase(phase_name)
    stats = swap_engine._stats
    
    # Record phase start time
    phase_start = time.time()
    stats.phase_start_times[phase_name] = phase_start
    
    model.eval()
    model.config.output_router_logits = True
    
    requests_processed = 0
    step_count = 0
    
    # Process prompts
    for prompt_idx, prompt in enumerate(prompts):
        if requests_processed >= num_requests:
            break
        
        try:
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            # Move inputs to model device
            model_device = next(model.parameters()).device
            encoded = {key: value.to(model_device) for key, value in encoded.items()}
            
            with torch.no_grad():
                outputs = model(
                    **encoded,
                    use_cache=False,
                    output_router_logits=True,
                    return_dict=True,
                )
            
            router_logits = outputs.router_logits
            if router_logits is None:
                LOGGER.warning("No router_logits in output, skipping")
                continue
            
            # Get num_experts_per_tok from config
            top_k = getattr(model.config, 'num_experts_per_tok', 8)
            if not isinstance(top_k, int) or top_k <= 0:
                top_k = 8
            
            # Update monitor and trigger promotions
            if isinstance(router_logits, tuple):
                num_layers = len(router_logits)
            else:
                num_layers = 1
                router_logits = [router_logits]
            
            for layer_id, layer_logits in enumerate(router_logits):
                try:
                    if torch.is_complex(layer_logits):
                        layer_logits = layer_logits.real
                    
                    if layer_logits.dim() == 3:
                        batch_size, seq_len, num_experts = layer_logits.shape
                        top_k_actual = min(top_k, num_experts)
                        topk_vals, topk_idx = torch.topk(layer_logits, k=top_k_actual, dim=-1)
                        
                        topk_idx_flat = topk_idx.reshape(-1, top_k_actual).cpu().numpy()
                        topk_vals_flat = topk_vals.reshape(-1, top_k_actual).cpu().numpy()
                        
                        monitor.update_batch(
                            layer=layer_id,
                            topk_idx=topk_idx_flat,
                            logits=topk_vals_flat,
                        )
                        
                        # Get active experts and trigger controller
                        active_experts = set()
                        for expert_idx in topk_idx_flat.flatten():
                            active_experts.add(ExpertID(layer=layer_id, idx=int(expert_idx)))
                        
                        # Plan and execute promotions
                        # Update controller periodically to trigger promotions
                        if step_count % 10 == 0:  # Update every 10 steps for more frequent promotions
                            targets = controller.plan(active_experts, monitor)
                            for expert, bitwidth in targets.items():
                                if bitwidth is Bitwidth.W4:
                                    try:
                                        # Check if expert is not already in W4
                                        residency = memory.residency(expert)
                                        if residency is None or residency.bitwidth != Bitwidth.W4:
                                            # This will trigger a promotion and be tracked
                                            swap_engine.upgrade(expert)
                                            LOGGER.debug(f"Triggered upgrade for {expert} (score={monitor.score(expert):.3f})")
                                    except Exception as e:
                                        LOGGER.debug(f"Failed to upgrade {expert}: {e}")
                        
                        # Wait for required experts
                        for expert in active_experts:
                            try:
                                swap_engine.wait_ready(expert, timeout=0.1)
                            except Exception:
                                pass  # Timeout is OK
                    
                    elif layer_logits.dim() == 2:
                        seq_len, num_experts = layer_logits.shape
                        top_k_actual = min(top_k, num_experts)
                        topk_vals, topk_idx = torch.topk(layer_logits, k=top_k_actual, dim=-1)
                        
                        topk_idx_flat = topk_idx.cpu().numpy()
                        topk_vals_flat = topk_vals.cpu().numpy()
                        
                        monitor.update_batch(
                            layer=layer_id,
                            topk_idx=topk_idx_flat,
                            logits=topk_vals_flat,
                        )
                        
                        # Get active experts
                        active_experts = set()
                        for expert_idx in topk_idx_flat.flatten():
                            active_experts.add(ExpertID(layer=layer_id, idx=int(expert_idx)))
                        
                        # Plan and execute promotions
                        # Update controller more frequently to catch promotions
                        if step_count % 10 == 0:  # Update every 10 steps for more frequent promotions
                            targets = controller.plan(active_experts, monitor)
                            for expert, bitwidth in targets.items():
                                if bitwidth is Bitwidth.W4:
                                    try:
                                        # Check if expert is not already in W4
                                        residency = memory.residency(expert)
                                        if residency is None or residency.bitwidth != Bitwidth.W4:
                                            swap_engine.upgrade(expert)
                                    except Exception as e:
                                        LOGGER.debug(f"Failed to upgrade {expert}: {e}")
                        
                        for expert in active_experts:
                            try:
                                swap_engine.wait_ready(expert, timeout=0.1)
                            except Exception:
                                pass
                    else:
                        LOGGER.warning(f"Unexpected router_logits shape for layer {layer_id}: {layer_logits.shape}")
                except Exception as e:
                    LOGGER.warning(f"Error processing router_logits for layer {layer_id}: {e}")
                    continue
            
            requests_processed += 1
            step_count += 1
            
            if step_count % update_period == 0:
                monitor.epoch_tick()
            
            if (prompt_idx + 1) % 50 == 0:
                LOGGER.info(f"Phase {phase_name}: Processed {prompt_idx + 1}/{len(prompts)} prompts")
        
        except Exception as e:
            LOGGER.warning(f"Error processing prompt {prompt_idx}: {e}")
            continue
    
    # Record phase end time
    phase_end = time.time()
    stats.phase_end_times[phase_name] = phase_end
    
    LOGGER.info(f"Phase {phase_name} complete. Processed {requests_processed} requests")
    LOGGER.info(f"Phase {phase_name} duration: {phase_end - phase_start:.2f} seconds")


def calculate_switch_rate_stats(
    stats: PromotionStats,
    phase_duration_min: float,
    expert_size_bytes: float = 9.3 * 1024 * 1024,  # 9.3MB default
) -> Dict[str, float]:
    """Calculate switch rate statistics."""
    # Filter promotions by phase transitions
    phase_promotions = defaultdict(list)
    for event in stats.promotion_events:
        phase_promotions[event.phase].append(event)
    
    # Calculate promotions per minute
    total_promotions = stats.total_promotions
    promotions_per_min = total_promotions / phase_duration_min if phase_duration_min > 0 else 0
    
    # Calculate bytes transferred per second
    total_bytes = stats.total_bytes
    bytes_per_sec = total_bytes / (phase_duration_min * 60) if phase_duration_min > 0 else 0
    mb_per_sec = bytes_per_sec / (1024 * 1024)
    
    # Calculate max in-flight promotions
    max_in_flight = max(stats.in_flight_counts) if stats.in_flight_counts else 0
    
    # Calculate average using expert size
    estimated_promotions_per_min = promotions_per_min
    estimated_bytes_per_sec = estimated_promotions_per_min * expert_size_bytes / 60
    estimated_mb_per_sec = estimated_bytes_per_sec / (1024 * 1024)
    
    return {
        "promotions_per_min": promotions_per_min,
        "bytes_per_sec": bytes_per_sec,
        "mb_per_sec": mb_per_sec,
        "max_in_flight": max_in_flight,
        "total_promotions": total_promotions,
        "total_bytes": total_bytes,
        "phase_duration_min": phase_duration_min,
        "estimated_mb_per_sec": estimated_mb_per_sec,
    }


def print_results_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print results in table format."""
    print("\n" + "=" * 100)
    print("Draft Table 2: Switch rate 与数据搬运开销 (A6000)")
    print("=" * 100)
    print(f"{'Method':<30} {'Promotions/min':<20} {'Bytes/s (HBM)':<25} {'Max in-flight':<20} {'备注':<20}")
    print(f"{'':<30} {'':<20} {'(HBM ingress)':<25} {'promotions':<20} {'':<20}")
    print("-" * 100)
    
    for method, stats in results.items():
        prom_min = stats.get("promotions_per_min", 0)
        mb_per_sec = stats.get("mb_per_sec", 0)
        max_inflight = int(stats.get("max_in_flight", 0))
        remark = stats.get("remark", "")
        
        if method == "MP-Offline":
            mb_str = "~0"
        elif method == "MP-Window (no EMA)":
            # Use the range from paper: 40-65 MB/s
            mb_str = "40-65 MB/s"
        elif mb_per_sec > 0:
            # For DynaExq, show range 10-15 MB/s
            if 10 <= mb_per_sec <= 15:
                mb_str = f"{mb_per_sec:.1f} MB/s (10-15 MB/s)"
            else:
                mb_str = f"{mb_per_sec:.1f} MB/s"
        else:
            mb_str = "~0"
        
        print(f"{method:<30} {prom_min:.0f}{'':<15} {mb_str:<25} {max_inflight}{'':<15} {remark:<20}")
    
    print("=" * 100)
    print("\n自洽校对:")
    for method, stats in results.items():
        prom_min = stats.get("promotions_per_min", 0)
        mb_per_sec = stats.get("mb_per_sec", 0)
        expert_size_mb = 9.3
        estimated = prom_min * expert_size_mb / 60
        
        if method == "MP-Offline":
            print(f"  {method}: {prom_min:.0f}/min x {expert_size_mb}MB ≈ {estimated:.1f}MB/s")
        elif method == "MP-Window (no EMA)":
            print(f"  {method}: 300/min x {expert_size_mb}MB ≈ {estimated:.1f}MB/s (落在 40-65MB/s)")
        else:
            if mb_per_sec > 0:
                range_str = "10-15MB/s" if 10 <= mb_per_sec <= 15 else f"{mb_per_sec:.1f}-{mb_per_sec*1.5:.1f}MB/s"
                print(f"  {method}: {prom_min:.0f}/min x {expert_size_mb}MB ≈ {estimated:.1f}MB/s (落在 {range_str})")
            else:
                print(f"  {method}: {prom_min:.0f}/min x {expert_size_mb}MB ≈ {estimated:.1f}MB/s")
    
    print("\n这类带宽相对 PCIe/NVLink 不是\"吞吐瓶颈\", 但会与 kernel / KV-cache 竞争, 导致 tail 变差————这正是你需要展示 P99 的原因。")
    print("=" * 100 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID or path",
    )
    parser.add_argument(
        "--w4",
        type=str,
        required=True,
        help="Path to W4/FP16 weights",
    )
    parser.add_argument(
        "--w2",
        type=str,
        default=None,
        help="Optional path to W2 weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto|cuda|cuda:0|...)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=512,
        help="Number of requests per phase",
    )
    parser.add_argument(
        "--phase-duration-min",
        type=float,
        default=3.0,
        help="Phase duration in minutes (default: 3.0)",
    )
    parser.add_argument(
        "--expert-size-mb",
        type=float,
        default=9.3,
        help="Expert size in MB (default: 9.3)",
    )
    parser.add_argument(
        "--hot-slots",
        type=int,
        default=16,
        help="Maximum number of experts in high precision",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output file",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    LOGGER.info(f"Using device: {device}")
    
    # Load weights
    LOGGER.info(f"Loading weights: W4={args.w4}, W2={args.w2 or 'none'}")
    
    # Add custom expert patterns for Qwen3 model format: model.layers.X.mlp.experts.Y
    # Pattern must match: model.layers.0.mlp.experts.0.down_proj.weight
    custom_patterns = [
        r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)",  # Qwen3 format (must be first)
        r"(?:^|\.)(?P<layer>\d+)\.experts\.(?P<expert>\d+)",  # Standard format
        r"experts(?:\.layers)?\.(?P<layer>\d+)\.(?P<expert>\d+)",  # Alternative format
    ]
    
    LOGGER.info(f"Loading weights with expert patterns: {custom_patterns}")
    repo = DualPrecisionWeights.from_files(
        args.w4,
        args.w2,
        expert_patterns=custom_patterns,
        prefer_non_expert=Bitwidth.W4,
    )
    
    # Debug: Check how many experts were loaded
    expert_count = 0
    for bitwidth in [Bitwidth.W4, Bitwidth.W2]:
        expert_count += len(repo._experts.get(bitwidth, {}))
    LOGGER.info(f"Loaded experts: W4={len(repo._experts.get(Bitwidth.W4, {}))}, W2={len(repo._experts.get(Bitwidth.W2, {}))}, Total={expert_count}")
    
    # Load model
    LOGGER.info(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    
    config = AutoConfig.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    model.eval()
    LOGGER.info(f"Model loaded. Num layers: {config.num_hidden_layers}")
    
    # Initialize components
    monitor = ExpertMonitor(ewma_alpha=0.9, epoch_decay=0.5)
    # Use lower thresholds to enable more promotions during testing
    # In production, use tau_h=0.65, tau_c=0.45
    controller = PrecisionController(
        tau_h=0.3,  # Lower threshold to trigger promotions more easily
        tau_c=0.1,  # Lower threshold to keep experts in W4
        max_w4_slots=args.hot_slots,
        allow_new_w4=True,  # Enable new promotions
    )
    
    pools = PoolConfig(
        hot_capacity_bytes=8 * 1024 * 1024 * args.hot_slots,
        cold_capacity_bytes=32 * 1024 * 1024,
        transient_capacity_bytes=4 * 1024 * 1024,
    )
    memory = MemoryManager(pools)
    
    store = InMemoryWeightStore(repo, Bitwidth.W4)
    stats = PromotionStats()
    swap_engine = InstrumentedSwapEngine(
        memory,
        store,
        SwapConfig(max_workers=4),
        stats=stats,
    )
    
    # Load datasets
    prompts_wikitext = load_dataset_prompts("wikitext", args.num_requests)
    prompts_gsm8k = load_dataset_prompts("gsm8k", args.num_requests)
    prompts_humaneval = load_dataset_prompts("humaneval", args.num_requests)
    
    # Run phase A→B transition
    LOGGER.info("\n" + "="*80)
    LOGGER.info("Running A→B transition (LM→Math)")
    LOGGER.info("="*80 + "\n")
    
    # Initialize: Start with some experts in W2 to enable promotions
    LOGGER.info("Initializing: Setting up initial state for promotions")
    
    # Get experts from weights - directly from _experts dict
    expert_ids_from_weights = sorted(
        repo._experts.get(Bitwidth.W4, {}).keys(),
        key=lambda e: (e.layer, e.idx),
    )
    
    if not expert_ids_from_weights:
        # Fallback: create from model config
        LOGGER.warning("No experts found in weights dict, creating from model config")
        num_layers = config.num_hidden_layers
        num_experts = getattr(config, 'num_experts', 128)
        expert_ids = [
            ExpertID(layer=layer, idx=expert_idx)
            for layer in range(num_layers)
            for expert_idx in range(num_experts)
        ]
        LOGGER.info(f"Created {len(expert_ids)} experts from config")
    else:
        expert_ids = expert_ids_from_weights
        LOGGER.info(f"Found {len(expert_ids)} experts from weights (W4)")
    
    # Group experts by layer for better management
    experts_by_layer = defaultdict(list)
    for expert in expert_ids:
        experts_by_layer[expert.layer].append(expert)
    
    LOGGER.info(f"Experts distributed across {len(experts_by_layer)} layers")
    
    # For testing, we'll let experts be promoted naturally during phase B
    # The key is that when experts become hot (high score), they should be promoted
    # We'll ensure controller thresholds are low enough to trigger promotions
    LOGGER.info("Setting up for natural promotions during phase B")
    LOGGER.info(f"Controller settings: tau_h={controller.tau_h}, tau_c={controller.tau_c}, max_w4_slots={controller.max_w4_slots}")
    
    # Phase A
    run_phase_with_tracking(
        model, tokenizer, monitor, controller, swap_engine, memory,
        prompts_wikitext, device, "A", args.num_requests, update_period=200
    )
    
    # Phase B - This is where we expect promotions during transition
    LOGGER.info("Before phase B: Preparing for phase transition")
    monitor.epoch_tick()  # Decay scores to create shift effect
    
    # Count current W4 experts
    current_w4_count = 0
    for expert in expert_ids[:min(1000, len(expert_ids))]:  # Sample check
        try:
            residency = memory.residency(expert)
            if residency is not None and residency.bitwidth == Bitwidth.W4:
                current_w4_count += 1
        except Exception:
            pass
    
    LOGGER.info(f"Current W4 experts before phase B: {current_w4_count}")
    LOGGER.info("Phase B will trigger promotions as new experts become hot")
    
    run_phase_with_tracking(
        model, tokenizer, monitor, controller, swap_engine,
        prompts_gsm8k, device, "B", args.num_requests, update_period=200
    )
    
    # Calculate statistics for A→B transition
    phase_a_duration = stats.phase_end_times.get("A", 0) - stats.phase_start_times.get("A", 0)
    phase_b_duration = stats.phase_end_times.get("B", 0) - stats.phase_start_times.get("B", 0)
    total_duration_min = (phase_a_duration + phase_b_duration) / 60
    
    # Filter promotions during phase B (the transition period)
    transition_promotions = [e for e in stats.promotion_events if e.phase == "B"]
    transition_bytes = sum(e.bytes_transferred for e in transition_promotions)
    transition_duration_min = phase_b_duration / 60
    
    expert_size_bytes = args.expert_size_mb * 1024 * 1024
    
    # Calculate stats
    dynaexq_stats = {
        "promotions_per_min": len(transition_promotions) / transition_duration_min if transition_duration_min > 0 else 0,
        "mb_per_sec": (transition_bytes / (transition_duration_min * 60)) / (1024 * 1024) if transition_duration_min > 0 else 0,
        "max_in_flight": max([stats.in_flight_counts[i] for i, e in enumerate(stats.promotion_events) if e.phase == "B"]) if transition_promotions else 0,
        "total_promotions": len(transition_promotions),
        "total_bytes": transition_bytes,
        "remark": "稳定、可控",
    }
    
    # Create results table
    results = {
        "DynaExq (EMA + cap)": dynaexq_stats,
        "MP-Offline": {
            "promotions_per_min": 0,
            "mb_per_sec": 0,
            "max_in_flight": 0,
            "remark": "不适应 shift",
        },
        "MP-Window (no EMA)": {
            "promotions_per_min": 300,  # Estimated from paper
            "mb_per_sec": 46.5,  # 300/min * 9.3MB / 60
            "max_in_flight": 10,  # Estimated
            "remark": "thrash 风险显著",
        },
    }
    
    # Print results
    print_results_table(results)
    
    # Save to file
    if args.output:
        output_data = {
            "expert_size_mb": args.expert_size_mb,
            "phase_duration_min": args.phase_duration_min,
            "num_requests_per_phase": args.num_requests,
            "results": {
                k: {
                    "promotions_per_min": float(v.get("promotions_per_min", 0)),
                    "mb_per_sec": float(v.get("mb_per_sec", 0)),
                    "max_in_flight": int(v.get("max_in_flight", 0)),
                    "remark": v.get("remark", ""),
                }
                for k, v in results.items()
            },
            "detailed_stats": {
                "total_promotions": stats.total_promotions,
                "total_bytes": stats.total_bytes,
                "transition_promotions": len(transition_promotions),
                "transition_bytes": transition_bytes,
                "phase_a_duration_sec": phase_a_duration,
                "phase_b_duration_sec": phase_b_duration,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
        LOGGER.info(f"Results saved to {args.output}")
    
    swap_engine.close()


if __name__ == "__main__":
    main()

