#!/usr/bin/env python3
"""
Test hot-set overlap (Jaccard similarity) between different workload phases.

This script implements the experiment described in the paper:
- Phase A: WikiText (LM task)
- Phase B: GSM8K/AIME25 (Math task)
- Phase C: HumanEval (Code task)
- Measures hot-set overlap (Jaccard J) between phases A↔B, B↔C, A↔C
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from dynaexq.runtime import ExpertMonitor
from dynaexq.runtime.types import ExpertID


LOGGER = logging.getLogger("dynaexq.hotset")


def load_dataset_prompts(dataset_name: str, num_samples: int = 512, dataset_path: Path | None = None) -> List[str]:
    """Load prompts from a dataset file or generate synthetic prompts."""
    prompts = []
    
    # Try to load from file if path provided
    if dataset_path and dataset_path.exists():
        try:
            if dataset_path.suffix.lower() in {".jsonl", ".json"}:
                with dataset_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            # Try common keys
                            prompt = entry.get("prompt") or entry.get("text") or entry.get("input") or str(entry)
                            if isinstance(prompt, str) and prompt.strip():
                                prompts.append(prompt.strip())
                                if len(prompts) >= num_samples:
                                    break
                        except json.JSONDecodeError:
                            continue
            else:
                # Plain text file
                with dataset_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            prompts.append(line)
                            if len(prompts) >= num_samples:
                                break
            
            if prompts:
                LOGGER.info(f"Loaded {len(prompts)} prompts from {dataset_path}")
                return prompts[:num_samples]
        except Exception as e:
            LOGGER.warning(f"Failed to load from {dataset_path}: {e}, using synthetic prompts")
    
    # Fallback to synthetic prompts
    if dataset_name == "wikitext":
        # Simple WikiText-like prompts
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
        # Math problem prompts
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
        # Code generation prompts
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
    
    LOGGER.info(f"Generated {len(prompts)} synthetic prompts for {dataset_name}")
    return prompts


def get_hot_set_from_monitor(
    monitor: ExpertMonitor,
    layer: int,
    top_n: int = 16,
) -> Set[int]:
    """Extract top-n hot experts for a layer from ExpertMonitor."""
    # Access internal state safely
    with monitor._lock:
        layer_state = monitor._layers.get(layer)
        if layer_state is None or not layer_state.scores:
            return set()
        
        # Sort experts by score
        expert_scores = [
            (expert_idx, score)
            for expert_idx, score in layer_state.scores.items()
        ]
        expert_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-n
        top_experts = {expert_idx for expert_idx, _ in expert_scores[:top_n]}
        return top_experts


def run_phase(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    monitor: ExpertMonitor,
    prompts: List[str],
    device: torch.device,
    phase_name: str,
    num_requests: int = 512,
    update_period: int = 200,
) -> Dict[int, Set[int]]:
    """
    Run a phase and collect hot-set per layer.
    
    Returns:
        Dictionary mapping layer_id -> set of expert indices (hot-set)
    """
    LOGGER.info(f"Starting phase {phase_name} with {len(prompts)} prompts")
    
    model.eval()
    model.config.output_router_logits = True
    
    # Track requests processed
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
            # Move inputs to the same device as the model
            # For device_map="auto", we need to get the device from the model
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
            
            # Get num_experts_per_tok from config if available
            top_k = getattr(model.config, 'num_experts_per_tok', 8)
            if not isinstance(top_k, int) or top_k <= 0:
                top_k = 8
            
            # Update monitor with router outputs
            if isinstance(router_logits, tuple):
                num_layers = len(router_logits)
            else:
                num_layers = 1
                router_logits = [router_logits]
            
            for layer_id, layer_logits in enumerate(router_logits):
                try:
                    # Handle complex numbers if present
                    if torch.is_complex(layer_logits):
                        layer_logits = layer_logits.real
                    
                    # Handle different logit shapes
                    if layer_logits.dim() == 3:
                        # Shape: (batch, seq_len, num_experts)
                        batch_size, seq_len, num_experts = layer_logits.shape
                        top_k_actual = min(top_k, num_experts)
                        topk_vals, topk_idx = torch.topk(layer_logits, k=top_k_actual, dim=-1)
                        
                        # Flatten to (batch * seq_len, top_k)
                        topk_idx_flat = topk_idx.reshape(-1, top_k_actual).cpu().numpy()
                        topk_vals_flat = topk_vals.reshape(-1, top_k_actual).cpu().numpy()
                        
                        # Update monitor
                        monitor.update_batch(
                            layer=layer_id,
                            topk_idx=topk_idx_flat,
                            logits=topk_vals_flat,
                        )
                    elif layer_logits.dim() == 2:
                        # Shape: (seq_len, num_experts) - single batch
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
                    else:
                        LOGGER.warning(f"Unexpected router_logits shape for layer {layer_id}: {layer_logits.shape}")
                except Exception as e:
                    LOGGER.warning(f"Error processing router_logits for layer {layer_id}: {e}")
                    continue
            
            requests_processed += 1
            step_count += 1
            
            # Periodic update (EMA decay simulation)
            if step_count % update_period == 0:
                monitor.epoch_tick()
            
            if (prompt_idx + 1) % 50 == 0:
                LOGGER.info(f"Phase {phase_name}: Processed {prompt_idx + 1}/{len(prompts)} prompts")
        
        except Exception as e:
            LOGGER.warning(f"Error processing prompt {prompt_idx}: {e}")
            continue
    
    # Extract hot-set for each layer
    num_layers = model.config.num_hidden_layers
    hot_sets = {}
    for layer_id in range(num_layers):
        hot_set = get_hot_set_from_monitor(monitor, layer_id, top_n=16)
        hot_sets[layer_id] = hot_set
    
    LOGGER.info(f"Phase {phase_name} complete. Processed {requests_processed} requests")
    return hot_sets


def calculate_jaccard(set1: Set[int], set2: Set[int]) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_overlap_statistics(
    hot_sets_a: Dict[int, Set[int]],
    hot_sets_b: Dict[int, Set[int]],
    hot_sets_c: Dict[int, Set[int]],
) -> Dict[str, Tuple[float, float]]:
    """
    Compute Jaccard similarity statistics between phases.
    
    Returns:
        Dictionary with keys like "A↔B", "B↔C", "A↔C"
        Values are (mean, std) tuples
    """
    # Get common layers
    layers = set(hot_sets_a.keys()) & set(hot_sets_b.keys()) & set(hot_sets_c.keys())
    layers = sorted(layers)
    
    jaccards_ab = []
    jaccards_bc = []
    jaccards_ac = []
    
    for layer_id in layers:
        set_a = hot_sets_a.get(layer_id, set())
        set_b = hot_sets_b.get(layer_id, set())
        set_c = hot_sets_c.get(layer_id, set())
        
        j_ab = calculate_jaccard(set_a, set_b)
        j_bc = calculate_jaccard(set_b, set_c)
        j_ac = calculate_jaccard(set_a, set_c)
        
        jaccards_ab.append(j_ab)
        jaccards_bc.append(j_bc)
        jaccards_ac.append(j_ac)
    
    results = {
        "A↔B (LM→Math)": (
            np.mean(jaccards_ab),
            np.std(jaccards_ab),
        ),
        "B↔C (Math→Code)": (
            np.mean(jaccards_bc),
            np.std(jaccards_bc),
        ),
        "A↔C (LM→Code)": (
            np.mean(jaccards_ac),
            np.std(jaccards_ac),
        ),
    }
    
    return results


def print_results_table(results: Dict[str, Tuple[float, float]]) -> None:
    """Print results in table format matching the paper format."""
    print("\n" + "=" * 80)
    print("Draft Table 1: Routing shift intensity (hot-set overlap)")
    print("(对48个 MoE layer 取平均;括号给出跨layer 的 std)")
    print("=" * 80)
    print(f"{'Hot-set Jaccard J':<30} {'A↔B (LM→Math)':<20} {'B↔C (Math→Code)':<20} {'A↔C (LM→Code)':<20}")
    print("-" * 80)
    
    # Extract values
    ab_mean, ab_std = results.get("A↔B (LM→Math)", (0.0, 0.0))
    bc_mean, bc_std = results.get("B↔C (Math→Code)", (0.0, 0.0))
    ac_mean, ac_std = results.get("A↔C (LM→Code)", (0.0, 0.0))
    
    ab_str = f"{ab_mean:.2f} (±{ab_std:.2f})"
    bc_str = f"{bc_mean:.2f} (±{bc_std:.2f})"
    ac_str = f"{ac_mean:.2f} (±{ac_std:.2f})"
    print(f"{'DynaExq 在线观测':<30} {ab_str:<20} {bc_str:<20} {ac_str:<20}")
    
    print("=" * 80)
    print("\n解释(对应理论):")
    print(f"若n = 16,则每层切换需替换的 hot experts 数为:")
    print(f"  A→B: 16 * (1-{ab_mean:.2f}) / (1+{ab_mean:.2f}) = {16 * (1-ab_mean) / (1+ab_mean):.2f}")
    print(f"  B→C: 16 * (1-{bc_mean:.2f}) / (1+{bc_mean:.2f}) = {16 * (1-bc_mean) / (1+bc_mean):.2f}")
    print(f"  A→C: 16 * (1-{ac_mean:.2f}) / (1+{ac_mean:.2f}) = {16 * (1-ac_mean) / (1+ac_mean):.2f}")
    print("这会导出你后面 promotion rate 的合理量级。")
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID or path (e.g., Qwen/Qwen3-30B-A3B)",
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
        help="Number of requests per phase (default: 512)",
    )
    parser.add_argument(
        "--update-period",
        type=int,
        default=200,
        help="Update period for EMA (default: 200 steps)",
    )
    parser.add_argument(
        "--hot-set-size",
        type=int,
        default=16,
        help="Hot-set size n (default: 16)",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=3,
        help="Number of cycles (A→B→C) to run (default: 3)",
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
    parser.add_argument(
        "--dataset-wikitext",
        type=Path,
        default=None,
        help="Optional path to WikiText dataset file",
    )
    parser.add_argument(
        "--dataset-gsm8k",
        type=Path,
        default=None,
        help="Optional path to GSM8K dataset file",
    )
    parser.add_argument(
        "--dataset-humaneval",
        type=Path,
        default=None,
        help="Optional path to HumanEval dataset file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Resolve device - for large models, we'll use device_map="auto"
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    LOGGER.info(f"Using device: {device}")
    # Note: For large models, device_map="auto" will be used instead of explicit device
    
    # Load model and tokenizer
    LOGGER.info(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    
    config = AutoConfig.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    
    # Use device_map="auto" for better memory management
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    model.eval()
    LOGGER.info(f"Model loaded. Num layers: {config.num_hidden_layers}")
    
    # Initialize monitor with EMA alpha=0.9 (as per image requirements)
    monitor = ExpertMonitor(ewma_alpha=0.9, epoch_decay=0.5)
    
    # Load datasets
    prompts_wikitext = load_dataset_prompts("wikitext", args.num_requests, args.dataset_wikitext)
    prompts_gsm8k = load_dataset_prompts("gsm8k", args.num_requests, args.dataset_gsm8k)
    prompts_humaneval = load_dataset_prompts("humaneval", args.num_requests, args.dataset_humaneval)
    
    # Run multiple cycles and aggregate results
    all_jaccards_ab = []
    all_jaccards_bc = []
    all_jaccards_ac = []
    
    for cycle in range(args.num_cycles):
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"Cycle {cycle + 1}/{args.num_cycles}")
        LOGGER.info(f"{'='*80}\n")
        
        # Phase A: WikiText
        monitor.epoch_tick()  # Reset/decay before new phase
        hot_sets_a = run_phase(
            model, tokenizer, monitor, prompts_wikitext, device,
            "A (WikiText)", args.num_requests, args.update_period
        )
        
        # Phase B: GSM8K
        monitor.epoch_tick()
        hot_sets_b = run_phase(
            model, tokenizer, monitor, prompts_gsm8k, device,
            "B (GSM8K)", args.num_requests, args.update_period
        )
        
        # Phase C: HumanEval
        monitor.epoch_tick()
        hot_sets_c = run_phase(
            model, tokenizer, monitor, prompts_humaneval, device,
            "C (HumanEval)", args.num_requests, args.update_period
        )
        
        # Compute overlaps for this cycle
        layers = set(hot_sets_a.keys()) & set(hot_sets_b.keys()) & set(hot_sets_c.keys())
        for layer_id in layers:
            set_a = hot_sets_a.get(layer_id, set())
            set_b = hot_sets_b.get(layer_id, set())
            set_c = hot_sets_c.get(layer_id, set())
            
            all_jaccards_ab.append(calculate_jaccard(set_a, set_b))
            all_jaccards_bc.append(calculate_jaccard(set_b, set_c))
            all_jaccards_ac.append(calculate_jaccard(set_a, set_c))
    
    # Aggregate results across all cycles
    results = {
        "A↔B (LM→Math)": (
            np.mean(all_jaccards_ab),
            np.std(all_jaccards_ab),
        ),
        "B↔C (Math→Code)": (
            np.mean(all_jaccards_bc),
            np.std(all_jaccards_bc),
        ),
        "A↔C (LM→Code)": (
            np.mean(all_jaccards_ac),
            np.std(all_jaccards_ac),
        ),
    }
    
    # Print results
    print_results_table(results)
    
    # Save to file if requested
    if args.output:
        output_data = {
            "hot_set_size": args.hot_set_size,
            "num_requests_per_phase": args.num_requests,
            "num_cycles": args.num_cycles,
            "results": {
                k: {"mean": float(v[0]), "std": float(v[1])}
                for k, v in results.items()
            },
            "per_layer_stats": {
                "A↔B": {
                    "mean": float(np.mean(all_jaccards_ab)),
                    "std": float(np.std(all_jaccards_ab)),
                    "min": float(np.min(all_jaccards_ab)),
                    "max": float(np.max(all_jaccards_ab)),
                },
                "B↔C": {
                    "mean": float(np.mean(all_jaccards_bc)),
                    "std": float(np.std(all_jaccards_bc)),
                    "min": float(np.min(all_jaccards_bc)),
                    "max": float(np.max(all_jaccards_bc)),
                },
                "A↔C": {
                    "mean": float(np.mean(all_jaccards_ac)),
                    "std": float(np.std(all_jaccards_ac)),
                    "min": float(np.min(all_jaccards_ac)),
                    "max": float(np.max(all_jaccards_ac)),
                },
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
        LOGGER.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
