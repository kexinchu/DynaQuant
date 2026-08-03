"""
DynaExq core experiment: runtime dual-precision expert swap.

Demonstrates the paper's central mechanism on Qwen3-30B:
1. Load INT4 (AutoRound) model on GPU as the baseline
2. Load FP16 expert weights into CPU pinned memory (selective, not full model)
3. Run inference to identify "hot" experts via router statistics
4. Prefetch hot experts' FP16 weights: CPU pinned → GPU, replace QuantLinear with nn.Linear
5. Re-run inference with mixed INT4/FP16 experts
6. Measure: quality delta, prefetch latency, memory overhead

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/experiment_dual_precision.py \
        --int4-path /path/to/int4-model \
        --fp16-path /path/to/fp16-model \
        --n-hot-experts 8 \
        --prompt "What is the capital of France?"
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

nn  # suppress unused import warning from linters

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Step 1: Load INT4 model
# ---------------------------------------------------------------------------

def load_int4_model(path, device="cuda:0", max_memory="32GiB"):
    print(f"[1/6] Loading INT4 model from {path} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map=device,
        max_memory={"0": max_memory},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"      Loaded in {time.time()-t0:.1f}s, GPU mem: {mem:.2f}GB")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Step 2: Load FP16 expert weights into CPU pinned memory
# ---------------------------------------------------------------------------

def load_fp16_expert_weights_to_cpu(fp16_path, layers, expert_ids_per_layer):
    """
    Selectively load expert MLP weights from FP16 safetensors into CPU
    pinned memory. Only loads the experts we actually need.

    Args:
        fp16_path: Path to the FP16 model directory
        layers: List of layer indices to load
        expert_ids_per_layer: dict[layer_idx, list[expert_idx]]

    Returns:
        dict[(layer, expert), {"gate_proj": Tensor, "up_proj": Tensor, "down_proj": Tensor}]
        All tensors are in CPU pinned memory, fp16.
    """
    print(f"[2/6] Loading FP16 expert weights into CPU pinned memory ...")
    t0 = time.time()

    # Build the set of keys we need
    needed_keys = {}
    for layer in layers:
        for expert in expert_ids_per_layer.get(layer, []):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                needed_keys[key] = (layer, expert, proj)

    # Scan safetensors files
    st_files = sorted(Path(fp16_path).glob("*.safetensors"))
    result = defaultdict(dict)
    loaded = 0

    for st_file in st_files:
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in needed_keys:
                    layer, expert, proj = needed_keys[key]
                    tensor = f.get_tensor(key).to(torch.float16).contiguous()
                    # Pin to host memory for fast H2D transfer
                    pinned = torch.empty_like(tensor, pin_memory=True)
                    pinned.copy_(tensor)
                    result[(layer, expert)][proj] = pinned
                    loaded += 1
                    del tensor

    total_mb = sum(
        t.numel() * t.element_size() / 1e6
        for expert_weights in result.values()
        for t in expert_weights.values()
    )
    print(f"      Loaded {loaded} tensors ({total_mb:.1f} MB) for "
          f"{len(result)} experts in {time.time()-t0:.1f}s")
    return dict(result)


# ---------------------------------------------------------------------------
# Step 3: Run inference + collect router statistics
# ---------------------------------------------------------------------------

def collect_router_stats(model, tokenizer, prompts, device="cuda:0"):
    """Run inference and collect which experts are activated per layer."""
    print(f"[3/6] Running inference to collect router statistics ...")
    t0 = time.time()

    activation_counts = defaultdict(Counter)
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # Handle multiple MoE block output formats:
            # 1. Standard Qwen3: (hidden_states, router_logits)
            # 2. Auto-round patched: (hidden_states,) or just hidden_states
            # For case 2, we hook the gate/router submodule directly.
            router_logits = None
            if isinstance(output, tuple):
                for item in output:
                    if isinstance(item, torch.Tensor) and item.dim() == 2:
                        # Could be router_logits if shape is (tokens, n_experts)
                        n_experts = getattr(module, "num_experts", None)
                        if n_experts and item.shape[-1] == n_experts:
                            router_logits = item
                            break
            if router_logits is None:
                # Try to get from module's gate submodule
                gate = getattr(module, "gate", None)
                if gate is not None and hasattr(gate, "_last_logits"):
                    router_logits = gate._last_logits
            if router_logits is not None:
                topk = min(8, router_logits.shape[-1])
                _, topk_indices = torch.topk(router_logits, topk, dim=-1)
                for idx in topk_indices.flatten().cpu().tolist():
                    activation_counts[layer_idx][idx] += 1
        return hook_fn

    # Register hooks on MoE blocks AND on gate/router submodules
    # (auto-round's patched block may not pass router_logits through)
    def make_gate_hook(layer_idx):
        def hook_fn(module, input, output):
            # Gate output: router_logits (tokens, n_experts) or tuple
            logits = output
            if isinstance(output, tuple):
                logits = output[0]
            if isinstance(logits, torch.Tensor) and logits.dim() == 2:
                topk = min(8, logits.shape[-1])
                _, topk_indices = torch.topk(logits, topk, dim=-1)
                for idx in topk_indices.flatten().cpu().tolist():
                    activation_counts[layer_idx][idx] += 1
        return hook_fn

    for layer_idx, layer in enumerate(model.model.layers):
        moe = layer.mlp
        # Try gate/router submodule first (most reliable for patched blocks)
        gate = getattr(moe, "gate", None)
        if gate is not None and isinstance(gate, nn.Module):
            h = gate.register_forward_hook(make_gate_hook(layer_idx))
            hooks.append(h)
        elif hasattr(moe, "experts"):
            h = moe.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

    # Run prompts
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=512).input_ids.to(device)
        with torch.no_grad():
            model(input_ids)

    # Clean up hooks
    for h in hooks:
        h.remove()

    total_activations = sum(sum(c.values()) for c in activation_counts.values())
    print(f"      Collected {total_activations} expert activations across "
          f"{len(activation_counts)} layers in {time.time()-t0:.1f}s")
    return dict(activation_counts)


def select_hot_experts(activation_counts, n_hot_per_layer=8):
    """Select the top-N most activated experts per layer."""
    hot = {}
    for layer_idx, counts in activation_counts.items():
        top_experts = [e for e, _ in counts.most_common(n_hot_per_layer)]
        hot[layer_idx] = top_experts
    total = sum(len(v) for v in hot.values())
    print(f"[3/6] Selected {total} hot experts ({n_hot_per_layer}/layer)")
    return hot


# ---------------------------------------------------------------------------
# Step 4: Prefetch FP16 weights → GPU, replace QuantLinear with nn.Linear
# ---------------------------------------------------------------------------

def replace_experts_with_fp16(model, fp16_weights, hot_experts, device="cuda:0"):
    """
    For each hot expert, replace its QuantLinear modules with
    standard nn.Linear holding FP16 weights from CPU pinned memory.

    Returns: replacement stats dict
    """
    print(f"[4/6] Prefetching FP16 weights to GPU + replacing QuantLinear ...")
    t0 = time.time()
    replaced = 0
    total_h2d_ms = 0.0
    total_bytes = 0

    for layer_idx, expert_ids in hot_experts.items():
        layer = model.model.layers[layer_idx]
        moe = layer.mlp

        for expert_idx in expert_ids:
            key = (layer_idx, expert_idx)
            if key not in fp16_weights:
                continue

            expert_module = moe.experts[expert_idx]
            weights = fp16_weights[key]

            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                if proj_name not in weights:
                    continue

                cpu_weight = weights[proj_name]
                quant_linear = getattr(expert_module, proj_name)

                # Determine shapes from the FP16 weight
                out_features, in_features = cpu_weight.shape

                # Create a standard nn.Linear (no bias, matching the original)
                new_linear = nn.Linear(in_features, out_features, bias=False,
                                       dtype=torch.float16, device="meta")

                # H2D transfer: pinned CPU → GPU (timed)
                torch.cuda.synchronize()
                h2d_start = time.perf_counter()
                gpu_weight = cpu_weight.to(device, non_blocking=True)
                torch.cuda.synchronize()
                h2d_ms = (time.perf_counter() - h2d_start) * 1000
                total_h2d_ms += h2d_ms
                total_bytes += gpu_weight.numel() * gpu_weight.element_size()

                # Materialize the Linear on GPU with the FP16 weight
                new_linear = nn.Linear(in_features, out_features, bias=False,
                                       dtype=torch.float16, device=device)
                new_linear.weight.data.copy_(gpu_weight)
                del gpu_weight

                # Replace the QuantLinear with the FP16 nn.Linear
                setattr(expert_module, proj_name, new_linear)

            replaced += 1

    # Force GC to free old QuantLinear buffers
    gc.collect()
    torch.cuda.empty_cache()

    mem = torch.cuda.memory_allocated() / 1e9
    print(f"      Replaced {replaced} experts, "
          f"H2D total: {total_h2d_ms:.1f}ms ({total_bytes/1e6:.1f}MB), "
          f"GPU mem: {mem:.2f}GB")
    return {
        "replaced_experts": replaced,
        "total_h2d_ms": round(total_h2d_ms, 1),
        "total_h2d_mb": round(total_bytes / 1e6, 1),
        "gpu_mem_after_gb": round(mem, 2),
    }


# ---------------------------------------------------------------------------
# Step 5 & 6: Compare quality and measure latency
# ---------------------------------------------------------------------------

def run_quality_comparison(model, tokenizer, prompts, device="cuda:0"):
    """Generate responses and return them for comparison."""
    print(f"[5/6] Running inference ...")
    results = []
    total_ms = 0.0
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=512).input_ids.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=32, do_sample=False)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        total_ms += elapsed

        new_tokens = out[0][input_ids.shape[1]:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append({
            "prompt": prompt[:100],
            "response": reply[:200],
            "latency_ms": round(elapsed, 1),
            "output_tokens": len(new_tokens),
        })

    avg_ms = total_ms / max(len(prompts), 1)
    print(f"      {len(prompts)} prompts, avg latency: {avg_ms:.1f}ms")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DynaExq dual-precision experiment")
    parser.add_argument("--int4-path", required=True, help="Path to INT4 model")
    parser.add_argument("--fp16-path", required=True, help="Path to FP16 model")
    parser.add_argument("--n-hot-experts", type=int, default=8,
                        help="Number of hot experts per layer to promote to FP16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-memory", default="32GiB")
    parser.add_argument("--output", default="results/dual_precision_experiment.json")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices (default: all)")
    args = parser.parse_args()

    test_prompts = [
        "What is the capital of France? Answer concisely.",
        "Solve: If 3x + 7 = 22, what is x?",
        "What are the main causes of climate change? Be brief.",
    ]

    # Step 1: Load INT4 model
    model, tokenizer = load_int4_model(args.int4_path, args.device, args.max_memory)
    baseline_mem = torch.cuda.memory_allocated() / 1e9

    # Step 5a: Baseline (pure INT4) quality
    print("\n=== Phase A: Pure INT4 baseline ===")
    int4_results = run_quality_comparison(model, tokenizer, test_prompts, args.device)

    # Step 3: Collect router statistics
    activation_counts = collect_router_stats(model, tokenizer, test_prompts, args.device)
    hot_experts = select_hot_experts(activation_counts, args.n_hot_experts)

    # Determine which layers to promote
    if args.layers:
        target_layers = [int(x) for x in args.layers.split(",")]
    else:
        target_layers = sorted(hot_experts.keys())

    # Step 2: Load FP16 weights for hot experts
    fp16_weights = load_fp16_expert_weights_to_cpu(
        args.fp16_path,
        target_layers,
        hot_experts,
    )

    # Step 4: Replace hot experts with FP16
    swap_stats = replace_experts_with_fp16(model, fp16_weights, hot_experts, args.device)

    # Step 5b: Mixed precision (INT4 cold + FP16 hot) quality
    print("\n=== Phase B: Mixed INT4/FP16 (hot experts promoted) ===")
    mixed_results = run_quality_comparison(model, tokenizer, test_prompts, args.device)

    # Step 6: Summary
    print("\n" + "=" * 60)
    print("=== Results Summary ===")
    print(f"Baseline GPU mem:     {baseline_mem:.2f} GB")
    print(f"After swap GPU mem:   {swap_stats['gpu_mem_after_gb']:.2f} GB")
    print(f"Memory overhead:      {swap_stats['gpu_mem_after_gb'] - baseline_mem:.2f} GB")
    print(f"Experts replaced:     {swap_stats['replaced_experts']}")
    print(f"H2D transfer total:   {swap_stats['total_h2d_ms']:.1f} ms ({swap_stats['total_h2d_mb']:.1f} MB)")
    print(f"H2D bandwidth:        {swap_stats['total_h2d_mb'] / max(swap_stats['total_h2d_ms']/1000, 0.001):.0f} MB/s")
    print()
    print("--- Responses comparison ---")
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt: {prompt[:80]}")
        print(f"  INT4:  {int4_results[i]['response'][:100]}... ({int4_results[i]['latency_ms']:.0f}ms)")
        print(f"  Mixed: {mixed_results[i]['response'][:100]}... ({mixed_results[i]['latency_ms']:.0f}ms)")

    output = {
        "config": {
            "int4_path": args.int4_path,
            "fp16_path": args.fp16_path,
            "n_hot_experts": args.n_hot_experts,
            "n_layers_promoted": len(target_layers),
        },
        "swap_stats": swap_stats,
        "baseline_mem_gb": round(baseline_mem, 2),
        "int4_results": int4_results,
        "mixed_results": mixed_results,
        "hot_experts_per_layer": {str(k): v for k, v in hot_experts.items()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[6/6] Results saved to {args.output}")


if __name__ == "__main__":
    main()
