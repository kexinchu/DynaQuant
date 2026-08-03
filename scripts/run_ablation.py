"""
DynaExq component ablation experiment (HPDC review response).

Measures accuracy + throughput under 5 configurations:
  1. Full DynaExq           — async pipeline, online scheduler, VER, δ-hysteresis
  2. w/o Online Scheduler   — freeze hot set after warmup, never re-schedule
  3. w/o VER (blocking)     — synchronous migration, forward waits for H2D
  4. w/o Async Pipeline     — no background migration, all transitions synchronous
  5. w/o Hysteresis          — δ=0, strict top-N every tick

Design:
  - Warmup phase: run a set of prompts to collect router stats and do initial promotion
  - Workload-shift phase: switch to a different prompt distribution to trigger re-scheduling
  - Eval phase: run accuracy (MMLU-Pro MC, fast) + throughput (tokens/s on generation)
  - Each config loads the model ONCE and resets expert states between configs

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/run_ablation.py \
        --int4-path /path/to/int4 \
        --fp16-path /path/to/fp16 \
        --n-hot 8 \
        --output results/ablation_qwen3_30b.json
"""

import argparse
import copy
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ===================================================================
# Workload prompts
# ===================================================================

WARMUP_PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "What is the difference between TCP and UDP protocols?",
    "Describe the process of photosynthesis step by step.",
    "What are the key principles of object-oriented programming?",
    "Explain how a neural network learns through backpropagation.",
]

SHIFT_PROMPTS = [
    "Solve: Find the derivative of f(x) = x^3 * sin(x).",
    "Calculate the eigenvalues of the matrix [[2,1],[1,2]].",
    "Prove that the square root of 2 is irrational.",
    "What is the integral of e^(-x^2) from 0 to infinity?",
    "Solve the differential equation dy/dx = y * cos(x).",
]

EVAL_PROMPTS_GEN = [
    "What is quantum entanglement? Explain briefly.",
    "Write a brief comparison of Python and Rust.",
    "Explain the CAP theorem in distributed systems.",
    "What causes aurora borealis?",
    "Describe the Turing test and its significance.",
    "What is the difference between a stack and a queue?",
    "Explain how CRISPR gene editing works.",
    "What is the Monty Hall problem?",
]

# For MC accuracy (MMLU-Pro style, fast eval)
EVAL_MC_SAMPLES = [
    {"q": "Which of the following is NOT a principle of object-oriented programming?",
     "options": ["Encapsulation", "Polymorphism", "Compilation", "Inheritance"],
     "answer": "C"},
    {"q": "What is the time complexity of binary search?",
     "options": ["O(n)", "O(log n)", "O(n log n)", "O(n^2)"],
     "answer": "B"},
    {"q": "Which data structure uses FIFO ordering?",
     "options": ["Stack", "Queue", "Tree", "Graph"],
     "answer": "B"},
    {"q": "What does HTTP stand for?",
     "options": ["HyperText Transfer Protocol", "High Transfer Text Protocol",
                 "HyperText Transmission Process", "High Text Transfer Protocol"],
     "answer": "A"},
    {"q": "Which sorting algorithm has the best average-case time complexity?",
     "options": ["Bubble Sort", "Quick Sort", "Selection Sort", "Insertion Sort"],
     "answer": "B"},
    {"q": "What is the derivative of ln(x)?",
     "options": ["x", "1/x", "e^x", "ln(x)/x"],
     "answer": "B"},
    {"q": "Which layer of the OSI model handles routing?",
     "options": ["Data Link", "Network", "Transport", "Session"],
     "answer": "B"},
    {"q": "What is the primary function of mitochondria?",
     "options": ["Protein synthesis", "Energy production", "Cell division", "DNA replication"],
     "answer": "B"},
]


# ===================================================================
# Helpers: model loading + expert management
# ===================================================================

def load_model(path, device="cuda:0", max_memory="32GiB"):
    print(f"  Loading model from {path} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map=device,
        max_memory={"0": max_memory}, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded in {time.time()-t0:.1f}s, GPU: {mem:.2f}GB")
    return model, tokenizer


def load_fp16_weights_cpu(fp16_path, layer_expert_pairs):
    """Load specific expert FP16 weights from safetensors into CPU pinned memory."""
    needed = {}
    for layer, expert in layer_expert_pairs:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
            needed[key] = (layer, expert, proj)

    result = defaultdict(dict)
    for st_file in sorted(Path(fp16_path).glob("*.safetensors")):
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in needed:
                    layer, expert, proj = needed[key]
                    t = f.get_tensor(key).to(torch.float16).contiguous()
                    pinned = torch.empty_like(t, pin_memory=True)
                    pinned.copy_(t)
                    result[(layer, expert)][proj] = pinned
                    del t
    return dict(result)


def collect_router_stats(model, tokenizer, prompts, device="cuda:0"):
    """Run prompts, hook gate modules, return per-layer expert activation counts."""
    counts = defaultdict(Counter)
    hooks = []

    def make_gate_hook(layer_idx):
        def fn(module, inp, out):
            logits = out[0] if isinstance(out, tuple) else out
            if isinstance(logits, torch.Tensor) and logits.dim() == 2:
                topk = min(8, logits.shape[-1])
                _, indices = torch.topk(logits, topk, dim=-1)
                for idx in indices.flatten().cpu().tolist():
                    counts[layer_idx][idx] += 1
        return fn

    for li, layer in enumerate(model.model.layers):
        gate = getattr(layer.mlp, "gate", None)
        if gate is not None and isinstance(gate, nn.Module):
            hooks.append(gate.register_forward_hook(make_gate_hook(li)))

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512).input_ids.to(device)
        with torch.no_grad():
            model(ids)

    for h in hooks:
        h.remove()
    return dict(counts)


def get_hot_experts(counts, n_hot):
    """Return {layer: [expert_ids]} for top-N per layer."""
    hot = {}
    for layer, c in counts.items():
        hot[layer] = [e for e, _ in c.most_common(n_hot)]
    return hot


# ===================================================================
# Expert promotion (the core DynaExq mechanism)
# ===================================================================

class ExpertSwapper:
    """Manages promoting/demoting experts between INT4 and FP16."""

    def __init__(self, model, fp16_weights, device="cuda:0"):
        self.model = model
        self.fp16_weights = fp16_weights
        self.device = device
        # Save original QuantLinear modules for restoration
        self._originals = {}
        self._promoted = set()

    def promote(self, layer, expert, blocking=True):
        """Replace expert's QuantLinear modules with FP16 nn.Linear."""
        key = (layer, expert)
        if key in self._promoted or key not in self.fp16_weights:
            return 0.0

        expert_mod = self.model.model.layers[layer].mlp.experts[expert]
        weights = self.fp16_weights[key]

        h2d_ms = 0.0
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            if proj_name not in weights:
                continue
            cpu_w = weights[proj_name]
            out_f, in_f = cpu_w.shape

            # Save original
            if key not in self._originals:
                self._originals[key] = {}
            self._originals[key][proj_name] = getattr(expert_mod, proj_name)

            # H2D transfer (timed)
            if blocking:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            gpu_w = cpu_w.to(self.device, non_blocking=not blocking)
            if blocking:
                torch.cuda.synchronize()
            h2d_ms += (time.perf_counter() - t0) * 1000

            new_linear = nn.Linear(in_f, out_f, bias=False,
                                   dtype=torch.float16, device=self.device)
            new_linear.weight.data.copy_(gpu_w)
            del gpu_w
            setattr(expert_mod, proj_name, new_linear)

        self._promoted.add(key)
        return h2d_ms

    def demote(self, layer, expert):
        """Restore original QuantLinear modules."""
        key = (layer, expert)
        if key not in self._promoted:
            return
        if key in self._originals:
            expert_mod = self.model.model.layers[layer].mlp.experts[expert]
            for proj_name, orig in self._originals[key].items():
                setattr(expert_mod, proj_name, orig)
            del self._originals[key]
        self._promoted.discard(key)

    def reset(self):
        """Demote all promoted experts back to INT4."""
        for layer, expert in list(self._promoted):
            self.demote(layer, expert)
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def n_promoted(self):
        return len(self._promoted)


# ===================================================================
# Measurement functions
# ===================================================================

def measure_mc_accuracy(model, tokenizer, samples, device="cuda:0"):
    """Quick MC accuracy via logit comparison."""
    correct = 0
    for s in samples:
        options_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(s["options"]))
        prompt = f"Question: {s['q']}\n{options_text}\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :]
        best = max(
            [(chr(65+i), tokenizer.encode(chr(65+i), add_special_tokens=False))
             for i in range(len(s["options"]))],
            key=lambda lt: logits[lt[1][0]].item() if lt[1] else -999
        )[0]
        if best == s["answer"]:
            correct += 1
    return correct / max(len(samples), 1)


def measure_throughput(model, tokenizer, prompts, max_new_tokens=32,
                       device="cuda:0"):
    """Measure tokens/s and avg latency on generation."""
    total_tokens = 0
    latencies = []

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=256).input_ids.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens,
                                 do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_new = out.shape[1] - ids.shape[1]
        total_tokens += n_new
        latencies.append(elapsed * 1000)

    total_time = sum(latencies) / 1000
    tput = total_tokens / max(total_time, 1e-6)
    latencies.sort()
    p99_idx = min(int(len(latencies) * 0.99), len(latencies) - 1)

    return {
        "throughput_tok_s": round(tput, 2),
        "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 1),
        "p99_latency_ms": round(latencies[p99_idx], 1),
        "total_tokens": total_tokens,
    }


# ===================================================================
# Ablation configurations
# ===================================================================

def run_full_dynaexq(model, tokenizer, swapper, n_hot, delta, device):
    """Full system: async promotion, online scheduling, VER, δ-hysteresis."""
    print("    Warmup phase ...")
    stats_warmup = collect_router_stats(model, tokenizer, WARMUP_PROMPTS, device)
    hot = get_hot_experts(stats_warmup, n_hot)

    # Initial async promotion (non-blocking H2D)
    total_h2d = 0.0
    for layer, experts in hot.items():
        for e in experts:
            total_h2d += swapper.promote(layer, e, blocking=False)
    torch.cuda.synchronize()
    print(f"    Promoted {swapper.n_promoted} experts (H2D: {total_h2d:.1f}ms)")

    # Workload shift → online re-scheduling
    print("    Workload shift phase ...")
    stats_shift = collect_router_stats(model, tokenizer, SHIFT_PROMPTS, device)
    new_hot = get_hot_experts(stats_shift, n_hot)

    # Demote old, promote new (async, with δ-hysteresis simulation)
    old_set = {(l, e) for l, es in hot.items() for e in es}
    new_set = {(l, e) for l, es in new_hot.items() for e in es}
    to_demote = old_set - new_set
    to_promote = new_set - old_set

    # With hysteresis: only promote if score gap > delta (simplified)
    if delta > 0:
        # Keep some of the old set (hysteresis reduces churn)
        to_promote_filtered = set()
        for l, e in to_promote:
            old_score = stats_warmup.get(l, {}).get(e, 0)
            new_score = stats_shift.get(l, {}).get(e, 0)
            if new_score - old_score > delta:
                to_promote_filtered.add((l, e))
        actual_demoted = len(to_demote) if not to_promote_filtered else len(to_promote_filtered)
        to_promote = to_promote_filtered

    for l, e in to_demote:
        swapper.demote(l, e)
    for l, e in to_promote:
        swapper.promote(l, e, blocking=False)
    torch.cuda.synchronize()
    print(f"    After shift: {swapper.n_promoted} experts promoted")

    # Eval
    print("    Evaluating ...")
    acc = measure_mc_accuracy(model, tokenizer, EVAL_MC_SAMPLES, device)
    perf = measure_throughput(model, tokenizer, EVAL_PROMPTS_GEN, device=device)
    return acc, perf


def run_without_scheduler(model, tokenizer, swapper, n_hot, device):
    """Freeze hot set after warmup, never re-schedule."""
    print("    Warmup phase (one-shot scheduling) ...")
    stats = collect_router_stats(model, tokenizer, WARMUP_PROMPTS, device)
    hot = get_hot_experts(stats, n_hot)

    total_h2d = 0.0
    for layer, experts in hot.items():
        for e in experts:
            total_h2d += swapper.promote(layer, e, blocking=False)
    torch.cuda.synchronize()
    print(f"    Promoted {swapper.n_promoted} experts (frozen, no re-scheduling)")

    # Workload shift happens but NO re-scheduling
    print("    Workload shift (ignored, scheduler frozen) ...")
    for prompt in SHIFT_PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=256).input_ids.to(device)
        with torch.no_grad():
            model(ids)

    print("    Evaluating ...")
    acc = measure_mc_accuracy(model, tokenizer, EVAL_MC_SAMPLES, device)
    perf = measure_throughput(model, tokenizer, EVAL_PROMPTS_GEN, device=device)
    return acc, perf


def run_without_ver_blocking(model, tokenizer, swapper, n_hot, device):
    """Blocking migration: forward waits for H2D completion."""
    print("    Warmup phase ...")
    stats_warmup = collect_router_stats(model, tokenizer, WARMUP_PROMPTS, device)
    hot = get_hot_experts(stats_warmup, n_hot)

    # BLOCKING promotion (synchronous H2D, forward must wait)
    total_h2d = 0.0
    for layer, experts in hot.items():
        for e in experts:
            total_h2d += swapper.promote(layer, e, blocking=True)
    print(f"    Promoted {swapper.n_promoted} experts BLOCKING (H2D: {total_h2d:.1f}ms)")

    # Workload shift with BLOCKING re-scheduling
    print("    Workload shift (blocking migration) ...")
    stats_shift = collect_router_stats(model, tokenizer, SHIFT_PROMPTS, device)
    new_hot = get_hot_experts(stats_shift, n_hot)

    old_set = {(l, e) for l, es in hot.items() for e in es}
    new_set = {(l, e) for l, es in new_hot.items() for e in es}
    for l, e in old_set - new_set:
        swapper.demote(l, e)
    h2d_shift = 0.0
    for l, e in new_set - old_set:
        h2d_shift += swapper.promote(l, e, blocking=True)
    print(f"    Shift migration BLOCKING: {h2d_shift:.1f}ms")

    print("    Evaluating ...")
    acc = measure_mc_accuracy(model, tokenizer, EVAL_MC_SAMPLES, device)
    perf = measure_throughput(model, tokenizer, EVAL_PROMPTS_GEN, device=device)
    return acc, perf


def run_without_async_pipeline(model, tokenizer, swapper, n_hot, device):
    """No background pipeline: all migrations synchronous, batched between forwards."""
    print("    Warmup phase ...")
    stats_warmup = collect_router_stats(model, tokenizer, WARMUP_PROMPTS, device)
    hot = get_hot_experts(stats_warmup, n_hot)

    # ALL promotions done synchronously in one batch, blocking everything
    torch.cuda.synchronize()
    batch_start = time.perf_counter()
    total_h2d = 0.0
    for layer, experts in hot.items():
        for e in experts:
            total_h2d += swapper.promote(layer, e, blocking=True)
    torch.cuda.synchronize()
    batch_ms = (time.perf_counter() - batch_start) * 1000
    print(f"    Batch promotion: {swapper.n_promoted} experts in {batch_ms:.1f}ms (no pipeline)")

    # Workload shift: synchronous batch migration
    print("    Workload shift (synchronous batch) ...")
    stats_shift = collect_router_stats(model, tokenizer, SHIFT_PROMPTS, device)
    new_hot = get_hot_experts(stats_shift, n_hot)

    old_set = {(l, e) for l, es in hot.items() for e in es}
    new_set = {(l, e) for l, es in new_hot.items() for e in es}
    torch.cuda.synchronize()
    shift_start = time.perf_counter()
    for l, e in old_set - new_set:
        swapper.demote(l, e)
    for l, e in new_set - old_set:
        swapper.promote(l, e, blocking=True)
    torch.cuda.synchronize()
    shift_ms = (time.perf_counter() - shift_start) * 1000
    print(f"    Shift batch: {shift_ms:.1f}ms")

    print("    Evaluating ...")
    acc = measure_mc_accuracy(model, tokenizer, EVAL_MC_SAMPLES, device)
    perf = measure_throughput(model, tokenizer, EVAL_PROMPTS_GEN, device=device)
    return acc, perf


def run_without_hysteresis(model, tokenizer, swapper, n_hot, device):
    """δ=0: strict top-N every tick, more thrashing."""
    print("    Warmup phase ...")
    stats_warmup = collect_router_stats(model, tokenizer, WARMUP_PROMPTS, device)
    hot = get_hot_experts(stats_warmup, n_hot)

    total_h2d = 0.0
    for layer, experts in hot.items():
        for e in experts:
            total_h2d += swapper.promote(layer, e, blocking=False)
    torch.cuda.synchronize()
    print(f"    Promoted {swapper.n_promoted} experts (δ=0)")

    # Workload shift with δ=0 (strict top-N, no hysteresis dampening)
    print("    Workload shift (δ=0, strict top-N) ...")
    stats_shift = collect_router_stats(model, tokenizer, SHIFT_PROMPTS, device)
    new_hot = get_hot_experts(stats_shift, n_hot)

    old_set = {(l, e) for l, es in hot.items() for e in es}
    new_set = {(l, e) for l, es in new_hot.items() for e in es}
    n_changes = len(old_set ^ new_set)
    for l, e in old_set - new_set:
        swapper.demote(l, e)
    for l, e in new_set - old_set:
        swapper.promote(l, e, blocking=False)
    torch.cuda.synchronize()
    print(f"    δ=0 caused {n_changes} expert changes (vs hysteresis which dampens)")

    print("    Evaluating ...")
    acc = measure_mc_accuracy(model, tokenizer, EVAL_MC_SAMPLES, device)
    perf = measure_throughput(model, tokenizer, EVAL_PROMPTS_GEN, device=device)
    return acc, perf


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="DynaExq ablation study")
    parser.add_argument("--int4-path", required=True)
    parser.add_argument("--fp16-path", required=True)
    parser.add_argument("--n-hot", type=int, default=8,
                        help="Hot experts per layer")
    parser.add_argument("--delta", type=float, default=5.0,
                        help="Hysteresis score margin")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-memory", default="32GiB")
    parser.add_argument("--output", default="results/ablation_qwen3_30b.json")
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.int4_path, args.device, args.max_memory)

    # Preload FP16 weights for ALL experts (top-32 per layer to cover all configs)
    print("\n  Pre-scanning hot experts across all layers ...")
    all_stats = collect_router_stats(model, tokenizer,
                                     WARMUP_PROMPTS + SHIFT_PROMPTS, args.device)
    all_hot = get_hot_experts(all_stats, min(32, 128))
    pairs = [(l, e) for l, es in all_hot.items() for e in es]
    print(f"  Loading FP16 weights for {len(pairs)} experts ...")
    fp16_weights = load_fp16_weights_cpu(args.fp16_path, pairs)
    print(f"  Loaded {len(fp16_weights)} expert weight sets")

    swapper = ExpertSwapper(model, fp16_weights, args.device)

    # Run each configuration
    configs = [
        ("Full DynaExq", lambda: run_full_dynaexq(
            model, tokenizer, swapper, args.n_hot, args.delta, args.device)),
        ("w/o Online Scheduler", lambda: run_without_scheduler(
            model, tokenizer, swapper, args.n_hot, args.device)),
        ("w/o VER (blocking)", lambda: run_without_ver_blocking(
            model, tokenizer, swapper, args.n_hot, args.device)),
        ("w/o Async Pipeline", lambda: run_without_async_pipeline(
            model, tokenizer, swapper, args.n_hot, args.device)),
        ("w/o Hysteresis", lambda: run_without_hysteresis(
            model, tokenizer, swapper, args.n_hot, args.device)),
    ]

    results = {}
    for name, run_fn in configs:
        print(f"\n{'='*60}")
        print(f"  Config: {name}")
        print(f"{'='*60}")

        swapper.reset()
        gc.collect()
        torch.cuda.empty_cache()

        acc, perf = run_fn()
        results[name] = {
            "accuracy": round(acc * 100, 2),
            "throughput_tok_s": perf["throughput_tok_s"],
            "avg_latency_ms": perf["avg_latency_ms"],
            "p99_latency_ms": perf["p99_latency_ms"],
            "n_promoted": swapper.n_promoted,
            "gpu_mem_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        }
        print(f"  -> Acc: {results[name]['accuracy']:.1f}% | "
              f"Tput: {perf['throughput_tok_s']:.1f} tok/s | "
              f"Lat: {perf['avg_latency_ms']:.0f}ms (p99: {perf['p99_latency_ms']:.0f}ms)")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Configuration':<25} {'Acc(%)':<8} {'Tput(tok/s)':<12} "
          f"{'AvgLat(ms)':<12} {'P99Lat(ms)':<12} {'GPU(GB)':<8}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<25} {r['accuracy']:<8.1f} {r['throughput_tok_s']:<12.1f} "
              f"{r['avg_latency_ms']:<12.0f} {r['p99_latency_ms']:<12.0f} "
              f"{r['gpu_mem_gb']:<8.2f}")

    # Save
    output = {
        "model": args.int4_path,
        "n_hot_per_layer": args.n_hot,
        "delta": args.delta,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
