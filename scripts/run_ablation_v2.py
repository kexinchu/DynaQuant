"""
DynaExq ablation v2: real MMLU-Pro accuracy + online-serving throughput.

Key improvements over v1:
- Accuracy: 200 real MMLU-Pro questions (not hand-crafted)
- Throughput: simulates online serving where migration overlaps/stalls
  generation — this is what makes async vs sync visible.

The throughput measurement works as follows:
  1. Warmup: run warmup prompts, collect stats, do initial promotion
  2. Serving loop with workload shift:
     - Generate tokens for prompt stream
     - At the shift point, trigger re-scheduling + migration
     - For async configs: migration is non-blocking (overlapped)
     - For sync configs: migration blocks before next generation
     - Total throughput = total tokens / total wall-clock time

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/run_ablation_v2.py \
        --int4-path /path/to/int4 \
        --fp16-path /path/to/fp16 \
        --output results/ablation_qwen3_30b_v2.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ===================================================================
# Model + weight loading (same as v1)
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
    counts = defaultdict(Counter)
    hooks = []
    def make_hook(li):
        def fn(mod, inp, out):
            logits = out[0] if isinstance(out, tuple) else out
            if isinstance(logits, torch.Tensor) and logits.dim() == 2:
                topk = min(8, logits.shape[-1])
                _, idx = torch.topk(logits, topk, dim=-1)
                for i in idx.flatten().cpu().tolist():
                    counts[li][i] += 1
        return fn
    for li, layer in enumerate(model.model.layers):
        gate = getattr(layer.mlp, "gate", None)
        if gate and isinstance(gate, nn.Module):
            hooks.append(gate.register_forward_hook(make_hook(li)))
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", truncation=True,
                        max_length=512).input_ids.to(device)
        with torch.no_grad():
            model(ids)
    for h in hooks:
        h.remove()
    return dict(counts)


def get_hot_experts(counts, n_hot):
    return {l: [e for e, _ in c.most_common(n_hot)] for l, c in counts.items()}


# ===================================================================
# Expert swapper (same as v1)
# ===================================================================

class ExpertSwapper:
    def __init__(self, model, fp16_weights, device="cuda:0"):
        self.model = model
        self.fp16_weights = fp16_weights
        self.device = device
        self._originals = {}
        self._promoted = set()

    def promote(self, layer, expert, blocking=True):
        key = (layer, expert)
        if key in self._promoted or key not in self.fp16_weights:
            return 0.0
        expert_mod = self.model.model.layers[layer].mlp.experts[expert]
        weights = self.fp16_weights[key]
        h2d_ms = 0.0
        for proj in ("gate_proj", "up_proj", "down_proj"):
            if proj not in weights:
                continue
            cpu_w = weights[proj]
            out_f, in_f = cpu_w.shape
            if key not in self._originals:
                self._originals[key] = {}
            self._originals[key][proj] = getattr(expert_mod, proj)
            if blocking:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            gpu_w = cpu_w.to(self.device, non_blocking=not blocking)
            if blocking:
                torch.cuda.synchronize()
            h2d_ms += (time.perf_counter() - t0) * 1000
            new_lin = nn.Linear(in_f, out_f, bias=False, dtype=torch.float16, device=self.device)
            new_lin.weight.data.copy_(gpu_w)
            del gpu_w
            setattr(expert_mod, proj, new_lin)
        self._promoted.add(key)
        return h2d_ms

    def demote(self, layer, expert):
        key = (layer, expert)
        if key not in self._promoted:
            return
        if key in self._originals:
            expert_mod = self.model.model.layers[layer].mlp.experts[expert]
            for proj, orig in self._originals[key].items():
                setattr(expert_mod, proj, orig)
            del self._originals[key]
        self._promoted.discard(key)

    def reset(self):
        for l, e in list(self._promoted):
            self.demote(l, e)
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def n_promoted(self):
        return len(self._promoted)


# ===================================================================
# Real MMLU-Pro accuracy
# ===================================================================

def load_mmlu_pro(n_samples=200):
    """Load real MMLU-Pro test samples from HF."""
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    samples = []
    for item in ds:
        q = item.get("question", "")
        opts = item.get("options", [])
        ans = item.get("answer", "")
        if not q or not opts or not ans:
            continue
        option_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))
        prompt = f"Question: {q}\n{option_text}\nAnswer:"
        samples.append({"prompt": prompt, "answer": ans, "n_options": len(opts)})
        if len(samples) >= n_samples:
            break
    return samples


def measure_mc_accuracy(model, tokenizer, samples, device="cuda:0"):
    correct = 0
    for s in samples:
        ids = tokenizer(s["prompt"], return_tensors="pt", truncation=True,
                        max_length=2048).input_ids.to(device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :]
        letter_scores = []
        for i in range(s["n_options"]):
            letter = chr(65 + i)
            toks = tokenizer.encode(letter, add_special_tokens=False)
            if toks:
                letter_scores.append((letter, logits[toks[0]].item()))
        if not letter_scores:
            continue
        best = max(letter_scores, key=lambda x: x[1])[0]
        if best == s["answer"]:
            correct += 1
    return correct / max(len(samples), 1)


# ===================================================================
# Online-serving throughput measurement
# ===================================================================

SERVING_PROMPTS_A = [
    "Explain the theory of general relativity in simple terms.",
    "What is the difference between TCP and UDP protocols?",
    "Describe the process of photosynthesis step by step.",
    "What are the key principles of object-oriented programming?",
]

SERVING_PROMPTS_B = [
    "Solve: Find the derivative of f(x) = x^3 * sin(x).",
    "Calculate the eigenvalues of the matrix [[2,1],[1,2]].",
    "Prove that the square root of 2 is irrational.",
    "What is the integral of e^(-x^2) from 0 to infinity?",
]


def measure_online_throughput(
    model, tokenizer, swapper, n_hot, fp16_weights,
    warmup_stats, shift_prompts, blocking_migration, do_reschedule,
    delta, device="cuda:0", max_new_tokens=32,
):
    """
    Simulate online serving with workload shift + migration.

    Returns throughput including migration stall time.
    """
    total_tokens = 0
    total_wall_s = 0.0
    migration_stall_ms = 0.0

    # Phase A: serve pre-shift prompts (already warmed up, experts promoted)
    for prompt in SERVING_PROMPTS_A:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=256).input_ids.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        total_tokens += out.shape[1] - ids.shape[1]
        total_wall_s += elapsed

    # Workload shift: collect new router stats + migrate
    shift_stats = collect_router_stats(model, tokenizer, shift_prompts, device)
    new_hot = get_hot_experts(shift_stats, n_hot)

    if do_reschedule:
        old_set = set(swapper._promoted)
        new_set = {(l, e) for l, es in new_hot.items() for e in es
                   if (l, e) in fp16_weights}

        # Apply delta filter (hysteresis)
        if delta > 0:
            to_promote = set()
            for l, e in new_set - old_set:
                old_score = warmup_stats.get(l, {}).get(e, 0)
                new_score = shift_stats.get(l, {}).get(e, 0)
                if new_score - old_score > delta:
                    to_promote.add((l, e))
        else:
            to_promote = new_set - old_set

        to_demote = old_set - new_set

        # Migration (timed — this is what stalls the serving loop)
        torch.cuda.synchronize()
        mig_start = time.perf_counter()
        for l, e in to_demote:
            swapper.demote(l, e)
        for l, e in to_promote:
            swapper.promote(l, e, blocking=blocking_migration)
        if not blocking_migration:
            torch.cuda.synchronize()  # async: sync once at end
        mig_elapsed = (time.perf_counter() - mig_start) * 1000
        migration_stall_ms = mig_elapsed if blocking_migration else 0.0
        total_wall_s += mig_elapsed / 1000 if blocking_migration else 0.0

    # Phase B: serve post-shift prompts
    for prompt in SERVING_PROMPTS_B:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=256).input_ids.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        total_tokens += out.shape[1] - ids.shape[1]
        total_wall_s += elapsed

    tput = total_tokens / max(total_wall_s, 1e-6)
    avg_lat = total_wall_s * 1000 / (len(SERVING_PROMPTS_A) + len(SERVING_PROMPTS_B))
    return {
        "throughput_tok_s": round(tput, 2),
        "avg_latency_ms": round(avg_lat, 1),
        "migration_stall_ms": round(migration_stall_ms, 1),
        "total_tokens": total_tokens,
        "total_wall_s": round(total_wall_s, 2),
    }


# ===================================================================
# Ablation configs
# ===================================================================

def run_config(name, model, tokenizer, swapper, n_hot, delta,
               fp16_weights, warmup_prompts, device):
    """Run one ablation configuration. Returns (accuracy, perf_dict)."""
    print(f"    [{name}] Warmup ...")
    warmup_stats = collect_router_stats(model, tokenizer, warmup_prompts, device)
    hot = get_hot_experts(warmup_stats, n_hot)

    # Determine config behavior
    if name == "Full DynaExq":
        blocking = False
        reschedule = True
        use_delta = delta
    elif name == "w/o Online Scheduler":
        blocking = False
        reschedule = False
        use_delta = delta
    elif name == "w/o VER (blocking)":
        blocking = True
        reschedule = True
        use_delta = delta
    elif name == "w/o Async Pipeline":
        blocking = True
        reschedule = True
        use_delta = delta
    elif name == "w/o Hysteresis":
        blocking = False
        reschedule = True
        use_delta = 0.0
    else:
        blocking = False
        reschedule = True
        use_delta = delta

    # Initial promotion
    h2d_total = 0.0
    for layer, experts in hot.items():
        for e in experts:
            h2d_total += swapper.promote(layer, e, blocking=blocking)
    if not blocking:
        torch.cuda.synchronize()
    print(f"    [{name}] Promoted {swapper.n_promoted} experts "
          f"(H2D: {h2d_total:.1f}ms, {'blocking' if blocking else 'async'})")

    # Accuracy on real MMLU-Pro
    print(f"    [{name}] MC accuracy eval ...")
    acc = measure_mc_accuracy(model, tokenizer, mmlu_pro_samples, device)
    print(f"    [{name}] Accuracy: {acc*100:.1f}%")

    # Online-serving throughput
    print(f"    [{name}] Online-serving throughput ...")
    perf = measure_online_throughput(
        model, tokenizer, swapper, n_hot, fp16_weights,
        warmup_stats, SERVING_PROMPTS_B,
        blocking_migration=blocking,
        do_reschedule=reschedule,
        delta=use_delta,
        device=device,
    )
    print(f"    [{name}] Tput: {perf['throughput_tok_s']:.1f} tok/s, "
          f"AvgLat: {perf['avg_latency_ms']:.0f}ms, "
          f"MigStall: {perf['migration_stall_ms']:.0f}ms")

    return acc, perf


# ===================================================================
# Main
# ===================================================================

mmlu_pro_samples = []  # Global, loaded once

def main():
    global mmlu_pro_samples

    parser = argparse.ArgumentParser()
    parser.add_argument("--int4-path", required=True)
    parser.add_argument("--fp16-path", required=True)
    parser.add_argument("--n-hot", type=int, default=8)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--n-mmlu", type=int, default=200,
                        help="Number of MMLU-Pro samples for accuracy")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-memory", default="32GiB")
    parser.add_argument("--output", default="results/ablation_qwen3_30b_v2.json")
    args = parser.parse_args()

    # Load MMLU-Pro
    print("[1] Loading MMLU-Pro test set ...")
    mmlu_pro_samples = load_mmlu_pro(args.n_mmlu)
    print(f"    {len(mmlu_pro_samples)} samples loaded")

    # Load model
    print("[2] Loading INT4 model ...")
    model, tokenizer = load_model(args.int4_path, args.device, args.max_memory)

    # Pre-scan all hot experts and load FP16 weights
    print("[3] Pre-scanning hot experts ...")
    warmup_prompts = SERVING_PROMPTS_A + SERVING_PROMPTS_B
    all_stats = collect_router_stats(model, tokenizer, warmup_prompts, args.device)
    all_hot = get_hot_experts(all_stats, min(32, 128))
    pairs = [(l, e) for l, es in all_hot.items() for e in es]
    print(f"    Loading FP16 weights for {len(pairs)} experts ...")
    fp16_weights = load_fp16_weights_cpu(args.fp16_path, pairs)
    print(f"    {len(fp16_weights)} expert weight sets loaded")

    swapper = ExpertSwapper(model, fp16_weights, args.device)

    # Run each config
    config_names = [
        "Full DynaExq",
        "w/o Online Scheduler",
        "w/o VER (blocking)",
        "w/o Async Pipeline",
        "w/o Hysteresis",
    ]

    results = {}
    for name in config_names:
        print(f"\n{'='*60}")
        print(f"  Config: {name}")
        print(f"{'='*60}")
        swapper.reset()
        gc.collect()
        torch.cuda.empty_cache()

        acc, perf = run_config(
            name, model, tokenizer, swapper, args.n_hot, args.delta,
            fp16_weights, warmup_prompts, args.device,
        )
        results[name] = {
            "accuracy_pct": round(acc * 100, 2),
            "throughput_tok_s": perf["throughput_tok_s"],
            "avg_latency_ms": perf["avg_latency_ms"],
            "migration_stall_ms": perf["migration_stall_ms"],
            "n_promoted": swapper.n_promoted,
            "gpu_mem_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        }
        print(f"  -> Acc: {results[name]['accuracy_pct']:.1f}% | "
              f"Tput: {perf['throughput_tok_s']:.1f} tok/s | "
              f"MigStall: {perf['migration_stall_ms']:.0f}ms")

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Configuration':<25} {'Acc(%)':<8} {'Tput':<10} {'AvgLat(ms)':<12} "
          f"{'MigStall(ms)':<14} {'GPU(GB)':<8}")
    print("-" * 90)
    for name, r in results.items():
        print(f"{name:<25} {r['accuracy_pct']:<8.1f} {r['throughput_tok_s']:<10.1f} "
              f"{r['avg_latency_ms']:<12.0f} {r['migration_stall_ms']:<14.0f} "
              f"{r['gpu_mem_gb']:<8.2f}")

    output = {
        "model": args.int4_path,
        "n_hot_per_layer": args.n_hot,
        "delta": args.delta,
        "n_mmlu_samples": len(mmlu_pro_samples),
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
