"""test_fused_gu.py — Verify gate+up fusion correctness + measure speedup.

Runs on CUDA 0. Tests:
  1. Per-layer numerical correctness: fused path matches unfused for one MLP
  2. Full-model decode speed: bf16 + attn dequant + fused gate/up
"""
import os
import sys
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.patch_autoround_quantlinear import (
    apply_patch,
    dequantize_int8_attention,
    fuse_gate_up_experts,
)

MODEL = "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"
DEV = "cuda:0"

apply_patch()
print("[load] loading model in BF16 …")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map=DEV,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()
print(f"[load] done in {time.time()-t0:.1f}s  GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

# Attention dequant (matches model's bf16)
dequantize_int8_attention(model)

# ── Step 1: verify fused-vs-unfused numerical equivalence on ONE expert ──
print("\n[verify] fused vs unfused single-expert correctness")
sample_mlp = model.model.layers[1].mlp.experts[0]
hidden_size = sample_mlp.gate_proj.infeatures
x_test = torch.randn(3, hidden_size, dtype=torch.bfloat16, device=DEV)

with torch.no_grad():
    ref = sample_mlp(x_test)            # unfused path (current _fast_forward × 3)

# Fuse all experts — now the MLP forward uses the fused path
n = fuse_gate_up_experts(model)
print(f"  fused {n} experts  GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

with torch.no_grad():
    fused = sample_mlp(x_test)

diff = (ref - fused).abs()
print(f"  ref  norm: {ref.abs().mean().item():.4f}")
print(f"  diff max:  {diff.max().item():.6f}")
print(f"  diff mean: {diff.mean().item():.6f}")
print(f"  rel err:   {diff.mean().item() / ref.abs().mean().item():.4%}")

# ── Step 2: decode speed benchmark ────────────────────────────────────────
prompts_b4 = [
    "Explain the transformer architecture in detail.",
    "Write a Python implementation of quicksort.",
    "Describe how photosynthesis works at a molecular level.",
    "What are the key results of general relativity?",
]

def timed_decode(label, n_new=32, n_runs=3):
    enc = tok(prompts_b4, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=8, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=n_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n_gen_total = (out.shape[1] - enc["input_ids"].shape[1]) * len(prompts_b4)
        times.append(dt / n_gen_total * 1000)
    avg = sum(times) / len(times)
    print(f"  {label:40s}: {avg:7.2f} ms/tok  ({1000/avg:6.1f} tok/s)")
    return avg

print("\n[bench] b=4 decode with fused gate+up")
t_fused = timed_decode("b=4 bf16 + attn-dq + fused-gu")

print("\n[summary]")
print(f"  original (fp16, int8 attn Triton):           167.11 ms/tok")
print(f"  + attn dequant (fp16):                       153.51 ms/tok")
print(f"  + bf16 cast-free int4pack:                   124.67 ms/tok")
print(f"  + FUSED gate+up:                             {t_fused:6.2f} ms/tok")
print(f"  overall speedup vs original: {167.11/t_fused:.2f}x")
