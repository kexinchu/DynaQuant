"""test_dequant_attn.py — Verify INT8 attention dequant + decode speedup.

Runs on CUDA 1 (so the main eval on CUDA 0 is unaffected). Loads the model,
times a short decode before and after dequantizing INT8 attention to fp16.
"""
import os
import sys
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.patch_autoround_quantlinear import (
    apply_patch,
    dequantize_int8_attention,
    dequantize_quantlinear_to_fp16,
)

MODEL = "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"
DEV = "cuda:0"  # CUDA_VISIBLE_DEVICES=1 remaps

apply_patch()
print("[load] loading model …")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map=DEV,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()
print(f"[load] done in {time.time()-t0:.1f}s  GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

# ── Step 1: validate dequant math for a single attention weight ───────────
print("\n[verify] single-layer dequant correctness")
sample_qproj = model.model.layers[0].self_attn.q_proj
print(f"  q_proj: bits={sample_qproj.bits} K={sample_qproj.infeatures} "
      f"N={sample_qproj.outfeatures} g={sample_qproj.group_size}")
x_test = torch.randn(1, sample_qproj.infeatures, dtype=torch.float16, device=DEV)

with torch.no_grad():
    ref = sample_qproj(x_test)  # through Triton (or fallback) path
    w_fp16 = dequantize_quantlinear_to_fp16(sample_qproj)
    manual = x_test @ w_fp16.T
    if sample_qproj.bias is not None:
        manual = manual + sample_qproj.bias.to(torch.float16)

diff = (ref - manual).abs()
print(f"  ref  norm: {ref.abs().mean().item():.4f}")
print(f"  diff max:  {diff.max().item():.4f}")
print(f"  diff mean: {diff.mean().item():.4f}")
print(f"  rel err:   {diff.mean().item() / ref.abs().mean().item():.4%}")

# ── Step 2: time decode BEFORE dequant ────────────────────────────────────
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

prompts_b1 = ["Explain the transformer architecture in detail."]
prompts_b4 = [
    "Explain the transformer architecture in detail.",
    "Write a Python implementation of quicksort.",
    "Describe how photosynthesis works at a molecular level.",
    "What are the key results of general relativity?",
]

def timed_decode(label, prompts, n_new=32, n_runs=3):
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEV)
    # warmup
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
        n_gen_total = (out.shape[1] - enc["input_ids"].shape[1]) * len(prompts)
        times.append(dt / n_gen_total * 1000)   # ms per generated token (avg)
    avg = sum(times) / len(times)
    tps = 1000 / avg
    print(f"  {label:40s}: {avg:7.2f} ms/tok  ({tps:6.1f} tok/s total)")
    return avg

print("\n[bench] decode BEFORE dequant")
t_before_b4 = timed_decode("b=4 int8-attn (Triton fallback)", prompts_b4)
torch.cuda.empty_cache()

# ── Step 3: dequant and re-time ───────────────────────────────────────────
print("\n[dequant] replacing int8 self_attn → fp16 …")
n_replaced = dequantize_int8_attention(model)
print(f"  replaced {n_replaced} layers. GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

print("\n[bench] decode AFTER dequant")
t_after_b4 = timed_decode("b=4 fp16-attn (cuBLAS)", prompts_b4)

print("\n[summary]")
print(f"  b=4 BEFORE: {t_before_b4:.2f} ms/tok   AFTER: {t_after_b4:.2f} ms/tok   "
      f"speedup: {t_before_b4/t_after_b4:.2f}x")
