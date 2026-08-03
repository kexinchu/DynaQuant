"""test_bf16_fastpath.py — Benchmark the bf16 cast-free int4pack path.

Runs on CUDA 0 (independent of the eval running on CUDA 1). Loads the model
in bf16 so the int4pack fast path skips all per-call casts. Reports ms/tok
at batch=4 to compare directly against the earlier fp16 baseline
(167 ms/tok int8-attn / 154 ms/tok fp16-attn).
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

# Verify model is actually bf16
sample_param = next(model.parameters())
print(f"[check] model dtype: {sample_param.dtype}")

# Dequant int8 attention (should pick bf16 automatically)
n = dequantize_int8_attention(model)
print(f"[check] attention dequant: {n} layers, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

# Confirm new attention Linear is bf16
sample_qproj = model.model.layers[0].self_attn.q_proj
print(f"[check] q_proj dtype: {sample_qproj.weight.dtype}")

# ── Benchmark b=4 decode ──────────────────────────────────────────────────
prompts_b4 = [
    "Explain the transformer architecture in detail.",
    "Write a Python implementation of quicksort.",
    "Describe how photosynthesis works at a molecular level.",
    "What are the key results of general relativity?",
]

def timed_decode(label, prompts, n_new=32, n_runs=3):
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEV)
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
        times.append(dt / n_gen_total * 1000)
    avg = sum(times) / len(times)
    tps = 1000 / avg
    print(f"  {label:40s}: {avg:7.2f} ms/tok  ({tps:6.1f} tok/s)")
    return avg

print("\n[bench] b=4 decode with bf16 cast-free fast path")
t_b4 = timed_decode("b=4 bf16 int4pack (cast-free)", prompts_b4)

print("\n[summary]")
print(f"  Earlier baseline (fp16, int8 attn Triton):   167.11 ms/tok")
print(f"  Earlier baseline (fp16, fp16 attn dequant):  153.51 ms/tok")
print(f"  NEW (bf16, fp16 attn dequant, cast-free):    {t_b4:.2f} ms/tok")
print(f"  speedup vs fp16+dequant:  {153.51/t_b4:.2f}x")
print(f"  speedup vs original:      {167.11/t_b4:.2f}x")
