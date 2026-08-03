"""test_batch_scaling.py — Measure throughput at different batch sizes."""
import os, sys
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, str(Path(__file__).parent.parent))

import time, gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.patch_autoround_quantlinear import (
    apply_patch, dequantize_int8_attention, fuse_gate_up_experts,
)

MODEL = "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"
DEV = "cuda:0"

apply_patch()
print("[load] loading model in BF16 …")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map=DEV,
    trust_remote_code=True, low_cpu_mem_usage=True,
)
model.eval()
dequantize_int8_attention(model)
fuse_gate_up_experts(model)
print(f"[load] GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

SEED_PROMPTS = [
    "Explain the transformer architecture.",
    "Write quicksort in Python.",
    "What causes the northern lights?",
    "Describe photosynthesis.",
    "What is the Pythagorean theorem?",
    "Explain general relativity simply.",
    "Write binary search in Python.",
    "Describe the water cycle.",
    "What are prime numbers?",
    "Explain how neural networks learn.",
    "Write merge sort in Python.",
    "Describe black holes.",
    "What is the Fibonacci sequence?",
    "Explain quantum computing basics.",
    "Write a stack implementation in Python.",
    "Describe how vaccines work.",
    "What is the theory of evolution?",
    "Explain how GPUs differ from CPUs.",
    "Write a linked list in Python.",
    "Describe the greenhouse effect.",
    "What is machine learning?",
    "Explain recursion with an example.",
    "Write a hash table in Python.",
    "Describe how airplanes fly.",
    "What is dark matter?",
    "Explain the Fourier transform.",
    "Write breadth-first search in Python.",
    "Describe the human immune system.",
    "What is a neural network layer?",
    "Explain convex optimization briefly.",
    "Write depth-first search in Python.",
    "Describe CRISPR gene editing.",
]

def bench(batch_size, n_new=32, n_runs=2):
    prompts = SEED_PROMPTS[:batch_size]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=512).to(DEV)

    # warmup
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=4, do_sample=False,
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
        n_gen = (out.shape[1] - enc["input_ids"].shape[1]) * batch_size
        times.append(dt / n_gen * 1000)

    avg = sum(times) / len(times)
    tps = 1000 / avg * 1  # throughput (tok/s for ALL samples combined = batch_size / avg)
    # Actually: tps = 1000/avg where avg is ms per token across all samples
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  batch={batch_size:>2d}:  {avg:7.2f} ms/tok  "
          f"({1000/avg:6.1f} tok/s)  peak_GPU={mem:.1f}GB")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return avg

print("\n[bench] batch size scaling (n_new=32, bf16 + attn-dq + fused-gu)")
results = {}
for bs in [1, 2, 4, 8, 16, 32]:
    try:
        results[bs] = bench(bs)
    except torch.cuda.OutOfMemoryError:
        print(f"  batch={bs:>2d}:  OOM")
        gc.collect()
        torch.cuda.empty_cache()
        break

print("\n[summary]")
if 4 in results:
    base = results[4]
    for bs, t in results.items():
        print(f"  batch={bs:>2d}: {t:7.2f} ms/tok  speedup vs b=4: {base/t:.2f}x")
