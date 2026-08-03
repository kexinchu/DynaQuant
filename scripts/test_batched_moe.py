"""test_batched_moe.py — Prototype batched-dequant + bmm MoE forward.

Replaces the Python expert loop with:
  1. Sort tokens by expert (one kernel)
  2. Batch-dequant all hit experts' weights (vectorized torch ops)
  3. ONE torch.bmm for gate+up, ONE for down
  4. Scatter results (one kernel)

Benchmarks against the per-expert int4pack loop.
"""
import os, sys, gc, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.patch_autoround_quantlinear import (
    apply_patch, dequantize_int8_attention,
)

MODEL = "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"
DEV = "cuda:0"

# ═══════════════════════════════════════════════════════════════════════════
# Batched dequant helper
# ═══════════════════════════════════════════════════════════════════════════

def batch_dequant_experts(expert_indices, experts, proj_names, device, dtype):
    """Dequant multiple experts' weights in one vectorized GPU op.

    Args:
        expert_indices: 1-D int tensor of expert indices to dequant (on GPU)
        experts: nn.ModuleList of expert MLPs
        proj_names: list of ("gate_proj", "up_proj") or ("down_proj",)
        device, dtype: target

    Returns:
        Stacked weight (N_experts, total_out_features, in_features) in dtype.
    """
    # Collect buffers (Python loop — just pointer collection, no compute)
    qw_list, sc_list, qz_list = [], [], []
    idx_cpu = expert_indices.cpu().tolist()
    for e_idx in idx_cpu:
        for pn in proj_names:
            ql = getattr(experts[e_idx], pn)
            qw_list.append(ql.qweight)
            sc_list.append(ql.scales)
            qz_list.append(ql.qzeros)

    n_total = len(qw_list)
    # Stack on GPU — one memcpy each
    qw = torch.stack(qw_list)    # (N, K/8, out_f)
    sc = torch.stack(sc_list)    # (N, n_grp, out_f)
    qz = torch.stack(qz_list)   # (N, n_grp, out_f/8)

    bits = 4
    shifts = torch.arange(0, 32, bits, device=device, dtype=torch.int32)

    # Unpack qweight → int4 values
    w = ((qw.unsqueeze(-1) >> shifts) & 0xF).to(torch.int32)  # (N, K/8, out_f, 8)
    N_exp, K8, out_f, pf = w.shape
    K = K8 * pf
    w = w.permute(0, 1, 3, 2).reshape(N_exp, K, out_f)         # (N, K, out_f)

    # Unpack qzeros → actual zeros
    z = ((qz.unsqueeze(-1) >> shifts) & 0xF).to(torch.int32)   # (N, n_grp, out_f/8, 8)
    n_grp = z.shape[1]
    z = z.reshape(N_exp, n_grp, out_f) + 1

    # Dequant per group: vectorized
    g = K // n_grp
    w = w.reshape(N_exp, n_grp, g, out_f)
    w_fp = (w.float() - z.float().unsqueeze(2)) * sc.float().unsqueeze(2)
    w_fp = w_fp.reshape(N_exp, K, out_f)

    # → (N_experts, total_out_features, K)  for bmm: out = input @ W.T
    w_out = w_fp.to(dtype).permute(0, 2, 1).contiguous()

    # If multiple proj_names, reshape: (N_experts * n_proj, out_f, K) → (N_experts, total_out_f, K)
    n_proj = len(proj_names)
    if n_proj > 1:
        n_exp = len(idx_cpu)
        w_out = w_out.reshape(n_exp, n_proj, out_f, K)
        w_out = w_out.reshape(n_exp, n_proj * out_f, K)

    return w_out.contiguous()


# ═══════════════════════════════════════════════════════════════════════════
# Batched MoE forward
# ═══════════════════════════════════════════════════════════════════════════

def batched_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    M = hidden_states.shape[0]
    dev = hidden_states.device
    dtype = hidden_states.dtype

    # ── Router ────────────────────────────────────────────────────────────
    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(
        routing_weights, self.top_k, dim=-1
    )
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(dtype)

    # ── Flatten + sort by expert ──────────────────────────────────────────
    flat_experts = selected_experts.view(-1)                       # (M*top_k,)
    flat_token_idx = (
        torch.arange(M, device=dev)
        .unsqueeze(1).expand(-1, self.top_k).reshape(-1)
    )                                                               # (M*top_k,)
    flat_weights = routing_weights.view(-1)                         # (M*top_k,)

    sort_idx = flat_experts.argsort(stable=True)
    sorted_experts = flat_experts[sort_idx]
    sorted_tokens = flat_token_idx[sort_idx]
    sorted_weights = flat_weights[sort_idx]
    sorted_hidden = hidden_states[sorted_tokens]                    # (M*top_k, D)

    # ── Expert boundaries ─────────────────────────────────────────────────
    unique_experts, counts = torch.unique_consecutive(
        sorted_experts, return_counts=True
    )
    n_hit = unique_experts.shape[0]
    max_m = counts.max().item()

    # ── Pad inputs → (n_hit, max_m, D) ───────────────────────────────────
    offsets = counts.cumsum(0)
    padded = torch.zeros(n_hit, max_m, hidden_dim, device=dev, dtype=dtype)
    starts = torch.cat([torch.zeros(1, device=dev, dtype=offsets.dtype), offsets[:-1]])
    for i in range(n_hit):
        s, c = starts[i].item(), counts[i].item()
        padded[i, :c] = sorted_hidden[s:s+c]

    # ── Batch dequant gate+up → (n_hit, 2*interm, D) ─────────────────────
    interm = self.experts[0].gate_proj.outfeatures
    gate_up_w = batch_dequant_experts(
        unique_experts, self.experts, ["gate_proj", "up_proj"], dev, dtype
    )   # (n_hit, 2*interm, D)

    # ── Grouped GEMM: gate+up ─────────────────────────────────────────────
    gu_out = torch.bmm(padded, gate_up_w.transpose(-1, -2))        # (n_hit, max_m, 2*interm)

    # ── SwiGLU ────────────────────────────────────────────────────────────
    gate_out = gu_out[:, :, :interm]
    up_out = gu_out[:, :, interm:]
    h = F.silu(gate_out) * up_out                                   # (n_hit, max_m, interm)

    # ── Batch dequant down → (n_hit, D, interm) ──────────────────────────
    down_w = batch_dequant_experts(
        unique_experts, self.experts, ["down_proj"], dev, dtype
    )   # (n_hit, D, interm)

    # ── Grouped GEMM: down ────────────────────────────────────────────────
    down_out = torch.bmm(h, down_w.transpose(-1, -2))              # (n_hit, max_m, D)

    # ── Scatter back with routing weights ─────────────────────────────────
    final = torch.zeros(M, hidden_dim, device=dev, dtype=dtype)
    for i in range(n_hit):
        s, c = starts[i].item(), counts[i].item()
        tok_idx = sorted_tokens[s:s+c]
        w = sorted_weights[s:s+c].unsqueeze(-1)
        final.index_add_(0, tok_idx, down_out[i, :c] * w)

    return final.reshape(batch_size, sequence_length, hidden_dim)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

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
print(f"[load] GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

# ── Step 1: verify correctness on one layer ──────────────────────────────
print("\n[verify] batched vs original MoE block output")
sample_moe = model.model.layers[1].mlp   # LinearQwen3MoeSparseMoeBlock
x_test = torch.randn(4, 8, 2048, dtype=torch.bfloat16, device=DEV) * 0.1

with torch.no_grad():
    ref = sample_moe(x_test)

# Monkey-patch forward
orig_fwd = type(sample_moe).forward
type(sample_moe).forward = batched_moe_forward

with torch.no_grad():
    batched = sample_moe(x_test)

diff = (ref - batched).abs()
print(f"  ref  norm: {ref.abs().mean().item():.6f}")
print(f"  diff max:  {diff.max().item():.6f}")
print(f"  diff mean: {diff.mean().item():.6f}")
print(f"  rel err:   {diff.mean().item() / max(ref.abs().mean().item(), 1e-8):.4%}")

# ── Step 2: benchmark full decode ─────────────────────────────────────────
prompts_b32 = [
    "Explain the transformer architecture.", "Write quicksort in Python.",
    "What causes the northern lights?", "Describe photosynthesis.",
    "What is the Pythagorean theorem?", "Explain general relativity simply.",
    "Write binary search in Python.", "Describe the water cycle.",
    "What are prime numbers?", "Explain how neural networks learn.",
    "Write merge sort in Python.", "Describe black holes.",
    "What is the Fibonacci sequence?", "Explain quantum computing basics.",
    "Write a stack implementation in Python.", "Describe how vaccines work.",
    "What is the theory of evolution?", "Explain how GPUs differ from CPUs.",
    "Write a linked list in Python.", "Describe the greenhouse effect.",
    "What is machine learning?", "Explain recursion with an example.",
    "Write a hash table in Python.", "Describe how airplanes fly.",
    "What is dark matter?", "Explain the Fourier transform.",
    "Write breadth-first search in Python.", "Describe the human immune system.",
    "What is a neural network layer?", "Explain convex optimization briefly.",
    "Write depth-first search in Python.", "Describe CRISPR gene editing.",
]

def bench(label, n_new=32, n_runs=2, batch_size=32):
    prompts = prompts_b32[:batch_size]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=512).to(DEV)
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
    print(f"  {label:45s}: {avg:7.2f} ms/tok  ({1000/avg:6.1f} tok/s)")
    return avg

print("\n[bench] batched-dequant + bmm MoE (batch=32)")
t_batched = bench("b=32 batched-dequant+bmm")

print("\n[summary]")
print(f"  original per-expert int4pack (batch=32):        34.16 ms/tok")
print(f"  batched-dequant + bmm        (batch=32):      {t_batched:6.2f} ms/tok")
print(f"  speedup: {34.16/t_batched:.2f}x")
print(f"  total speedup vs first baseline (167 ms/tok): {167.11/t_batched:.1f}x")
