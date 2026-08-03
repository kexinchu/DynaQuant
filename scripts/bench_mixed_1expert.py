"""
bench_mixed_1expert.py — 诊断 INT4 混合精度推理速度

测试场景:
  A. 全 INT4 基线 (当前路径)
  B. 1 个 FP16 expert (最热 expert 提升)
  C. 逐层 1 个 FP16 expert (每层最热)

同时输出:
  - 单次 forward pass 耗时 breakdown
  - token/s 吞吐
  - QuantLinear forward 耗时 vs nn.Linear forward 耗时

用法:
  CUDA_VISIBLE_DEVICES=1 python scripts/bench_mixed_1expert.py
"""

import gc
import sys
import time
from collections import Counter, defaultdict

import torch
import torch.nn as nn

MODEL_PATH = "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"
DEVICE = "cuda:0"   # CUDA_VISIBLE_DEVICES=1 remaps to 0
PROMPT = "Explain the transformer architecture and how attention works."
WARMUP = 3
ITERS  = 10
N_NEW_TOKENS = 64   # short enough for quick timing

# ─── Load model ──────────────────────────────────────────────────────────────
print("[load] Loading model …")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map=DEVICE,
    trust_remote_code=True,
)
model.eval()
print(f"[load] Done in {time.time()-t0:.1f}s  |  GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ─── Identify QuantLinear and nn.Linear types ─────────────────────────────
# Find the QuantLinear class used by auto_round
sample_expert = model.model.layers[1].mlp.experts[0]
QuantLinear = type(sample_expert.gate_proj)
print(f"[info] Expert linear type: {QuantLinear.__name__} from {QuantLinear.__module__}")

# ─── Tokenize prompt ─────────────────────────────────────────────────────────
enc = tok(PROMPT, return_tensors="pt").to(DEVICE)
input_ids = enc["input_ids"]

# ─── Measure single QuantLinear forward time ─────────────────────────────────
print("\n[probe] Timing single QuantLinear.forward vs nn.Linear.forward …")
gp = sample_expert.gate_proj  # QuantLinear
N_out, N_in = gp.weight.shape if hasattr(gp, 'weight') else (768, 2048)
# get actual shapes from scales
N_out = gp.scales.shape[1]      # out_features
N_in  = gp.qweight.shape[0] * 8  # in_features (gptq: K//8 * 8)
print(f"  gate_proj: in={N_in} out={N_out}")

x_probe = torch.randn(1, N_in, dtype=torch.float16, device=DEVICE)

# warm up
for _ in range(5): gp(x_probe)
torch.cuda.synchronize()

N = 200
t0 = time.time()
for _ in range(N): gp(x_probe)
torch.cuda.synchronize()
t_quant = (time.time()-t0)/N*1e3
print(f"  QuantLinear forward (1 token): {t_quant:.3f} ms")

# nn.Linear comparison
lin = nn.Linear(N_in, N_out, bias=False, dtype=torch.float16, device=DEVICE)
for _ in range(5): lin(x_probe)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N): lin(x_probe)
torch.cuda.synchronize()
t_fp16 = (time.time()-t0)/N*1e3
print(f"  nn.Linear  forward (1 token): {t_fp16:.3f} ms")
print(f"  speedup if using nn.Linear: {t_quant/t_fp16:.2f}x")

del lin

# ─── Collect router stats to find hottest expert ──────────────────────────────
print("\n[calib] Collecting router statistics …")
counts: dict[int, Counter] = defaultdict(Counter)
hooks = []

def make_hook(li):
    def fn(mod, inp, out):
        logits = out[0] if isinstance(out, tuple) else out
        if isinstance(logits, torch.Tensor) and logits.dim() == 2:
            _, idx = torch.topk(logits, min(8, logits.shape[-1]), dim=-1)
            for i in idx.flatten().cpu().tolist():
                counts[li][i] += 1
    return fn

for li, layer in enumerate(model.model.layers):
    gate = getattr(layer.mlp, "gate", None)
    if gate is not None:
        hooks.append(gate.register_forward_hook(make_hook(li)))

calib_prompts = [
    "Explain the transformer architecture.",
    "Write a Python function for binary search.",
    "What is the Pythagorean theorem?",
    "Describe the water cycle.",
]
for p in calib_prompts:
    ids = tok(p, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad(): model(ids)

for h in hooks: h.remove()
print(f"  Tracked {sum(len(v) for v in counts.values())} expert-activation slots")

# Global hottest expert
all_acts = [(cnt, li, ei) for li, c in counts.items() for ei, cnt in c.items()]
all_acts.sort(reverse=True)
hottest_global = [(li, ei) for _, li, ei in all_acts[:1]]
hottest_per_layer = {li: max(c, key=c.get) for li, c in counts.items()}

print(f"  Hottest global: layer={hottest_global[0][0]}, expert={hottest_global[0][1]}")
print(f"  Hottest per-layer: {len(hottest_per_layer)} layers")

# ─── Helper: promote/demote ────────────────────────────────────────────────────
promoted: dict = {}
originals: dict = {}

def promote_expert(layer: int, expert: int):
    key = (layer, expert)
    if key in promoted:
        return
    exp_mod = model.model.layers[layer].mlp.experts[expert]
    originals[key] = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        qlin = getattr(exp_mod, proj)
        out_f = qlin.scales.shape[1]
        in_f  = qlin.qweight.shape[0] * 8
        # dequantize to FP16
        w_fp16 = _gptq_dequant(qlin, out_f, in_f).to(DEVICE)
        new_lin = nn.Linear(in_f, out_f, bias=False, dtype=torch.float16, device=DEVICE)
        new_lin.weight.data.copy_(w_fp16)
        originals[key][proj] = qlin
        setattr(exp_mod, proj, new_lin)
    promoted[key] = True
    print(f"  Promoted layer={layer} expert={expert}")

def demote_expert(layer: int, expert: int):
    key = (layer, expert)
    if key not in promoted:
        return
    exp_mod = model.model.layers[layer].mlp.experts[expert]
    for proj, orig in originals[key].items():
        setattr(exp_mod, proj, orig)
    del promoted[key], originals[key]


def _gptq_dequant(qlin, out_f, in_f, bits=4, group_size=64):
    """Fast GPTQ dequant: (K//8, N) int32 → (N, K) fp16."""
    qw = qlin.qweight   # (K//8, N) int32 or (K//k, N)
    scales = qlin.scales # (K//g, N)
    qzeros = qlin.qzeros # (K//g, N//8)
    n_groups = in_f // group_size
    elems = 32 // bits  # 8

    # Unpack qweight → (K, N) int32 values in [0,15]
    w = torch.zeros(in_f, out_f, dtype=torch.int32, device=qw.device)
    for i in range(elems):
        w[i::elems] = (qw >> (i * bits)) & 0xF

    # Unpack qzeros → (n_groups, N) int32
    z = torch.zeros(n_groups, out_f, dtype=torch.int32, device=qzeros.device)
    for i in range(elems):
        z[:, i::elems] = (qzeros >> (i * bits)) & 0xF

    # Dequant: (w - z) * scale per group
    out = torch.zeros(in_f, out_f, dtype=torch.float32, device=qw.device)
    for g in range(n_groups):
        ks, ke = g * group_size, (g+1) * group_size
        out[ks:ke] = ((w[ks:ke].float() - z[g].float().unsqueeze(0)) *
                      scales[g].float().unsqueeze(0))
    return out.half().T.contiguous()  # (N, K)


# ─── Benchmark helper ─────────────────────────────────────────────────────────
def bench_generate(label: str, n_new=N_NEW_TOKENS, iters=ITERS, warmup=WARMUP):
    # warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=8, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=n_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n_gen = out.shape[1] - input_ids.shape[1]
        times.append(dt / n_gen * 1000)  # ms per token

    avg = sum(times) / len(times)
    tps = 1000 / avg
    print(f"  {label:40s}: {avg:7.2f} ms/tok  ({tps:6.1f} tok/s)")
    return avg


# ─── Scenario A: All INT4 ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("Scenario A: All INT4 (GPTQ baseline)")
print("="*60)
t_all_int4 = bench_generate("all INT4")

# ─── Scenario B: 1 FP16 expert (global hottest) ───────────────────────────────
print("\n" + "="*60)
print("Scenario B: 1 FP16 expert (global hottest)")
print("="*60)
li_h, ei_h = hottest_global[0]
promote_expert(li_h, ei_h)
t_1fp16 = bench_generate(f"1 FP16 (layer={li_h}, expert={ei_h})")
demote_expert(li_h, ei_h)
gc.collect(); torch.cuda.empty_cache()

# ─── Scenario C: 1 FP16 expert per layer (N_layers FP16 total) ───────────────
print("\n" + "="*60)
print(f"Scenario C: 1 FP16 expert per layer ({len(hottest_per_layer)} experts FP16)")
print("="*60)
for li, ei in hottest_per_layer.items():
    promote_expert(li, ei)
print(f"  Total promoted: {len(promoted)}")
t_per_layer = bench_generate(f"1 FP16/layer ({len(promoted)} total)")
for li, ei in list(hottest_per_layer.items()):
    demote_expert(li, ei)
gc.collect(); torch.cuda.empty_cache()

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"  QuantLinear forward (1 tok):  {t_quant:.3f} ms")
print(f"  nn.Linear   forward (1 tok):  {t_fp16:.3f} ms")
print(f"  Linear speedup:               {t_quant/t_fp16:.2f}x")
print()
print(f"  All INT4:             {t_all_int4:.2f} ms/tok  ({1000/t_all_int4:.1f} tok/s)")
print(f"  1 FP16 expert:        {t_1fp16:.2f} ms/tok  ({1000/t_1fp16:.1f} tok/s)")
print(f"  1 FP16/layer:         {t_per_layer:.2f} ms/tok  ({1000/t_per_layer:.1f} tok/s)")
print()

n_experts_total = 128  # Qwen3-30B-A3B has 128 experts
n_layers = len(counts)
print(f"  Model: {n_layers} MoE layers, {n_experts_total} experts each")
print(f"  Hottest 1 expert covers ~{all_acts[0][0]/sum(c for _,_,c in all_acts)*100:.1f}% "
      f"of all activations")
print()
print("[diagnosis]")
if t_quant / t_fp16 > 3:
    print("  ⚠ GPTQ QuantLinear is significantly slower than nn.Linear.")
    print("  → Root cause: no gptqmodel installed, using slow fallback.")
    print("  → Fix: pip install -v 'gptqmodel>=2.0' --no-build-isolation")
    print("         OR use our _weight_int4pack_mm path (DynaExQ)")
if t_1fp16 / t_all_int4 > 0.95:
    print("  ⚠ Promoting 1 expert gives negligible speedup.")
    print("  → Expected: the single hot expert is only a small fraction of total time.")
