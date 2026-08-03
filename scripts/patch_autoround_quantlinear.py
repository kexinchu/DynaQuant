"""
patch_autoround_quantlinear.py — Monkey-patch auto_round QuantLinear to use
PyTorch native _weight_int4pack_mm kernel instead of Triton/slow-torch.

Why this helps:
  The auto_round:tritonv2_zp backend is selected for this model, but the Triton
  kernel has high per-call overhead for the tiny per-expert batch sizes seen
  during MoE decode (often 0-2 tokens per expert). The PyTorch-native
  _weight_int4pack_mm kernel is faster for M=1..few token batches.

Usage — call apply_patch() BEFORE loading the model, or AFTER loading:

  from scripts.patch_autoround_quantlinear import apply_patch
  apply_patch()                                # patches all 4-bit QuantLinear
  model = AutoModelForCausalLM.from_pretrained(...)

  # Or after loading:
  model = AutoModelForCausalLM.from_pretrained(...)
  apply_patch(model)   # warm-caches all layers immediately (optional)

The patch is lightweight — it lazily builds the int4pack cache on the first
forward call for each layer and reuses it forever.  The original quantized
weights (qweight/scales/qzeros) are kept so ExpertSwapper can still demote
layers back to nn.Linear if needed.

GPTQ packing ↔ int4pack conventions
-------------------------------------
GPTQ qweight (K//8, N) int32:
  bit 4j … 4j+3 of word [k//8, n]  =  weight(k = (k//8)*8+j, n)

int4pack kernel nibble convention (empirically verified in DynaExQ):
  low nibble  (bits 0-3) = weight at odd   K index  (k+1)
  high nibble (bits 4-7) = weight at even  K index  (k)

So the uint8 byte for kernel at (n, k//2) is:
  byte = w_int4[k+1, n] | (w_int4[k, n] << 4)
       = w_odd << 0 | w_even << 4

GPTQ zero convention:
  packed qzero = actual_zero - 1  (add 1 when unpacking, as in forward())
  int4pack kernel formula: out = (w - 8) * scale + zero_fp
  ⟹ zero_fp = (8 - actual_zero) * scale
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

_INT4PACK_AVAILABLE: bool = (
    hasattr(torch.ops.aten, "_weight_int4pack_mm")
    and hasattr(torch.ops.aten, "_convert_weight_to_int4pack")
)

# Sentinel attribute name stored on each patched QuantLinear instance
_CACHE_ATTR = "_int4pack_cache"   # tuple (int4pack_weight, scales_and_zeros) or None


# ─── Core conversion (GPTQ → int4pack) ────────────────────────────────────────

def _build_int4pack_cache(ql) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a GPTQ QuantLinear's buffers to _weight_int4pack_mm format.

    Supports bits=4 only (the common case for MoE experts).
    """
    bits = ql.bits
    if bits != 4:
        raise ValueError(f"_weight_int4pack_mm only supports bits=4, got {bits}")

    K      = ql.infeatures
    N      = ql.outfeatures
    g      = ql.group_size
    dev    = ql.qweight.device
    n_grp  = K // g

    # ── Step 1: Unpack qweight (K//8, N) int32 → (K, N) uint8 ────────────────
    qw = ql.qweight  # (K//8, N)
    shifts = torch.arange(0, 32, bits, device=dev, dtype=torch.int32)  # [0,4,8,…,28]
    # unsqueeze: (K//8, N, 1) >> (8,) → (K//8, N, 8)
    w_int4 = ((qw.unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8)  # (K//8, N, 8)
    # Reshape: (K, N) — note we permute dims 0 and 2 then reshape
    w_int4 = w_int4.permute(0, 2, 1).reshape(K, N)  # (K, N)

    # ── Step 2: Repack → (N, K//2) uint8 for the kernel ──────────────────────
    # Kernel convention: low nibble = odd K index, high nibble = even K index
    w_even = w_int4[0::2, :]   # (K//2, N) even k indices
    w_odd  = w_int4[1::2, :]   # (K//2, N) odd  k indices
    uint8_kernel = ((w_even << 4) | w_odd).to(torch.uint8)  # (K//2, N)
    uint8_kernel = uint8_kernel.T.contiguous()               # (N, K//2)

    # ── Step 3: Convert to int4pack internal format ───────────────────────────
    int4pack_w = torch.ops.aten._convert_weight_to_int4pack(uint8_kernel, innerKTiles=2)

    # ── Step 4: Unpack qzeros (n_grp, N//8) int32 → actual zeros (n_grp, N) ──
    qz = ql.qzeros   # (n_grp, N//8)
    zeros_raw = ((qz.unsqueeze(-1) >> shifts) & 0xF).to(torch.int32)  # (n_grp, N//8, 8)
    zeros_raw = zeros_raw.reshape(n_grp, N)                            # (n_grp, N)
    actual_zeros = zeros_raw + 1   # undo the -1 that pack() applies

    # ── Step 5: Build scales_and_zeros (n_grp, N, 2) bf16 ────────────────────
    # Kernel:  out = (w - 8) * scale + zero_fp
    # Match:   (w - z_actual) * scale  ⟹  zero_fp = (8 - z_actual) * scale
    sc = ql.scales  # (n_grp, N) fp16
    zero_fp = (8.0 - actual_zeros.float()) * sc.float()  # (n_grp, N)
    scales_and_zeros = torch.stack(
        [sc.to(torch.bfloat16), zero_fp.to(torch.bfloat16)], dim=-1
    ).contiguous()  # (n_grp, N, 2)

    return int4pack_w, scales_and_zeros


# ─── Patched forward ───────────────────────────────────────────────────────────

def _fast_forward(self, x: torch.Tensor) -> torch.Tensor:
    # Only accelerate INT4 CUDA layers; fall back to original for INT8, INT2, CPU
    if getattr(self, "bits", 4) != 4 or not self.qweight.is_cuda:
        return self.__class__._original_forward(self, x)

    cache = getattr(self, _CACHE_ATTR, None)
    if cache is None:
        cache = _build_int4pack_cache(self)
        setattr(self, _CACHE_ATTR, cache)

    int4pack_w, scales_and_zeros = cache
    orig_dtype = x.dtype
    out_shape = x.shape[:-1] + (self.outfeatures,)
    x_flat = x.reshape(-1, x.shape[-1])

    # _weight_int4pack_mm requires bf16 input/output. If the model is already
    # loaded in bf16, both casts below become no-ops — that's the entire point
    # of loading the model in bf16 for the mixed-precision HF path.
    if x_flat.dtype != torch.bfloat16:
        x_flat = x_flat.to(torch.bfloat16)

    out = torch.ops.aten._weight_int4pack_mm(
        x_flat,
        int4pack_w,
        self.group_size,
        scales_and_zeros,
    )   # (M, N) bf16

    if orig_dtype != torch.bfloat16:
        out = out.to(orig_dtype)

    out = out.reshape(out_shape)
    if self.bias is not None:
        out = out + self.bias.to(out.dtype)
    return out


def _get_quant_linear_classes() -> list[type]:
    """Return all QuantLinear classes from backends that the model might use."""
    classes = []
    try:
        from auto_round_extension.triton.qlinear_tritonv2_zp import QuantLinear as TQL
        classes.append(TQL)
    except ImportError:
        pass
    try:
        from auto_round_extension.triton.qlinear_tritonv2 import QuantLinear as TQL2
        classes.append(TQL2)
    except ImportError:
        pass
    try:
        from auto_round_extension.torch.qlinear_torch_zp import QuantLinear as ThQL
        classes.append(ThQL)
    except ImportError:
        pass
    try:
        from auto_round_extension.torch.qlinear_torch import QuantLinear as ThQL2
        classes.append(ThQL2)
    except ImportError:
        pass
    return classes


# ─── Public API ────────────────────────────────────────────────────────────────

def apply_patch(model: Optional[nn.Module] = None) -> int:
    """Monkey-patch all QuantLinear.forward methods to use _weight_int4pack_mm.

    Args:
        model: If provided, also immediately warms up the int4pack cache for
               every INT4 QuantLinear in the model (skips layers with bits≠4).
               If None, only patches the class-level forward (affects future
               QuantLinear instantiations and already-instantiated ones sharing
               the same class).

    Returns:
        Number of QuantLinear classes patched.
    """
    if not _INT4PACK_AVAILABLE:
        print("[patch_autoround] _weight_int4pack_mm not available — skipping patch")
        return 0

    classes = _get_quant_linear_classes()
    if not classes:
        print("[patch_autoround] No auto_round QuantLinear classes found — skipping patch")
        return 0

    patched = 0
    for cls in classes:
        if getattr(cls, "_int4pack_patched", False):
            continue
        cls._original_forward = cls.forward
        cls.forward = _fast_forward
        cls._int4pack_patched = True
        patched += 1
        print(f"[patch_autoround] Patched {cls.__module__}.{cls.__name__}.forward "
              f"→ _weight_int4pack_mm")

    # Optional: eagerly build caches for all loaded INT4 layers
    if model is not None:
        n_warmed = _warm_caches(model)
        print(f"[patch_autoround] Warmed int4pack cache for {n_warmed} INT4 layers")

    return patched


def _warm_caches(model: nn.Module) -> int:
    """Eagerly build int4pack caches for all INT4 QuantLinear layers in model."""
    n = 0
    for module in model.modules():
        if not getattr(type(module), "_int4pack_patched", False):
            continue
        if getattr(module, "bits", None) != 4:
            continue
        if not module.qweight.is_cuda:
            continue
        try:
            cache = _build_int4pack_cache(module)
            setattr(module, _CACHE_ATTR, cache)
            n += 1
        except Exception as e:
            print(f"[patch_autoround] Warning: cache build failed for a layer: {e}")
    return n


# ─── Universal GPTQ dequantization (any bits ∈ {2,4,8}) ───────────────────────

def dequantize_quantlinear_to_fp16(ql) -> torch.Tensor:
    """Fully dequantize a GPTQ QuantLinear to a dense fp16 weight (N, K).

    Works for bits ∈ {2, 4, 8}. Uses fully-vectorized GPU ops, no Python loops.
    Returns a contiguous (out_features, in_features) fp16 tensor on ql's device.
    """
    bits = ql.bits
    assert bits in (2, 4, 8), f"Unsupported bits={bits}"
    K       = ql.infeatures
    N       = ql.outfeatures
    g       = ql.group_size
    dev     = ql.qweight.device
    n_grp   = K // g
    pack_f  = 32 // bits       # values per int32 word
    mask    = (1 << bits) - 1

    shifts = torch.arange(0, 32, bits, device=dev, dtype=torch.int32)  # (pack_f,)

    # ── Unpack qweight (K/pack_f, N) int32 → (K, N) int ─────────────────────
    qw = ql.qweight
    w_int = ((qw.unsqueeze(-1) >> shifts) & mask).to(torch.int32)  # (K/pf, N, pf)
    w_int = w_int.permute(0, 2, 1).reshape(K, N)                    # (K, N)

    # ── Unpack qzeros (n_grp, N/pack_f) int32 → (n_grp, N) int ──────────────
    qz = ql.qzeros
    z_int = ((qz.unsqueeze(-1) >> shifts) & mask).to(torch.int32)   # (n_grp, N/pf, pf)
    z_int = z_int.reshape(n_grp, N)                                  # (n_grp, N)
    actual_zeros = z_int + 1   # undo pack's -1

    # ── Dequant per group: (w - z) * scale ──────────────────────────────────
    sc = ql.scales  # (n_grp, N) fp16
    w_int = w_int.reshape(n_grp, g, N)
    # broadcast: (n_grp, 1, N)
    w_fp = (w_int.float() - actual_zeros.float().unsqueeze(1)) * sc.float().unsqueeze(1)
    w_fp = w_fp.reshape(K, N)   # (K, N)

    return w_fp.to(torch.float16).T.contiguous()   # (N, K)


# ─── Gate+Up fusion for Qwen3MoeMLP experts ───────────────────────────────────

def _build_fused_gate_up_int4pack_cache(gate_ql, up_ql):
    """Build a single int4pack cache that computes (gate_proj(x) | up_proj(x))
    — the concatenation along output dim — in one _weight_int4pack_mm call.

    Requires gate_ql and up_ql to share bits, group_size, and infeatures.
    """
    assert gate_ql.bits == 4 and up_ql.bits == 4
    assert gate_ql.group_size == up_ql.group_size
    assert gate_ql.infeatures == up_ql.infeatures

    bits = 4
    K      = gate_ql.infeatures
    g      = gate_ql.group_size
    dev    = gate_ql.qweight.device
    n_grp  = K // g
    N_gate = gate_ql.outfeatures
    N_up   = up_ql.outfeatures
    N_tot  = N_gate + N_up

    shifts = torch.arange(0, 32, bits, device=dev, dtype=torch.int32)

    def _unpack_w(ql):
        qw = ql.qweight  # (K/8, N) int32
        w = ((qw.unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8)  # (K/8, N, 8)
        return w.permute(0, 2, 1).reshape(K, ql.outfeatures)       # (K, N)

    def _unpack_z(ql):
        qz = ql.qzeros  # (n_grp, N/8) int32
        z = ((qz.unsqueeze(-1) >> shifts) & 0xF).to(torch.int32)   # (n_grp, N/8, 8)
        return z.reshape(n_grp, ql.outfeatures) + 1                # actual zeros

    # ── Concat int4 weights along N ──────────────────────────────────────────
    w_gate = _unpack_w(gate_ql)                    # (K, N_gate)
    w_up   = _unpack_w(up_ql)                      # (K, N_up)
    w_tot  = torch.cat([w_gate, w_up], dim=1)      # (K, N_tot)

    # ── Repack for int4pack kernel ───────────────────────────────────────────
    w_even = w_tot[0::2, :]                        # (K/2, N_tot)
    w_odd  = w_tot[1::2, :]
    uint8_kernel = ((w_even << 4) | w_odd).to(torch.uint8).T.contiguous()  # (N_tot, K/2)
    int4pack_w = torch.ops.aten._convert_weight_to_int4pack(uint8_kernel, innerKTiles=2)

    # ── Concat scales/zeros ──────────────────────────────────────────────────
    z_gate = _unpack_z(gate_ql)                    # (n_grp, N_gate)
    z_up   = _unpack_z(up_ql)                      # (n_grp, N_up)
    z_tot  = torch.cat([z_gate, z_up], dim=1)      # (n_grp, N_tot)
    sc_tot = torch.cat([gate_ql.scales, up_ql.scales], dim=1)  # (n_grp, N_tot)
    zero_fp = (8.0 - z_tot.float()) * sc_tot.float()
    scales_and_zeros = torch.stack(
        [sc_tot.to(torch.bfloat16), zero_fp.to(torch.bfloat16)], dim=-1
    ).contiguous()                                 # (n_grp, N_tot, 2)

    return int4pack_w, scales_and_zeros


def _fused_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Patched Qwen3MoeMLP.forward that uses the fused gate+up int4pack cache.

    Falls through to the original forward when the fused cache is absent
    (e.g., for experts promoted to FP16 by ExpertSwapper).
    """
    cache = getattr(self, "_fused_gu_cache", None)
    if cache is None:
        # FP16-promoted expert or non-INT4 path
        return self.__class__._original_mlp_forward(self, x)

    int4pack_w, scales_and_zeros = cache
    N_gate = self._fused_gu_N_gate
    g      = self._fused_gu_g

    orig_dtype = x.dtype
    x_flat = x.reshape(-1, x.shape[-1])
    x_bf16 = x_flat if x_flat.dtype == torch.bfloat16 else x_flat.to(torch.bfloat16)

    # ── One int4pack call covers gate AND up ────────────────────────────────
    gu_out = torch.ops.aten._weight_int4pack_mm(
        x_bf16, int4pack_w, g, scales_and_zeros,
    )   # (M, N_gate + N_up) bf16

    gate_out = gu_out[:, :N_gate]
    up_out   = gu_out[:, N_gate:]

    if orig_dtype != torch.bfloat16:
        gate_out = gate_out.to(orig_dtype)
        up_out   = up_out.to(orig_dtype)

    hidden = self.act_fn(gate_out) * up_out        # SwiGLU
    out    = self.down_proj(hidden)                # goes through int4pack fast path
    return out.reshape(x.shape[:-1] + (out.shape[-1],))


def fuse_gate_up_experts(model: nn.Module, verbose: bool = True) -> int:
    """Fuse gate_proj and up_proj into a single int4pack cache for every
    INT4 Qwen3MoeMLP expert in the model. Cuts MoE INT4 expert kernel
    launches from 3 → 2 per expert.

    Memory cost: roughly equal to the combined gate+up int4pack buffers
    (which replace the per-Linear caches — those can be dropped by the
    MLP forward override since gate_proj/up_proj are no longer called
    individually).
    """
    QL_classes = tuple(_get_quant_linear_classes())
    if not QL_classes:
        return 0

    # Discover the MLP class holding the expert (gate_proj, up_proj, down_proj)
    MLP_cls = None
    for mod in model.modules():
        if (hasattr(mod, "gate_proj") and hasattr(mod, "up_proj")
                and hasattr(mod, "down_proj") and hasattr(mod, "act_fn")):
            if isinstance(getattr(mod, "gate_proj"), QL_classes):
                MLP_cls = type(mod)
                break

    if MLP_cls is None:
        if verbose:
            print("[patch_autoround] No INT4 Qwen3MoeMLP experts found — skipping fusion")
        return 0

    n_fused = 0
    with torch.no_grad():
        for mod in model.modules():
            if not isinstance(mod, MLP_cls):
                continue
            gate = getattr(mod, "gate_proj", None)
            up   = getattr(mod, "up_proj", None)
            if not isinstance(gate, QL_classes) or not isinstance(up, QL_classes):
                continue
            if gate.bits != 4 or up.bits != 4:
                continue
            if not gate.qweight.is_cuda:
                continue
            try:
                c = _build_fused_gate_up_int4pack_cache(gate, up)
                mod._fused_gu_cache = c
                mod._fused_gu_N_gate = gate.outfeatures
                mod._fused_gu_g = gate.group_size
                # Drop the per-Linear int4pack caches since gate_proj/up_proj
                # won't be called individually anymore.
                if hasattr(gate, _CACHE_ATTR):
                    setattr(gate, _CACHE_ATTR, None)
                if hasattr(up, _CACHE_ATTR):
                    setattr(up, _CACHE_ATTR, None)
                n_fused += 1
            except Exception as e:
                print(f"[patch_autoround] Fuse build failed for one expert: {e}")

    # Class-level forward patch (one-time)
    if n_fused > 0 and not getattr(MLP_cls, "_gate_up_fused_patched", False):
        MLP_cls._original_mlp_forward = MLP_cls.forward
        MLP_cls.forward = _fused_mlp_forward
        MLP_cls._gate_up_fused_patched = True

    torch.cuda.empty_cache()
    if verbose:
        print(f"[patch_autoround] Fused gate+up in {n_fused} INT4 experts "
              f"(3 linear calls/expert → 2)")
    return n_fused


def dequantize_int8_attention(
    model: nn.Module,
    target_dtype: Optional[torch.dtype] = None,
    verbose: bool = True,
) -> int:
    """Replace every INT8 self_attn QuantLinear with a plain nn.Linear.

    INT8 attention projections (q/k/v/o) cannot use _weight_int4pack_mm and
    fall back to the slow Triton path. Dequantizing them to a dense Linear
    trades a small amount of GPU memory (~1 GB for a 48-layer Qwen3-30B)
    for much faster cuBLAS GEMM — removes Triton launch overhead.

    Args:
        target_dtype: dtype to use for the replacement nn.Linear. If None,
            inferred from the model's existing parameters (matches whichever
            of fp16/bf16 the rest of the model uses — important so the new
            Linear doesn't force per-call casts).
    """
    QL_classes = tuple(_get_quant_linear_classes())
    if not QL_classes:
        return 0

    if target_dtype is None:
        for p in model.parameters():
            if p.dtype in (torch.float16, torch.bfloat16):
                target_dtype = p.dtype
                break
        else:
            target_dtype = torch.float16

    replaced = 0
    with torch.no_grad():
        for mod_name, module in list(model.named_modules()):
            if "self_attn" not in mod_name:
                continue
            for child_name, child in list(module.named_children()):
                if not isinstance(child, QL_classes):
                    continue
                if child_name not in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    continue
                if getattr(child, "bits", 4) == 4:
                    # INT4 attention would use int4pack fast path already
                    continue
                # Dequantize → (N, K) fp16, then cast to target dtype
                w_dense = dequantize_quantlinear_to_fp16(child).to(target_dtype)
                out_f, in_f = w_dense.shape
                new_lin = nn.Linear(
                    in_f, out_f,
                    bias=(child.bias is not None),
                    dtype=target_dtype,
                    device=w_dense.device,
                )
                new_lin.weight.data.copy_(w_dense)
                if child.bias is not None:
                    new_lin.bias.data.copy_(child.bias.to(target_dtype))
                del w_dense
                setattr(module, child_name, new_lin)
                replaced += 1
    torch.cuda.empty_cache()
    if verbose:
        print(f"[patch_autoround] Dequantized {replaced} INT8 attention "
              f"QuantLinear → {target_dtype} nn.Linear")
    return replaced
