#!/usr/bin/env python3
"""
Quantize Qwen3-Next-80B  Int4 → Int2  using Intel AutoRound (W2A16).
Compatible with AutoRound 0.12.1.

Strategy
--------
1. Load Int4 model on **CPU** (avoids GPU OOM during dequantization).
2. Patch _init_weights so QuantLinear layers don't trigger AttributeError.
3. Patch all QuantLinear subclasses (torch/triton variants) to expose a
   .weight property (on-the-fly dequant) and auto-infer missing attrs
   (bits, maxq, wf, group_size, …).
4. Replace every QuantLinear in-place with a plain nn.Linear (BF16).
   System has ~938 GB RAM; 80B × 2 B ≈ 160 GB fits comfortably.
5. Pass the fully-float model + local calibration data to AutoRound.
   AutoRound now sees normal nn.Linear layers and can apply W2A16.

Mixed precision
---------------
  Expert weights   : W2A16 (default scheme)
  Attention projections / mlp.gate: W4A16 (via layer_config)
  shared_expert_gate: skipped (fp, critical for routing)

Calibration data
----------------
  Local file: calibration_datasets/compiled/w2a16_requests.jsonl
  Prompts are short (~109 tokens median), so we concatenate them all
  and slice into seqlen-length chunks (seqlen=512, nsamples=128).
"""

import json
import os
import random
import sys
import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/home/kec23008/Models/Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound"
OUTPUT_PATH = "/home/kec23008/Models/Qwen3-Next-80B-A3B-Instruct-int2-mixed-AutoRound"
CALIB_JSONL = "/home/kec23008/DynaQuant/calibration_datasets/compiled/w2a16_requests.jsonl"

NUM_LAYERS       = 48
SELF_ATTN_PERIOD = 4   # layer_idx % 4 == 3  →  self_attn
SEQLEN           = 512
NSAMPLES         = 128


# ─── Fix 1: patch _init_weights ───────────────────────────────────────────────
def patch_qwen3next_init_weights():
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextPreTrainedModel
    except ImportError:
        return
    orig = Qwen3NextPreTrainedModel._init_weights
    def safe(self, module):
        if hasattr(module, "qweight") and not hasattr(module, "weight"):
            return
        try:
            orig(self, module)
        except AttributeError:
            pass
    Qwen3NextPreTrainedModel._init_weights = safe
    logger.info("Patched Qwen3NextPreTrainedModel._init_weights.")


# ─── Fix 2: dequant helpers (standalone, no class dependency) ─────────────────
def _infer_bits(module) -> int:
    """Infer bit-width from qzeros / qweight shape."""
    out_features = module.qweight.shape[1]
    bits = int(module.qzeros.shape[1] * 32 // out_features)
    return bits if bits in (2, 3, 4, 8) else 4


def dequant_to_linear(module) -> torch.nn.Linear:
    """
    Convert any QuantLinear (GPTQ-packed) to nn.Linear with BF16 weights.
    Works for 2/4/8-bit symmetric and asymmetric quantization.
    """
    bits      = _infer_bits(module)
    maxq      = 2**bits - 1
    dt        = torch.int16 if bits == 8 else torch.int8
    qweight   = module.qweight  # [in//32*bits, out]
    scales    = module.scales   # [num_groups, out]
    qzeros    = module.qzeros   # [num_groups, out//32*bits]
    bias      = getattr(module, "bias", None)
    g_idx     = getattr(module, "g_idx", None)
    if g_idx is not None and (not torch.is_tensor(g_idx) or g_idx.numel() == 0):
        g_idx = None

    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32,
                      device=qweight.device).unsqueeze(0)

    # Unpack zeros → [num_groups, out]
    zeros = torch.bitwise_right_shift(
        qzeros.unsqueeze(2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)
    ).to(dt)
    zeros = (zeros & maxq).reshape(scales.shape) + 1  # +1: GPTQ convention

    # Unpack weights → [in, out]
    weight = torch.bitwise_right_shift(
        qweight.unsqueeze(1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)
    ).to(dt)
    weight = (weight & maxq).reshape(-1, qweight.shape[1])

    in_f = weight.shape[0]
    if g_idx is not None and g_idx.numel() == in_f:
        w_f = scales[g_idx.long()] * (weight.float() - zeros[g_idx.long()])
    else:
        gs = in_f // scales.shape[0]
        rs = scales.repeat_interleave(gs, 0)
        rz = zeros.repeat_interleave(gs, 0)
        w_f = rs * (weight.float() - rz)

    w_bf16 = w_f.T.to(torch.bfloat16).cpu()  # [out, in]

    linear = torch.nn.Linear(
        in_features=w_bf16.shape[1],
        out_features=w_bf16.shape[0],
        bias=bias is not None,
        dtype=torch.bfloat16,
    )
    linear.weight.data = w_bf16
    if bias is not None:
        linear.bias.data = bias.cpu().to(torch.bfloat16)
    return linear


def dequantize_model(model) -> int:
    """Replace every QuantLinear (has qweight) with nn.Linear in-place."""
    count = 0
    for name, module in list(model.named_modules()):
        if not hasattr(module, "qweight"):
            continue
        parts = name.rsplit(".", 1)
        parent     = model.get_submodule(parts[0]) if len(parts) == 2 else model
        child_name = parts[-1]
        new_linear = dequant_to_linear(module)
        setattr(parent, child_name, new_linear)
        count += 1
        if count % 100 == 0:
            logger.info("  dequantized %d layers so far…", count)
    logger.info("Dequantized %d QuantLinear → nn.Linear (BF16).", count)
    return count


# ─── Calibration data ─────────────────────────────────────────────────────────
def build_calib_dataset(tokenizer, path: str, seqlen: int, nsamples: int) -> list[str]:
    """
    Concatenate all prompts from the JSONL, tokenize, slice into
    seqlen-length chunks, return up to nsamples text strings.
    """
    texts = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            t = obj.get("prompt") or obj.get("text") or obj.get("content", "")
            if t:
                texts.append(t)

    random.seed(42)
    random.shuffle(texts)

    combined = "\n\n".join(texts)
    token_ids = tokenizer.encode(combined, add_special_tokens=False)
    logger.info("Total tokens in calibration corpus: %d", len(token_ids))

    chunks = []
    for i in range(0, len(token_ids) - seqlen, seqlen):
        chunk_text = tokenizer.decode(token_ids[i : i + seqlen], skip_special_tokens=True)
        chunks.append(chunk_text)
        if len(chunks) >= nsamples:
            break

    logger.info(
        "Built %d calibration chunks × %d tokens (needed %d).",
        len(chunks), seqlen, nsamples,
    )
    if len(chunks) < nsamples:
        logger.warning(
            "Only %d chunks available (< %d). Consider a larger dataset or lower seqlen.",
            len(chunks), nsamples,
        )
    return chunks


# ─── Layer config ─────────────────────────────────────────────────────────────
def build_layer_config(num_layers: int):
    layer_config = {}
    ignore_layers = []
    for idx in range(num_layers):
        is_sa = (idx % SELF_ATTN_PERIOD == 3)
        if is_sa:
            for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                layer_config[f"model.layers.{idx}.self_attn.{p}"] = {"bits": 4}
        else:
            for p in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
                layer_config[f"model.layers.{idx}.linear_attn.{p}"] = {"bits": 4}
        layer_config[f"model.layers.{idx}.mlp.gate"] = {"bits": 4}
        ignore_layers.append(f"model.layers.{idx}.mlp.shared_expert_gate")
    return layer_config, ignore_layers


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        from auto_round import AutoRound
        from auto_round.autoround import ExtraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.error("Import error: %s", e)
        sys.exit(1)

    # 1. Patch _init_weights before any model loading
    patch_qwen3next_init_weights()

    # 2. Load model on CPU (938 GB RAM available; avoids GPU OOM during dequant)
    logger.info("Loading Int4 model to CPU …")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 3. Dequantize ALL QuantLinear → nn.Linear (BF16) in CPU RAM
    logger.info("Dequantizing model in CPU RAM (~160 GB)…")
    n = dequantize_model(model)
    if n == 0:
        logger.warning("No QuantLinear found – model may already be float.")

    # 4. Build calibration dataset from local JSONL
    calib = build_calib_dataset(tokenizer, CALIB_JSONL, seqlen=SEQLEN, nsamples=NSAMPLES)
    actual_n = len(calib)

    # 5. Per-layer config
    layer_config, ignore_layers = build_layer_config(NUM_LAYERS)
    logger.info("layer_config: %d overrides, ignore_layers: %d", len(layer_config), len(ignore_layers))
    logger.info("Model  : %s", MODEL_PATH)
    logger.info("Output : %s", OUTPUT_PATH)
    logger.info("Scheme : W2A16 (attn/gate=4-bit, shared_expert_gate=skip)")

    extra_config = ExtraConfig(
        ignore_layers=",".join(ignore_layers),
        enable_alg_ext=True,   # AutoRound docs: recommended for INT2
    )

    # 6. Run AutoRound – pass pre-loaded float model; device_map="auto" moves
    #    blocks to GPU 0 (48.7 GB free) one at a time during calibration.
    ar = AutoRound(
        model=model,
        tokenizer=tokenizer,
        scheme="W2A16",
        layer_config=layer_config,
        extra_config=extra_config,
        dataset=calib,
        iters=200,
        nsamples=actual_n,
        seqlen=SEQLEN,
        batch_size=1,
        device_map="auto",
        low_gpu_mem_usage=True,
    )

    logger.info("Starting quantization (several hours for 80 B)…")
    ar.quantize_and_save(OUTPUT_PATH)
    logger.info("Done! Saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
