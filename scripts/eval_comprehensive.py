#!/usr/bin/env python3
"""
eval_comprehensive.py  —  Qwen3-30B-A3B Precision Ablation Benchmark Suite

Evaluates MMLU-Pro, GPQA, AIME, GSM8K, HumanEval across:
  fp16            — Full FP16 via vLLM (tensor_parallel=2)
  int4            — INT4 (AutoRound) via HF Transformers
  mixed_10/20/30/40 — INT4 base + top-X% hot experts promoted to FP16

Notes:
  • GPQA requires HF_TOKEN (gated dataset). Set env var HF_TOKEN to enable.
  • AIME uses AI-MO/aimo-validation-aime (real AIME competition problems).
  • HumanEval uses code execution (subprocess with 10s timeout per test).
  • Mixed configs load FP16 safetensors for the top max_ratio experts into CPU RAM
    (~9 MB/expert; 40% of 6144 experts ≈ 22 GB CPU RAM).

Usage (full sweep, 2 GPUs):
  CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_comprehensive.py \\
    --fp16-path  /home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507 \\
    --int4-path  /home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound \\
    --configs    fp16,int4,mixed_10,mixed_20,mixed_30,mixed_40 \\
    --n-samples  200 \\
    --output     results/comprehensive_eval.json

  # FP16 only (needs 2 GPUs):
  CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_comprehensive.py \\
    --fp16-path ... --configs fp16

  # INT4 + mixed only (1 GPU OK, but 2 GPUs handle 40% mixed more comfortably):
  CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_comprehensive.py \\
    --fp16-path ... --int4-path ... --configs int4,mixed_10,mixed_20,mixed_30,mixed_40
"""

import argparse
import gc
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── HuggingFace login (for gated datasets like GPQA) ──────────────────────────
_HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if _HF_TOKEN:
    try:
        from huggingface_hub import login as hf_login
        hf_login(token=_HF_TOKEN, add_to_git_credential=False)
        print("[init] HF token found, logged in to Hugging Face Hub.")
    except Exception as _e:
        print(f"[init] HF login failed: {_e}")
else:
    print("[init] No HF_TOKEN set — gated datasets (e.g. GPQA) will be skipped.")

from datasets import load_dataset as hf_load_dataset  # noqa: E402  (after token setup)

# NOTE: autoround QuantLinear patch is applied lazily (only before HF model
# load for mixed configs) to avoid initializing CUDA at import time.
# Early CUDA init prevents vLLM v1 from forking its EngineCore subprocess.
def _apply_autoround_patch():
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.patch_autoround_quantlinear import apply_patch as _p
        _p()
        print("[hf] auto_round QuantLinear patched → _weight_int4pack_mm")
    except Exception as _e:
        print(f"[hf] QuantLinear patch skipped: {_e}")

# ──────────────────────────────────────────────────────────────────────────────
# CALIBRATION PROMPTS  (diverse; used for expert-activation router statistics)
# ──────────────────────────────────────────────────────────────────────────────
CALIB_PROMPTS = [
    # Math / science
    "Prove that √2 is irrational.",
    "Find all eigenvalues of the 3×3 matrix [[2,1,0],[1,2,1],[0,1,2]].",
    "Solve: ∫₀^∞ e^{-x²} dx.",
    "A ball is thrown upward at 20 m/s. How high does it go?",
    "Calculate the pH of a 0.01 M HCl solution.",
    # Reasoning / knowledge
    "Explain the theory of general relativity in simple terms.",
    "What are the main causes and effects of the 2008 financial crisis?",
    "Describe the central limit theorem and its significance.",
    "What is the difference between deductive and inductive reasoning?",
    # Code / logic
    "Write a Python function to implement binary search.",
    "Implement a trie data structure in Python.",
    "Explain the time complexity of heapsort.",
    "Write Python code to detect a cycle in a linked list.",
    # MMLU-style
    "Which of the following is NOT a property of enzymes? (A) They are catalysts (B) They are consumed in reactions (C) They lower activation energy (D) They are proteins",
    "What is the primary function of mitochondria in eukaryotic cells?",
    "In economics, what does 'comparative advantage' mean?",
]

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ──────────────────────────────────────────────────────────────────────────────
SYS_MC = (
    "You are a highly knowledgeable expert. Reason carefully step by step inside "
    "<think>...</think>, then state your final answer on the very last line as "
    "'Answer: X' where X is exactly one capital letter (A, B, C, …)."
)
SYS_MATH = (
    "You are an expert mathematician. Show complete reasoning inside "
    "<think>...</think>, then state the final integer answer on the last line as "
    "'Answer: N' where N is an integer (no units, no words)."
)
SYS_CODE = (
    "You are an expert Python programmer. Think through the algorithm inside "
    "<think>...</think>, then output only the complete Python function "
    "implementation — no markdown fences, no extra explanation, just code."
)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_mmlu_pro(n: int = 200) -> list[dict]:
    print(f"  [mmlu_pro] Loading {n} samples …")
    ds = hf_load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    out = []
    for it in ds:
        q, opts, ans = (
            it.get("question", "").strip(),
            it.get("options", []),
            it.get("answer", "").strip(),
        )
        if not q or not opts or not ans:
            continue
        out.append({
            "bench": "mmlu_pro", "type": "mc",
            "question": q, "options": opts,
            "answer": ans, "n_options": len(opts),
        })
        if len(out) >= n:
            break
    print(f"  [mmlu_pro] {len(out)} samples loaded")
    return out


def load_gpqa(n: int = 200) -> list[dict]:
    """Graduate-level Professional Q&A (Diamond subset). Requires HF_TOKEN."""
    if not _HF_TOKEN:
        print("  [gpqa] SKIPPED — set HF_TOKEN env var to access this gated dataset.")
        return []
    print(f"  [gpqa] Loading gpqa_diamond {n} samples …")
    try:
        ds = hf_load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception as exc:
        print(f"  [gpqa] Load failed: {exc}")
        return []
    rng = random.Random(42)
    out = []
    for it in ds:
        q = it.get("Question", "").strip()
        correct = it.get("Correct Answer", "").strip()
        wrongs = [it.get(f"Incorrect Answer {i}", "").strip() for i in range(1, 4)]
        if not q or not correct:
            continue
        opts = [correct] + [w for w in wrongs if w]
        rng.shuffle(opts)
        ans_letter = chr(65 + opts.index(correct))
        out.append({
            "bench": "gpqa", "type": "mc",
            "question": q, "options": opts,
            "answer": ans_letter, "n_options": len(opts),
        })
        if len(out) >= n:
            break
    print(f"  [gpqa] {len(out)} samples loaded")
    return out


def load_aime(n: int = 30) -> list[dict]:
    """AIME competition problems (AI-MO validation set, integer answers 0-999)."""
    print(f"  [aime] Loading AIME competition samples (n≤{n}) …")
    ds = hf_load_dataset("AI-MO/aimo-validation-aime", split="train")
    out = []
    for it in ds:
        q = it.get("problem", "").strip()
        a = str(it.get("answer", "")).strip()
        if not q or not a:
            continue
        try:
            a = str(int(float(a)))
        except (ValueError, TypeError):
            pass
        out.append({"bench": "aime", "type": "numeric", "question": q, "answer": a})
        if len(out) >= n:
            break
    print(f"  [aime] {len(out)} samples loaded")
    return out


def load_gsm8k(n: int = 200) -> list[dict]:
    print(f"  [gsm8k] Loading {n} test samples …")
    ds = hf_load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for it in ds:
        q = it["question"].strip()
        m = re.search(r"####\s*([\d,]+)", it["answer"])
        if not m:
            continue
        a = m.group(1).replace(",", "")
        out.append({"bench": "gsm8k", "type": "numeric", "question": q, "answer": a})
        if len(out) >= n:
            break
    print(f"  [gsm8k] {len(out)} samples loaded")
    return out


def load_humaneval(n: int | None = None) -> list[dict]:
    print("  [humaneval] Loading HumanEval (164 problems) …")
    ds = hf_load_dataset("openai_humaneval", split="test")
    out = []
    for it in ds:
        out.append({
            "bench": "humaneval", "type": "code",
            "task_id": it["task_id"],
            "prompt": it["prompt"],          # Python signature + docstring
            "test": it["test"],              # test function: check(candidate)
            "entry_point": it["entry_point"],
            "canonical_solution": it["canonical_solution"],
        })
        if n and len(out) >= n:
            break
    print(f"  [humaneval] {len(out)} samples loaded")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _apply_chat(tokenizer, system: str, user: str, thinking: bool) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kw = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(msgs, **kw, enable_thinking=thinking)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, **kw)


def build_prompts(tokenizer, samples: list[dict], thinking: bool = True) -> list[str]:
    prompts = []
    for s in samples:
        if s["type"] == "mc":
            opts = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(s["options"]))
            user = (
                f"Question: {s['question']}\n\n"
                f"Options:\n{opts}\n\n"
                "Reason step by step, then write 'Answer: X' on the last line."
            )
            prompts.append(_apply_chat(tokenizer, SYS_MC, user, thinking))
        elif s["type"] == "numeric":
            user = (
                f"{s['question']}\n\n"
                "Think carefully, then write 'Answer: N' on the last line "
                "where N is the integer answer."
            )
            prompts.append(_apply_chat(tokenizer, SYS_MATH, user, thinking))
        elif s["type"] == "code":
            user = (
                "Complete the following Python function. "
                "Output only the implementation — no markdown fences.\n\n"
                f"```python\n{s['prompt']}\n```"
            )
            prompts.append(_apply_chat(tokenizer, SYS_CODE, user, thinking))
        else:
            prompts.append(s.get("question", ""))
    return prompts


# ══════════════════════════════════════════════════════════════════════════════
# ANSWER EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_mc(text: str, n_options: int) -> str | None:
    c = _strip_think(text)
    valid = {chr(65 + i) for i in range(n_options)}

    # "Answer: B"
    m = re.search(r"[Aa]nswer\s*[:：]\s*([A-J])", c)
    if m and m.group(1) in valid:
        return m.group(1)
    # "the answer is B"
    m = re.search(r"(?i)the answer is\s+([A-J])", c)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    # Last standalone letter on its own line
    for line in reversed(c.splitlines()):
        tok = line.strip().rstrip(".).:")
        if tok in valid:
            return tok
    # Last capital letter in clean text
    for ch in reversed(re.findall(r"\b([A-J])\b", c)):
        if ch in valid:
            return ch
    return None


def extract_number(text: str) -> str | None:
    c = _strip_think(text)

    # "Answer: 42" or "Answer: 3.14"
    m = re.search(r"[Aa]nswer\s*[:：]\s*([-\d,\.]+)", c)
    if m:
        try:
            return str(int(round(float(m.group(1).replace(",", "")))))
        except (ValueError, OverflowError):
            pass
    # "#### 42"
    m = re.search(r"####\s*([\d,]+)", c)
    if m:
        return m.group(1).replace(",", "")
    # Last integer (up to 7 digits) in clean text
    nums = re.findall(r"\b(\d{1,7})\b", c)
    return nums[-1] if nums else None


def execute_humaneval(generated: str, sample: dict, timeout: int = 10) -> bool:
    """Execute generated code + HumanEval tests. Returns True if all pass."""
    # Remove <think>...</think> block but preserve indentation in the rest.
    # _strip_think() calls .strip() which destroys leading spaces (function-body indent),
    # so we do it manually here.
    code = re.sub(r"<think>.*?</think>", "", generated, flags=re.DOTALL)
    # Remove markdown code fences without touching surrounding whitespace
    code = re.sub(r"```python[ \t]*\n?", "", code)
    code = re.sub(r"\n?```[ \t]*", "", code)
    # Strip only blank lines at the very start/end, NOT leading spaces (preserves indent)
    code = code.strip("\n\r")

    # Always prepend the prompt (contains imports + function signature + docstring).
    # If the model regenerated a full "def ...", Python simply uses the last definition.
    # If the model only generated the function body, it continues the function from prompt.
    full_code = sample["prompt"] + "\n" + code
    full_code += "\n\n" + sample["test"] + f"\n\ncheck({sample['entry_point']})\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fp:
        fp.write(full_code)
        fname = fp.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True, timeout=timeout, text=True,
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def score_outputs(samples: list[dict], generated: list[str]) -> dict:
    """Score all outputs. Returns per-benchmark accuracy dicts."""
    bench: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0, "skipped": 0})
    for s, g in zip(samples, generated):
        bk = s["bench"]
        bench[bk]["total"] += 1
        t = s["type"]
        if t == "mc":
            pred = extract_mc(g, s["n_options"])
            if pred is None:
                bench[bk]["skipped"] += 1
                continue
            if pred == s["answer"]:
                bench[bk]["correct"] += 1
        elif t == "numeric":
            pred = extract_number(g)
            if pred is None:
                bench[bk]["skipped"] += 1
                continue
            if str(pred).strip() == str(s["answer"]).strip():
                bench[bk]["correct"] += 1
        elif t == "code":
            if execute_humaneval(g, s):
                bench[bk]["correct"] += 1

    results = {}
    for bk, st in bench.items():
        n, c, sk = st["total"], st["correct"], st["skipped"]
        results[bk] = {
            "accuracy_pct": round(100 * c / max(n, 1), 2),
            "correct": c, "total": n, "skipped": sk,
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# vLLM RUNNER  (FP16 baseline + INT4/GPTQ via tensor parallelism)
# ══════════════════════════════════════════════════════════════════════════════

def vllm_generate(
    model_path: str,
    prompts: list[str],
    thinking: bool,
    tp: int = 2,
    max_tokens: int = 4096,
    gpu_util: float = 0.90,
    max_model_len: int = 8192,
    quantization: str | None = None,   # e.g. "gptq" for AutoRound INT4 models
) -> tuple[list[str], dict]:
    from vllm import LLM, SamplingParams

    quant_tag = f", quant={quantization}" if quantization else ""
    print(f"  [vllm] Loading {Path(model_path).name} (tp={tp}{quant_tag}) …")
    t0 = time.time()
    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tp,
        dtype="float16",
        gpu_memory_utilization=gpu_util,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        disable_custom_all_reduce=True,   # RTX A6000 (cc=8.6) workaround
        enforce_eager=True,               # disable CUDA graphs — avoids illegal memory access with large batches
        trust_remote_code=True,
    )
    if quantization:
        llm_kwargs["quantization"] = quantization
    llm = LLM(**llm_kwargs)
    load_t = time.time() - t0
    print(f"  [vllm] Loaded in {load_t:.1f}s")

    sp = SamplingParams(
        temperature=0.6 if thinking else 0.0,
        top_p=0.95 if thinking else 1.0,
        top_k=20 if thinking else -1,
        max_tokens=max_tokens,
    )
    print(f"  [vllm] Generating {len(prompts)} prompts (max_tokens={max_tokens}) …")
    t1 = time.time()
    outputs = llm.generate(prompts, sp)
    gen_t = time.time() - t1
    texts = [o.outputs[0].text for o in outputs]
    print(f"  [vllm] Generation done in {gen_t:.1f}s "
          f"({len(prompts) / gen_t:.1f} prompts/s)")

    # --- Cleanup to free GPU before potential HF loading ---
    del llm, outputs
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()

    return texts, {"load_s": round(load_t, 1), "gen_s": round(gen_t, 1)}


# ══════════════════════════════════════════════════════════════════════════════
# HF RUNNER  (INT4 / mixed precision)
# ══════════════════════════════════════════════════════════════════════════════

def hf_generate(
    model,
    tokenizer,
    prompts: list[str],
    thinking: bool,
    max_new_tokens: int = 4096,
    batch_size: int = 4,
) -> list[str]:
    """Batched HF generation with left-padding."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Primary device: where the embedding / first layer lives
    first_device = next(model.parameters()).device

    gen_kw: dict = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if thinking:
        gen_kw.update(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
    else:
        gen_kw["do_sample"] = False

    all_texts = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for bi in range(0, len(prompts), batch_size):
        batch = prompts[bi : bi + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(first_device)
        with torch.no_grad():
            out_ids = model.generate(**enc, **gen_kw)
        for j in range(len(batch)):
            new_toks = out_ids[j][enc["input_ids"].shape[1]:]
            text = tokenizer.decode(new_toks, skip_special_tokens=False)
            all_texts.append(text)
        batch_idx = bi // batch_size + 1
        print(f"  [hf_gen] Batch {batch_idx}/{n_batches} done "
              f"({batch_idx * batch_size}/{len(prompts)} prompts)")
        gc.collect()
        torch.cuda.empty_cache()
    return all_texts


# ══════════════════════════════════════════════════════════════════════════════
# EXPERT SWAPPER  (runtime INT4 → FP16 expert promotion)
# ══════════════════════════════════════════════════════════════════════════════

class ExpertSwapper:
    def __init__(self, model, fp16_weights: dict):
        self.model = model
        self.fp16_weights = fp16_weights   # {(layer, expert): {proj: pinned_cpu_tensor}}
        self._originals: dict = {}
        self._promoted: set = set()
        # Match whatever dtype the rest of the model currently uses — if the
        # model is loaded in bf16, the promoted FP16 expert weights must be
        # cast to bf16 too, otherwise every expert call triggers a cast.
        self._target_dtype = torch.float16
        for p in model.parameters():
            if p.dtype in (torch.float16, torch.bfloat16):
                self._target_dtype = p.dtype
                break

    def _expert_device(self, layer: int, expert: int):
        mod = self.model.model.layers[layer].mlp.experts[expert]
        try:
            return next(mod.parameters()).device
        except StopIteration:
            return torch.device("cuda:0")

    def promote(self, layer: int, expert: int) -> None:
        key = (layer, expert)
        if key in self._promoted or key not in self.fp16_weights:
            return
        device = self._expert_device(layer, expert)
        expert_mod = self.model.model.layers[layer].mlp.experts[expert]
        weights = self.fp16_weights[key]
        self._originals[key] = {}
        for proj, cpu_w in weights.items():
            out_f, in_f = cpu_w.shape
            self._originals[key][proj] = getattr(expert_mod, proj)
            gpu_w = cpu_w.to(device=device, dtype=self._target_dtype, non_blocking=True)
            new_lin = nn.Linear(
                in_f, out_f, bias=False,
                dtype=self._target_dtype, device=device,
            )
            new_lin.weight.data.copy_(gpu_w)
            del gpu_w
            setattr(expert_mod, proj, new_lin)
        self._promoted.add(key)

    def demote(self, layer: int, expert: int) -> None:
        key = (layer, expert)
        if key not in self._promoted:
            return
        expert_mod = self.model.model.layers[layer].mlp.experts[expert]
        for proj, orig in self._originals.get(key, {}).items():
            setattr(expert_mod, proj, orig)
        self._originals.pop(key, None)
        self._promoted.discard(key)

    def reset(self) -> None:
        for l, e in list(self._promoted):
            self.demote(l, e)
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def n_promoted(self) -> int:
        return len(self._promoted)


def collect_router_stats(model, tokenizer, prompts: list[str], device) -> dict:
    """Run calibration prompts and collect per-layer expert activation counts."""
    counts: dict[int, Counter] = defaultdict(Counter)
    hooks = []

    def make_hook(li: int):
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
        ids = (
            tokenizer(p, return_tensors="pt", truncation=True, max_length=512)
            .input_ids.to(device)
        )
        with torch.no_grad():
            model(ids)

    for h in hooks:
        h.remove()
    return dict(counts)


def load_fp16_expert_weights(fp16_path: str, pairs: list[tuple]) -> dict:
    """
    Load FP16 (pinned CPU) weights for the given (layer, expert) pairs
    by scanning safetensors files once sequentially.
    """
    needed: dict[str, tuple] = {}
    for layer, expert in pairs:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
            needed[key] = (layer, expert, proj)

    result: dict = defaultdict(dict)
    st_files = sorted(Path(fp16_path).glob("*.safetensors"))
    for fi, stf in enumerate(st_files):
        with safe_open(str(stf), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in needed:
                    layer, expert, proj = needed[key]
                    t = f.get_tensor(key).to(torch.float16).contiguous()
                    pin = torch.empty_like(t, pin_memory=True)
                    pin.copy_(t)
                    del t
                    result[(layer, expert)][proj] = pin
        if (fi + 1) % 4 == 0 or (fi + 1) == len(st_files):
            print(
                f"  [weights] Scanned {fi+1}/{len(st_files)} shard(s), "
                f"{len(result)} expert weight sets ready"
            )
    return dict(result)


def select_top_pct_experts(
    router_stats: dict,
    ratio: float,
    fp16_weights: dict,
) -> list[tuple]:
    """Return global top-`ratio` fraction of experts by activation frequency."""
    all_experts = [
        (cnt, layer, expert)
        for layer, counts in router_stats.items()
        for expert, cnt in counts.items()
        if (layer, expert) in fp16_weights
    ]
    all_experts.sort(reverse=True)
    n_select = max(1, int(len(all_experts) * ratio))
    return [(l, e) for _, l, e in all_experts[:n_select]]


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    total = sum(
        torch.cuda.memory_allocated(i)
        for i in range(torch.cuda.device_count())
    )
    return round(total / 1e9, 2)


def _save(results: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results},
            f, indent=2, ensure_ascii=False,
        )


def _bench_summary(r: dict) -> str:
    bk = r.get("benchmarks", {})
    return " | ".join(f"{k}={v.get('accuracy_pct','?')}%" for k, v in bk.items())


def print_summary_table(all_results: dict) -> None:
    benchmarks = ["mmlu_pro", "gpqa", "aime", "gsm8k", "humaneval"]
    cols = ["Config", "MMLU-Pro", "GPQA", "AIME", "GSM8K", "HumanEval", "GPU(GB)", "#Promoted"]
    widths = [14, 9, 9, 9, 9, 11, 9, 10]
    header = "".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print("\n" + "=" * sum(widths))
    print(header)
    print("-" * sum(widths))
    for cfg, r in all_results.items():
        bk = r.get("benchmarks", {})
        row = [cfg]
        for b in benchmarks:
            v = bk.get(b, {}).get("accuracy_pct", "—")
            row.append(f"{v}%" if v != "—" else "—")
        row.append(str(r.get("gpu_mem_gb", "—")))
        row.append(str(r.get("n_promoted", "—")))
        print("".join(f"{c:<{w}}" for c, w in zip(row, widths)))
    print("=" * sum(widths))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--fp16-path",
        default="/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507",
    )
    parser.add_argument(
        "--int4-path",
        default="/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound",
    )
    parser.add_argument(
        "--configs",
        default="fp16,int4,mixed_10,mixed_20,mixed_30,mixed_40",
        help="Comma-separated list of configs to run",
    )
    parser.add_argument(
        "--benchmarks",
        default="mmlu_pro,gpqa,aime,gsm8k,humaneval",
        help="Benchmarks to include (comma-separated)",
    )
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Samples per MC/numeric benchmark (HumanEval always uses all 164)")
    parser.add_argument("--n-aime", type=int, default=30,
                        help="AIME samples (max 90)")
    parser.add_argument("--tp", type=int, default=2,
                        help="vLLM tensor parallel size (for fp16 config)")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Max tokens to generate per sample (including thinking)")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="vLLM KV-cache size")
    parser.add_argument("--gpu-mem-util", type=float, default=0.90,
                        help="vLLM GPU memory utilization fraction")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="HF generate batch size (for int4 / mixed configs)")
    parser.add_argument("--n-calib", type=int, default=16,
                        help="Number of calibration prompts for router statistics")
    parser.add_argument("--max-memory", default="44GiB",
                        help="Max GPU memory for INT4/mixed single-GPU loading")
    parser.add_argument("--output", default="results/comprehensive_eval.json")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",")]
    bench_names = [b.strip() for b in args.benchmarks.split(",")]
    n = args.n_samples

    # ── 1. Load all datasets ──────────────────────────────────────────────────
    print("\n[Step 1] Loading datasets …")
    LOADERS = {
        "mmlu_pro":  lambda: load_mmlu_pro(n),
        "gpqa":      lambda: load_gpqa(n),
        "aime":      lambda: load_aime(args.n_aime),
        "gsm8k":     lambda: load_gsm8k(n),
        "humaneval": lambda: load_humaneval(n),
    }
    all_samples: dict[str, list[dict]] = {}
    for bk in bench_names:
        if bk in LOADERS:
            samples = LOADERS[bk]()
            if samples:
                all_samples[bk] = samples
        else:
            print(f"  [warn] Unknown benchmark: {bk}")

    def get_samples(subset: list[str] | None = None) -> list[dict]:
        keys = subset if subset else bench_names
        return [s for bk in keys if bk in all_samples for s in all_samples[bk]]

    all_results: dict = {}

    # ── 2. FP16 baseline via vLLM ─────────────────────────────────────────────
    if "fp16" in configs:
        print("\n[Step 2] === FP16 Baseline (vLLM) ===")
        tokenizer_fp16 = AutoTokenizer.from_pretrained(args.fp16_path, trust_remote_code=True)
        samples_fp16 = get_samples()
        prompts_fp16 = build_prompts(tokenizer_fp16, samples_fp16, thinking=True)
        del tokenizer_fp16

        texts, timing = vllm_generate(
            args.fp16_path, prompts_fp16, thinking=True,
            tp=args.tp, max_tokens=args.max_new_tokens,
            gpu_util=args.gpu_mem_util, max_model_len=args.max_model_len,
        )
        bench_results = score_outputs(samples_fp16, texts)
        all_results["fp16"] = {
            "config": "fp16",
            "gpu_mem_gb": "N/A (vLLM)",
            "n_promoted": 0,
            "timing": timing,
            "benchmarks": bench_results,
        }
        _save(all_results, args.output)
        print(f"  [fp16] {_bench_summary(all_results['fp16'])}")

    # ── 3. INT4 baseline (vLLM + GPTQ) ───────────────────────────────────────
    if "int4" in configs:
        print("\n[Step 3] === INT4 Baseline (vLLM + GPTQ) ===")
        tokenizer_int4 = AutoTokenizer.from_pretrained(args.int4_path, trust_remote_code=True)
        samples_int4 = get_samples()
        prompts_int4 = build_prompts(tokenizer_int4, samples_int4, thinking=True)
        del tokenizer_int4

        texts, timing = vllm_generate(
            args.int4_path, prompts_int4, thinking=True,
            tp=args.tp,                     # controlled via --tp (default 1 for INT4)
            max_tokens=args.max_new_tokens,
            gpu_util=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            quantization="gptq",            # AutoRound uses auto_gptq packing
        )
        bench_results = score_outputs(samples_int4, texts)
        all_results["int4"] = {
            "config": "int4",
            "gpu_mem_gb": "N/A (vLLM)",
            "n_promoted": 0,
            "timing": timing,
            "benchmarks": bench_results,
        }
        _save(all_results, args.output)
        print(f"  [int4] {_bench_summary(all_results['int4'])}")

    # ── 4. Mixed precision (HF Transformers + ExpertSwapper) ─────────────────
    hf_configs = [c for c in configs if c.startswith("mixed_")]
    if not hf_configs:
        print_summary_table(all_results)
        _save(all_results, args.output)
        print(f"\n[done] Results saved → {args.output}")
        return

    print("\n[Step 4] Loading INT4 model (HF Transformers, for mixed precision) …")
    _apply_autoround_patch()   # safe here: vLLM calls are done, CUDA already init'd
    tokenizer = AutoTokenizer.from_pretrained(args.int4_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Mixed always runs on a single GPU so two independent jobs
    # can run simultaneously on the two GPUs (use CUDA_VISIBLE_DEVICES to pick).
    device_map = "cuda:0"
    max_memory = {"0": args.max_memory}

    # Load HF path in bfloat16 (the model's native training dtype) so that
    # our int4pack fast path can skip per-call fp16↔bf16 casts. Combined
    # with the INT8-attention dequant below, this is the main performance
    # path for mixed_* configs — see patch_autoround_quantlinear.py.
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.int4_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_t = round(time.time() - t0, 1)
    first_device = next(model.parameters()).device
    print(f"  [hf] INT4 loaded in {load_t}s — GPU: {gpu_mem_gb():.2f} GB")

    # ── Speed fix 1: dequantize INT8 attention (q/k/v/o) → bf16 nn.Linear ──
    # INT8 attention falls back to slow Triton in our int4pack patch; replacing
    # it with cuBLAS bf16 GEMM removes one of the decode bottlenecks.
    try:
        from scripts.patch_autoround_quantlinear import (
            dequantize_int8_attention,
            fuse_gate_up_experts,
        )
        dequantize_int8_attention(model)
        print(f"  [hf] after INT8 attn dequant — GPU: {gpu_mem_gb():.2f} GB")
    except Exception as _e:
        print(f"  [hf] INT8 attention dequant skipped: {_e}")

    # ── Speed fix 2: fuse gate_proj + up_proj into one int4pack call ────────
    # Cuts MoE INT4 expert kernel launches from 3 → 2 per expert. Combined
    # with bf16 load + attn dequant, delivers ~1.37× vs the original path.
    try:
        fuse_gate_up_experts(model)
        print(f"  [hf] after gate+up fusion — GPU: {gpu_mem_gb():.2f} GB")
    except Exception as _e:
        print(f"  [hf] gate+up fusion skipped: {_e}")

    samples_hf = get_samples()

    # ── 4b. Mixed precision configs ───────────────────────────────────────────
    mixed_configs = [c for c in hf_configs if c.startswith("mixed_")]
    if mixed_configs:
        ratios = sorted({int(c.split("_")[1]) / 100 for c in mixed_configs})
        max_ratio = max(ratios)

        print(f"\n[Step 5] Collecting router statistics (mixed precision) "
              f"({args.n_calib} calibration prompts) …")
        calib = CALIB_PROMPTS[: args.n_calib]
        router_stats = collect_router_stats(model, tokenizer, calib, first_device)
        total_tracked = sum(len(v) for v in router_stats.values())
        print(f"  [calib] Tracked {total_tracked} expert-activation slots "
              f"across {len(router_stats)} MoE layers")

        # Determine which pairs to preload (top max_ratio fraction globally)
        all_expert_acts = [
            (cnt, layer, expert)
            for layer, counts in router_stats.items()
            for expert, cnt in counts.items()
        ]
        all_expert_acts.sort(reverse=True)
        n_preload = max(1, int(len(all_expert_acts) * max_ratio))
        pairs_to_load = [(l, e) for _, l, e in all_expert_acts[:n_preload]]

        est_gb = n_preload * 3 * 768 * 2048 * 2 / 1e9
        print(f"\n[Step 6] Pre-loading FP16 weights for {n_preload} experts "
              f"(top {max_ratio*100:.0f}%, est. {est_gb:.1f} GB CPU RAM) …")
        fp16_weights = load_fp16_expert_weights(args.fp16_path, pairs_to_load)
        print(f"  [weights] {len(fp16_weights)} expert sets loaded into pinned CPU RAM")

        swapper = ExpertSwapper(model, fp16_weights)

        # ── Run each mixed ratio ──────────────────────────────────────────────
        for ratio in ratios:
            pct = int(ratio * 100)
            config_name = f"mixed_{pct}"
            if config_name not in configs:
                continue

            print(f"\n[Step 7] === Mixed {pct}% FP16 Experts ===")
            swapper.reset()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            to_promote = select_top_pct_experts(router_stats, ratio, fp16_weights)
            for l, e in to_promote:
                swapper.promote(l, e)
            torch.cuda.synchronize()
            mem_after = gpu_mem_gb()
            print(f"  [{config_name}] Promoted {swapper.n_promoted} experts → "
                  f"GPU: {mem_after:.2f} GB")

            prompts_mixed = build_prompts(tokenizer, samples_hf, thinking=True)
            t1 = time.time()
            texts = hf_generate(model, tokenizer, prompts_mixed, thinking=True,
                                max_new_tokens=args.max_new_tokens,
                                batch_size=args.batch_size)
            gen_t = round(time.time() - t1, 1)
            bench_results = score_outputs(samples_hf, texts)

            all_results[config_name] = {
                "config": config_name,
                "expert_ratio_pct": pct,
                "n_promoted": swapper.n_promoted,
                "total_tracked_experts": total_tracked,
                "gpu_mem_gb": mem_after,
                "timing": {"load_s": load_t, "gen_s": gen_t},
                "benchmarks": bench_results,
            }
            _save(all_results, args.output)
            print(f"  [{config_name}] GPU={mem_after:.2f}GB, "
                  f"promoted={swapper.n_promoted} | {_bench_summary(all_results[config_name])}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print_summary_table(all_results)
    _save(all_results, args.output)
    print(f"\n[done] Full results → {args.output}")


if __name__ == "__main__":
    main()
