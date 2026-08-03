#!/usr/bin/env python3
"""
eval_qwen3next_80b.py — Evaluate Qwen3-Next-80B-A3B INT4 (AutoRound) on accuracy benchmarks.

Uses HF Transformers + AutoRound backend (vLLM GPTQ can't handle the mixed
4/8-bit precision in this model).

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_qwen3next_80b.py \
      --n-samples 200 --batch-size 16 \
      --output results/qwen3next_80b_int4.json
"""
import argparse, gc, json, os, random, re, subprocess, sys, tempfile, time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset as hf_load_dataset

_HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if _HF_TOKEN:
    try:
        from huggingface_hub import login as hf_login
        hf_login(token=_HF_TOKEN, add_to_git_credential=False)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════════════

SYS_MC = (
    "You are a highly knowledgeable expert. Reason carefully step by step, "
    "then state your final answer on the very last line as "
    "'Answer: X' where X is exactly one capital letter (A, B, C, …)."
)
SYS_MATH = (
    "You are an expert mathematician. Show complete reasoning, "
    "then state the final integer answer on the last line as "
    "'Answer: N' where N is an integer (no units, no words)."
)
SYS_CODE = (
    "You are an expert Python programmer. Think through the algorithm, "
    "then output only the complete Python function — no markdown fences, "
    "no extra explanation, just code."
)

# ══════════════════════════════════════════════════════════════════════════════
# Dataset loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_mmlu_pro(n=200):
    print(f"  [mmlu_pro] Loading {n} samples …")
    ds = hf_load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    out = []
    for it in ds:
        q, opts, ans = it.get("question","").strip(), it.get("options",[]), it.get("answer","").strip()
        if not q or not opts or not ans: continue
        out.append({"bench":"mmlu_pro","type":"mc","question":q,"options":opts,"answer":ans,"n_options":len(opts)})
        if len(out) >= n: break
    print(f"  [mmlu_pro] {len(out)} samples loaded")
    return out

def load_gpqa(n=200):
    if not _HF_TOKEN:
        print("  [gpqa] SKIPPED — no HF_TOKEN"); return []
    print(f"  [gpqa] Loading {n} samples …")
    try: ds = hf_load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception as e: print(f"  [gpqa] failed: {e}"); return []
    rng = random.Random(42); out = []
    for it in ds:
        q = it.get("Question","").strip(); correct = it.get("Correct Answer","").strip()
        wrongs = [it.get(f"Incorrect Answer {i}","").strip() for i in range(1,4)]
        if not q or not correct: continue
        opts = [correct] + [w for w in wrongs if w]; rng.shuffle(opts)
        ans_letter = chr(65 + opts.index(correct))
        out.append({"bench":"gpqa","type":"mc","question":q,"options":opts,"answer":ans_letter,"n_options":len(opts)})
        if len(out) >= n: break
    print(f"  [gpqa] {len(out)} samples loaded"); return out

def load_aime(n=30):
    print(f"  [aime] Loading ≤{n} samples …")
    ds = hf_load_dataset("AI-MO/aimo-validation-aime", split="train"); out = []
    for it in ds:
        q = it.get("problem","").strip(); a = str(it.get("answer","")).strip()
        if not q or not a: continue
        try: a = str(int(float(a)))
        except: pass
        out.append({"bench":"aime","type":"numeric","question":q,"answer":a})
        if len(out) >= n: break
    print(f"  [aime] {len(out)} samples loaded"); return out

def load_gsm8k(n=200):
    print(f"  [gsm8k] Loading {n} samples …")
    ds = hf_load_dataset("openai/gsm8k", "main", split="test"); out = []
    for it in ds:
        q = it.get("question","").strip(); a_raw = it.get("answer","").strip()
        nums = re.findall(r"####\s*(-?[\d,]+)", a_raw)
        if not q or not nums: continue
        a = nums[-1].replace(",","")
        out.append({"bench":"gsm8k","type":"numeric","question":q,"answer":a})
        if len(out) >= n: break
    print(f"  [gsm8k] {len(out)} samples loaded"); return out

def load_humaneval(n=200):
    print(f"  [humaneval] Loading HumanEval …")
    ds = hf_load_dataset("openai/openai_humaneval", split="test"); out = []
    for it in ds:
        out.append({"bench":"humaneval","type":"code",
                    "question":it["prompt"],"answer":it.get("canonical_solution",""),
                    "test":it.get("test",""),"entry_point":it.get("entry_point","")})
        if len(out) >= n: break
    print(f"  [humaneval] {len(out)} samples loaded"); return out

# ══════════════════════════════════════════════════════════════════════════════
# Prompts + scoring
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(tokenizer, s):
    if s["type"] == "mc":
        opts_str = "\n".join(f"({chr(65+i)}) {o}" for i,o in enumerate(s["options"]))
        user_msg = f"{s['question']}\n\n{opts_str}"
        sys_msg = SYS_MC
    elif s["type"] == "numeric":
        user_msg = s["question"]; sys_msg = SYS_MATH
    elif s["type"] == "code":
        user_msg = s["question"]; sys_msg = SYS_CODE
    else:
        return None
    msgs = [{"role":"system","content":sys_msg},{"role":"user","content":user_msg}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

def extract_mc(text, n_opts=4):
    text = re.sub(r"<\|.*?\|>", "", text)
    m = re.findall(r"(?:Answer|answer|ANSWER)\s*[:：]\s*\(?([A-Z])\)?", text)
    if m: return m[-1]
    letters = re.findall(r"\b([A-Z])\b", text[-200:])
    valid = [l for l in letters if ord(l)-65 < n_opts]
    return valid[-1] if valid else None

def extract_number(text):
    text = re.sub(r"<\|.*?\|>", "", text)
    m = re.findall(r"(?:Answer|answer|ANSWER)\s*[:：]\s*(-?[\d,]+)", text)
    if m: return m[-1].replace(",","")
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else None

def execute_humaneval(generated, sample):
    code = re.sub(r"```python\s*\n?", "", generated)
    code = re.sub(r"```\s*$", "", code)
    full = sample["question"] + code + "\n" + sample["test"] + f"\ncheck({sample['entry_point']})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full); f.flush()
        try:
            r = subprocess.run([sys.executable, f.name], capture_output=True, timeout=10)
            return r.returncode == 0
        except: return False
        finally: os.unlink(f.name)

def score_outputs(samples, texts):
    bench = defaultdict(lambda: {"correct":0,"total":0,"skipped":0})
    for s,g in zip(samples, texts):
        bk = s["bench"]; bench[bk]["total"] += 1
        if s["type"] == "mc":
            pred = extract_mc(g, s["n_options"])
            if pred is None: bench[bk]["skipped"] += 1; continue
            if pred == s["answer"]: bench[bk]["correct"] += 1
        elif s["type"] == "numeric":
            pred = extract_number(g)
            if pred is None: bench[bk]["skipped"] += 1; continue
            if str(pred).strip() == str(s["answer"]).strip(): bench[bk]["correct"] += 1
        elif s["type"] == "code":
            if execute_humaneval(g, s): bench[bk]["correct"] += 1
    results = {}
    for bk, st in bench.items():
        n, c, sk = st["total"], st["correct"], st["skipped"]
        results[bk] = {"accuracy_pct": round(100*c/max(n,1), 2), "correct":c, "total":n, "skipped":sk}
    return results

# ══════════════════════════════════════════════════════════════════════════════
# HF batched generation
# ══════════════════════════════════════════════════════════════════════════════

def hf_generate_batched(model, tokenizer, prompts, max_new_tokens=2048, batch_size=16, device="cuda:0"):
    texts = []
    n_total = len(prompts)
    t_start = time.time()

    for i in range(0, n_total, batch_size):
        batch = prompts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=512, padding_side="left").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only newly generated tokens
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        texts.extend(decoded)

        elapsed = time.time() - t_start
        done = min(i + batch_size, n_total)
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  [{done}/{n_total}] {elapsed:.0f}s elapsed  ({rate:.1f} samples/s)", flush=True)

        gc.collect()
        torch.cuda.empty_cache()

    return texts

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/kec23008/Models/Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound")
    parser.add_argument("--benchmarks", default="mmlu_pro,gpqa,gsm8k,aime,humaneval")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-aime", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device-map", default="auto",
                        help="HF device_map: 'auto' (split across GPUs) or 'cuda:0'")
    parser.add_argument("--output", default="results/qwen3next_80b_int4.json")
    args = parser.parse_args()

    bench_names = [b.strip() for b in args.benchmarks.split(",")]

    print("\n[Step 1] Loading datasets …")
    LOADERS = {
        "mmlu_pro":  lambda: load_mmlu_pro(args.n_samples),
        "gpqa":      lambda: load_gpqa(args.n_samples),
        "aime":      lambda: load_aime(args.n_aime),
        "gsm8k":     lambda: load_gsm8k(args.n_samples),
        "humaneval": lambda: load_humaneval(args.n_samples),
    }
    all_samples = {}
    for bk in bench_names:
        if bk in LOADERS:
            s = LOADERS[bk]()
            if s: all_samples[bk] = s

    samples = [s for bk in bench_names if bk in all_samples for s in all_samples[bk]]
    print(f"  Total: {len(samples)} samples across {len(all_samples)} benchmarks")

    print(f"\n[Step 2] Loading model (HF + AutoRound, device_map={args.device_map}) …")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_t = time.time() - t0
    mem_gb = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9
    print(f"  Loaded in {load_t:.1f}s  total_GPU={mem_gb:.1f}GB")

    print(f"\n[Step 3] Building prompts …")
    prompts = []
    valid_samples = []
    for s in samples:
        p = build_prompt(tokenizer, s)
        if p is not None:
            prompts.append(p)
            valid_samples.append(s)
    print(f"  {len(prompts)} prompts built")

    # Determine the first device of the model for input encoding
    first_device = next(model.parameters()).device

    # Quick sanity check: print one sample output
    print(f"\n[Sanity] Generating 1 sample to verify output format …")
    enc = tokenizer([prompts[0]], return_tensors="pt", padding=True,
                    truncation=True, max_length=256).to(first_device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=100, do_sample=False,
                             temperature=None, top_p=None,
                             pad_token_id=tokenizer.eos_token_id)
    sanity_text = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Output: {repr(sanity_text[:300])}")
    if not sanity_text.strip():
        print("  WARNING: empty output — model may not be generating correctly!")

    print(f"\n[Step 4] Generating (batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens}) …")
    t_gen = time.time()
    texts = hf_generate_batched(
        model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=str(first_device),
    )
    gen_t = time.time() - t_gen
    print(f"  Generation done in {gen_t:.1f}s")

    print(f"\n[Step 5] Scoring …")
    bench_results = score_outputs(valid_samples, texts)

    # Debug: print first 3 outputs
    print("\n[Debug] First 3 outputs:")
    for i in range(min(3, len(texts))):
        print(f"  [{i}] {repr(texts[i][:200])}")

    print("\n" + "="*60)
    print(f"{'Benchmark':15s} {'Accuracy':>8s} {'Correct':>8s} {'Total':>6s} {'Skip':>5s}")
    print("-"*60)
    for bk in bench_names:
        if bk in bench_results:
            r = bench_results[bk]
            print(f"{bk:15s} {r['accuracy_pct']:7.2f}% {r['correct']:>8d} {r['total']:>6d} {r['skipped']:>5d}")
    print("="*60)

    output = {
        "model": args.model_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "timing": {"load_s": round(load_t,1), "gen_s": round(gen_t,1)},
        "gpu_mem_gb": round(mem_gb, 2),
        "benchmarks": bench_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
