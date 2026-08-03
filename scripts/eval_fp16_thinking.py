"""
FP16 MMLU-Pro accuracy evaluation for Qwen3-30B with thinking mode.

Uses vLLM with tensor_parallel_size=2 for full FP16 inference.
Enables Qwen3 thinking mode (<think>...</think>) and extracts the
final answer letter after the reasoning block.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_fp16_thinking.py \
        --model-path /home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507 \
        --n-mmlu 200 \
        --output results/fp16_mmlu_thinking.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert assistant. For each multiple-choice question, "
    "reason carefully step by step inside <think>...</think>, then output "
    "your final answer as a single capital letter on the last line, "
    "e.g. 'Answer: B'."
)


def build_prompt(tokenizer, question: str, options: list[str], enable_thinking: bool) -> str:
    option_text = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
    user_content = (
        f"Question: {question}\n\n"
        f"Options:\n{option_text}\n\n"
        "Think step by step, then state your final answer as 'Answer: X' "
        "where X is the letter of the correct option."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Older tokenizer that doesn't support enable_thinking kwarg
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str, n_options: int) -> str | None:
    """
    Extract the answer letter from generated text.
    Tries several patterns in order of specificity.
    """
    # Remove thinking block if present
    text_no_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text_clean = text_no_think.strip()

    valid = [chr(65 + i) for i in range(n_options)]

    # Pattern 1: "Answer: X" or "answer: X"
    m = re.search(r"[Aa]nswer\s*[:：]\s*([A-Z])", text_clean)
    if m and m.group(1) in valid:
        return m.group(1)

    # Pattern 2: "The answer is X"
    m = re.search(r"[Tt]he answer is\s+([A-Z])", text_clean)
    if m and m.group(1) in valid:
        return m.group(1)

    # Pattern 3: last standalone letter on its own line
    for line in reversed(text_clean.strip().splitlines()):
        line = line.strip().rstrip(".)")
        if line in valid:
            return line

    # Pattern 4: last capital letter anywhere in clean text
    letters = re.findall(r"\b([A-Z])\b", text_clean)
    for l in reversed(letters):
        if l in valid:
            return l

    return None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_mmlu_pro(n_samples: int) -> list[dict]:
    print(f"[data] Loading MMLU-Pro (n={n_samples}) ...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    samples = []
    for item in ds:
        q = item.get("question", "").strip()
        opts = item.get("options", [])
        ans = item.get("answer", "").strip()
        if not q or not opts or not ans:
            continue
        samples.append({
            "question": q,
            "options": opts,
            "answer": ans,
            "n_options": len(opts),
        })
        if len(samples) >= n_samples:
            break
    print(f"[data] {len(samples)} samples loaded")
    return samples


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    samples = load_mmlu_pro(args.n_mmlu)

    print(f"\n[vllm] Loading FP16 model with tp=2 ...")
    t0 = time.time()
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=2,
        dtype="float16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )
    load_time = time.time() - t0
    print(f"[vllm] Model loaded in {load_time:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Build prompts
    print(f"[eval] Building {len(samples)} prompts (thinking={args.thinking}) ...")
    prompts = [
        build_prompt(tokenizer, s["question"], s["options"], args.thinking)
        for s in samples
    ]

    sampling_params = SamplingParams(
        temperature=0.6 if args.thinking else 0.0,
        top_p=0.95 if args.thinking else 1.0,
        top_k=20 if args.thinking else -1,
        max_tokens=args.max_new_tokens,
        stop=None,
    )

    print(f"[eval] Running generation (max_tokens={args.max_new_tokens}) ...")
    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t1
    print(f"[eval] Generation done in {gen_time:.1f}s")

    # Score
    correct = 0
    skipped = 0
    wrong_samples = []

    for i, (s, out) in enumerate(zip(samples, outputs)):
        generated = out.outputs[0].text
        pred = extract_answer(generated, s["n_options"])
        if pred is None:
            skipped += 1
            pred = "?"
        if pred == s["answer"]:
            correct += 1
        elif len(wrong_samples) < 5:
            wrong_samples.append({
                "q": s["question"][:120],
                "gold": s["answer"],
                "pred": pred,
                "generated_tail": generated[-300:],
            })

    total = len(samples)
    accuracy = correct / total
    print(f"\n[result] Correct: {correct}/{total} = {accuracy*100:.1f}%")
    print(f"[result] Skipped (no parseable answer): {skipped}")

    # Print a few failures for debugging
    if wrong_samples:
        print("\n[debug] First wrong predictions:")
        for ex in wrong_samples:
            print(f"  Q: {ex['q']}")
            print(f"  Gold={ex['gold']}  Pred={ex['pred']}")
            print(f"  ...{ex['generated_tail'][-200:]}")
            print()

    output = {
        "model": args.model_path,
        "thinking": args.thinking,
        "n_mmlu_samples": total,
        "accuracy_pct": round(accuracy * 100, 2),
        "correct": correct,
        "skipped": skipped,
        "gen_time_s": round(gen_time, 1),
        "load_time_s": round(load_time, 1),
        "max_new_tokens": args.max_new_tokens,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[done] Results saved to {args.output}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",
                        default="/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--n-mmlu", type=int, default=200,
                        help="Number of MMLU-Pro samples to evaluate")
    parser.add_argument("--thinking", action="store_true", default=True,
                        help="Enable Qwen3 thinking mode (default: True)")
    parser.add_argument("--no-thinking", dest="thinking", action="store_false",
                        help="Disable thinking mode (greedy, fast)")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Max tokens to generate per question (including thinking)")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max model sequence length for vLLM KV cache")
    parser.add_argument("--gpu-mem-util", type=float, default=0.90,
                        help="vLLM GPU memory utilization fraction")
    parser.add_argument("--output", default="results/fp16_mmlu_thinking.json")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Model   : {args.model_path}")
    print(f"  Samples : {args.n_mmlu}")
    print(f"  Thinking: {args.thinking}")
    print(f"  MaxToks : {args.max_new_tokens}")
    print("=" * 60)

    evaluate(args)


if __name__ == "__main__":
    main()
