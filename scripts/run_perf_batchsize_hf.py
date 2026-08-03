#!/usr/bin/env python3
"""
TTFT across num_of_tokens using HF Transformers (for AutoRound models when vLLM unavailable).
Tests: 1,2,4,8,16,32,64,128,192,256,320,384,512,576,640,704,768,832,896,960,1024
10 groups per (model, num_tokens). Output: TTFT avg, p95, p99.
"""
import gc
import json
import logging
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("run_perf_batchsize_hf")

DATASET = Path(__file__).resolve().parent.parent / "ShareGPT_V3_unfiltered_cleaned_split.json"
NUM_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
GROUPS = 10


def load_prompts(tokenizer, target_len: int, max_n: int):
    with open(DATASET) as f:
        data = json.load(f)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    samples = []
    for item in data:
        if len(samples) >= max_n:
            break
        if "conversations" not in item:
            continue
        text = None
        for c in item["conversations"]:
            if c.get("from") == "human" and "value" in c:
                text = c["value"]
                break
        if not text or not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids = (ids + [pad_id] * (target_len - len(ids)))[:target_len]
        samples.append(tokenizer.decode(ids, skip_special_tokens=True))
    return samples


def percentile(vals, q):
    if not vals:
        return None
    o = sorted(vals)
    idx = (len(o) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(o) - 1)
    return float(o[lo]) if lo == hi else o[lo] + (o[hi] - o[lo]) * (idx - lo)


def run_ttft(model, tokenizer, text, device):
    enc = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=1024, return_attention_mask=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000.0


def main():
    models = [
        "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound",
        "/home/kec23008/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound",
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = []

    for model_path in models:
        LOG.info("Loading %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
        )
        model.eval()

        try:
            for num_tokens in NUM_TOKENS:
                samples = load_prompts(tokenizer, num_tokens, GROUPS + 4)
                samples = samples[:GROUPS]
                if not samples:
                    continue
                ttft_list = [run_ttft(model, tokenizer, s, device) for s in samples[:GROUPS]]

                summary = {
                    "model": model_path,
                    "num_tokens": num_tokens,
                    "groups": len(ttft_list),
                    "ttft_avg_ms": sum(ttft_list) / len(ttft_list),
                    "ttft_p95_ms": percentile(ttft_list, 0.95),
                    "ttft_p99_ms": percentile(ttft_list, 0.99),
                }
                all_results.append(summary)
                LOG.info("num_tokens=%d | TTFT avg=%.1f p95=%.1f p99=%.1f ms", num_tokens,
                    summary["ttft_avg_ms"], summary["ttft_p95_ms"] or 0, summary["ttft_p99_ms"] or 0)

                out_path = Path("scripts/results/perf_batchsize_results.json")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps({"results": all_results, "num_tokens_tested": NUM_TOKENS}, indent=2), encoding="utf-8")
        except Exception as e:
            LOG.exception("Error: %s", e)
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out_path = Path("scripts/results/perf_batchsize_results.json")
    out_path.write_text(json.dumps({"results": all_results, "num_tokens_tested": NUM_TOKENS}, indent=2), encoding="utf-8")
    LOG.info("Results saved to %s", out_path)
    print(json.dumps({"results": all_results}, indent=2))


if __name__ == "__main__":
    main()
