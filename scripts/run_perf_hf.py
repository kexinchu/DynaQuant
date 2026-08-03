#!/usr/bin/env python3
"""Run perf test using HF Transformers for AutoRound quantized models."""
import gc
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("run_perf_hf")

DATASET = Path(__file__).resolve().parent.parent / "ShareGPT_V3_unfiltered_cleaned_split.json"
PROMPT_LEN = 512


@dataclass
class PromptSample:
    sample_id: str
    text: str
    token_count: int


@dataclass
class Measurement:
    sample_id: str
    prompt_tokens: int
    decode_tokens: int
    ttft_ms: float
    tpop_ms: float
    end2end_ms: float


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
        samples.append(PromptSample(
            sample_id=str(item.get("id", len(samples))),
            text=tokenizer.decode(ids, skip_special_tokens=True),
            token_count=len(ids),
        ))
    return samples


def percentile(vals, q):
    if not vals:
        return None
    o = sorted(vals)
    idx = (len(o) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(o) - 1)
    return float(o[lo]) if lo == hi else o[lo] + (o[hi] - o[lo]) * (idx - lo)


def run_batch(model, tokenizer, samples, device, max_new_tokens):
    texts = [s.text for s in samples]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=PROMPT_LEN, return_attention_mask=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    batch_size = enc["input_ids"].shape[0]

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=True)
    sync()
    ttft_ms = (time.perf_counter() - t0) * 1000
    past_kv = out.past_key_values
    logits = out.logits[:, -1, :]
    next_tok = torch.argmax(logits, dim=-1, keepdim=True)
    gen = next_tok.clone()
    eos = tokenizer.eos_token_id
    decode_times = []
    for _ in range(max_new_tokens - 1):
        sync()
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
        sync()
        decode_times.append(time.perf_counter() - t1)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        gen = torch.cat([gen, next_tok], dim=1)
        if eos is not None and (next_tok == eos).all().item():
            break
    n_decode = len(decode_times) + 1
    tpop_ms = (sum(decode_times) / n_decode) * 1000 if n_decode else 0
    end2end_ms = ttft_ms + sum(decode_times) * 1000
    return [Measurement(s.sample_id, enc["input_ids"].shape[1], n_decode, ttft_ms, tpop_ms, end2end_ms) for s in samples]


def main():
    models = [
        # "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound",
        # "/home/kec23008/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound",
        "/home/kec23008/Models/Phi-3.5-MoE-instruct-mixed-AutoRound",
    ]
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    all_results = []

    for model_path in models:
        LOG.info("Loading %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        samples = load_prompts(tokenizer, PROMPT_LEN, 80)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
        model.eval()

        batch_sizes = [1, 2, 4, 8, 16, 32]
        max_new = 32

        try:
            for bs in batch_sizes:
                ms_all = []
                for i in range(10):
                    batch = samples[i * bs : (i + 1) * bs]
                    if len(batch) < bs:
                        batch = samples[:bs]
                    ms_all.extend(run_batch(model, tokenizer, batch, device, max_new))
                ttft = [m.ttft_ms for m in ms_all]
                tpop = [m.tpop_ms for m in ms_all if m.tpop_ms > 0]
                e2e = [m.end2end_ms for m in ms_all]
                summary = {
                    "model": model_path,
                    "batch_size": bs,
                    "ttft_avg_ms": sum(ttft) / len(ttft) if ttft else None,
                    "ttft_p95_ms": percentile(ttft, 0.95),
                    "ttft_p99_ms": percentile(ttft, 0.99),
                    "tpop_avg_ms": sum(tpop) / len(tpop) if tpop else None,
                    "tpop_p95_ms": percentile(tpop, 0.95) if tpop else None,
                    "tpop_p99_ms": percentile(tpop, 0.99) if tpop else None,
                    "end2end_avg_ms": sum(e2e) / len(e2e) if e2e else None,
                    "end2end_p95_ms": percentile(e2e, 0.95),
                    "end2end_p99_ms": percentile(e2e, 0.99),
                }
                all_results.append(summary)
                LOG.info("bs=%d | TTFT avg=%.1f p95=%.1f | TPOP avg=%.1f | E2E avg=%.1f ms",
                    bs, summary["ttft_avg_ms"] or 0, summary["ttft_p95_ms"] or 0,
                    summary["tpop_avg_ms"] or 0, summary["end2end_avg_ms"] or 0)
        except Exception as e:
            LOG.exception("Error with model %s: %s", model_path, e)
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        out_path = Path("scripts/results/perf_hf_results_phi35moe.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"results": all_results}, indent=2), encoding="utf-8")
        LOG.info("Intermediate results saved to %s", out_path)

    out_path = Path("scripts/results/perf_hf_results_phi35moe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": all_results}, indent=2), encoding="utf-8")
    LOG.info("Results saved to %s", out_path)
    print(json.dumps({"results": all_results}, indent=2))


if __name__ == "__main__":
    main()
