"""
Run all quality benchmarks on Qwen3-30B in a single model-load session.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_qwen30b_eval.py \
        --model-path /path/to/model \
        --tag int4 \
        --n-samples 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynaexq.experiments.datasets import load_dataset
from dynaexq.experiments.eval_quality import (
    compute_mc_accuracy,
    compute_perplexity,
    compute_string_match_accuracy,
    compute_pass_at_1,
    EVALUATORS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tag", default="default", help="Output filename tag (e.g. 'int4', 'fp16')")
    parser.add_argument("--benchmarks", default="wikitext,mmlu_pro,gsm8k",
                        help="Comma-separated benchmark names")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-memory", default="32GiB", help="Max GPU memory")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(args.dtype, torch.float16)

    print(f"[main] Loading model {args.model_path} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device,
        max_memory={"0": args.max_memory},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_time = time.time() - t0
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"[main] Loaded in {load_time:.1f}s, GPU mem: {mem_gb:.2f}GB")

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    results = {}
    out_dir = Path("results/quality")
    out_dir.mkdir(parents=True, exist_ok=True)

    for bench_name in benchmarks:
        if bench_name not in EVALUATORS:
            print(f"[WARN] Unknown benchmark {bench_name}")
            continue

        task_type, eval_fn = EVALUATORS[bench_name]
        kwargs = {}
        if args.n_samples is not None:
            kwargs["n_samples"] = args.n_samples

        print(f"\n[eval] === {bench_name} ===")
        try:
            data = load_dataset(bench_name, **kwargs)
        except Exception as e:
            print(f"[eval] Failed to load {bench_name}: {e}")
            results[bench_name] = {"error": str(e)}
            continue

        print(f"[eval] {len(data)} samples loaded, running...")
        t1 = time.time()
        try:
            score = eval_fn(model, tokenizer, data, device=args.device)
            elapsed = time.time() - t1
            results[bench_name] = round(score, 4)

            if task_type == "ppl":
                print(f"[eval] {bench_name}: PPL = {score:.2f} ({elapsed:.1f}s)")
            else:
                print(f"[eval] {bench_name}: accuracy = {score*100:.1f}% ({elapsed:.1f}s)")
        except Exception as e:
            print(f"[eval] {bench_name} failed: {e}")
            results[bench_name] = {"error": str(e)}

    output = {
        "model": args.model_path,
        "tag": args.tag,
        "dtype": args.dtype,
        "gpu_mem_gb": round(mem_gb, 2),
        "load_time_s": round(load_time, 1),
        "benchmarks": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path = out_dir / f"qwen3-30b_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[main] Results saved to {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
