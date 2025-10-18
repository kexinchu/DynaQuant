#!/usr/bin/env python3
"""
Memory and Throughput Benchmarking Tool
========================================
Benchmark memory usage and inference throughput for quantized models.

Usage:
    python tools/bench_mem.py \
        --models fp16_model w4a16_model w2a16_model \
        --labels FP16 W4A16 W2A16 \
        --output benchmark_results.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
import GPUtil

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_memory_usage():
    """Get current GPU and CPU memory usage."""
    memory_info = {}

    # GPU memory
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_memory = sum(gpu.memoryUsed for gpu in gpus)
            memory_info['gpu_mb'] = gpu_memory
            memory_info['gpu_gb'] = gpu_memory / 1024
        else:
            memory_info['gpu_mb'] = 0
            memory_info['gpu_gb'] = 0
    except:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**2)
            memory_info['gpu_mb'] = gpu_memory
            memory_info['gpu_gb'] = gpu_memory / 1024
        else:
            memory_info['gpu_mb'] = 0
            memory_info['gpu_gb'] = 0

    # CPU memory
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / (1024**2)
    memory_info['cpu_mb'] = cpu_memory
    memory_info['cpu_gb'] = cpu_memory / 1024

    return memory_info


def measure_model_size(model_path: str) -> Dict:
    """Measure disk size of model files."""
    model_path = Path(model_path)

    total_size = 0
    file_count = 0

    for file in model_path.rglob('*'):
        if file.is_file():
            total_size += file.stat().st_size
            file_count += 1

    return {
        'bytes': total_size,
        'mb': total_size / (1024**2),
        'gb': total_size / (1024**3),
        'files': file_count,
    }


def benchmark_inference(
    model,
    tokenizer,
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 100,
    num_runs: int = 10,
    warmup_runs: int = 3,
    device: str = 'cuda'
) -> Dict:
    """
    Benchmark inference throughput.

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Number of tokens to generate
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        device: Device

    Returns:
        Dictionary with throughput metrics
    """
    model.eval()

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_length = inputs.input_ids.shape[1]

    # Warmup
    print(f"  Warmup ({warmup_runs} runs)...", end=' ', flush=True)
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    print("✓")

    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...", end=' ', flush=True)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    print("✓")

    # Compute metrics
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    tokens_per_second = max_new_tokens / avg_time

    return {
        'total_time_s': total_time,
        'avg_time_s': avg_time,
        'tokens_per_second': tokens_per_second,
        'num_runs': num_runs,
        'max_new_tokens': max_new_tokens,
    }


def benchmark_model(
    model_path: str,
    label: str = None,
    device: str = 'cuda',
    num_runs: int = 10
) -> Dict:
    """
    Full benchmark of a model.

    Args:
        model_path: Path to model
        label: Label for this model
        device: Device
        num_runs: Number of benchmark runs

    Returns:
        Dictionary with all benchmark results
    """
    if label is None:
        label = Path(model_path).name

    print("\n" + "="*80)
    print(f"Benchmarking: {label}")
    print(f"Path: {model_path}")
    print("="*80)

    results = {
        'label': label,
        'path': model_path,
    }

    # Measure disk size
    print("\n[1/3] Measuring disk size...")
    disk_size = measure_model_size(model_path)
    results['disk_size'] = disk_size
    print(
        f"  Total size: {disk_size['gb']:.2f} GB ({disk_size['files']} files)")

    # Load model and measure memory
    print("\n[2/3] Loading model and measuring memory...")

    # Measure before loading
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    mem_before = get_memory_usage()

    # Load
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    # Measure after loading
    mem_after = get_memory_usage()
    mem_used = {
        'gpu_mb': mem_after['gpu_mb'] - mem_before['gpu_mb'],
        'gpu_gb': mem_after['gpu_gb'] - mem_before['gpu_gb'],
        'cpu_mb': mem_after['cpu_mb'] - mem_before['cpu_mb'],
        'cpu_gb': mem_after['cpu_gb'] - mem_before['cpu_gb'],
    }

    results['memory_usage'] = mem_used
    print(f"  GPU memory: {mem_used['gpu_gb']:.2f} GB")
    print(f"  CPU memory: {mem_used['cpu_gb']:.2f} GB")

    # Benchmark inference
    print("\n[3/3] Benchmarking inference throughput...")
    throughput = benchmark_inference(
        model, tokenizer, num_runs=num_runs, device=device
    )
    results['throughput'] = throughput
    print(f"  Tokens/second: {throughput['tokens_per_second']:.2f}")
    print(f"  Avg latency: {throughput['avg_time_s']*1000:.2f} ms")

    return results


def compare_models(
    model_paths: List[str],
    labels: List[str] = None,
    output_file: str = None,
    device: str = 'cuda',
    num_runs: int = 10
):
    """
    Compare multiple models.

    Args:
        model_paths: List of model paths
        labels: List of labels for models
        output_file: Output JSON file
        device: Device
        num_runs: Number of benchmark runs
    """
    if labels is None:
        labels = [Path(p).name for p in model_paths]

    assert len(model_paths) == len(
        labels), "Number of paths and labels must match"

    print("\n" + "="*80)
    print("Memory & Throughput Benchmark")
    print("="*80)
    print(f"Models: {len(model_paths)}")
    print(f"Device: {device}")
    print(f"Runs: {num_runs}")
    print("="*80)

    # Benchmark each model
    results = []
    for model_path, label in zip(model_paths, labels):
        result = benchmark_model(model_path, label, device, num_runs)
        results.append(result)

        # Clear GPU memory between models
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print comparison table
    print("\n" + "="*80)
    print("Comparison Results")
    print("="*80)

    print("\n{:<15} {:<12} {:<12} {:<12} {:<12}".format(
        "Model", "Disk (GB)", "GPU (GB)", "Tokens/s", "Latency (ms)"
    ))
    print("-" * 80)

    for result in results:
        print("{:<15} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f}".format(
            result['label'],
            result['disk_size']['gb'],
            result['memory_usage']['gpu_gb'],
            result['throughput']['tokens_per_second'],
            result['throughput']['avg_time_s'] * 1000
        ))

    # Compute speedup/compression relative to first model (assumed to be baseline)
    if len(results) > 1:
        baseline = results[0]

        print("\n{:<15} {:<12} {:<12} {:<12}".format(
            "Model", "Compression", "Speedup", "GPU Savings"
        ))
        print("-" * 80)

        for result in results:
            compression = baseline['disk_size']['gb'] / \
                result['disk_size']['gb']
            speedup = result['throughput']['tokens_per_second'] / \
                baseline['throughput']['tokens_per_second']
            gpu_savings = 1 - \
                (result['memory_usage']['gpu_gb'] /
                 baseline['memory_usage']['gpu_gb'])

            print("{:<15} {:<12.2f}x {:<12.2f}x {:<12.1f}%".format(
                result['label'],
                compression,
                speedup,
                gpu_savings * 100
            ))

    print("="*80)

    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        output_data = {
            'results': results,
            'device': device,
            'num_runs': num_runs,
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark model memory and throughput")
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='Paths to models to benchmark')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Labels for models (optional)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for computation')
    parser.add_argument('--num-runs', type=int, default=10,
                        help='Number of benchmark runs')

    args = parser.parse_args()

    compare_models(
        model_paths=args.models,
        labels=args.labels,
        output_file=args.output,
        device=args.device,
        num_runs=args.num_runs
    )


if __name__ == '__main__':
    main()
