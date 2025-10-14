"""
Benchmarking and evaluation script for DynaQuant.
Measures accuracy, latency, throughput, and resource usage.
"""

from dynaquant.hooks import inject_dynaquant_into_sglang
import os
import sys
import argparse
import yaml
import torch
import time
from pathlib import Path
import logging
import json
from tqdm import tqdm
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_prompts(prompts_dir: str, prompt_sets: list) -> dict:
    """Load evaluation prompts."""
    prompts = {}

    for prompt_set in prompt_sets:
        prompt_file = os.path.join(prompts_dir, f"{prompt_set}.txt")

        if os.path.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                prompts[prompt_set] = [line.strip()
                                       for line in f if line.strip()]
            logger.info(
                f"Loaded {len(prompts[prompt_set])} prompts from {prompt_set}")
        else:
            logger.warning(
                f"Prompt file not found: {prompt_file}, using dummy prompts")
            prompts[prompt_set] = [
                f"Dummy {prompt_set} prompt {i}" for i in range(10)
            ]

    return prompts


def load_model(config: dict, precision: str = "w2a4"):
    """Load model for benchmarking."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config['model']['name']
    cache_dir = config['model'].get('cache_dir', None)

    logger.info(f"Loading model: {model_name} (precision: {precision})")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Inject DynaQuant if not FP16
    hook_manager = None
    if precision != "fp16":
        logger.info(f"Injecting DynaQuant for {precision}")

        # Set default precision
        if 'precision_scheduler' not in config:
            config['precision_scheduler'] = {}
        config['precision_scheduler']['default_precision'] = precision

        hook_manager = inject_dynaquant_into_sglang(
            model=model,
            config=config,
            enable_rcg=True,
            enable_ps=True,
            enable_ec=True,
        )

    return model, tokenizer, hook_manager


def benchmark_latency(model, tokenizer, prompts, config):
    """Benchmark latency metrics (TTFT, throughput)."""
    logger.info("Benchmarking latency")

    bench_config = config['benchmark']
    input_len = bench_config['input_len']
    output_len = bench_config['output_len']

    device = next(model.parameters()).device
    model.eval()

    results = {
        'ttft': [],  # Time to first token
        'latency': [],  # Total latency
        'throughput': [],  # Tokens per second
    }

    with torch.no_grad():
        for prompt in tqdm(prompts[:min(len(prompts), 100)], desc="Latency benchmark"):
            # Tokenize
            inputs = tokenizer(
                prompt,
                max_length=input_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            ).to(device)

            # Measure TTFT (time to first token)
            start_time = time.time()

            # First forward pass
            outputs = model(inputs['input_ids'])

            ttft = time.time() - start_time
            results['ttft'].append(ttft)

            # Generate full output
            start_time = time.time()

            try:
                output_ids = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=output_len,
                    do_sample=False,
                )

                latency = time.time() - start_time
                results['latency'].append(latency)

                # Throughput (tokens/sec)
                num_tokens = output_ids.shape[1] - inputs['input_ids'].shape[1]
                throughput = num_tokens / latency
                results['throughput'].append(throughput)

            except Exception as e:
                logger.warning(f"Generation failed: {e}")
                continue

    # Compute statistics
    import numpy as np

    stats = {}
    for metric, values in results.items():
        if values:
            stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'p50': float(np.percentile(values, 50)),
                'p90': float(np.percentile(values, 90)),
                'p95': float(np.percentile(values, 95)),
                'p99': float(np.percentile(values, 99)),
            }

    return stats


def benchmark_accuracy(model, tokenizer, prompts, prompt_set_name, config):
    """Benchmark accuracy on specific task."""
    logger.info(f"Benchmarking accuracy on {prompt_set_name}")

    # This is a simplified accuracy benchmark
    # In practice, would use proper evaluation metrics for each task

    device = next(model.parameters()).device
    model.eval()

    results = []

    with torch.no_grad():
        for prompt in tqdm(prompts[:min(len(prompts), 50)], desc=f"Accuracy - {prompt_set_name}"):
            inputs = tokenizer(
                prompt,
                max_length=512,
                truncation=True,
                return_tensors='pt',
            ).to(device)

            try:
                # Generate response
                output_ids = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=128,
                    do_sample=False,
                )

                response = tokenizer.decode(
                    output_ids[0], skip_special_tokens=True)

                results.append({
                    'prompt': prompt,
                    'response': response,
                })

            except Exception as e:
                logger.warning(f"Generation failed: {e}")
                continue

    return results


def benchmark_resource_usage(model, hook_manager):
    """Benchmark VRAM and bandwidth usage."""
    logger.info("Measuring resource usage")

    # VRAM usage
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        vram_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    else:
        vram_allocated = 0.0
        vram_reserved = 0.0

    stats = {
        'vram_allocated_gb': vram_allocated,
        'vram_reserved_gb': vram_reserved,
    }

    # Get DynaQuant statistics if available
    if hook_manager is not None:
        dynaquant_stats = hook_manager.get_statistics()
        stats['dynaquant'] = dynaquant_stats

    return stats


def run_benchmark(precision: str, config: dict):
    """Run full benchmark for a specific precision."""
    logger.info(f"Running benchmark for precision: {precision}")

    # Load model
    model, tokenizer, hook_manager = load_model(config, precision=precision)

    # Load prompts
    bench_config = config['benchmark']
    prompts_dict = load_prompts(
        bench_config['prompts_dir'], bench_config['prompt_sets'])

    results = {
        'precision': precision,
        'latency': {},
        'accuracy': {},
        'resources': {},
    }

    # Benchmark latency (use first prompt set)
    first_prompt_set = bench_config['prompt_sets'][0]
    if bench_config['metrics'].get('ttft', False) or bench_config['metrics'].get('throughput', False):
        latency_stats = benchmark_latency(
            model, tokenizer, prompts_dict[first_prompt_set], config
        )
        results['latency'] = latency_stats

    # Benchmark accuracy on each prompt set
    if bench_config['metrics'].get('accuracy', False):
        for prompt_set_name, prompts in prompts_dict.items():
            accuracy_results = benchmark_accuracy(
                model, tokenizer, prompts, prompt_set_name, config
            )
            results['accuracy'][prompt_set_name] = accuracy_results

    # Measure resource usage
    if bench_config['metrics'].get('vram_usage', False):
        resource_stats = benchmark_resource_usage(model, hook_manager)
        results['resources'] = resource_stats

    return results


def generate_html_summary(all_results: dict, output_path: str):
    """Generate HTML summary of results."""
    html = """
    <html>
    <head>
        <title>DynaQuant Benchmark Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>DynaQuant Benchmark Results</h1>
    """

    # Latency comparison
    html += "<h2>Latency Metrics</h2>"
    html += "<table><tr><th>Precision</th><th>TTFT (ms)</th><th>Throughput (tok/s)</th><th>P95 Latency (s)</th></tr>"

    for precision, results in all_results.items():
        if 'latency' in results and results['latency']:
            ttft = results['latency'].get('ttft', {}).get('mean', 0) * 1000
            throughput = results['latency'].get(
                'throughput', {}).get('mean', 0)
            p95_latency = results['latency'].get('latency', {}).get('p95', 0)

            html += f"<tr><td>{precision}</td><td>{ttft:.2f}</td><td>{throughput:.2f}</td><td>{p95_latency:.3f}</td></tr>"

    html += "</table>"

    # Resource usage
    html += "<h2>Resource Usage</h2>"
    html += "<table><tr><th>Precision</th><th>VRAM Allocated (GB)</th><th>VRAM Reserved (GB)</th></tr>"

    for precision, results in all_results.items():
        if 'resources' in results:
            vram_alloc = results['resources'].get('vram_allocated_gb', 0)
            vram_res = results['resources'].get('vram_reserved_gb', 0)

            html += f"<tr><td>{precision}</td><td>{vram_alloc:.2f}</td><td>{vram_res:.2f}</td></tr>"

    html += "</table>"

    html += "</body></html>"

    with open(output_path, 'w') as f:
        f.write(html)

    logger.info(f"Generated HTML summary: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark DynaQuant model')
    parser.add_argument('--config', type=str, default='experiments/config_ptq_qat.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--precision', type=str, nargs='+',
                        default=['fp16', 'w4a4', 'w2a4'],
                        help='Precisions to benchmark')

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    if args.output_dir:
        config['benchmark']['results_dir'] = args.output_dir

    # Run benchmarks
    all_results = {}

    for precision in args.precision:
        try:
            results = run_benchmark(precision, config)
            all_results[precision] = results
        except Exception as e:
            logger.error(f"Benchmark failed for {precision}: {e}")
            continue

    # Save results
    results_dir = config['benchmark']['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    # Save JSONL
    if config['benchmark']['save_jsonl']:
        jsonl_path = os.path.join(results_dir, 'benchmark_results.jsonl')
        with open(jsonl_path, 'w') as f:
            for precision, results in all_results.items():
                json.dump({'precision': precision, 'results': results}, f)
                f.write('\n')
        logger.info(f"Saved results to {jsonl_path}")

    # Generate HTML summary
    if config['benchmark']['generate_html_summary']:
        html_path = os.path.join(results_dir, 'benchmark_summary.html')
        generate_html_summary(all_results, html_path)

    logger.info("Benchmarking complete!")


if __name__ == '__main__':
    main()
