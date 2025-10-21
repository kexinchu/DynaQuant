#!/usr/bin/env python3
"""
多GPU并行评估脚本 - 动态量化MoE模型
利用多个GPU并行评估不同的数据集，加速评估过程
"""

import os
import json
import torch
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import torch.multiprocessing as mp
from queue import Queue
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_on_gpu(
    gpu_id: int,
    fp16_model_path: str,
    int4_model_path: str,
    dataset_names: List[str],
    data_dir: str,
    max_length: int,
    time_window: float,
    hot_ratio: float,
    disable_dynamic_routing: bool,
    num_samples: Dict[str, int],
    result_queue: mp.Queue
):
    """在指定GPU上评估数据集"""
    try:
        # 设置GPU
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"

        logger.info(f"GPU {gpu_id}: Loading models...")

        # 导入评估器（在子进程中导入以避免CUDA初始化问题）
        from evaluate_dynamic_quant import DynamicQuantEvaluator

        evaluator = DynamicQuantEvaluator(
            fp16_model_path=fp16_model_path,
            int4_model_path=int4_model_path,
            device=device,
            max_length=max_length,
            time_window=time_window,
            hot_ratio=hot_ratio,
            enable_dynamic_routing=not disable_dynamic_routing
        )

        results = {}

        for dataset in dataset_names:
            logger.info(f"GPU {gpu_id}: Evaluating {dataset}")
            start_time = time.time()

            try:
                if dataset.lower() == "wikitext":
                    result = evaluator.evaluate_wikitext(
                        num_samples=num_samples.get('wikitext', 1000)
                    )
                elif dataset.lower() == "mmlu":
                    result = evaluator.evaluate_mmlu(
                        data_dir=f"{data_dir}/MMLU",
                        num_samples=num_samples.get('mmlu', 100)
                    )
                elif dataset.lower() == "gsm8k":
                    result = evaluator.evaluate_gsm8k(
                        data_dir=f"{data_dir}/GSM8K",
                        num_samples=num_samples.get('gsm8k', 100)
                    )
                elif dataset.lower() == "hellaswag":
                    result = evaluator.evaluate_hellaswag(
                        data_dir=f"{data_dir}/HELLASWAG",
                        num_samples=num_samples.get('hellaswag', 100)
                    )
                else:
                    logger.warning(f"GPU {gpu_id}: Unknown dataset {dataset}")
                    continue

                result['evaluation_time'] = time.time() - start_time
                result['gpu_id'] = gpu_id
                results[dataset] = result

                logger.info(
                    f"GPU {gpu_id}: Completed {dataset} in {result['evaluation_time']:.2f}s")

            except Exception as e:
                logger.error(
                    f"GPU {gpu_id}: Failed to evaluate {dataset}: {e}")
                results[dataset] = {"error": str(e), "gpu_id": gpu_id}

        # 获取模型统计信息
        model_stats = evaluator.get_model_statistics()

        result_queue.put({
            'gpu_id': gpu_id,
            'results': results,
            'model_statistics': model_stats
        })

        logger.info(f"GPU {gpu_id}: All evaluations completed")

    except Exception as e:
        logger.error(f"GPU {gpu_id}: Fatal error: {e}")
        result_queue.put({
            'gpu_id': gpu_id,
            'error': str(e)
        })


def distribute_datasets_to_gpus(datasets: List[str], num_gpus: int) -> List[List[str]]:
    """将数据集分配到不同的GPU上"""
    # 数据集评估时间权重（相对值）
    dataset_weights = {
        'wikitext': 2,
        'mmlu': 4,
        'gsm8k': 3,
        'hellaswag': 3
    }

    # 为每个数据集创建带权重的条目
    weighted_datasets = [(ds, dataset_weights.get(ds.lower(), 1))
                         for ds in datasets]

    # 排序：权重大的优先
    weighted_datasets.sort(key=lambda x: x[1], reverse=True)

    # 使用贪心算法分配
    gpu_loads = [0] * num_gpus
    gpu_datasets = [[] for _ in range(num_gpus)]

    for dataset, weight in weighted_datasets:
        # 找到当前负载最小的GPU
        min_load_gpu = gpu_loads.index(min(gpu_loads))
        gpu_datasets[min_load_gpu].append(dataset)
        gpu_loads[min_load_gpu] += weight

    return gpu_datasets


def main():
    parser = argparse.ArgumentParser(
        description="Parallel evaluation of dynamic quantization MoE models"
    )
    parser.add_argument("--fp16-model", type=str, required=True,
                        help="Path to the FP16 model")
    parser.add_argument("--int4-model", type=str, required=True,
                        help="Path to the INT4 model")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["wikitext", "mmlu", "gsm8k", "hellaswag"],
                        help="Datasets to evaluate on")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs to use (default: all available)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root directory for datasets")
    parser.add_argument("--time-window", type=float, default=20.0,
                        help="Time window for expert tracking (seconds)")
    parser.add_argument("--hot-ratio", type=float, default=0.1,
                        help="Hot expert ratio (default: 0.1 = 10 percent)")
    parser.add_argument("--disable-dynamic-routing", action="store_true",
                        help="Disable dynamic routing")

    # 每个数据集的样本数
    parser.add_argument("--num-samples-wikitext", type=int, default=1000)
    parser.add_argument("--num-samples-mmlu", type=int, default=100)
    parser.add_argument("--num-samples-gsm8k", type=int, default=100)
    parser.add_argument("--num-samples-hellaswag", type=int, default=100)

    args = parser.parse_args()

    # 确定可用的GPU数量
    num_available_gpus = torch.cuda.device_count()
    if num_available_gpus == 0:
        logger.error("No CUDA GPUs available!")
        return

    num_gpus = args.num_gpus if args.num_gpus else num_available_gpus
    num_gpus = min(num_gpus, num_available_gpus, len(args.datasets))

    logger.info(f"Using {num_gpus} GPUs for parallel evaluation")

    # 分配数据集到GPU
    gpu_datasets = distribute_datasets_to_gpus(args.datasets, num_gpus)

    print(f"\n{'='*60}")
    print(f"Parallel Evaluation - Dynamic Quantization MoE")
    print(f"FP16 Model: {args.fp16_model}")
    print(f"INT4 Model: {args.int4_model}")
    print(f"Using {num_gpus} GPUs")
    print(f"Time Window: {args.time_window}s, Hot Ratio: {args.hot_ratio}")
    print(
        f"Dynamic Routing: {'enabled' if not args.disable_dynamic_routing else 'disabled'}")
    print(f"\nDataset distribution:")
    for i, datasets in enumerate(gpu_datasets):
        print(f"  GPU {i}: {', '.join(datasets)}")
    print(f"{'='*60}\n")

    # 准备样本数配置
    num_samples = {
        'wikitext': args.num_samples_wikitext,
        'mmlu': args.num_samples_mmlu,
        'gsm8k': args.num_samples_gsm8k,
        'hellaswag': args.num_samples_hellaswag
    }

    # 创建结果队列
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()

    # 启动评估进程
    processes = []
    start_time = time.time()

    for gpu_id in range(num_gpus):
        if not gpu_datasets[gpu_id]:
            continue

        p = mp.Process(
            target=evaluate_on_gpu,
            args=(
                gpu_id,
                args.fp16_model,
                args.int4_model,
                gpu_datasets[gpu_id],
                args.data_dir,
                args.max_length,
                args.time_window,
                args.hot_ratio,
                args.disable_dynamic_routing,
                num_samples,
                result_queue
            )
        )
        p.start()
        processes.append(p)
        logger.info(f"Started evaluation process on GPU {gpu_id}")

    # 收集结果
    all_results = {}
    all_model_stats = {}

    for _ in range(len(processes)):
        result = result_queue.get()
        gpu_id = result['gpu_id']

        if 'error' in result:
            logger.error(f"GPU {gpu_id} failed: {result['error']}")
            continue

        all_results.update(result['results'])
        all_model_stats[f'gpu_{gpu_id}'] = result['model_statistics']

    # 等待所有进程完成
    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    # 合并结果
    final_results = {
        "fp16_model": args.fp16_model,
        "int4_model": args.int4_model,
        "num_gpus_used": num_gpus,
        "max_length": args.max_length,
        "time_window": args.time_window,
        "hot_ratio": args.hot_ratio,
        "dynamic_routing_enabled": not args.disable_dynamic_routing,
        "total_evaluation_time": total_time,
        "evaluations": all_results,
        "model_statistics_per_gpu": all_model_stats,
        "evaluation_timestamp": datetime.now().isoformat()
    }

    # 打印结果
    print(f"\n{'='*60}")
    print("Evaluation Results:")
    print(f"{'='*60}\n")

    for dataset, result in all_results.items():
        print(f"{dataset.upper()}:")
        if "error" in result:
            print(f"  Error: {result['error']}")
        elif "perplexity" in result:
            print(f"  Perplexity: {result['perplexity']:.2f}")
            print(
                f"  Time: {result.get('evaluation_time', 0):.2f}s (GPU {result.get('gpu_id', 'N/A')})")
        elif "accuracy" in result:
            print(
                f"  Accuracy: {result['accuracy']:.4f} ({result.get('correct', 0)}/{result.get('total', 0)})")
            print(
                f"  Time: {result.get('evaluation_time', 0):.2f}s (GPU {result.get('gpu_id', 'N/A')})")
        print()

    # 打印统计信息摘要
    print(f"{'='*60}")
    print("Model Statistics Summary:")
    print(f"{'='*60}\n")

    # 聚合所有GPU的统计信息
    total_tokens = 0
    total_hot_calls = 0
    total_cold_calls = 0
    total_fp16_calls = 0
    total_int4_calls = 0

    for gpu_stats in all_model_stats.values():
        inf_stats = gpu_stats['inference_stats']
        total_tokens += inf_stats['total_tokens']
        total_hot_calls += inf_stats['hot_expert_calls']
        total_cold_calls += inf_stats['cold_expert_calls']
        total_fp16_calls += inf_stats['fp16_expert_calls']
        total_int4_calls += inf_stats['int4_expert_calls']

    print(f"Total tokens processed: {total_tokens}")
    print(f"Total hot expert calls: {total_hot_calls}")
    print(f"Total cold expert calls: {total_cold_calls}")
    print(f"Total FP16 expert calls: {total_fp16_calls}")
    print(f"Total INT4 expert calls: {total_int4_calls}")

    if total_hot_calls + total_cold_calls > 0:
        actual_hot_ratio = total_hot_calls / \
            (total_hot_calls + total_cold_calls)
        print(f"Actual hot ratio: {actual_hot_ratio:.4f}")

    print(f"\nTotal evaluation time: {total_time:.2f}s")
    print(f"Average time per dataset: {total_time / len(all_results):.2f}s")

    # 保存结果
    if args.output:
        output_path = args.output
    else:
        model_name = Path(args.fp16_model).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"eval_results_parallel_{model_name}_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Evaluation completed!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
