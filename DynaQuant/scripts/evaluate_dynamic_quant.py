#!/usr/bin/env python3
"""
动态量化MoE模型评估脚本 - 支持多GPU并行评估
评估动态量化模型在多个数据集上的准确度和困惑度
支持的数据集：WikiText-2, MMLU, GSM8K, HellaSwag
"""

import os
import json
import torch
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import time

from dynamic_quant_moe import DynamicQuantMoEModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DynamicQuantEvaluator:
    """动态量化模型评估器"""

    def __init__(
        self,
        fp16_model_path: str,
        int4_model_path: str,
        device: str = "cuda",
        max_length: int = 2048,
        time_window: float = 20.0,
        hot_ratio: float = 0.1,
        enable_dynamic_routing: bool = True
    ):
        """
        初始化评估器

        Args:
            fp16_model_path: FP16模型路径
            int4_model_path: INT4模型路径
            device: 设备
            max_length: 最大序列长度
            time_window: 时间窗口（秒）
            hot_ratio: hot专家比例
            enable_dynamic_routing: 是否启用动态路由
        """
        logger.info(f"Loading dynamic quantization model...")
        self.device = device
        self.max_length = max_length

        self.model = DynamicQuantMoEModel(
            fp16_model_path=fp16_model_path,
            int4_model_path=int4_model_path,
            device=device,
            time_window=time_window,
            hot_ratio=hot_ratio,
            enable_dynamic_routing=enable_dynamic_routing
        )

        self.tokenizer = self.model.tokenizer
        logger.info("Model loaded successfully")

    def evaluate_perplexity(self, texts: List[str], batch_size: int = 1) -> Dict:
        """
        评估困惑度

        Args:
            texts: 文本列表
            batch_size: 批量大小

        Returns:
            包含PPL和其他统计信息的字典
        """
        logger.info(f"Evaluating perplexity on {len(texts)} samples")

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Computing PPL"):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)

                try:
                    # Forward pass
                    outputs = self.model.forward(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=inputs["input_ids"]
                    )
                    loss = outputs.loss

                    # Accumulate
                    num_tokens = (inputs["attention_mask"] == 1).sum().item()
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens

                except Exception as e:
                    logger.warning(f"Failed to process batch {i}: {e}")
                    continue

        if total_tokens == 0:
            return {"perplexity": float('inf'), "total_tokens": 0}

        perplexity = np.exp(total_loss / total_tokens)

        return {
            "perplexity": float(perplexity),
            "total_tokens": total_tokens,
            "avg_loss": total_loss / total_tokens
        }

    def evaluate_wikitext(self, split: str = "test", num_samples: int = 1000) -> Dict:
        """评估WikiText-2数据集"""
        logger.info("Loading WikiText-2 dataset")
        try:
            dataset = load_dataset(
                "wikitext", "wikitext-2-raw-v1", split=split)
            texts = [text for text in dataset["text"] if text.strip()]
            texts = texts[:num_samples]

            result = self.evaluate_perplexity(texts)
            result["dataset"] = "wikitext-2"
            result["split"] = split
            result["num_samples"] = len(texts)
            return result
        except Exception as e:
            logger.error(f"Failed to evaluate WikiText: {e}")
            return {"dataset": "wikitext-2", "error": str(e)}

    def evaluate_mmlu(self, data_dir: str = "data/MMLU", num_samples: int = 100) -> Dict:
        """
        评估MMLU数据集

        Args:
            data_dir: MMLU数据目录
            num_samples: 每个subject的样本数
        """
        logger.info(f"Evaluating MMLU from {data_dir}")

        test_dir = Path(data_dir) / "data" / "test"
        if not test_dir.exists():
            logger.error(f"MMLU test directory not found: {test_dir}")
            return {"dataset": "mmlu", "error": "Data not found"}

        test_files = list(test_dir.glob("*.csv"))
        if not test_files:
            logger.error(f"No test files found in {test_dir}")
            return {"dataset": "mmlu", "error": "No test files"}

        total_correct = 0
        total_questions = 0
        subject_results = {}

        for test_file in tqdm(test_files[:10], desc="MMLU subjects"):
            subject = test_file.stem.replace("_test", "")

            try:
                df = pd.read_csv(test_file, header=None)
                df = df.head(num_samples)

                correct = 0
                for _, row in df.iterrows():
                    question = row[0]
                    choices = [row[1], row[2], row[3], row[4]]
                    answer = row[5]

                    # 构建prompt
                    prompt = f"Question: {question}\n"
                    for idx, choice in enumerate(choices):
                        prompt += f"{chr(65+idx)}. {choice}\n"
                    prompt += "Answer: "

                    # 生成答案
                    pred_text = self.model.generate(
                        prompt, max_new_tokens=1, do_sample=False)
                    pred = pred_text[len(prompt):].strip().upper()

                    if pred and pred[0] == chr(65 + answer):
                        correct += 1

                    total_questions += 1

                accuracy = correct / len(df) if len(df) > 0 else 0
                subject_results[subject] = {
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": len(df)
                }
                total_correct += correct

            except Exception as e:
                logger.warning(f"Failed to evaluate {subject}: {e}")
                continue

        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0

        return {
            "dataset": "mmlu",
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "num_subjects": len(subject_results),
            "subject_results": subject_results
        }

    def evaluate_gsm8k(self, data_dir: str = "data/GSM8K", num_samples: int = 100) -> Dict:
        """评估GSM8K数据集"""
        logger.info(f"Evaluating GSM8K from {data_dir}")

        try:
            # 尝试从本地加载
            parquet_file = Path(data_dir) / "test-00000-of-00001.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                df = df.head(num_samples)
            else:
                # 从HuggingFace加载
                dataset = load_dataset("gsm8k", "main", split="test")
                df = pd.DataFrame(dataset)
                df = df.head(num_samples)

            correct = 0
            total = len(df)

            for _, row in tqdm(df.iterrows(), total=len(df), desc="GSM8K"):
                question = row["question"] if "question" in row else row[0]
                answer = row["answer"] if "answer" in row else row[1]

                # 提取数字答案
                try:
                    gold_answer = answer.split(
                        "####")[-1].strip().replace(",", "")
                    gold_answer = float(gold_answer)
                except:
                    continue

                # 生成答案
                prompt = f"Question: {question}\nAnswer: Let's think step by step.\n"
                pred = self.model.generate(
                    prompt, max_new_tokens=256, do_sample=False)

                # 提取预测答案
                try:
                    import re
                    numbers = re.findall(r'-?\d+\.?\d*', pred[len(prompt):])
                    if numbers:
                        pred_answer = float(numbers[-1].replace(",", ""))
                        if abs(pred_answer - gold_answer) < 1e-3:
                            correct += 1
                except:
                    continue

            accuracy = correct / total if total > 0 else 0

            return {
                "dataset": "gsm8k",
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            }

        except Exception as e:
            logger.error(f"Failed to evaluate GSM8K: {e}")
            return {"dataset": "gsm8k", "error": str(e)}

    def evaluate_hellaswag(self, data_dir: str = "data/HELLASWAG", num_samples: int = 100) -> Dict:
        """评估HellaSwag数据集"""
        logger.info(f"Evaluating HellaSwag from {data_dir}")

        try:
            # 尝试从本地加载
            parquet_file = Path(data_dir) / "data" / \
                "test-00000-of-00001.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                df = df.head(num_samples)
            else:
                # 从HuggingFace加载
                dataset = load_dataset("hellaswag", split="validation")
                df = pd.DataFrame(dataset)
                df = df.head(num_samples)

            correct = 0
            total = len(df)

            for _, row in tqdm(df.iterrows(), total=len(df), desc="HellaSwag"):
                ctx = row["ctx"] if "ctx" in row else row.get("context", "")
                endings = row["endings"] if "endings" in row else []
                label = int(row["label"]) if "label" in row else 0

                if not endings:
                    continue

                # 计算每个ending的概率（使用loss作为度量）
                max_prob = -float('inf')
                pred_idx = 0

                for idx, ending in enumerate(endings):
                    text = ctx + " " + ending
                    inputs = self.tokenizer(
                        text, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        outputs = self.model.forward(
                            input_ids=inputs["input_ids"],
                            labels=inputs["input_ids"]
                        )
                        loss = outputs.loss.item()

                    if -loss > max_prob:
                        max_prob = -loss
                        pred_idx = idx

                if pred_idx == label:
                    correct += 1

            accuracy = correct / total if total > 0 else 0

            return {
                "dataset": "hellaswag",
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            }

        except Exception as e:
            logger.error(f"Failed to evaluate HellaSwag: {e}")
            return {"dataset": "hellaswag", "error": str(e)}

    def get_model_statistics(self) -> Dict:
        """获取模型统计信息"""
        return self.model.get_statistics()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dynamic quantization MoE models")
    parser.add_argument("--fp16-model", type=str, required=True,
                        help="Path to the FP16 model")
    parser.add_argument("--int4-model", type=str, required=True,
                        help="Path to the INT4 model")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["wikitext", "mmlu", "gsm8k", "hellaswag"],
                        help="Datasets to evaluate on")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run evaluation on")
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
    parser.add_argument("--num-samples-wikitext", type=int, default=1000,
                        help="Number of samples for WikiText")
    parser.add_argument("--num-samples-mmlu", type=int, default=100,
                        help="Number of samples per subject for MMLU")
    parser.add_argument("--num-samples-gsm8k", type=int, default=100,
                        help="Number of samples for GSM8K")
    parser.add_argument("--num-samples-hellaswag", type=int, default=100,
                        help="Number of samples for HellaSwag")

    args = parser.parse_args()

    # 创建评估器
    evaluator = DynamicQuantEvaluator(
        fp16_model_path=args.fp16_model,
        int4_model_path=args.int4_model,
        device=args.device,
        max_length=args.max_length,
        time_window=args.time_window,
        hot_ratio=args.hot_ratio,
        enable_dynamic_routing=not args.disable_dynamic_routing
    )

    # 运行评估
    results = {
        "fp16_model": args.fp16_model,
        "int4_model": args.int4_model,
        "device": args.device,
        "max_length": args.max_length,
        "time_window": args.time_window,
        "hot_ratio": args.hot_ratio,
        "dynamic_routing_enabled": not args.disable_dynamic_routing,
        "evaluations": {}
    }

    print(f"\n{'='*60}")
    print(f"Evaluating Dynamic Quantization MoE Model")
    print(f"FP16 Model: {args.fp16_model}")
    print(f"INT4 Model: {args.int4_model}")
    print(f"Time Window: {args.time_window}s, Hot Ratio: {args.hot_ratio}")
    print(
        f"Dynamic Routing: {'enabled' if not args.disable_dynamic_routing else 'disabled'}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for dataset in args.datasets:
        print(f"\n--- Evaluating {dataset.upper()} ---")

        if dataset.lower() == "wikitext":
            result = evaluator.evaluate_wikitext(
                num_samples=args.num_samples_wikitext)
        elif dataset.lower() == "mmlu":
            result = evaluator.evaluate_mmlu(
                data_dir=f"{args.data_dir}/MMLU",
                num_samples=args.num_samples_mmlu
            )
        elif dataset.lower() == "gsm8k":
            result = evaluator.evaluate_gsm8k(
                data_dir=f"{args.data_dir}/GSM8K",
                num_samples=args.num_samples_gsm8k
            )
        elif dataset.lower() == "hellaswag":
            result = evaluator.evaluate_hellaswag(
                data_dir=f"{args.data_dir}/HELLASWAG",
                num_samples=args.num_samples_hellaswag
            )
        else:
            logger.warning(f"Unknown dataset: {dataset}")
            continue

        results["evaluations"][dataset] = result

        # 打印结果
        if "error" in result:
            print(f"  Error: {result['error']}")
        elif "perplexity" in result:
            print(f"  Perplexity: {result['perplexity']:.2f}")
        elif "accuracy" in result:
            print(
                f"  Accuracy: {result['accuracy']:.4f} ({result.get('correct', 0)}/{result.get('total', 0)})")

    end_time = time.time()

    # 获取模型统计信息
    model_stats = evaluator.get_model_statistics()
    results["model_statistics"] = model_stats
    results["total_evaluation_time"] = end_time - start_time

    # 打印统计信息
    print(f"\n{'='*60}")
    print("Model Statistics:")
    print(
        f"  Total tokens processed: {model_stats['inference_stats']['total_tokens']}")
    print(
        f"  Hot expert calls: {model_stats['inference_stats']['hot_expert_calls']}")
    print(
        f"  Cold expert calls: {model_stats['inference_stats']['cold_expert_calls']}")
    print(
        f"  FP16 expert calls: {model_stats['inference_stats']['fp16_expert_calls']}")
    print(
        f"  INT4 expert calls: {model_stats['inference_stats']['int4_expert_calls']}")

    tracker_stats = model_stats['tracker_stats']
    print(f"\nTracker Statistics:")
    print(f"  Time window: {tracker_stats['time_window']}s")
    print(f"  Total activations: {tracker_stats['total_activations']}")
    print(f"  Hot activations: {tracker_stats['total_hot_activations']}")
    print(f"  Cold activations: {tracker_stats['total_cold_activations']}")
    print(f"  Hot ratio: {tracker_stats['hot_ratio']:.4f}")
    print(f"  Num hot experts: {tracker_stats['num_hot_experts']}")

    print(f"\nTotal evaluation time: {results['total_evaluation_time']:.2f}s")

    # 保存结果
    if args.output:
        output_path = args.output
    else:
        model_name = Path(args.fp16_model).name
        output_path = f"eval_results_dynamic_quant_{model_name}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Evaluation completed!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
