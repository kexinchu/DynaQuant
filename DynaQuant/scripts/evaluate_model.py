"""
Model Evaluation Script
========================
评估量化模型在多个数据集上的准确度和困惑度(PPL)

支持的数据集：
- WikiText-2 (PPL)
- MMLU (Accuracy)
- GSM8K (Accuracy)
- HellaSwag (Accuracy)
- HumanEval (Accuracy)
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model_path: str, device: str = "cuda", max_length: int = 2048):
        """
        初始化评估器

        Args:
            model_path: 模型路径
            device: 设备 (cuda/cpu)
            max_length: 最大序列长度
        """
        logger.info(f"Loading model from {model_path}")
        self.device = device
        self.max_length = max_length

        # 加载模型和tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
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
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
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

    def evaluate_wikitext(self, split: str = "test") -> Dict:
        """评估WikiText-2数据集"""
        logger.info("Loading WikiText-2 dataset")
        try:
            dataset = load_dataset(
                "wikitext", "wikitext-2-raw-v1", split=split)
            texts = [text for text in dataset["text"] if text.strip()]
            texts = texts[:1000]  # 限制样本数量以加速评估

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

        # 查找所有test文件
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

        for test_file in tqdm(test_files[:10], desc="MMLU subjects"):  # 限制subject数量
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
                    inputs = self.tokenizer(
                        prompt, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=1,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )

                    pred = self.tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    pred = pred.strip().upper()

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
                inputs = self.tokenizer(
                    prompt, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                pred = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                # 提取预测答案
                try:
                    # 简单提取最后一个数字
                    import re
                    numbers = re.findall(r'-?\d+\.?\d*', pred)
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

                # 计算每个ending的概率
                max_prob = -float('inf')
                pred_idx = 0

                for idx, ending in enumerate(endings):
                    text = ctx + " " + ending
                    inputs = self.tokenizer(
                        text, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        outputs = self.model(
                            **inputs, labels=inputs["input_ids"])
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantized models")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model to evaluate")
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
    args = parser.parse_args()

    # 创建评估器
    evaluator = ModelEvaluator(
        args.model, device=args.device, max_length=args.max_length)

    # 运行评估
    results = {
        "model": args.model,
        "device": args.device,
        "max_length": args.max_length,
        "evaluations": {}
    }

    print(f"\n{'='*60}")
    print(f"Evaluating Model: {args.model}")
    print(f"{'='*60}\n")

    for dataset in args.datasets:
        print(f"\n--- Evaluating {dataset.upper()} ---")

        if dataset.lower() == "wikitext":
            result = evaluator.evaluate_wikitext()
        elif dataset.lower() == "mmlu":
            result = evaluator.evaluate_mmlu(data_dir=f"{args.data_dir}/MMLU")
        elif dataset.lower() == "gsm8k":
            result = evaluator.evaluate_gsm8k(
                data_dir=f"{args.data_dir}/GSM8K")
        elif dataset.lower() == "hellaswag":
            result = evaluator.evaluate_hellaswag(
                data_dir=f"{args.data_dir}/HELLASWAG")
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

    # 保存结果
    if args.output:
        output_path = args.output
    else:
        model_name = Path(args.model).name
        output_path = f"eval_results_{model_name}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Evaluation completed!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
