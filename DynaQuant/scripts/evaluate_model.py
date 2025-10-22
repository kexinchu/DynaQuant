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
            skipped = 0

            # Few-shot examples (标准的5个示例)
            few_shot_examples = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
#### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $10.
#### 10

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: In the beginning, Betty has only 100 / 2 = $50.
Betty's grandparents gave her 15 * 2 = $30.
This means, Betty needs 100 - 50 - 15 - 30 = $5 more.
#### 5

Question: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
Answer: Maila read 12 x 2 = 24 pages today.
So she was able to read a total of 12 + 24 = 36 pages since yesterday.
There are 120 - 36 = 84 pages left to be read.
Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = 42 pages.
#### 42

Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Answer: He writes each friend 3*2=6 pages a week
So he writes 6*2=12 pages every week
That means he writes 12*52=624 pages a year
#### 624

"""

            for _, row in tqdm(df.iterrows(), total=len(df), desc="GSM8K"):
                question = row["question"] if "question" in row else row[0]
                answer = row["answer"] if "answer" in row else row[1]

                # 提取数字答案
                try:
                    gold_answer = answer.split(
                        "####")[-1].strip().replace(",", "")
                    gold_answer = float(gold_answer)
                except:
                    skipped += 1
                    continue

                # 使用 chat template 构建 prompt
                question_text = few_shot_examples + \
                    f"Question: {question}\nAnswer:"

                # 尝试使用 apply_chat_template
                try:
                    messages = [{"role": "user", "content": question_text}]
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    # 如果 apply_chat_template 失败，使用简单格式
                    logger.debug(
                        f"apply_chat_template failed, using simple format: {e}")
                    prompt = question_text

                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=self.max_length
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,  # 增加生成长度
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                pred = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]
                        :], skip_special_tokens=True
                )

                # 改进的答案提取逻辑
                try:
                    import re
                    # 首先尝试查找 #### 格式的答案
                    if "####" in pred:
                        pred_str = pred.split("####")[-1].strip()
                        pred_str = re.sub(r'[,\s]', '', pred_str)
                        # 提取第一个数字
                        numbers = re.findall(r'-?\d+\.?\d*', pred_str)
                        if numbers:
                            pred_answer = float(numbers[0])
                        else:
                            raise ValueError("No number found after ####")
                    else:
                        # 否则查找所有数字，优先使用最后一行的数字
                        lines = pred.strip().split('\n')
                        pred_answer = None

                        # 从后往前查找包含数字的行
                        for line in reversed(lines):
                            # 移除逗号和空格
                            line_clean = line.replace(',', '').replace(' ', '')
                            # 查找数字（包括负数和小数）
                            numbers = re.findall(r'-?\d+\.?\d*', line_clean)
                            if numbers:
                                # 使用最后一个数字
                                pred_answer = float(numbers[-1])
                                break

                        if pred_answer is None:
                            raise ValueError("No number found in prediction")

                    # 比较答案（允许小的浮点误差）
                        if abs(pred_answer - gold_answer) < 1e-2:
                            correct += 1

                except Exception as e:
                    logger.debug(
                        f"Failed to extract answer: {e}, pred: {pred[:100]}")
                    skipped += 1
                    continue

            accuracy = correct / total if total > 0 else 0

            return {
                "dataset": "gsm8k",
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "skipped": skipped
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
