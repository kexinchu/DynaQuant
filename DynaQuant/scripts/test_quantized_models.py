#!/usr/bin/env python3
"""
快速测试量化模型

测试W2A2和W4A4模型的性能和精度
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
from pathlib import Path
import argparse
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantizedModelTester:
    """量化模型测试器"""

    def __init__(
        self,
        model_path: str,
        quantized_weights_path: str,
        device: str = "cuda:0"
    ):
        self.model_path = model_path
        self.quantized_weights_path = quantized_weights_path
        self.device = device

        # 加载模型和tokenizer
        logger.info(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()

        # 加载量化权重
        logger.info(
            f"Loading quantized weights from {quantized_weights_path}...")
        self.quant_data = torch.load(quantized_weights_path)
        self.quantized_weights = self.quant_data["quantized_weights"]

        logger.info(f"Loaded {len(self.quantized_weights)} quantized modules")

    def apply_quantized_weights(self):
        """应用量化权重到模型（简化版，仅用于测试）"""
        logger.info("Applying quantized weights...")

        # 注意：这里是简化实现，实际应用需要替换为QuantizedLinear模块
        # 这里仅用于快速测试权重是否正确加载

        applied_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 尝试找到对应的量化权重
                for key in self.quantized_weights:
                    if name in key or key.replace("_", ".") in name:
                        weight_data = self.quantized_weights[key]
                        if "W_absorbed" in weight_data:
                            # 使用吸收了激活变换的权重
                            quant_weight = weight_data["W_absorbed"]
                            if quant_weight.shape == module.weight.shape:
                                module.weight.data.copy_(
                                    quant_weight.to(self.device))
                                applied_count += 1
                                break

        logger.info(f"Applied {applied_count} quantized weights")
        return applied_count > 0

    def test_generation(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        num_runs: int = 3
    ):
        """测试生成性能"""
        logger.info(f"Testing generation with prompt: '{prompt[:50]}...'")

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        latencies = []
        outputs_list = []

        for run in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            torch.cuda.synchronize()
            end = time.time()

            latency = end - start
            latencies.append(latency)

            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            outputs_list.append(generated_text)

            if run == 0:
                logger.info(
                    f"Generated (run {run+1}): {generated_text[:100]}...")

        avg_latency = sum(latencies) / len(latencies)
        tokens_per_sec = max_new_tokens / avg_latency

        return {
            "avg_latency": avg_latency,
            "tokens_per_sec": tokens_per_sec,
            "latencies": latencies,
            "outputs": outputs_list
        }

    def test_perplexity(self, test_texts: list):
        """测试困惑度"""
        logger.info(f"Testing perplexity on {len(test_texts)} samples...")

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in test_texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                num_tokens = (inputs["attention_mask"] == 1).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()

        return {
            "perplexity": perplexity,
            "total_tokens": total_tokens
        }

    def test_memory(self):
        """测试内存使用"""
        torch.cuda.reset_peak_memory_stats()

        # 运行一次推理
        prompt = "测试内存使用的示例文本"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=50)

        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

        return {
            "peak_memory_gb": peak_memory
        }

    def run_all_tests(self, output_file: str = None):
        """运行所有测试"""
        results = {
            "model_path": self.model_path,
            "quantized_weights_path": self.quantized_weights_path,
            "config": {
                "expert_config": str(self.quant_data.get("expert_config")),
                "router_config": str(self.quant_data.get("router_config"))
            }
        }

        # 应用量化权重
        applied = self.apply_quantized_weights()
        results["weights_applied"] = applied

        # 测试生成
        test_prompts = [
            "人工智能是",
            "量子计算的未来",
            "气候变化带来的挑战"
        ]

        generation_results = []
        for prompt in test_prompts:
            gen_result = self.test_generation(
                prompt, max_new_tokens=50, num_runs=1)
            generation_results.append({
                "prompt": prompt,
                **gen_result
            })

        results["generation"] = generation_results

        # 测试困惑度
        ppl_texts = [
            "深度学习在计算机视觉领域取得了突破性进展。",
            "自然语言处理技术正在改变人机交互的方式。",
            "强化学习为智能决策系统提供了新的可能。"
        ]
        ppl_result = self.test_perplexity(ppl_texts)
        results["perplexity"] = ppl_result

        # 测试内存
        mem_result = self.test_memory()
        results["memory"] = mem_result

        # 打印摘要
        logger.info("\n" + "="*60)
        logger.info("测试结果摘要")
        logger.info("="*60)
        logger.info(f"模型: {self.model_path}")
        logger.info(f"量化权重: {self.quantized_weights_path}")
        logger.info(f"权重应用: {'成功' if applied else '失败'}")
        logger.info(f"困惑度: {ppl_result['perplexity']:.2f}")
        logger.info(f"内存使用: {mem_result['peak_memory_gb']:.2f} GB")
        logger.info(
            f"生成速度: {generation_results[0]['tokens_per_sec']:.2f} tokens/s")
        logger.info("="*60)

        # 保存结果
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"结果已保存到: {output_file}")

        return results


def main():
    parser = argparse.ArgumentParser(description="测试量化模型")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="原始模型路径"
    )
    parser.add_argument(
        "--w2a2-weights",
        type=str,
        help="W2A2量化权重路径"
    )
    parser.add_argument(
        "--w4a4-weights",
        type=str,
        help="W4A4量化权重路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_results",
        help="测试结果输出目录"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 测试W2A2
    if args.w2a2_weights:
        logger.info("\n" + "="*60)
        logger.info("测试W2A2模型")
        logger.info("="*60)

        tester_w2a2 = QuantizedModelTester(
            args.model,
            args.w2a2_weights,
            device="cuda:0"
        )

        results_w2a2 = tester_w2a2.run_all_tests(
            output_file=str(output_dir / "w2a2_test_results.json")
        )

    # 测试W4A4
    if args.w4a4_weights:
        logger.info("\n" + "="*60)
        logger.info("测试W4A4模型")
        logger.info("="*60)

        tester_w4a4 = QuantizedModelTester(
            args.model,
            args.w4a4_weights,
            device="cuda:0"
        )

        results_w4a4 = tester_w4a4.run_all_tests(
            output_file=str(output_dir / "w4a4_test_results.json")
        )

    logger.info("\n✅ 所有测试完成！")


if __name__ == "__main__":
    main()
