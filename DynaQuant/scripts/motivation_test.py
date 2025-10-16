#!/usr/bin/env python3
"""
Motivation Test: 测试量化对MoE Router激活模式的影响

对照组: 全FP16
实验组1: Experts全Int4
实验组2: Hot 10% experts FP16 + Cold 90% Int4
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from tqdm import tqdm
import pickle


class MixedPrecisionMoE:
    """动态混合精度MoE模型 - 真正实现非expert层FP16 + expert层可配置"""

    def __init__(
        self,
        fp16_model_path: str,
        int4_model_path: Optional[str] = None,
        device: str = "cuda"
    ):
        self.device = device
        self.fp16_model_path = fp16_model_path
        self.int4_model_path = int4_model_path

        print(f"加载FP16模型: {fp16_model_path}")
        self.fp16_model = AutoModelForCausalLM.from_pretrained(
            fp16_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            fp16_model_path,
            trust_remote_code=True
        )

        if int4_model_path:
            print(f"加载Int4模型: {int4_model_path}")
            self.int4_model = AutoModelForCausalLM.from_pretrained(
                int4_model_path,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.int4_model = None

        # 获取模型配置
        self.config = self.fp16_model.config
        self.num_layers = self.config.num_hidden_layers
        self.num_experts = getattr(self.config, 'num_experts', 64)

        print(f"模型配置: {self.num_layers} 层, 每层 {self.num_experts} 个experts")

        # 存储原始expert权重的备份
        self.original_expert_modules = {}
        self.current_expert_config = None

    def _get_expert_modules(self, model):
        """获取所有expert相关的模块"""
        expert_modules = {}
        for layer_idx in range(self.num_layers):
            if hasattr(model.model, 'layers'):
                layer = model.model.layers[layer_idx]

                # 检查MoE结构 - 可能是mlp.experts或其他结构
                if hasattr(layer, 'mlp'):
                    mlp = layer.mlp
                    # 查找experts属性
                    if hasattr(mlp, 'experts'):
                        expert_modules[layer_idx] = mlp.experts
                    elif hasattr(mlp, 'shared_expert_gate'):
                        # DeepSeekV2风格的MoE
                        expert_modules[layer_idx] = mlp

        return expert_modules

    def replace_experts_with_int4(self):
        """将所有expert层替换为Int4版本"""
        if self.int4_model is None:
            print("警告: 没有Int4模型，无法替换expert")
            return

        print("替换expert层为Int4版本...")

        # 备份原始FP16 experts
        if not self.original_expert_modules:
            self.original_expert_modules = self._get_expert_modules(
                self.fp16_model)
            print(f"  备份了 {len(self.original_expert_modules)} 层的FP16 experts")

        # 获取Int4 experts
        int4_expert_modules = self._get_expert_modules(self.int4_model)

        # 替换每一层的experts
        replaced_count = 0
        for layer_idx in range(self.num_layers):
            if layer_idx not in int4_expert_modules:
                continue

            fp16_layer = self.fp16_model.model.layers[layer_idx]
            int4_experts = int4_expert_modules[layer_idx]

            # 替换experts模块
            if hasattr(fp16_layer.mlp, 'experts'):
                fp16_layer.mlp.experts = int4_experts
                replaced_count += 1
            elif hasattr(fp16_layer, 'mlp'):
                # 整个mlp替换（DeepSeekV2风格）
                # 保存非expert部分
                old_gate = fp16_layer.mlp.gate if hasattr(
                    fp16_layer.mlp, 'gate') else None
                # 替换experts
                fp16_layer.mlp.experts = int4_experts
                # 恢复gate
                if old_gate is not None:
                    fp16_layer.mlp.gate = old_gate
                replaced_count += 1

        print(f"  ✓ 替换了 {replaced_count} 层的experts为Int4")
        self.current_expert_config = "all_int4"

    def restore_fp16_experts(self):
        """恢复所有expert层为FP16版本"""
        if not self.original_expert_modules:
            print("警告: 没有备份的FP16 experts")
            return

        print("恢复expert层为FP16版本...")

        restored_count = 0
        for layer_idx, fp16_experts in self.original_expert_modules.items():
            fp16_layer = self.fp16_model.model.layers[layer_idx]

            if hasattr(fp16_layer.mlp, 'experts'):
                fp16_layer.mlp.experts = fp16_experts
                restored_count += 1

        print(f"  ✓ 恢复了 {restored_count} 层的experts为FP16")
        self.current_expert_config = "fp16"

    def verify_expert_precision(self):
        """验证当前expert的精度配置"""
        print("\n验证Expert精度配置...")

        for layer_idx in range(min(3, self.num_layers)):  # 检查前3层
            layer = self.fp16_model.model.layers[layer_idx]

            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                experts = layer.mlp.experts

                # 检查experts的第一个expert的权重类型
                if hasattr(experts, 'experts') and len(experts.experts) > 0:
                    first_expert = experts.experts[0]
                    # 查找第一个参数
                    for name, param in first_expert.named_parameters():
                        print(f"  Layer {layer_idx} - {name}:")
                        print(f"    dtype: {param.dtype}")
                        print(f"    device: {param.device}")
                        if hasattr(param, 'qweight'):
                            print(f"    量化类型: AWQ (qweight存在)")
                        break
                    break

            # 同时检查非expert层（attention）的精度
            if hasattr(layer, 'self_attn'):
                for name, param in layer.self_attn.named_parameters():
                    print(f"  Layer {layer_idx} - Attention {name}:")
                    print(f"    dtype: {param.dtype}")
                    break

        print(f"当前配置: {self.current_expert_config}\n")

    def setup_expert_hooks(self, expert_config: str = "fp16"):
        """
        设置expert层的配置并注册hooks
        expert_config: "fp16" | "all_int4" | "mixed"
        """
        self.expert_config = expert_config
        self.expert_activations = {i: [] for i in range(self.num_layers)}
        self.router_weights = {i: [] for i in range(self.num_layers)}

        # 根据配置替换expert权重
        if expert_config == "all_int4":
            self.replace_experts_with_int4()
        elif expert_config == "fp16":
            if self.current_expert_config == "all_int4":
                self.restore_fp16_experts()

        # 注册hook来记录expert激活
        self.hooks = []

        for layer_idx in range(self.num_layers):
            # 获取对应的MoE层
            if hasattr(self.fp16_model.model, 'layers'):
                layer = self.fp16_model.model.layers[layer_idx]
            else:
                continue

            # 检查是否有MoE结构
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                hook = layer.mlp.gate.register_forward_hook(
                    self._create_router_hook(layer_idx)
                )
                self.hooks.append(hook)

        print(f"  ✓ 注册了 {len(self.hooks)} 个router hooks")

    def _create_router_hook(self, layer_idx: int):
        """创建router hook来记录expert选择"""
        def hook(module, input, output):
            # output是router的logits
            # 获取top-k expert indices
            if isinstance(output, tuple):
                router_logits = output[0]
            else:
                router_logits = output

            # 获取每个token选择的expert
            top_k_experts = torch.topk(router_logits, k=2, dim=-1).indices

            # 记录激活的expert
            activated_experts = top_k_experts.cpu().numpy()
            self.expert_activations[layer_idx].append(activated_experts)

            # 记录router weights
            router_probs = torch.softmax(router_logits, dim=-1)
            self.router_weights[layer_idx].append(router_probs.cpu().numpy())

        return hook

    def clear_hooks(self):
        """清除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate(self, prompt: str, max_new_tokens: int = 100) -> Tuple[str, Dict]:
        """生成文本并记录expert激活"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 清空之前的记录
        for layer_idx in range(self.num_layers):
            self.expert_activations[layer_idx] = []
            self.router_weights[layer_idx] = []

        # 生成
        with torch.no_grad():
            outputs = self.fp16_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)

        # 整理激活数据
        activation_data = {
            'expert_activations': self.expert_activations,
            'router_weights': self.router_weights
        }

        return generated_text, activation_data


def load_expert_statistics(stats_dir: str) -> Dict:
    """加载之前的expert激活统计"""
    stats_files = list(Path(stats_dir).glob("*_summary.json"))

    if not stats_files:
        raise FileNotFoundError(f"未找到expert统计文件: {stats_dir}")

    # 合并所有数据集的统计
    all_stats = {}
    for stats_file in stats_files:
        with open(stats_file, 'r') as f:
            data = json.load(f)
            dataset_name = stats_file.stem.replace('_summary', '')
            all_stats[dataset_name] = data

    return all_stats


def identify_hot_experts(expert_stats: Dict, hot_ratio: float = 0.1) -> Dict[int, List[int]]:
    """识别每层的hot experts"""
    hot_experts = {}

    # 假设stats结构: {dataset: {layer_idx: {expert_idx: count}}}
    # 合并所有数据集的统计
    merged_stats = {}

    for dataset, stats in expert_stats.items():
        for layer_str, layer_stats in stats.items():
            if 'layer_' not in layer_str:
                continue
            layer_idx = int(layer_str.replace('layer_', ''))

            if layer_idx not in merged_stats:
                merged_stats[layer_idx] = {}

            if isinstance(layer_stats, dict):
                for expert_str, count in layer_stats.items():
                    if 'expert_' in expert_str:
                        expert_idx = int(expert_str.replace('expert_', ''))
                        merged_stats[layer_idx][expert_idx] = \
                            merged_stats[layer_idx].get(expert_idx, 0) + count

    # 对每层选择top hot_ratio的experts
    for layer_idx, expert_counts in merged_stats.items():
        sorted_experts = sorted(
            expert_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        num_hot = max(1, int(len(sorted_experts) * hot_ratio))
        hot_experts[layer_idx] = [
            exp_idx for exp_idx, _ in sorted_experts[:num_hot]]

    return hot_experts


def load_test_datasets(num_samples: int = 256) -> Dict[str, List[str]]:
    """加载测试数据集 - 只加载GSM8K"""
    datasets_dict = {}

    print(f"加载测试数据集: GSM8K (每个{num_samples}样本)...")

    # 只加载GSM8K
    try:
        # 首先尝试从parquet文件加载
        parquet_path = "./data/GSM8K/test-00000-of-00001.parquet"
        if os.path.exists(parquet_path):
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            # 获取question列
            if 'question' in df.columns:
                texts = df['question'].tolist()
            else:
                texts = df.iloc[:, 0].tolist()
            texts = [str(t) for t in texts if len(str(t).strip()) > 10]
            datasets_dict['gsm8k'] = texts[:num_samples]
            print(f"  ✓ GSM8K (从parquet): {len(datasets_dict['gsm8k'])} 样本")
        else:
            # 从HuggingFace加载
            gsm8k = load_dataset("gsm8k", "main", split="test")
            texts = [item['question'] for item in gsm8k]
            datasets_dict['gsm8k'] = texts[:num_samples]
            print(
                f"  ✓ GSM8K (从HuggingFace): {len(datasets_dict['gsm8k'])} 样本")
    except Exception as e:
        print(f"  ✗ GSM8K加载失败: {e}")

    return datasets_dict


def convert_to_expert_activation_format(expert_activations_list, num_layers, num_experts):
    """
    将expert激活数据转换为与之前一致的JSON格式
    输出格式: [layer0_dict, layer1_dict, ...]
    每个layer_dict: {"expert_id": count, ...}
    """
    # 初始化每层的计数器
    layer_counts = [defaultdict(int) for _ in range(num_layers)]

    # 遍历所有样本的激活数据
    for sample_activation in expert_activations_list:
        for layer_idx, activations_list in sample_activation.items():
            if not isinstance(layer_idx, int):
                continue
            if layer_idx >= num_layers:
                continue

            # 统计这个样本在该层激活的experts
            for act_array in activations_list:
                if isinstance(act_array, np.ndarray):
                    # act_array shape: (seq_len, top_k)
                    expert_ids = act_array.flatten()
                    for exp_id in expert_ids:
                        if 0 <= exp_id < num_experts:
                            layer_counts[layer_idx][str(int(exp_id))] += 1

    # 转换为标准dict格式
    result = []
    for layer_dict in layer_counts:
        result.append(dict(layer_dict))

    return result


def run_experiment(
    model: MixedPrecisionMoE,
    test_data: Dict[str, List[str]],
    test_group: str,
    output_dir: str,
    max_new_tokens: int = 100
):
    """运行实验并保存为与之前expert activation一致的JSON格式"""
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name, prompts in test_data.items():
        print(f"\n处理数据集: {dataset_name}")

        # 收集所有激活数据
        all_expert_activations = []

        for prompt in tqdm(prompts, desc=f"  {dataset_name}"):
            try:
                generated_text, activation_data = model.generate(
                    prompt,
                    max_new_tokens=max_new_tokens
                )

                all_expert_activations.append(
                    activation_data['expert_activations']
                )

            except Exception as e:
                print(f"    错误: {e}")
                continue

        print(f"  收集了 {len(all_expert_activations)} 个样本的激活数据")

        # 转换为与之前一致的JSON格式
        print(f"  转换为标准格式...")
        expert_counts_json = convert_to_expert_activation_format(
            all_expert_activations,
            model.num_layers,
            model.num_experts
        )

        # 保存为JSON (与之前格式一致)
        json_output = os.path.join(
            output_dir, f"{dataset_name}_{test_group}.json")
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(expert_counts_json, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 保存JSON: {json_output}")

        # 打印统计信息
        total_layers_with_data = sum(
            1 for layer_dict in expert_counts_json if layer_dict)
        print(
            f"  统计: {total_layers_with_data}/{len(expert_counts_json)} 层有激活数据")

        # 显示第一层的top-3专家
        if expert_counts_json and expert_counts_json[0]:
            first_layer = expert_counts_json[0]
            sorted_experts = sorted(
                first_layer.items(), key=lambda x: x[1], reverse=True)[:3]
            print(
                f"  第1层 top-3: {[(exp_id, count) for exp_id, count in sorted_experts]}")

    # 保存元数据
    metadata = {
        'test_group': test_group,
        'num_layers': model.num_layers,
        'num_experts': model.num_experts,
        'datasets': list(test_data.keys()),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'output_format': 'expert_activation_compatible'
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 实验完成，结果保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Motivation Test for Quantization Impact")
    parser.add_argument('--model_path', type=str,
                        required=True, help='FP16模型路径')
    parser.add_argument('--int4_model_path', type=str, help='Int4模型路径')
    parser.add_argument('--test_group', type=str, required=True,
                        choices=['control', 'exp1_all_int4', 'exp2_mixed'],
                        help='测试组')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--num_samples', type=int,
                        default=256, help='每个数据集的样本数')
    parser.add_argument('--quantize_experts', type=str, default='none',
                        choices=['none', 'all', 'mixed'],
                        help='Expert量化策略')
    parser.add_argument('--hot_expert_ratio', type=float, default=0.1,
                        help='Hot expert比例（用于mixed模式）')
    parser.add_argument('--expert_stats_dir', type=str,
                        help='Expert统计目录（用于mixed模式）')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                        help='生成的最大token数')

    args = parser.parse_args()

    print("========================================")
    print(f"Motivation Test - {args.test_group}")
    print("========================================")
    print(f"FP16模型: {args.model_path}")
    if args.int4_model_path:
        print(f"Int4模型: {args.int4_model_path}")
    print(f"Expert配置: {args.quantize_experts}")
    print(f"输出目录: {args.output_dir}")
    print("========================================\n")

    # 加载模型
    model = MixedPrecisionMoE(
        fp16_model_path=args.model_path,
        int4_model_path=args.int4_model_path
    )

    # 设置expert配置
    if args.test_group == 'exp2_mixed' and args.expert_stats_dir:
        print("加载expert统计数据...")
        expert_stats = load_expert_statistics(args.expert_stats_dir)
        hot_experts = identify_hot_experts(expert_stats, args.hot_expert_ratio)
        print(f"识别了 {len(hot_experts)} 层的hot experts")
        for layer_idx in list(hot_experts.keys())[:3]:
            print(
                f"  Layer {layer_idx}: {len(hot_experts[layer_idx])} hot experts")
        # 这里可以进一步配置模型使用mixed precision

    # 设置hooks
    expert_config = "fp16" if args.test_group == "control" else args.quantize_experts
    model.setup_expert_hooks(expert_config=expert_config)

    # 验证expert配置
    model.verify_expert_precision()

    # 加载测试数据
    test_data = load_test_datasets(num_samples=args.num_samples)

    if not test_data:
        print("错误: 没有加载到任何测试数据")
        return

    # 运行实验
    run_experiment(
        model=model,
        test_data=test_data,
        test_group=args.test_group,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens
    )

    # 清理
    model.clear_hooks()

    print("\n实验完成!")


if __name__ == "__main__":
    main()
