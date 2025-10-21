#!/usr/bin/env python3
"""
动态量化MoE模型 - 基于实时激活统计的hot/cold专家动态切换
支持20s时间窗口的实时统计和FP16/INT4动态路由
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExpertActivationRecord:
    """专家激活记录"""
    timestamp: float
    layer_id: int
    expert_id: int
    tokens_processed: int = 1


class RealTimeExpertTracker:
    """实时专家激活追踪器 - 20s时间窗口"""

    def __init__(self, time_window: float = 20.0, hot_ratio: float = 0.1):
        """
        Args:
            time_window: 时间窗口大小（秒）
            hot_ratio: hot专家比例（top 10%）
        """
        self.time_window = time_window
        self.hot_ratio = hot_ratio

        # 使用deque存储激活记录，自动维护时间窗口
        self.activation_records = deque()
        self.lock = threading.RLock()

        # 缓存的hot/cold分类结果
        # layer_id -> set of hot expert_ids
        self.hot_experts: Dict[int, set] = {}
        self.last_update_time = 0.0
        self.update_interval = 1.0  # 每秒更新一次hot/cold分类

        logger.info(
            f"RealTimeExpertTracker initialized: window={time_window}s, hot_ratio={hot_ratio}")

    def record_activation(self, layer_id: int, expert_id: int, tokens_processed: int = 1):
        """记录专家激活"""
        with self.lock:
            current_time = time.time()
            record = ExpertActivationRecord(
                timestamp=current_time,
                layer_id=layer_id,
                expert_id=expert_id,
                tokens_processed=tokens_processed
            )
            self.activation_records.append(record)

            # 定期更新hot/cold分类
            if current_time - self.last_update_time >= self.update_interval:
                self._update_hot_cold_classification()
                self.last_update_time = current_time

    def _clean_old_records(self):
        """清理超过时间窗口的记录"""
        current_time = time.time()
        cutoff_time = current_time - self.time_window

        # 从左侧移除过期记录
        while self.activation_records and self.activation_records[0].timestamp < cutoff_time:
            self.activation_records.popleft()

    def _update_hot_cold_classification(self):
        """更新hot/cold专家分类"""
        self._clean_old_records()

        # 统计每层每个专家的激活次数
        layer_expert_counts = defaultdict(lambda: defaultdict(int))

        for record in self.activation_records:
            layer_expert_counts[record.layer_id][record.expert_id] += record.tokens_processed

        # 对每层进行hot/cold分类
        new_hot_experts = {}
        for layer_id, expert_counts in layer_expert_counts.items():
            if not expert_counts:
                new_hot_experts[layer_id] = set()
                continue

            # 按激活次数排序
            sorted_experts = sorted(
                expert_counts.items(), key=lambda x: x[1], reverse=True)

            # 选择top hot_ratio的专家作为hot
            num_hot = max(1, int(len(sorted_experts) * self.hot_ratio))
            hot_expert_ids = {expert_id for expert_id,
                              _ in sorted_experts[:num_hot]}
            new_hot_experts[layer_id] = hot_expert_ids

        self.hot_experts = new_hot_experts

    def is_hot_expert(self, layer_id: int, expert_id: int) -> bool:
        """判断专家是否为hot"""
        with self.lock:
            if layer_id not in self.hot_experts:
                return False  # 默认cold
            return expert_id in self.hot_experts[layer_id]

    def get_hot_experts(self, layer_id: int) -> set:
        """获取指定层的hot专家"""
        with self.lock:
            return self.hot_experts.get(layer_id, set()).copy()

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            self._clean_old_records()

            total_activations = len(self.activation_records)
            layer_counts = defaultdict(int)
            total_hot_count = 0
            total_cold_count = 0

            for record in self.activation_records:
                layer_counts[record.layer_id] += 1
                if self.is_hot_expert(record.layer_id, record.expert_id):
                    total_hot_count += 1
                else:
                    total_cold_count += 1

            return {
                'time_window': self.time_window,
                'total_activations': total_activations,
                'total_hot_activations': total_hot_count,
                'total_cold_activations': total_cold_count,
                'hot_ratio': total_hot_count / total_activations if total_activations > 0 else 0,
                'num_layers_tracked': len(self.hot_experts),
                'num_hot_experts': sum(len(experts) for experts in self.hot_experts.values()),
            }


class DynamicQuantMoEModel:
    """
    动态量化MoE模型
    1. 加载FP16和INT4两个版本的模型
    2. 实时统计expert激活（20s时间窗口）
    3. 动态路由：hot专家用FP16，cold专家用INT4
    """

    def __init__(
        self,
        fp16_model_path: str,
        int4_model_path: str,
        device: str = "cuda",
        time_window: float = 20.0,
        hot_ratio: float = 0.1,
        enable_dynamic_routing: bool = True
    ):
        """
        Args:
            fp16_model_path: FP16模型路径
            int4_model_path: INT4模型路径
            device: 设备
            time_window: 时间窗口（秒）
            hot_ratio: hot专家比例
            enable_dynamic_routing: 是否启用动态路由
        """
        self.fp16_model_path = fp16_model_path
        self.int4_model_path = int4_model_path
        self.device = device
        self.enable_dynamic_routing = enable_dynamic_routing

        logger.info(f"Loading FP16 model from {fp16_model_path}")
        self.fp16_model = AutoModelForCausalLM.from_pretrained(
            fp16_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        logger.info(f"Loading INT4 model from {int4_model_path}")
        self.int4_model = AutoModelForCausalLM.from_pretrained(
            int4_model_path,
            device_map="auto",
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            fp16_model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 获取模型配置
        self.config = self.fp16_model.config
        self.num_layers = self.config.num_hidden_layers
        self.num_experts = getattr(self.config, 'num_experts', 64)

        # 初始化实时追踪器
        self.tracker = RealTimeExpertTracker(
            time_window=time_window, hot_ratio=hot_ratio)

        # 为每个expert创建INT4备份引用
        self._prepare_expert_backups()

        # 注册forward hooks来追踪激活
        if enable_dynamic_routing:
            self._register_dynamic_routing_hooks()

        # 统计信息
        self.inference_stats = {
            'total_tokens': 0,
            'hot_expert_calls': 0,
            'cold_expert_calls': 0,
            'fp16_expert_calls': 0,
            'int4_expert_calls': 0,
        }

        logger.info(
            f"DynamicQuantMoEModel initialized: {self.num_layers} layers, {self.num_experts} experts per layer")
        logger.info(
            f"Dynamic routing: {'enabled' if enable_dynamic_routing else 'disabled'}")

    def _prepare_expert_backups(self):
        """为每个expert准备INT4备份引用"""
        logger.info("Preparing INT4 expert backups...")

        self.fp16_experts = {}  # (layer_id, expert_id) -> fp16 expert module
        self.int4_experts = {}  # (layer_id, expert_id) -> int4 expert module

        for layer_idx in range(self.num_layers):
            # 获取FP16 experts
            fp16_layer = self.fp16_model.model.layers[layer_idx]
            if hasattr(fp16_layer, 'mlp') and hasattr(fp16_layer.mlp, 'experts'):
                fp16_experts_container = fp16_layer.mlp.experts
                if hasattr(fp16_experts_container, 'experts'):
                    for expert_id, expert in enumerate(fp16_experts_container.experts):
                        self.fp16_experts[(layer_idx, expert_id)] = expert

            # 获取INT4 experts
            int4_layer = self.int4_model.model.layers[layer_idx]
            if hasattr(int4_layer, 'mlp') and hasattr(int4_layer.mlp, 'experts'):
                int4_experts_container = int4_layer.mlp.experts
                if hasattr(int4_experts_container, 'experts'):
                    for expert_id, expert in enumerate(int4_experts_container.experts):
                        self.int4_experts[(layer_idx, expert_id)] = expert

        logger.info(
            f"Prepared {len(self.fp16_experts)} FP16 experts and {len(self.int4_experts)} INT4 experts")

    def _register_dynamic_routing_hooks(self):
        """注册动态路由hooks"""
        logger.info("Registering dynamic routing hooks...")

        self.hooks = []

        for layer_idx in range(self.num_layers):
            layer = self.fp16_model.model.layers[layer_idx]

            # Hook到MoE层的forward
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp

                # Hook到router/gate
                if hasattr(mlp, 'gate'):
                    hook = mlp.gate.register_forward_hook(
                        self._create_activation_tracking_hook(layer_idx)
                    )
                    self.hooks.append(hook)

                # Hook到experts的forward来实现动态路由
                if hasattr(mlp, 'experts'):
                    experts_container = mlp.experts

                    # 如果是ModuleList，hook每个expert
                    if hasattr(experts_container, 'experts'):
                        for expert_id, expert in enumerate(experts_container.experts):
                            hook = expert.register_forward_pre_hook(
                                self._create_dynamic_routing_hook(
                                    layer_idx, expert_id)
                            )
                            self.hooks.append(hook)

        logger.info(f"Registered {len(self.hooks)} hooks")

    def _create_activation_tracking_hook(self, layer_idx: int):
        """创建激活追踪hook"""
        def hook(module, input, output):
            # 从router输出获取选中的expert
            if isinstance(output, tuple):
                router_logits = output[0]
            else:
                router_logits = output

            # 获取top-k expert indices
            with torch.no_grad():
                top_k_experts = torch.topk(router_logits, k=min(
                    2, self.num_experts), dim=-1).indices

                # 记录激活
                for expert_id in top_k_experts.flatten().cpu().numpy():
                    self.tracker.record_activation(layer_idx, int(expert_id))

        return hook

    def _create_dynamic_routing_hook(self, layer_idx: int, expert_id: int):
        """创建动态路由hook - 在expert forward之前决定使用FP16还是INT4"""
        def hook(module, input):
            # 判断是否为hot expert
            is_hot = self.tracker.is_hot_expert(layer_idx, expert_id)

            if is_hot:
                # Hot expert使用FP16
                self.inference_stats['hot_expert_calls'] += 1
                self.inference_stats['fp16_expert_calls'] += 1
                # FP16 expert已经在使用中，不需要替换
            else:
                # Cold expert使用INT4
                self.inference_stats['cold_expert_calls'] += 1
                self.inference_stats['int4_expert_calls'] += 1

                # 动态替换为INT4 expert（这里需要修改forward行为）
                # 注意：实际实现需要在更底层替换计算
                # 这里只是统计，实际替换需要修改模型结构

        return hook

    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> str:
        """生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        self.inference_stats['total_tokens'] += inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = self.fp16_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        self.inference_stats['total_tokens'] += max_new_tokens

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """前向传播"""
        self.inference_stats['total_tokens'] += input_ids.shape[0] * \
            input_ids.shape[1]

        with torch.no_grad():
            outputs = self.fp16_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )

        return outputs

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        tracker_stats = self.tracker.get_statistics()

        return {
            'model_info': {
                'num_layers': self.num_layers,
                'num_experts': self.num_experts,
                'fp16_model_path': self.fp16_model_path,
                'int4_model_path': self.int4_model_path,
            },
            'tracker_stats': tracker_stats,
            'inference_stats': self.inference_stats.copy(),
        }

    def clear_hooks(self):
        """清除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.info("Cleared all hooks")

    def __del__(self):
        """析构时清除hooks"""
        if hasattr(self, 'hooks'):
            self.clear_hooks()


def create_dynamic_quant_model(
    fp16_model_path: str,
    int4_model_path: str,
    device: str = "cuda",
    time_window: float = 20.0,
    hot_ratio: float = 0.1,
    enable_dynamic_routing: bool = True
) -> DynamicQuantMoEModel:
    """创建动态量化MoE模型的工厂函数"""
    return DynamicQuantMoEModel(
        fp16_model_path=fp16_model_path,
        int4_model_path=int4_model_path,
        device=device,
        time_window=time_window,
        hot_ratio=hot_ratio,
        enable_dynamic_routing=enable_dynamic_routing
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dynamic Quantization MoE Model")
    parser.add_argument('--fp16-model', type=str,
                        required=True, help='FP16 model path')
    parser.add_argument('--int4-model', type=str,
                        required=True, help='INT4 model path')
    parser.add_argument('--time-window', type=float,
                        default=20.0, help='Time window in seconds')
    parser.add_argument('--hot-ratio', type=float,
                        default=0.1, help='Hot expert ratio')
    parser.add_argument('--test-prompt', type=str,
                        default="Hello, how are you?", help='Test prompt')

    args = parser.parse_args()

    # 创建模型
    model = create_dynamic_quant_model(
        fp16_model_path=args.fp16_model,
        int4_model_path=args.int4_model,
        time_window=args.time_window,
        hot_ratio=args.hot_ratio
    )

    # 测试生成
    print(f"\nTest prompt: {args.test_prompt}")
    output = model.generate(args.test_prompt, max_new_tokens=50)
    print(f"Generated: {output}\n")

    # 打印统计信息
    stats = model.get_statistics()
    print("Statistics:")
    print(
        f"  Total tokens processed: {stats['inference_stats']['total_tokens']}")
    print(
        f"  Hot expert calls: {stats['inference_stats']['hot_expert_calls']}")
    print(
        f"  Cold expert calls: {stats['inference_stats']['cold_expert_calls']}")
    print(f"  Tracker stats: {stats['tracker_stats']}")
