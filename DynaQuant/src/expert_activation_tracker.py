#!/usr/bin/env python3
"""
专家激活统计器
用于跟踪和统计每个expert的激活次数
"""

import time
import json
import threading
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import torch
import numpy as np


@dataclass
class ExpertActivationInfo:
    """专家激活信息"""
    layer_id: int
    expert_id: int
    activation_count: int
    total_tokens: int
    activation_rate: float
    last_activation_time: float
    avg_activation_per_token: float


class ExpertActivationTracker:
    """专家激活统计器"""
    
    def __init__(self):
        """初始化统计器"""
        self.lock = threading.Lock()
        self.reset()
    
    def reset(self):
        """重置统计数据"""
        with self.lock:
            # 专家激活计数: {layer_id: {expert_id: count}}
            self.expert_activations = defaultdict(lambda: defaultdict(int))
            
            # 总token计数
            self.total_tokens = 0
            
            # 请求统计
            self.request_count = 0
            
            # 时间统计
            self.start_time = time.time()
            self.last_activation_times = defaultdict(lambda: defaultdict(float))
            
            # 详细激活记录
            self.activation_history = []
    
    def record_expert_activation(self, layer_id: int, expert_id: int, token_count: int = 1):
        """
        记录专家激活
        
        Args:
            layer_id: 层ID
            expert_id: 专家ID
            token_count: token数量
        """
        with self.lock:
            self.expert_activations[layer_id][expert_id] += token_count
            self.total_tokens += token_count
            self.last_activation_times[layer_id][expert_id] = time.time()
            
            # 记录详细历史
            self.activation_history.append({
                'timestamp': time.time(),
                'layer_id': layer_id,
                'expert_id': expert_id,
                'token_count': token_count
            })
    
    def record_request(self, token_count: int):
        """记录请求"""
        with self.lock:
            self.request_count += 1
            self.total_tokens += token_count
    
    def get_expert_activation_info(self, layer_id: int, expert_id: int) -> ExpertActivationInfo:
        """获取专家激活信息"""
        with self.lock:
            activation_count = self.expert_activations[layer_id][expert_id]
            last_time = self.last_activation_times[layer_id][expert_id]
            
            activation_rate = 0.0
            avg_activation_per_token = 0.0
            
            if self.total_tokens > 0:
                activation_rate = activation_count / self.total_tokens
                avg_activation_per_token = activation_count / max(self.request_count, 1)
            
            return ExpertActivationInfo(
                layer_id=layer_id,
                expert_id=expert_id,
                activation_count=activation_count,
                total_tokens=self.total_tokens,
                activation_rate=activation_rate,
                last_activation_time=last_time,
                avg_activation_per_token=avg_activation_per_token
            )
    
    def get_all_expert_info(self) -> Dict[str, Any]:
        """获取所有专家信息"""
        with self.lock:
            expert_info = {}
            
            for layer_id in self.expert_activations:
                expert_info[f"layer_{layer_id}"] = {}
                for expert_id in self.expert_activations[layer_id]:
                    info = self.get_expert_activation_info(layer_id, expert_id)
                    expert_info[f"layer_{layer_id}"][f"expert_{expert_id}"] = asdict(info)
            
            return expert_info
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """获取摘要统计"""
        with self.lock:
            total_activations = sum(
                sum(expert_counts.values()) 
                for expert_counts in self.expert_activations.values()
            )
            
            total_experts = sum(
                len(expert_counts) 
                for expert_counts in self.expert_activations.values()
            )
            
            total_layers = len(self.expert_activations)
            
            runtime = time.time() - self.start_time
            
            return {
                'total_activations': total_activations,
                'total_tokens': self.total_tokens,
                'total_requests': self.request_count,
                'total_layers': total_layers,
                'total_experts': total_experts,
                'runtime_seconds': runtime,
                'activations_per_second': total_activations / max(runtime, 1),
                'tokens_per_second': self.total_tokens / max(runtime, 1),
                'requests_per_second': self.request_count / max(runtime, 1)
            }
    
    def get_top_experts(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """获取激活次数最多的专家"""
        with self.lock:
            all_experts = []
            
            for layer_id in self.expert_activations:
                for expert_id in self.expert_activations[layer_id]:
                    info = self.get_expert_activation_info(layer_id, expert_id)
                    all_experts.append(asdict(info))
            
            # 按激活次数排序
            all_experts.sort(key=lambda x: x['activation_count'], reverse=True)
            
            return all_experts[:top_k]
    
    def get_layer_stats(self) -> Dict[str, Any]:
        """获取每层统计"""
        with self.lock:
            layer_stats = {}
            
            for layer_id in self.expert_activations:
                layer_activations = sum(self.expert_activations[layer_id].values())
                layer_experts = len(self.expert_activations[layer_id])
                
                layer_stats[f"layer_{layer_id}"] = {
                    'total_activations': layer_activations,
                    'expert_count': layer_experts,
                    'avg_activations_per_expert': layer_activations / max(layer_experts, 1),
                    'activation_rate': layer_activations / max(self.total_tokens, 1)
                }
            
            return layer_stats
    
    def export_stats(self, file_path: str):
        """导出统计数据到文件"""
        with self.lock:
            stats = {
                'summary': self.get_summary_stats(),
                'layer_stats': self.get_layer_stats(),
                'expert_info': self.get_all_expert_info(),
                'top_experts': self.get_top_experts(20),
                'export_time': time.time()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
    
    def get_recent_activations(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """获取最近的激活记录"""
        with self.lock:
            cutoff_time = time.time() - (minutes * 60)
            recent = [
                record for record in self.activation_history
                if record['timestamp'] >= cutoff_time
            ]
            return recent


# 全局统计器实例
global_tracker = ExpertActivationTracker()


def get_global_tracker() -> ExpertActivationTracker:
    """获取全局统计器实例"""
    return global_tracker


def reset_global_tracker():
    """重置全局统计器"""
    global_tracker.reset()


def record_expert_activation(layer_id: int, expert_id: int, token_count: int = 1):
    """记录专家激活（全局函数）"""
    global_tracker.record_expert_activation(layer_id, expert_id, token_count)


def record_request(token_count: int):
    """记录请求（全局函数）"""
    global_tracker.record_request(token_count)
