#!/usr/bin/env python3
"""
增强版专家激活跟踪器
支持hot-cold分数计算和实时跟踪
"""

import torch
import torch.nn as nn
import logging
import time
import json
import threading
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExpertActivationRecord:
    """专家激活记录"""
    timestamp: float
    layer_id: int
    expert_id: int
    tokens_processed: int
    activation_strength: float = 1.0


@dataclass
class ExpertHotColdStats:
    """专家hot-cold统计"""
    layer_id: int
    expert_id: int
    total_activations: int = 0
    total_tokens: int = 0
    last_activation_time: float = 0.0
    activation_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    hot_cold_score: float = 0.0
    
    def update_score(self, decay_factor: float = 0.95):
        """更新hot-cold分数"""
        if not self.activation_history:
            self.hot_cold_score = 0.0
            return
        
        current_time = time.time()
        recent_activations = 0
        total_weight = 0.0
        
        for record in self.activation_history:
            time_diff = current_time - record.timestamp
            weight = np.exp(-time_diff / decay_factor)
            recent_activations += record.activation_strength * weight
            total_weight += weight
        
        if total_weight > 0:
            self.hot_cold_score = min(1.0, recent_activations / total_weight)
        else:
            self.hot_cold_score = 0.0


class EnhancedExpertTracker:
    """增强版专家跟踪器"""
    
    def __init__(self, max_history: int = 10000, decay_factor: float = 0.95):
        self.expert_stats: Dict[Tuple[int, int], ExpertHotColdStats] = {}
        self.activation_history: deque = deque(maxlen=max_history)
        self.request_history: deque = deque(maxlen=max_history)
        self.lock = threading.RLock()
        self.decay_factor = decay_factor
    
    def record_expert_activation(self, layer_id: int, expert_id: int, 
                               tokens_processed: int = 1, activation_strength: float = 1.0):
        """记录专家激活"""
        with self.lock:
            key = (layer_id, expert_id)
            if key not in self.expert_stats:
                self.expert_stats[key] = ExpertHotColdStats(layer_id, expert_id)
            
            stats = self.expert_stats[key]
            stats.total_activations += 1
            stats.total_tokens += tokens_processed
            stats.last_activation_time = time.time()
            
            record = ExpertActivationRecord(
                timestamp=time.time(),
                layer_id=layer_id,
                expert_id=expert_id,
                tokens_processed=tokens_processed,
                activation_strength=activation_strength
            )
            
            stats.activation_history.append(record)
            self.activation_history.append(record)
            stats.update_score(self.decay_factor)
    
    def record_request(self, request_data: Dict[str, Any]):
        """记录请求"""
        with self.lock:
            self.request_history.append({
                'timestamp': time.time(),
                'data': request_data
            })
    
    def get_expert_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取专家统计数据，返回格式化的字典"""
        with self.lock:
            stats = {}
            for key, expert_stat in self.expert_stats.items():
                layer_id, expert_id = key
                stats[f"layer_{layer_id}_expert_{expert_id}"] = {
                    'layer_id': layer_id,
                    'expert_id': expert_id,
                    'activation_count': expert_stat.total_activations,
                    'total_tokens_processed': expert_stat.total_tokens,
                    'hot_cold_score': round(expert_stat.hot_cold_score, 4),
                    'last_activation_time': expert_stat.last_activation_time
                }
            return stats
    
    def get_expert_stats_by_layer(self) -> Dict[str, Dict[str, Any]]:
        """获取按层分组的专家统计数据，计算正确的hot_cold_score"""
        with self.lock:
            # 按层分组
            layer_experts = {}
            for key, expert_stat in self.expert_stats.items():
                layer_id, expert_id = key
                if layer_id not in layer_experts:
                    layer_experts[layer_id] = []
                layer_experts[layer_id].append({
                    'expert_id': expert_id,
                    'activation_count': expert_stat.total_activations,
                    'total_tokens': expert_stat.total_tokens,
                    'last_activation_time': expert_stat.last_activation_time
                })
            
            # 计算每层的hot_cold_score
            result = {}
            for layer_id, experts in layer_experts.items():
                if not experts:
                    continue
                
                # 找到该层激活次数最多的expert
                max_activation_count = max(expert['activation_count'] for expert in experts)
                
                layer_data = {
                    'experts': {}
                }
                
                for expert in experts:
                    # 计算hot_cold_score: activation_count / max_activation_count
                    if max_activation_count == 0:
                        hot_cold_score = 0.0
                    else:
                        hot_cold_score = expert['activation_count'] / max_activation_count
                    
                    layer_data['experts'][str(expert['expert_id'])] = {
                        'activation_count': expert['activation_count'],
                        'total_tokens': expert['total_tokens'],
                        'hot_cold_score': round(hot_cold_score, 4)
                    }
                
                result[f'layer_{layer_id}'] = layer_data
            
            return result
    
    def get_top_experts(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """获取激活次数最多的前N个专家"""
        with self.lock:
            if not self.expert_stats:
                return []
            
            # 按激活次数排序
            sorted_experts = sorted(
                self.expert_stats.items(),
                key=lambda x: x[1].total_activations,
                reverse=True
            )
            
            top_experts = []
            for key, stats in sorted_experts[:top_n]:
                layer_id, expert_id = key
                top_experts.append({
                    'layer_id': layer_id,
                    'expert_id': expert_id,
                    'activation_count': stats.total_activations,
                    'total_tokens': stats.total_tokens,
                    'hot_cold_score': round(stats.hot_cold_score, 4)
                })
            
            return top_experts
    
    def get_expert_hot_cold_scores(self) -> Dict[str, Dict]:
        """获取专家hot-cold分数"""
        with self.lock:
            scores = {}
            for key, stats in self.expert_stats.items():
                scores[f"layer_{stats.layer_id}_expert_{stats.expert_id}"] = {
                    'layer_id': stats.layer_id,
                    'expert_id': stats.expert_id,
                    'hot_cold_score': round(stats.hot_cold_score, 4),
                    'total_activations': stats.total_activations,
                    'total_tokens': stats.total_tokens
                }
            return scores
    
    def export_hot_cold_report(self, file_path: str):
        """导出hot-cold报告"""
        with self.lock:
            report = {
                'export_time': time.time(),
                'expert_scores': self.get_expert_hot_cold_scores()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Hot-cold report exported to {file_path}")


# 全局实例
_global_expert_tracker: Optional[EnhancedExpertTracker] = None


def get_global_expert_tracker() -> Optional[EnhancedExpertTracker]:
    """获取全局专家跟踪器"""
    global _global_expert_tracker
    return _global_expert_tracker


def init_global_expert_tracker() -> EnhancedExpertTracker:
    """初始化全局专家跟踪器"""
    global _global_expert_tracker
    if _global_expert_tracker is None:
        _global_expert_tracker = EnhancedExpertTracker()
        logger.info("Global expert tracker initialized")
    return _global_expert_tracker


def record_expert_activation(layer_id: int, expert_id: int, 
                           tokens_processed: int = 1, activation_strength: float = 1.0):
    """记录专家激活"""
    tracker = get_global_expert_tracker()
    if tracker:
        tracker.record_expert_activation(layer_id, expert_id, tokens_processed, activation_strength)
