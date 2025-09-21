#!/usr/bin/env python3
"""
动态量化管理器
根据expert激活score动态调整量化精度
"""

import os
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 全局量化管理器
_global_quantization_manager: Optional['DynamicQuantizationManager'] = None


@dataclass
class QuantizationThresholds:
    """量化阈值配置"""
    high_threshold: float = 0.5      # score > 0.5 使用 fp16
    medium_threshold: float = 0.1    # 0.1 < score <= 0.5 使用 fp8
    # score <= 0.1 使用 gptq-int4


@dataclass
class ModelPaths:
    """不同精度模型路径配置"""
    fp16_path: str = ""
    fp8_path: str = ""
    gptq_int4_path: str = ""


class ExpertQuantizationState:
    """单个expert的量化状态"""
    
    def __init__(self, expert_id: int, layer_id: int):
        self.expert_id = expert_id
        self.layer_id = layer_id
        self.current_precision = "fp8"  # 默认精度
        self.target_precision = "fp8"
        self.last_score = 0.0
        self.activation_count = 0
        self.last_update_time = time.time()
        self.needs_reload = False


class DynamicQuantizationManager:
    """动态量化管理器"""
    
    def __init__(self, 
                 thresholds: QuantizationThresholds,
                 model_paths: ModelPaths,
                 time_window: int = 300):
        self.thresholds = thresholds
        self.model_paths = model_paths
        self.time_window = time_window
        
        # Expert量化状态
        self.expert_states: Dict[Tuple[int, int], ExpertQuantizationState] = {}  # (layer_id, expert_id) -> state
        self.quantization_lock = threading.RLock()
        
        # 统计信息
        self.quantization_stats = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'precision_distribution': defaultdict(int)
        }
        
        logger.info(f"DynamicQuantizationManager initialized with thresholds: {thresholds}")
    
    def update_expert_score(self, layer_id: int, expert_id: int, score: float):
        """更新expert的激活分数"""
        with self.quantization_lock:
            key = (layer_id, expert_id)
            
            if key not in self.expert_states:
                self.expert_states[key] = ExpertQuantizationState(expert_id, layer_id)
            
            self.expert_states[key].last_score = score
            self.expert_states[key].activation_count += 1
            self.expert_states[key].last_update_time = time.time()
            
            # 确定目标精度
            target_precision = self._determine_precision(score)
            
            if target_precision != self.expert_states[key].current_precision:
                self.expert_states[key].target_precision = target_precision
                self.expert_states[key].needs_reload = True
    
    def _determine_precision(self, score: float) -> str:
        """根据分数确定量化精度"""
        if score > self.thresholds.high_threshold:
            return "fp16"
        elif score > self.thresholds.medium_threshold:
            return "fp8"
        else:
            return "gptq_int4"
    
    def get_experts_needing_reload(self) -> List[Tuple[int, int, str]]:
        """获取需要重新加载的expert列表"""
        with self.quantization_lock:
            reload_list = []
            for (layer_id, expert_id), state in self.expert_states.items():
                if state.needs_reload:
                    reload_list.append((layer_id, expert_id, state.target_precision))
            return reload_list
    
    def mark_expert_reloaded(self, layer_id: int, expert_id: int):
        """标记expert已重新加载"""
        with self.quantization_lock:
            key = (layer_id, expert_id)
            if key in self.expert_states:
                state = self.expert_states[key]
                state.current_precision = state.target_precision
                state.needs_reload = False
                self.quantization_stats['precision_distribution'][state.current_precision] += 1
                self.quantization_stats['successful_updates'] += 1
    
    def mark_expert_reload_failed(self, layer_id: int, expert_id: int):
        """标记expert重新加载失败"""
        with self.quantization_lock:
            key = (layer_id, expert_id)
            if key in self.expert_states:
                self.quantization_stats['failed_updates'] += 1
    
    def get_model_path(self, precision: str) -> str:
        """获取指定精度的模型路径"""
        if precision == "fp16":
            return self.model_paths.fp16_path
        elif precision == "fp8":
            return self.model_paths.fp8_path
        elif precision == "gptq_int4":
            return self.model_paths.gptq_int4_path
        else:
            raise ValueError(f"Unknown precision: {precision}")
    
    def get_expert_precision(self, layer_id: int, expert_id: int) -> str:
        """获取expert的当前精度"""
        with self.quantization_lock:
            key = (layer_id, expert_id)
            if key in self.expert_states:
                return self.expert_states[key].current_precision
            return "fp16"  # 默认精度
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        """获取量化统计信息"""
        with self.quantization_lock:
            stats = self.quantization_stats.copy()
            stats['total_experts'] = len(self.expert_states)
            stats['experts_needing_reload'] = len([s for s in self.expert_states.values() if s.needs_reload])
            return stats
    
    def set_thresholds(self, high_threshold: float = None, medium_threshold: float = None):
        """设置量化阈值"""
        if high_threshold is not None:
            self.thresholds.high_threshold = high_threshold
        if medium_threshold is not None:
            self.thresholds.medium_threshold = medium_threshold
        logger.info(f"Updated thresholds: {self.thresholds}")
    
    def export_quantization_report(self) -> Dict[str, Any]:
        """导出量化报告"""
        with self.quantization_lock:
            report = {
                'thresholds': {
                    'high_threshold': self.thresholds.high_threshold,
                    'medium_threshold': self.thresholds.medium_threshold
                },
                'statistics': self.get_quantization_stats(),
                'expert_precisions': {}
            }
            
            # 按层分组expert精度
            layer_precisions = defaultdict(lambda: defaultdict(int))
            for (layer_id, expert_id), state in self.expert_states.items():
                layer_precisions[layer_id][state.current_precision] += 1
            
            report['layer_precisions'] = dict(layer_precisions)
            return report


def init_global_quantization_manager(
    high_threshold: float = 0.5,
    medium_threshold: float = 0.1,
    fp16_path: str = "",
    fp8_path: str = "",
    gptq_int4_path: str = "",
    time_window: int = 300
) -> DynamicQuantizationManager:
    """初始化全局量化管理器"""
    global _global_quantization_manager
    
    thresholds = QuantizationThresholds(
        high_threshold=high_threshold,
        medium_threshold=medium_threshold
    )
    
    model_paths = ModelPaths(
        fp16_path=fp16_path,
        fp8_path=fp8_path,
        gptq_int4_path=gptq_int4_path
    )
    
    _global_quantization_manager = DynamicQuantizationManager(
        thresholds=thresholds,
        model_paths=model_paths,
        time_window=time_window
    )
    
    logger.info("Global quantization manager initialized")
    return _global_quantization_manager


def get_global_quantization_manager() -> Optional[DynamicQuantizationManager]:
    """获取全局量化管理器"""
    return _global_quantization_manager


def set_global_quantization_manager(manager: DynamicQuantizationManager):
    """设置全局量化管理器"""
    global _global_quantization_manager
    _global_quantization_manager = manager
