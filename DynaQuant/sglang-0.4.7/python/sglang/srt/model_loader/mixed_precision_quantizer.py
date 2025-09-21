#!/usr/bin/env python3
"""
混合精度量化器
基于MxMoE项目的混合精度量化实现
支持基于expert激活热度的动态量化策略
"""

import os
import torch
import torch.nn as nn
import logging
import json
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """量化配置"""
    weight_bits: int = 8
    activation_bits: int = 8
    group_size: int = 128
    symmetric: bool = True
    expert_id: Optional[int] = None
    layer_id: Optional[int] = None


@dataclass
class ExpertQuantizationProfile:
    """专家量化配置档案"""
    expert_id: int
    layer_id: int
    activation_frequency: float = 0.0
    hot_cold_score: float = 0.0
    quantization_config: QuantizationConfig = field(default_factory=QuantizationConfig)
    performance_impact: float = 0.0
    accuracy_impact: float = 0.0


class MixedPrecisionQuantizer:
    """混合精度量化器 - 基于MxMoE的实现"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化混合精度量化器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.expert_profiles: Dict[Tuple[int, int], ExpertQuantizationProfile] = {}
        self.quantization_cache: Dict[str, torch.Tensor] = {}
        self.lock = threading.RLock()
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            self._load_config()
        else:
            self._init_default_config()
        
        logger.info("Mixed precision quantizer initialized")
    
    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 加载专家量化配置
            expert_configs = config.get('expert_quantization', {})
            for key, profile_data in expert_configs.items():
                layer_id, expert_id = map(int, key.split('_'))
                profile = ExpertQuantizationProfile(
                    expert_id=expert_id,
                    layer_id=layer_id,
                    activation_frequency=profile_data.get('activation_frequency', 0.0),
                    hot_cold_score=profile_data.get('hot_cold_score', 0.0),
                    quantization_config=QuantizationConfig(**profile_data.get('quantization_config', {}))
                )
                self.expert_profiles[(layer_id, expert_id)] = profile
            
            logger.info(f"Loaded quantization config for {len(self.expert_profiles)} experts")
            
        except Exception as e:
            logger.error(f"Error loading quantization config: {e}")
            self._init_default_config()
    
    def _init_default_config(self):
        """初始化默认配置"""
        # 默认配置：所有expert使用相同的量化策略
        default_config = QuantizationConfig(
            weight_bits=8,
            activation_bits=8,
            group_size=128,
            symmetric=True
        )
        
        # 这里可以根据需要添加更多默认配置
        logger.info("Using default quantization configuration")
    
    def update_expert_profile(self, layer_id: int, expert_id: int, 
                            activation_frequency: float, hot_cold_score: float):
        """更新专家量化配置档案"""
        with self.lock:
            key = (layer_id, expert_id)
            
            if key not in self.expert_profiles:
                self.expert_profiles[key] = ExpertQuantizationProfile(
                    expert_id=expert_id,
                    layer_id=layer_id
                )
            
            profile = self.expert_profiles[key]
            profile.activation_frequency = activation_frequency
            profile.hot_cold_score = hot_cold_score
            
            # 根据激活热度调整量化策略
            self._update_quantization_strategy(profile)
    
    def _update_quantization_strategy(self, profile: ExpertQuantizationProfile):
        """根据激活热度更新量化策略"""
        # 基于MxMoE的策略：hot experts使用更高精度，cold experts使用更低精度
        if profile.hot_cold_score > 0.8:  # Hot expert
            profile.quantization_config.weight_bits = 8
            profile.quantization_config.activation_bits = 8
            profile.quantization_config.group_size = 128
        elif profile.hot_cold_score > 0.5:  # Medium expert
            profile.quantization_config.weight_bits = 6
            profile.quantization_config.activation_bits = 6
            profile.quantization_config.group_size = 64
        else:  # Cold expert
            profile.quantization_config.weight_bits = 4
            profile.quantization_config.activation_bits = 4
            profile.quantization_config.group_size = 32
        
        logger.debug(f"Updated quantization strategy for expert {profile.expert_id} in layer {profile.layer_id}: "
                    f"weight_bits={profile.quantization_config.weight_bits}, "
                    f"activation_bits={profile.quantization_config.activation_bits}")
    
    def quantize_expert_weight(self, weight: torch.Tensor, layer_id: int, expert_id: int) -> torch.Tensor:
        """量化专家权重"""
        key = (layer_id, expert_id)
        
        if key not in self.expert_profiles:
            # 如果没有配置，使用默认量化
            return self._default_quantize(weight)
        
        profile = self.expert_profiles[key]
        config = profile.quantization_config
        
        # 生成缓存键
        cache_key = f"{layer_id}_{expert_id}_{hash(weight.data_ptr())}"
        
        if cache_key in self.quantization_cache:
            return self.quantization_cache[cache_key]
        
        # 执行量化
        quantized_weight = self._quantize_weight(weight, config)
        
        # 缓存结果
        self.quantization_cache[cache_key] = quantized_weight
        
        return quantized_weight
    
    def _quantize_weight(self, weight: torch.Tensor, config: QuantizationConfig) -> torch.Tensor:
        """执行权重量化"""
        if config.weight_bits >= 8:
            # 8位或更高精度，使用简单的缩放量化
            return self._scale_quantize(weight, config.weight_bits)
        else:
            # 低精度量化，使用更复杂的量化方法
            return self._low_precision_quantize(weight, config)
    
    def _scale_quantize(self, weight: torch.Tensor, bits: int) -> torch.Tensor:
        """缩放量化"""
        # 计算量化参数
        scale = 2 ** (bits - 1) - 1
        weight_min = weight.min()
        weight_max = weight.max()
        
        # 对称量化
        if weight_min < 0 and weight_max > 0:
            max_abs = max(abs(weight_min), abs(weight_max))
            weight_scaled = weight / max_abs * scale
        else:
            weight_scaled = (weight - weight_min) / (weight_max - weight_min) * scale
        
        # 量化到整数
        weight_quantized = torch.round(weight_scaled).clamp(-scale, scale)
        
        # 反量化
        if weight_min < 0 and weight_max > 0:
            weight_dequantized = weight_quantized / scale * max_abs
        else:
            weight_dequantized = weight_quantized / scale * (weight_max - weight_min) + weight_min
        
        return weight_dequantized
    
    def _low_precision_quantize(self, weight: torch.Tensor, config: QuantizationConfig) -> torch.Tensor:
        """低精度量化"""
        # 使用分组量化
        group_size = config.group_size
        weight_shape = weight.shape
        
        if len(weight_shape) == 2:  # 线性层
            # 重塑为分组形式
            weight_flat = weight.view(-1, group_size)
            
            # 对每个组进行量化
            quantized_groups = []
            for i in range(weight_flat.shape[0]):
                group = weight_flat[i]
                quantized_group = self._quantize_group(group, config.weight_bits)
                quantized_groups.append(quantized_group)
            
            quantized_weight = torch.stack(quantized_groups).view(weight_shape)
        else:
            # 其他形状，使用简单量化
            quantized_weight = self._scale_quantize(weight, config.weight_bits)
        
        return quantized_weight
    
    def _quantize_group(self, group: torch.Tensor, bits: int) -> torch.Tensor:
        """量化一个组"""
        # 计算组的量化参数
        scale = 2 ** (bits - 1) - 1
        group_min = group.min()
        group_max = group.max()
        
        # 对称量化
        if group_min < 0 and group_max > 0:
            max_abs = max(abs(group_min), abs(group_max))
            group_scaled = group / max_abs * scale
        else:
            group_scaled = (group - group_min) / (group_max - group_min) * scale
        
        # 量化到整数
        group_quantized = torch.round(group_scaled).clamp(-scale, scale)
        
        # 反量化
        if group_min < 0 and group_max > 0:
            group_dequantized = group_quantized / scale * max_abs
        else:
            group_dequantized = group_quantized / scale * (group_max - group_min) + group_min
        
        return group_dequantized
    
    def _default_quantize(self, weight: torch.Tensor) -> torch.Tensor:
        """默认量化（8位）"""
        return self._scale_quantize(weight, 8)
    
    def get_expert_quantization_config(self, layer_id: int, expert_id: int) -> Optional[QuantizationConfig]:
        """获取专家量化配置"""
        key = (layer_id, expert_id)
        if key in self.expert_profiles:
            return self.expert_profiles[key].quantization_config
        return None
    
    def export_quantization_config(self, file_path: str):
        """导出量化配置"""
        with self.lock:
            config_data = {
                'expert_quantization': {},
                'export_time': time.time(),
                'total_experts': len(self.expert_profiles)
            }
            
            for (layer_id, expert_id), profile in self.expert_profiles.items():
                key = f"{layer_id}_{expert_id}"
                config_data['expert_quantization'][key] = {
                    'expert_id': profile.expert_id,
                    'layer_id': profile.layer_id,
                    'activation_frequency': profile.activation_frequency,
                    'hot_cold_score': profile.hot_cold_score,
                    'quantization_config': {
                        'weight_bits': profile.quantization_config.weight_bits,
                        'activation_bits': profile.quantization_config.activation_bits,
                        'group_size': profile.quantization_config.group_size,
                        'symmetric': profile.quantization_config.symmetric
                    },
                    'performance_impact': profile.performance_impact,
                    'accuracy_impact': profile.accuracy_impact
                }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Quantization config exported to {file_path}")
    
    def load_quantization_config(self, file_path: str):
        """加载量化配置"""
        if os.path.exists(file_path):
            self.config_path = file_path
            self._load_config()
            logger.info(f"Quantization config loaded from {file_path}")
        else:
            logger.warning(f"Quantization config file not found: {file_path}")
    
    def get_quantization_summary(self) -> Dict[str, Any]:
        """获取量化摘要"""
        with self.lock:
            summary = {
                'total_experts': len(self.expert_profiles),
                'quantization_distribution': defaultdict(int),
                'hot_experts': 0,
                'cold_experts': 0,
                'medium_experts': 0
            }
            
            for profile in self.expert_profiles.values():
                # 统计量化位宽分布
                weight_bits = profile.quantization_config.weight_bits
                summary['quantization_distribution'][f'weight_{weight_bits}bit'] += 1
                
                # 统计热度分布
                if profile.hot_cold_score > 0.8:
                    summary['hot_experts'] += 1
                elif profile.hot_cold_score > 0.5:
                    summary['medium_experts'] += 1
                else:
                    summary['cold_experts'] += 1
            
            return summary


class ExpertQuantizationManager:
    """专家量化管理器 - 集成expert tracking和quantization"""
    
    def __init__(self, quantizer: MixedPrecisionQuantizer):
        """
        初始化专家量化管理器
        
        Args:
            quantizer: 混合精度量化器
        """
        self.quantizer = quantizer
        self.expert_tracker = None
        self.quantization_enabled = False
        
        logger.info("Expert quantization manager initialized")
    
    def set_expert_tracker(self, tracker):
        """设置专家跟踪器"""
        self.expert_tracker = tracker
        logger.info("Expert tracker set for quantization manager")
    
    def enable_quantization(self, enable: bool = True):
        """启用或禁用量化"""
        self.quantization_enabled = enable
        logger.info(f"Expert quantization {'enabled' if enable else 'disabled'}")
    
    def update_expert_profiles_from_tracker(self):
        """从专家跟踪器更新量化配置档案"""
        if not self.expert_tracker or not self.quantization_enabled:
            return
        
        try:
            # 获取专家统计信息
            expert_stats = self.expert_tracker.get_expert_stats()
            
            for layer_key, layer_data in expert_stats.items():
                if layer_key.startswith('layer_'):
                    layer_id = int(layer_key.split('_')[1])
                    
                    for expert_key, expert_data in layer_data.items():
                        if expert_key.startswith('expert_'):
                            expert_id = int(expert_key.split('_')[1])
                            
                            # 获取激活频率和热度分数
                            activation_frequency = expert_data.get('activity', 0)
                            hot_cold_score = expert_data.get('hot_cold_score', 0.0)
                            
                            # 更新量化配置档案
                            self.quantizer.update_expert_profile(
                                layer_id, expert_id, activation_frequency, hot_cold_score
                            )
            
            logger.info("Expert quantization profiles updated from tracker")
            
        except Exception as e:
            logger.error(f"Error updating expert profiles from tracker: {e}")
    
    def quantize_expert_weights(self, model: nn.Module) -> Dict[str, Any]:
        """量化模型中的专家权重"""
        if not self.quantization_enabled:
            logger.info("Quantization disabled, skipping expert weight quantization")
            return {'quantized': 0, 'skipped': 0, 'errors': 0}
        
        stats = {'quantized': 0, 'skipped': 0, 'errors': 0, 'details': []}
        
        try:
            for name, module in model.named_modules():
                if 'expert' in name.lower() and hasattr(module, 'weight'):
                    # 提取层ID和专家ID
                    layer_id, expert_id = self._extract_expert_info(name)
                    
                    if layer_id is not None and expert_id is not None:
                        try:
                            # 量化权重
                            original_weight = module.weight.data.clone()
                            quantized_weight = self.quantizer.quantize_expert_weight(
                                original_weight, layer_id, expert_id
                            )
                            
                            # 更新模块权重
                            module.weight.data = quantized_weight
                            
                            stats['quantized'] += 1
                            stats['details'].append({
                                'name': name,
                                'layer_id': layer_id,
                                'expert_id': expert_id,
                                'original_shape': list(original_weight.shape),
                                'quantized_shape': list(quantized_weight.shape)
                            })
                            
                            logger.debug(f"Quantized expert weight: {name}")
                            
                        except Exception as e:
                            logger.error(f"Error quantizing expert weight {name}: {e}")
                            stats['errors'] += 1
                            stats['details'].append({
                                'name': name,
                                'error': str(e)
                            })
                    else:
                        stats['skipped'] += 1
                        stats['details'].append({
                            'name': name,
                            'reason': 'Could not extract expert info'
                        })
            
            logger.info(f"Expert weight quantization completed: {stats['quantized']} quantized, "
                       f"{stats['skipped']} skipped, {stats['errors']} errors")
            
        except Exception as e:
            logger.error(f"Error during expert weight quantization: {e}")
            stats['errors'] += 1
        
        return stats
    
    def _extract_expert_info(self, module_name: str) -> Tuple[Optional[int], Optional[int]]:
        """从模块名称提取层ID和专家ID"""
        try:
            # 假设模块名称格式为: model.layers.{layer_id}.mlp.experts.{expert_id}
            parts = module_name.split('.')
            
            layer_id = None
            expert_id = None
            
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        layer_id = int(parts[i + 1])
                    except ValueError:
                        pass
                elif part == 'experts' and i + 1 < len(parts):
                    try:
                        expert_id = int(parts[i + 1])
                    except ValueError:
                        pass
            
            return layer_id, expert_id
            
        except Exception as e:
            logger.debug(f"Could not extract expert info from {module_name}: {e}")
            return None, None
    
    def export_quantization_report(self, file_path: str):
        """导出量化报告"""
        try:
            # 更新专家配置档案
            self.update_expert_profiles_from_tracker()
            
            # 导出量化配置
            self.quantizer.export_quantization_config(file_path)
            
            # 添加摘要信息
            summary = self.quantizer.get_quantization_summary()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            config_data['summary'] = summary
            config_data['quantization_enabled'] = self.quantization_enabled
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Quantization report exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting quantization report: {e}")


# 全局量化器实例
_global_quantizer: Optional[MixedPrecisionQuantizer] = None
_global_quantization_manager: Optional[ExpertQuantizationManager] = None


def get_global_quantizer() -> Optional[MixedPrecisionQuantizer]:
    """获取全局量化器"""
    return _global_quantizer


def set_global_quantizer(quantizer: MixedPrecisionQuantizer):
    """设置全局量化器"""
    global _global_quantizer
    _global_quantizer = quantizer


def get_global_quantization_manager() -> Optional[ExpertQuantizationManager]:
    """获取全局量化管理器"""
    return _global_quantization_manager


def set_global_quantization_manager(manager: ExpertQuantizationManager):
    """设置全局量化管理器"""
    global _global_quantization_manager
    _global_quantization_manager = manager


def init_global_quantization_system(config_path: Optional[str] = None) -> ExpertQuantizationManager:
    """初始化全局量化系统"""
    global _global_quantizer, _global_quantization_manager
    
    if _global_quantizer is None:
        _global_quantizer = MixedPrecisionQuantizer(config_path)
    
    if _global_quantization_manager is None:
        _global_quantization_manager = ExpertQuantizationManager(_global_quantizer)
    
    logger.info("Global quantization system initialized")
    return _global_quantization_manager

