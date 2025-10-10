#!/usr/bin/env python3
"""
量化校准器
基于激活分布和专家激活频率优化量化参数
"""

import os
import torch
import torch.nn as nn
import logging
import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class CalibrationStats:
    """校准统计信息"""
    activation_mean: float = 0.0
    activation_std: float = 1.0
    activation_min: float = 0.0
    activation_max: float = 1.0
    weight_mean: float = 0.0
    weight_std: float = 1.0
    weight_min: float = 0.0
    weight_max: float = 1.0
    expert_activation_frequency: float = 0.0
    sample_count: int = 0


@dataclass
class CalibratedQuantizationParams:
    """校准后的量化参数"""
    scale: float = 1.0
    zero_point: float = 0.0
    symmetric: bool = True
    optimal_bits: int = 8
    calibration_confidence: float = 0.0


class QuantizationCalibrator:
    """量化校准器 - 基于激活分布优化量化参数"""
    
    def __init__(self, calibration_data_path: Optional[str] = None):
        """
        初始化量化校准器
        
        Args:
            calibration_data_path: 校准数据路径
        """
        self.calibration_data_path = calibration_data_path
        self.calibration_stats: Dict[str, CalibrationStats] = {}
        self.calibrated_params: Dict[str, CalibratedQuantizationParams] = {}
        self.lock = threading.RLock()
        
        # 加载校准数据
        if calibration_data_path and os.path.exists(calibration_data_path):
            self._load_calibration_data()
        
        logger.info("Quantization calibrator initialized")
    
    def _load_calibration_data(self):
        """加载校准数据"""
        try:
            with open(self.calibration_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for layer_name, stats_data in data.get('calibration_stats', {}).items():
                stats = CalibrationStats(**stats_data)
                self.calibration_stats[layer_name] = stats
            
            for layer_name, params_data in data.get('calibrated_params', {}).items():
                params = CalibratedQuantizationParams(**params_data)
                self.calibrated_params[layer_name] = params
            
            logger.info(f"Loaded calibration data for {len(self.calibration_stats)} layers")
            
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
    
    def collect_activation_stats(self, layer_name: str, activations: torch.Tensor, 
                                expert_activation_freq: float = 0.0):
        """收集激活统计信息"""
        with self.lock:
            try:
                with torch.no_grad():
                    # 计算激活统计
                    activation_mean = activations.mean().item()
                    activation_std = activations.std().item()
                    activation_min = activations.min().item()
                    activation_max = activations.max().item()
                    
                    # 更新或创建统计信息
                    if layer_name in self.calibration_stats:
                        stats = self.calibration_stats[layer_name]
                        # 更新统计（使用指数移动平均）
                        alpha = 0.1  # 学习率
                        stats.activation_mean = (1 - alpha) * stats.activation_mean + alpha * activation_mean
                        stats.activation_std = (1 - alpha) * stats.activation_std + alpha * activation_std
                        stats.activation_min = min(stats.activation_min, activation_min)
                        stats.activation_max = max(stats.activation_max, activation_max)
                        stats.expert_activation_frequency = expert_activation_freq
                        stats.sample_count += 1
                    else:
                        stats = CalibrationStats(
                            activation_mean=activation_mean,
                            activation_std=activation_std,
                            activation_min=activation_min,
                            activation_max=activation_max,
                            expert_activation_frequency=expert_activation_freq,
                            sample_count=1
                        )
                        self.calibration_stats[layer_name] = stats
                    
                    logger.debug(f"Updated activation stats for {layer_name}: mean={activation_mean:.4f}, std={activation_std:.4f}")
                    
            except Exception as e:
                logger.error(f"Error collecting activation stats for {layer_name}: {e}")
    
    def collect_weight_stats(self, layer_name: str, weight: torch.Tensor):
        """收集权重统计信息"""
        with self.lock:
            try:
                with torch.no_grad():
                    # 计算权重统计
                    weight_mean = weight.mean().item()
                    weight_std = weight.std().item()
                    weight_min = weight.min().item()
                    weight_max = weight.max().item()
                    
                    # 更新统计信息
                    if layer_name in self.calibration_stats:
                        stats = self.calibration_stats[layer_name]
                        stats.weight_mean = weight_mean
                        stats.weight_std = weight_std
                        stats.weight_min = weight_min
                        stats.weight_max = weight_max
                    else:
                        stats = CalibrationStats(
                            weight_mean=weight_mean,
                            weight_std=weight_std,
                            weight_min=weight_min,
                            weight_max=weight_max
                        )
                        self.calibration_stats[layer_name] = stats
                    
                    logger.debug(f"Updated weight stats for {layer_name}: mean={weight_mean:.4f}, std={weight_std:.4f}")
                    
            except Exception as e:
                logger.error(f"Error collecting weight stats for {layer_name}: {e}")
    
    def calibrate_quantization_params(self, layer_name: str, target_precision: str) -> CalibratedQuantizationParams:
        """校准量化参数"""
        with self.lock:
            try:
                if layer_name not in self.calibration_stats:
                    # 如果没有校准数据，返回默认参数
                    return CalibratedQuantizationParams()
                
                stats = self.calibration_stats[layer_name]
                
                # 基于激活频率调整量化策略
                activation_freq = stats.expert_activation_frequency
                
                if target_precision == "fp8":
                    params = self._calibrate_fp8_params(stats, activation_freq)
                elif target_precision == "gptq_int4":
                    params = self._calibrate_int4_params(stats, activation_freq)
                else:
                    params = CalibratedQuantizationParams()
                
                # 缓存校准参数
                self.calibrated_params[layer_name] = params
                
                logger.debug(f"Calibrated {target_precision} params for {layer_name}: scale={params.scale:.4f}, confidence={params.calibration_confidence:.4f}")
                
                return params
                
            except Exception as e:
                logger.error(f"Error calibrating quantization params for {layer_name}: {e}")
                return CalibratedQuantizationParams()
    
    def _calibrate_fp8_params(self, stats: CalibrationStats, activation_freq: float) -> CalibratedQuantizationParams:
        """校准FP8量化参数"""
        # FP8 E4M3格式的数值范围
        fp8_max = 448.0
        
        # 基于激活分布计算最优缩放因子
        activation_range = max(abs(stats.activation_max), abs(stats.activation_min))
        weight_range = max(abs(stats.weight_max), abs(stats.weight_min))
        
        # 考虑激活频率：高频激活的专家需要更保守的量化
        frequency_factor = 1.0 + (1.0 - activation_freq) * 0.5  # 低频专家可以更激进
        
        # 计算缩放因子
        max_range = max(activation_range, weight_range)
        scale = fp8_max / (max_range * frequency_factor)
        
        # 计算校准置信度
        confidence = min(1.0, stats.sample_count / 100.0)  # 基于样本数量
        
        return CalibratedQuantizationParams(
            scale=scale,
            zero_point=0.0,  # FP8通常使用对称量化
            symmetric=True,
            optimal_bits=8,
            calibration_confidence=confidence
        )
    
    def _calibrate_int4_params(self, stats: CalibrationStats, activation_freq: float) -> CalibratedQuantizationParams:
        """校准INT4量化参数"""
        # INT4的数值范围
        int4_max = 7.0  # 4位有符号整数的最大值
        
        # 基于激活分布计算最优缩放因子
        activation_range = max(abs(stats.activation_max), abs(stats.activation_min))
        weight_range = max(abs(stats.weight_max), abs(stats.weight_min))
        
        # 对于INT4，低频专家可以使用更激进的量化
        frequency_factor = 0.8 + activation_freq * 0.4  # 高频专家需要更保守
        
        # 计算缩放因子
        max_range = max(activation_range, weight_range)
        scale = int4_max / (max_range * frequency_factor)
        
        # 计算零点（非对称量化）
        zero_point = -stats.weight_min * scale
        
        # 计算校准置信度
        confidence = min(1.0, stats.sample_count / 200.0)  # INT4需要更多样本
        
        return CalibratedQuantizationParams(
            scale=scale,
            zero_point=zero_point,
            symmetric=False,  # INT4通常使用非对称量化
            optimal_bits=4,
            calibration_confidence=confidence
        )
    
    def get_calibrated_params(self, layer_name: str) -> Optional[CalibratedQuantizationParams]:
        """获取校准后的量化参数"""
        return self.calibrated_params.get(layer_name)
    
    def save_calibration_data(self, file_path: str):
        """保存校准数据"""
        try:
            with self.lock:
                data = {
                    'calibration_stats': {
                        name: {
                            'activation_mean': stats.activation_mean,
                            'activation_std': stats.activation_std,
                            'activation_min': stats.activation_min,
                            'activation_max': stats.activation_max,
                            'weight_mean': stats.weight_mean,
                            'weight_std': stats.weight_std,
                            'weight_min': stats.weight_min,
                            'weight_max': stats.weight_max,
                            'expert_activation_frequency': stats.expert_activation_frequency,
                            'sample_count': stats.sample_count
                        }
                        for name, stats in self.calibration_stats.items()
                    },
                    'calibrated_params': {
                        name: {
                            'scale': params.scale,
                            'zero_point': params.zero_point,
                            'symmetric': params.symmetric,
                            'optimal_bits': params.optimal_bits,
                            'calibration_confidence': params.calibration_confidence
                        }
                        for name, params in self.calibrated_params.items()
                    }
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"Saved calibration data to {file_path}")
                
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")
    
    def generate_calibration_report(self) -> Dict[str, Any]:
        """生成校准报告"""
        with self.lock:
            report = {
                'total_layers': len(self.calibration_stats),
                'calibrated_layers': len(self.calibrated_params),
                'layer_details': {}
            }
            
            for layer_name, stats in self.calibration_stats.items():
                params = self.calibrated_params.get(layer_name)
                
                layer_info = {
                    'activation_stats': {
                        'mean': stats.activation_mean,
                        'std': stats.activation_std,
                        'min': stats.activation_min,
                        'max': stats.activation_max,
                        'frequency': stats.expert_activation_frequency
                    },
                    'weight_stats': {
                        'mean': stats.weight_mean,
                        'std': stats.weight_std,
                        'min': stats.weight_min,
                        'max': stats.weight_max
                    },
                    'sample_count': stats.sample_count,
                    'has_calibrated_params': params is not None
                }
                
                if params:
                    layer_info['calibrated_params'] = {
                        'scale': params.scale,
                        'zero_point': params.zero_point,
                        'symmetric': params.symmetric,
                        'optimal_bits': params.optimal_bits,
                        'confidence': params.calibration_confidence
                    }
                
                report['layer_details'][layer_name] = layer_info
            
            return report


# 全局校准器实例
_global_calibrator: Optional[QuantizationCalibrator] = None


def get_global_calibrator() -> QuantizationCalibrator:
    """获取全局校准器实例"""
    global _global_calibrator
    if _global_calibrator is None:
        _global_calibrator = QuantizationCalibrator()
    return _global_calibrator


def set_global_calibrator(calibrator: QuantizationCalibrator):
    """设置全局校准器实例"""
    global _global_calibrator
    _global_calibrator = calibrator
