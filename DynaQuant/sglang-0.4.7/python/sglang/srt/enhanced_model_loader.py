#!/usr/bin/env python3
"""
增强的模型加载器
集成混合精度权重加载和专家激活跟踪功能
基于SGLang架构优化
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

from sglang.srt.model_loader.enhanced_mixed_precision_loader import (
    EnhancedMixedPrecisionWeightLoader,
    set_global_expert_tracker,
    get_global_expert_tracker,
    ExpertActivationTracker
)
from sglang.srt.models.moe_tracker import (
    track_expert_activations_in_model,
    analyze_moe_architecture
)

logger = logging.getLogger(__name__)


class EnhancedModelLoader:
    """增强的模型加载器"""
    
    def __init__(self, config_path: str, enable_expert_tracking: bool = True):
        """
        初始化增强的模型加载器
        
        Args:
            config_path: 配置文件路径
            enable_expert_tracking: 是否启用专家激活跟踪
        """
        self.config_path = config_path
        self.enable_expert_tracking = enable_expert_tracking
        
        # 初始化增强的混合精度权重加载器
        self.weight_loader = EnhancedMixedPrecisionWeightLoader(
            config_path, enable_expert_tracking
        )
        
        # 设置全局专家跟踪器
        if enable_expert_tracking:
            expert_tracker = self.weight_loader.get_expert_tracker()
            if expert_tracker:
                set_global_expert_tracker(expert_tracker)
                logger.info("Global expert tracker set")
        
        logger.info("Enhanced model loader initialized")
    
    def load_model(self, model: torch.nn.Module, 
                  enable_moe_tracking: bool = True) -> Dict[str, Any]:
        """
        加载模型权重并启用专家激活跟踪
        
        Args:
            model: 要加载的模型
            enable_moe_tracking: 是否启用MoE跟踪
            
        Returns:
            加载统计信息
        """
        # 1. 分析MoE架构
        moe_analysis = analyze_moe_architecture(model)
        logger.info(f"MoE architecture analysis: {moe_analysis['total_moe_layers']} MoE layers, {moe_analysis['total_experts']} total experts")
        
        # 2. 加载混合精度权重
        weight_stats = self.weight_loader.load_model_weights(model)
        logger.info(f"Weight loading completed: {weight_stats['loaded']} loaded, {weight_stats['skipped']} skipped")
        
        # 3. 启用专家激活跟踪
        moe_wrappers = None
        if enable_moe_tracking and self.enable_expert_tracking:
            moe_wrappers = track_expert_activations_in_model(model, True)
            if moe_wrappers:
                logger.info(f"Expert activation tracking enabled for {len(moe_wrappers)} MoE modules")
        
        # 4. 返回综合统计信息
        stats = {
            'weight_loading': weight_stats,
            'moe_analysis': moe_analysis,
            'expert_tracking_enabled': self.enable_expert_tracking and enable_moe_tracking,
            'moe_wrappers_count': len(moe_wrappers) if moe_wrappers else 0
        }
        
        return stats
    
    def get_expert_tracker(self) -> Optional[ExpertActivationTracker]:
        """获取专家激活跟踪器"""
        return self.weight_loader.get_expert_tracker()
    
    def get_global_expert_tracker(self) -> Optional[ExpertActivationTracker]:
        """获取全局专家激活跟踪器"""
        return get_global_expert_tracker()
    
    def enable_expert_tracking(self, enable: bool = True):
        """启用或禁用专家激活跟踪"""
        self.weight_loader.enable_expert_tracking(enable)
        if enable:
            expert_tracker = self.weight_loader.get_expert_tracker()
            if expert_tracker:
                set_global_expert_tracker(expert_tracker)
    
    def export_expert_stats(self, file_path: str):
        """导出专家统计信息"""
        expert_tracker = self.get_expert_tracker()
        if expert_tracker:
            expert_tracker.export_stats(file_path)
        else:
            logger.warning("No expert tracker available for export")
    
    def get_expert_stats(self, layer_id: Optional[int] = None, 
                        expert_id: Optional[int] = None) -> Dict:
        """获取专家统计信息"""
        expert_tracker = self.get_expert_tracker()
        if expert_tracker:
            return expert_tracker.get_expert_stats(layer_id, expert_id)
        return {}
    
    def get_top_experts(self, top_k: int = 10) -> list:
        """获取激活次数最多的专家"""
        expert_tracker = self.get_expert_tracker()
        if expert_tracker:
            return expert_tracker.get_top_experts(top_k)
        return []
    
    def get_layer_stats(self) -> Dict:
        """获取每层的统计信息"""
        expert_tracker = self.get_expert_tracker()
        if expert_tracker:
            return expert_tracker.get_layer_stats()
        return {}
    
    def reset_expert_stats(self):
        """重置专家统计信息"""
        expert_tracker = self.get_expert_tracker()
        if expert_tracker:
            expert_tracker.reset_stats()
            logger.info("Expert statistics reset")


def create_enhanced_model_loader(config_path: str, 
                               enable_expert_tracking: bool = True) -> EnhancedModelLoader:
    """创建增强的模型加载器（工厂函数）"""
    return EnhancedModelLoader(config_path, enable_expert_tracking)


# 便捷函数
def load_model_with_enhanced_features(model: torch.nn.Module, 
                                    config_path: str,
                                    enable_expert_tracking: bool = True,
                                    enable_moe_tracking: bool = True) -> Dict[str, Any]:
    """
    使用增强功能加载模型
    
    Args:
        model: 要加载的模型
        config_path: 配置文件路径
        enable_expert_tracking: 是否启用专家激活跟踪
        enable_moe_tracking: 是否启用MoE跟踪
        
    Returns:
        加载统计信息
    """
    loader = create_enhanced_model_loader(config_path, enable_expert_tracking)
    return loader.load_model(model, enable_moe_tracking)


def get_expert_activation_stats() -> Dict[str, Any]:
    """获取专家激活统计信息（全局函数）"""
    expert_tracker = get_global_expert_tracker()
    if expert_tracker:
        return {
            'expert_stats': expert_tracker.get_expert_stats(),
            'layer_stats': expert_tracker.get_layer_stats(),
            'top_experts': expert_tracker.get_top_experts(10)
        }
    return {}


def reset_expert_activation_stats():
    """重置专家激活统计信息（全局函数）"""
    expert_tracker = get_global_expert_tracker()
    if expert_tracker:
        expert_tracker.reset_stats()
        logger.info("Expert activation statistics reset")
    else:
        logger.warning("No global expert tracker found")


def export_expert_activation_stats(file_path: str):
    """导出专家激活统计信息（全局函数）"""
    expert_tracker = get_global_expert_tracker()
    if expert_tracker:
        expert_tracker.export_stats(file_path)
    else:
        logger.warning("No global expert tracker found")
