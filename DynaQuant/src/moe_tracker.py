#!/usr/bin/env python3
"""
MoE专家激活跟踪器
用于在推理过程中跟踪每个expert的激活情况
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from .expert_activation_tracker import record_expert_activation


class MoETracker:
    """MoE专家激活跟踪器"""
    
    @staticmethod
    def track_moe_forward(module: nn.Module, input_tensor: torch.Tensor, 
                         layer_id: int, original_forward_func) -> torch.Tensor:
        """
        跟踪MoE前向传播
        
        Args:
            module: MoE模块
            input_tensor: 输入张量
            layer_id: 层ID
            original_forward_func: 原始前向传播函数
            
        Returns:
            输出张量
        """
        # 调用原始前向传播
        output = original_forward_func(input_tensor)
        
        # 尝试提取专家激活信息
        MoETracker._extract_expert_activations(module, input_tensor, layer_id)
        
        return output
    
    @staticmethod
    def _extract_expert_activations(module: nn.Module, input_tensor: torch.Tensor, layer_id: int):
        """
        提取专家激活信息
        
        Args:
            module: MoE模块
            input_tensor: 输入张量
            layer_id: 层ID
        """
        try:
            # 尝试从模块中获取专家激活信息
            if hasattr(module, 'gate') and hasattr(module, 'experts'):
                # 获取门控权重
                gate_output = module.gate(input_tensor)
                
                # 获取top-k专家
                if hasattr(module, 'num_experts_per_tok'):
                    top_k = module.num_experts_per_tok
                else:
                    top_k = 2  # 默认值
                
                # 获取top-k专家索引
                top_k_weights, top_k_indices = torch.topk(gate_output, top_k, dim=-1)
                
                # 记录激活的专家
                batch_size, seq_len, _ = top_k_indices.shape
                for b in range(batch_size):
                    for s in range(seq_len):
                        for k in range(top_k):
                            expert_id = top_k_indices[b, s, k].item()
                            weight = top_k_weights[b, s, k].item()
                            
                            # 只记录权重大于阈值的激活
                            if weight > 0.01:  # 阈值可调整
                                record_expert_activation(layer_id, expert_id, 1)
            
            # 对于Qwen3的MoE结构
            elif hasattr(module, 'gate_proj') and hasattr(module, 'experts'):
                # Qwen3 MoE结构
                gate_output = module.gate_proj(input_tensor)
                
                # 获取专家数量
                num_experts = len(module.experts)
                
                # 计算每个token的专家分配
                expert_weights = F.softmax(gate_output, dim=-1)
                
                # 获取top-k专家
                top_k = min(2, num_experts)  # 通常使用top-2
                top_k_weights, top_k_indices = torch.topk(expert_weights, top_k, dim=-1)
                
                # 记录激活的专家
                batch_size, seq_len, _ = top_k_indices.shape
                for b in range(batch_size):
                    for s in range(seq_len):
                        for k in range(top_k):
                            expert_id = top_k_indices[b, s, k].item()
                            weight = top_k_weights[b, s, k].item()
                            
                            # 只记录权重大于阈值的激活
                            if weight > 0.01:
                                record_expert_activation(layer_id, expert_id, 1)
        
        except Exception as e:
            # 如果提取失败，记录一个默认激活
            print(f"Warning: Failed to extract expert activations for layer {layer_id}: {e}")
            # 记录一个默认激活（假设使用第一个专家）
            record_expert_activation(layer_id, 0, input_tensor.shape[1])


class MoEModuleWrapper(nn.Module):
    """MoE模块包装器"""
    
    def __init__(self, original_module: nn.Module, layer_id: int):
        """
        初始化MoE模块包装器
        
        Args:
            original_module: 原始MoE模块
            layer_id: 层ID
        """
        super().__init__()
        self.original_module = original_module
        self.layer_id = layer_id
        
        # 保存原始前向传播函数
        self.original_forward = original_module.forward
    
    def forward(self, *args, **kwargs):
        """包装的前向传播函数"""
        # 调用原始前向传播
        output = self.original_forward(*args, **kwargs)
        
        # 提取专家激活信息
        if len(args) > 0:
            input_tensor = args[0]
            MoETracker._extract_expert_activations(
                self.original_module, input_tensor, self.layer_id
            )
        
        return output


def wrap_moe_modules(model: nn.Module) -> nn.Module:
    """
    包装模型中的MoE模块以跟踪专家激活
    
    Args:
        model: 要包装的模型
        
    Returns:
        包装后的模型
    """
    def _wrap_module(module, layer_id=0):
        """递归包装模块"""
        for name, child in module.named_children():
            # 检查是否是MoE模块
            if _is_moe_module(child):
                # 包装MoE模块
                wrapped_child = MoEModuleWrapper(child, layer_id)
                setattr(module, name, wrapped_child)
                print(f"Wrapped MoE module at layer {layer_id}: {name}")
            
            # 递归处理子模块
            if hasattr(child, 'named_children'):
                _wrap_module(child, layer_id + 1)
    
    _wrap_module(model)
    return model


def _is_moe_module(module: nn.Module) -> bool:
    """
    检查模块是否是MoE模块
    
    Args:
        module: 要检查的模块
        
    Returns:
        是否是MoE模块
    """
    # 检查常见的MoE模块特征
    moe_indicators = [
        hasattr(module, 'experts'),
        hasattr(module, 'gate'),
        hasattr(module, 'gate_proj'),
        hasattr(module, 'num_experts'),
        hasattr(module, 'num_experts_per_tok'),
        'moe' in module.__class__.__name__.lower(),
        'mixture' in module.__class__.__name__.lower()
    ]
    
    return any(moe_indicators)


def track_expert_activations_in_model(model: nn.Module) -> nn.Module:
    """
    在模型中启用专家激活跟踪
    
    Args:
        model: 要跟踪的模型
        
    Returns:
        启用跟踪的模型
    """
    print("Enabling expert activation tracking...")
    wrapped_model = wrap_moe_modules(model)
    print("Expert activation tracking enabled!")
    return wrapped_model
