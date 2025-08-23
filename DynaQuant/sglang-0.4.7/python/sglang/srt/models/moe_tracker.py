#!/usr/bin/env python3
"""
MoE专家激活跟踪器
用于包装MoE模块并跟踪专家激活情况
基于SGLang架构优化
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from contextlib import contextmanager

from sglang.srt.model_loader.enhanced_mixed_precision_loader import (
    record_expert_activation, record_request, get_global_expert_tracker
)

logger = logging.getLogger(__name__)


class MoEModuleWrapper(nn.Module):
    """MoE模块包装器，用于跟踪专家激活"""
    
    def __init__(self, original_module: nn.Module, layer_id: int, module_name: str):
        super().__init__()
        self.original_module = original_module
        self.layer_id = layer_id
        self.module_name = module_name
        self.expert_tracker = get_global_expert_tracker()
        
        # 保存原始的前向传播方法
        self.original_forward = original_module.forward
        
        # 替换前向传播方法
        original_module.forward = self._tracked_forward
    
    def _tracked_forward(self, *args, **kwargs):
        """带跟踪的前向传播"""
        try:
            # 调用原始前向传播
            output = self.original_forward(*args, **kwargs)
            
            # 尝试提取专家激活信息
            if self.expert_tracker:
                self._extract_expert_activations(output, *args, **kwargs)
            
            return output
            
        except Exception as e:
            logger.warning(f"Error in tracked forward pass for {self.module_name}: {e}")
            # 如果跟踪失败，仍然返回原始输出
            return self.original_forward(*args, **kwargs)
    
    def _extract_expert_activations(self, output, *args, **kwargs):
        """提取专家激活信息"""
        try:
            # 尝试从输出中提取专家激活信息
            if hasattr(output, 'expert_activations'):
                # 如果输出包含专家激活信息
                expert_activations = output.expert_activations
                if isinstance(expert_activations, (list, tuple)):
                    for expert_id in expert_activations:
                        record_expert_activation(self.layer_id, expert_id)
                elif isinstance(expert_activations, torch.Tensor):
                    # 如果是张量，假设是one-hot编码
                    expert_ids = torch.nonzero(expert_activations).flatten().tolist()
                    for expert_id in expert_ids:
                        record_expert_activation(self.layer_id, expert_id)
            
            # 尝试从模块内部状态提取
            elif hasattr(self.original_module, 'experts'):
                # 检查是否有gate输出或路由信息
                if hasattr(self.original_module, '_gate_outputs'):
                    gate_outputs = self.original_module._gate_outputs
                    if isinstance(gate_outputs, torch.Tensor):
                        # 假设gate输出是专家选择概率
                        expert_ids = torch.argmax(gate_outputs, dim=-1).flatten().tolist()
                        for expert_id in expert_ids:
                            record_expert_activation(self.layer_id, expert_id)
                
                elif hasattr(self.original_module, '_selected_experts'):
                    selected_experts = self.original_module._selected_experts
                    if isinstance(selected_experts, (list, tuple)):
                        for expert_id in selected_experts:
                            record_expert_activation(self.layer_id, expert_id)
            
            # 尝试从输入参数中提取
            elif len(args) > 0 and isinstance(args[0], torch.Tensor):
                # 尝试从输入张量的形状推断处理的token数量
                input_tensor = args[0]
                if len(input_tensor.shape) >= 2:
                    tokens_processed = input_tensor.shape[1]  # 序列长度
                    # 假设每个token激活一个专家
                    record_expert_activation(self.layer_id, 0, tokens_processed)
            
        except Exception as e:
            logger.debug(f"Could not extract expert activations from {self.module_name}: {e}")
    
    def forward(self, *args, **kwargs):
        """前向传播（委托给原始模块）"""
        return self.original_module(*args, **kwargs)


class MoETracker:
    """MoE跟踪器静态类"""
    
    @staticmethod
    def _is_moe_module(module: nn.Module) -> bool:
        """检查是否是MoE模块"""
        # 检查是否有experts属性
        if hasattr(module, 'experts'):
            return True
        
        # 检查模块名称
        module_name = module.__class__.__name__.lower()
        moe_keywords = ['moe', 'mixture', 'expert', 'switch', 'gated']
        if any(keyword in module_name for keyword in moe_keywords):
            return True
        
        # 检查是否有gate相关的方法或属性
        gate_attributes = ['gate', 'router', 'switch', 'gating']
        if any(hasattr(module, attr) for attr in gate_attributes):
            return True
        
        return False
    
    @staticmethod
    def wrap_moe_modules(model: nn.Module, layer_id_map: Optional[Dict[str, int]] = None) -> Dict[str, MoEModuleWrapper]:
        """包装模型中的MoE模块"""
        wrapped_modules = {}
        
        if layer_id_map is None:
            # 自动生成层ID映射
            layer_id_map = {}
            layer_counter = 0
            
            for name, module in model.named_modules():
                if MoETracker._is_moe_module(module):
                    layer_id_map[name] = layer_counter
                    layer_counter += 1
        
        for name, module in model.named_modules():
            if MoETracker._is_moe_module(module):
                if name in layer_id_map:
                    layer_id = layer_id_map[name]
                else:
                    # 如果没有预定义的层ID，使用模块的哈希值
                    layer_id = hash(name) % 10000
                
                wrapper = MoEModuleWrapper(module, layer_id, name)
                wrapped_modules[name] = wrapper
                
                logger.info(f"Wrapped MoE module at layer {layer_id}: {name}")
        
        logger.info(f"Wrapped {len(wrapped_modules)} MoE modules")
        return wrapped_modules
    
    @staticmethod
    def track_expert_activations_in_model(model: nn.Module, 
                                        enable_tracking: bool = True,
                                        layer_id_map: Optional[Dict[str, int]] = None) -> Optional[Dict[str, MoEModuleWrapper]]:
        """在模型中启用专家激活跟踪"""
        if not enable_tracking:
            logger.info("Expert activation tracking disabled")
            return None
        
        expert_tracker = get_global_expert_tracker()
        if expert_tracker is None:
            logger.warning("No global expert tracker found, tracking will be disabled")
            return None
        
        logger.info("Enabling expert activation tracking in model")
        wrapped_modules = MoETracker.wrap_moe_modules(model, layer_id_map)
        
        return wrapped_modules


def track_expert_activations_in_model(model: nn.Module, 
                                    enable_tracking: bool = True,
                                    layer_id_map: Optional[Dict[str, int]] = None) -> Optional[Dict[str, MoEModuleWrapper]]:
    """在模型中启用专家激活跟踪（便捷函数）"""
    return MoETracker.track_expert_activations_in_model(model, enable_tracking, layer_id_map)


@contextmanager
def expert_tracking_context(request_id: str = None, input_length: int = 0, output_length: int = 0):
    """专家跟踪上下文管理器"""
    try:
        # 记录请求开始
        if request_id:
            record_request(request_id, input_length, output_length)
        
        yield
        
    except Exception as e:
        logger.error(f"Error in expert tracking context: {e}")
        raise
    finally:
        # 可以在这里添加清理逻辑
        pass


class ExpertActivationHook:
    """专家激活钩子，用于在特定点记录激活"""
    
    def __init__(self, layer_id: int, expert_id: int):
        self.layer_id = layer_id
        self.expert_id = expert_id
    
    def __call__(self, module, input, output):
        """钩子函数"""
        try:
            # 记录专家激活
            record_expert_activation(self.layer_id, self.expert_id)
        except Exception as e:
            logger.warning(f"Error in expert activation hook: {e}")


def register_expert_activation_hooks(model: nn.Module, 
                                   layer_expert_map: Dict[Tuple[str, int], int]) -> List[ExpertActivationHook]:
    """注册专家激活钩子"""
    hooks = []
    
    for (module_name, expert_id), layer_id in layer_expert_map.items():
        # 查找模块
        module = None
        for name, mod in model.named_modules():
            if name == module_name:
                module = mod
                break
        
        if module is not None:
            hook = ExpertActivationHook(layer_id, expert_id)
            module.register_forward_hook(hook)
            hooks.append(hook)
            logger.debug(f"Registered expert activation hook for {module_name}, expert {expert_id}, layer {layer_id}")
        else:
            logger.warning(f"Module {module_name} not found for expert activation hook")
    
    logger.info(f"Registered {len(hooks)} expert activation hooks")
    return hooks


def get_moe_module_info(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """获取MoE模块信息"""
    moe_info = {}
    
    for name, module in model.named_modules():
        if MoETracker._is_moe_module(module):
            info = {
                'module_type': module.__class__.__name__,
                'has_experts': hasattr(module, 'experts'),
                'expert_count': len(module.experts) if hasattr(module, 'experts') else 0,
                'has_gate': hasattr(module, 'gate') or hasattr(module, 'router'),
                'parameters': sum(p.numel() for p in module.parameters()),
                'trainable_parameters': sum(p.numel() for p in module.parameters() if p.requires_grad)
            }
            
            # 尝试获取更多详细信息
            if hasattr(module, 'experts'):
                expert_types = [type(expert).__name__ for expert in module.experts]
                info['expert_types'] = expert_types
            
            moe_info[name] = info
    
    return moe_info


def analyze_moe_architecture(model: nn.Module) -> Dict[str, Any]:
    """分析MoE架构"""
    moe_info = get_moe_module_info(model)
    
    analysis = {
        'total_moe_layers': len(moe_info),
        'moe_layers': moe_info,
        'total_experts': sum(info['expert_count'] for info in moe_info.values()),
        'total_parameters': sum(info['parameters'] for info in moe_info.values()),
        'architecture_summary': {}
    }
    
    # 生成架构摘要
    if moe_info:
        expert_counts = [info['expert_count'] for info in moe_info.values()]
        analysis['architecture_summary'] = {
            'min_experts_per_layer': min(expert_counts),
            'max_experts_per_layer': max(expert_counts),
            'avg_experts_per_layer': sum(expert_counts) / len(expert_counts),
            'layer_distribution': {}
        }
        
        # 按专家数量分组
        for info in moe_info.values():
            expert_count = info['expert_count']
            if expert_count not in analysis['architecture_summary']['layer_distribution']:
                analysis['architecture_summary']['layer_distribution'][expert_count] = 0
            analysis['architecture_summary']['layer_distribution'][expert_count] += 1
    
    return analysis
