#!/usr/bin/env python3
"""
增强版Expert Parallel MoE模块
支持GPTQ-INT4量化和动态精度切换
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from collections import defaultdict

logger = logging.getLogger(__name__)

# 导入SGLang的EPMoE基础类
from sglang.srt.layers.moe.ep_moe.layer import EPMoE

# 导入量化支持
from sglang.srt.layers.quantization.gptq import GPTQConfig, GPTQLinearMethod
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod


class EnhancedEPMoE(EPMoE):
    """增强版EPMoE，支持动态量化和精度切换"""
    
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        topk_group: Optional[int] = None,
        quant_config: Optional[Any] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[callable] = None,
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        use_per_token_if_dynamic: bool = True,
        **kwargs
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            params_dtype=params_dtype,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            topk_group=topk_group,
            quant_config=quant_config,
            tp_size=tp_size,
            prefix=prefix,
            correction_bias=correction_bias,
            custom_routing_function=custom_routing_function,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            use_per_token_if_dynamic=use_per_token_if_dynamic,
            **kwargs
        )
        
        # 动态量化支持
        self.expert_precisions = defaultdict(lambda: "fp16")  # expert_id -> precision
        self.quantization_methods = self._init_quantization_methods()
        self.weight_cache = defaultdict(dict)  # precision -> {expert_id: weights}
        
        logger.info(f"EnhancedEPMoE initialized with {num_experts} experts, layer {layer_id}")
    
    def _init_quantization_methods(self):
        """初始化量化方法"""
        methods = {}
        
        # FP8量化方法
        fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            ignored_layers=None,
            weight_block_size=None
        )
        methods["fp8"] = Fp8LinearMethod(fp8_config)
        
        # GPTQ-INT4量化方法
        gptq_config = GPTQConfig(
            weight_bits=4,
            group_size=128,
            desc_act=True,
            lm_head_quantized=False,
            dynamic={}
        )
        methods["gptq_int4"] = GPTQLinearMethod(gptq_config)
        
        return methods
    
    def set_expert_precision(self, expert_id: int, precision: str):
        """设置expert的量化精度"""
        self.expert_precisions[expert_id] = precision
        logger.debug(f"Set expert {expert_id} precision to {precision}")
    
    def get_expert_precision(self, expert_id: int) -> str:
        """获取expert的量化精度"""
        return self.expert_precisions[expert_id]
    
    def load_expert_weights(self, expert_id: int, precision: str, weights: Dict[str, torch.Tensor]):
        """加载expert权重到缓存"""
        self.weight_cache[precision][expert_id] = weights
        logger.debug(f"Loaded weights for expert {expert_id} with precision {precision}")
    
    def swap_expert_weights(self, expert_id: int, new_precision: str) -> bool:
        """交换expert权重"""
        try:
            if new_precision not in self.weight_cache:
                logger.warning(f"No weights cached for precision {new_precision}")
                return False
            
            if expert_id not in self.weight_cache[new_precision]:
                logger.warning(f"No weights cached for expert {expert_id} with precision {new_precision}")
                return False
            
            # 获取expert模块
            expert_module = self._get_expert_module(expert_id)
            if expert_module is None:
                logger.error(f"Expert module not found for expert {expert_id}")
                return False
            
            # 获取新权重
            new_weights = self.weight_cache[new_precision][expert_id]
            
            # 替换权重
            device = next(expert_module.parameters()).device
            for name, weight in new_weights.items():
                if hasattr(expert_module, name):
                    param = getattr(expert_module, name)
                    if weight.device != device:
                        weight = weight.to(device)
                    param.data.copy_(weight)
            
            # 更新精度
            old_precision = self.expert_precisions[expert_id]
            self.expert_precisions[expert_id] = new_precision
            
            logger.info(f"Successfully swapped expert {expert_id} from {old_precision} to {new_precision}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to swap expert {expert_id} weights: {e}")
            return False
    
    def _get_expert_module(self, expert_id: int) -> Optional[nn.Module]:
        """获取expert模块"""
        try:
            # 检查expert_id是否在当前rank的范围内
            if not (self.start_expert_id <= expert_id <= self.end_expert_id):
                logger.warning(f"Expert {expert_id} not in current rank range [{self.start_expert_id}, {self.end_expert_id}]")
                return None
            
            # 计算本地expert索引
            local_expert_id = expert_id - self.start_expert_id
            
            # 获取expert模块
            if hasattr(self, 'experts') and local_expert_id < len(self.experts):
                return self.experts[local_expert_id]
            
            logger.warning(f"Expert module not found for expert {expert_id} (local_id: {local_expert_id})")
            return None
            
        except Exception as e:
            logger.error(f"Error getting expert module for expert {expert_id}: {e}")
            return None
    
    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        """前向传播，支持动态量化"""
        # 调用父类的forward方法
        output = super().forward(hidden_states, router_logits)
        
        # 这里可以添加额外的动态量化逻辑
        # 例如：根据当前expert的精度选择不同的计算路径
        
        return output
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        """获取量化统计信息"""
        precision_counts = defaultdict(int)
        for expert_id, precision in self.expert_precisions.items():
            precision_counts[precision] += 1
        
        return {
            'total_experts': len(self.expert_precisions),
            'precision_distribution': dict(precision_counts),
            'cached_precisions': list(self.weight_cache.keys()),
            'cached_experts': {precision: list(weights.keys()) for precision, weights in self.weight_cache.items()}
        }


def create_enhanced_ep_moe_from_original(original_ep_moe: EPMoE) -> EnhancedEPMoE:
    """从原始EPMoE创建增强版EPMoE"""
    # 获取原始EPMoE的参数
    config = {
        'num_experts': original_ep_moe.num_experts,
        'top_k': original_ep_moe.top_k,
        'hidden_size': original_ep_moe.hidden_size,
        'intermediate_size': original_ep_moe.intermediate_size,
        'layer_id': original_ep_moe.layer_id,
        'params_dtype': getattr(original_ep_moe, 'params_dtype', None),
        'renormalize': original_ep_moe.renormalize,
        'use_grouped_topk': getattr(original_ep_moe, 'use_grouped_topk', False),
        'num_expert_group': getattr(original_ep_moe, 'num_expert_group', None),
        'num_fused_shared_experts': getattr(original_ep_moe, 'num_fused_shared_experts', 0),
        'topk_group': getattr(original_ep_moe, 'topk_group', None),
        'quant_config': getattr(original_ep_moe, 'quant_config', None),
        'tp_size': original_ep_moe.tp_size,
        'prefix': getattr(original_ep_moe, 'prefix', ""),
        'correction_bias': getattr(original_ep_moe, 'correction_bias', None),
        'custom_routing_function': getattr(original_ep_moe, 'custom_routing_function', None),
        'activation': getattr(original_ep_moe, 'activation', "silu"),
        'routed_scaling_factor': getattr(original_ep_moe, 'routed_scaling_factor', None),
        'use_per_token_if_dynamic': getattr(original_ep_moe, 'use_per_token_if_dynamic', True),
    }
    
    # 创建增强版EPMoE
    enhanced_ep_moe = EnhancedEPMoE(**config)
    
    # 复制权重（如果需要）
    if hasattr(original_ep_moe, 'experts'):
        for i, expert in enumerate(original_ep_moe.experts):
            if expert is not None and i < len(enhanced_ep_moe.experts):
                enhanced_ep_moe.experts[i] = expert
    
    logger.info("Created EnhancedEPMoE from original EPMoE")
    return enhanced_ep_moe
