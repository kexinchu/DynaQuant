#!/usr/bin/env python3
"""
混合精度EPMoE模块
支持对专家层进行混合精度量化，复用SGLang的EPMoE实现
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

# 导入SGLang的EPMoE实现
from sglang.srt.layers.moe.ep_moe.layer import EPMoE
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

logger = logging.getLogger(__name__)


@dataclass
class ExpertQuantizationConfig:
    """专家量化配置"""
    w13_precision: str  # w13_weight的量化精度
    w2_precision: str   # w2_weight的量化精度
    w13_compressed_weight: Optional[Any] = None
    w2_compressed_weight: Optional[Any] = None


class MixedPrecisionEPMoE(EPMoE):
    """混合精度EPMoE模块 - 支持专家层的混合精度量化"""
    
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
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Any] = None,
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        use_per_token_if_dynamic: bool = True,
        expert_quant_configs: Optional[Dict[int, ExpertQuantizationConfig]] = None,
    ):
        # 调用父类初始化
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
        )
        
        # 存储专家量化配置
        self.expert_quant_configs = expert_quant_configs or {}
        
        # 初始化量化方法
        self._init_quantization_methods()
        
        # 存储原始权重，用于混合精度处理
        self.original_w13_weights = {}
        self.original_w2_weights = {}
        
        logger.info(f"Initialized MixedPrecisionEPMoE with {len(self.expert_quant_configs)} expert quantization configs")
    
    def _init_quantization_methods(self):
        """初始化量化方法"""
        self.quantization_methods = {}
        
        # 创建FP8配置
        fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            ignored_layers=None,
            weight_block_size=None
        )
        
        # 添加SGLang自己的FP8方法
        self.quantization_methods['fp8'] = Fp8LinearMethod(fp8_config)
        
        # 添加vllm的方法（如果可用）
        try:
            from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
            from vllm.model_executor.layers.quantization.awq import AWQLinearMethod
            VLLM_AVAILABLE = True
        except ImportError:
            VLLM_AVAILABLE = False
            # 定义占位符类
            class GPTQLinearMethod:
                def __init__(self):
                    pass
                def apply(self, layer, x, bias=None):
                    raise NotImplementedError("GPTQLinearMethod requires vllm")
            
            class AWQLinearMethod:
                def __init__(self):
                    pass
                def apply(self, layer, x, bias=None):
                    raise NotImplementedError("AWQLinearMethod requires vllm")
        
        if VLLM_AVAILABLE:
            self.quantization_methods['gptq_int4'] = GPTQLinearMethod()
            self.quantization_methods['awq_int4'] = AWQLinearMethod()
        else:
            logger.warning("VLLM not available, GPTQ and AWQ quantization methods will not work")
            # 添加占位符
            self.quantization_methods['gptq_int4'] = GPTQLinearMethod()
            self.quantization_methods['awq_int4'] = AWQLinearMethod()
    
    def set_expert_quantization_config(self, expert_id: int, config: ExpertQuantizationConfig):
        """设置特定专家的量化配置"""
        self.expert_quant_configs[expert_id] = config
        logger.info(f"Set quantization config for expert {expert_id}: w13={config.w13_precision}, w2={config.w2_precision}")
    
    def _apply_expert_quantization(self, expert_id: int, weight_name: str, weight: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """应用专家量化"""
        if expert_id not in self.expert_quant_configs:
            return torch.matmul(input_tensor, weight)
        
        config = self.expert_quant_configs[expert_id]
        
        # 确定使用哪个权重和精度
        if weight_name == "w13_weight":
            precision = config.w13_precision
            compressed_weight = config.w13_compressed_weight
        elif weight_name == "w2_weight":
            precision = config.w2_precision
            compressed_weight = config.w2_compressed_weight
        else:
            return torch.matmul(input_tensor, weight)
        
        if compressed_weight is None:
            return torch.matmul(input_tensor, weight)
        
        try:
            if precision == "fp8":
                return self._apply_fp8_quantization(compressed_weight, input_tensor, weight)
            elif precision == "gptq_int4":
                return self._apply_gptq_quantization(compressed_weight, input_tensor, weight)
            elif precision == "awq_int4":
                return self._apply_awq_quantization(compressed_weight, input_tensor, weight)
            else:
                return torch.matmul(input_tensor, weight)
        except Exception as e:
            logger.error(f"Error applying {precision} quantization for expert {expert_id}: {e}")
            return torch.matmul(input_tensor, weight)
    
    def _apply_fp8_quantization(self, compressed_weight: Any, input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """应用FP8量化"""
        try:
            # 从压缩权重中提取FP8数据
            weight_data = compressed_weight.data.get('weight')
            scale_inv = compressed_weight.data.get('scale_inv')
            
            if weight_data is None:
                return torch.matmul(input_tensor, weight)
            
            # 使用SGLang的FP8方法
            quant_method = self.quantization_methods['fp8']
            
            # 创建临时层用于量化计算
            temp_layer = nn.Linear(weight_data.shape[1], weight_data.shape[0], bias=False)
            temp_layer.weight = nn.Parameter(weight_data)
            if scale_inv is not None:
                temp_layer.weight_scale_inv = nn.Parameter(scale_inv)
            
            # 应用量化
            result = quant_method.apply(temp_layer, input_tensor)
            return result
            
        except Exception as e:
            logger.error(f"Error in FP8 quantization: {e}")
            return torch.matmul(input_tensor, weight)
    
    def _apply_gptq_quantization(self, compressed_weight: Any, input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """应用GPTQ量化"""
        try:
            if not VLLM_AVAILABLE:
                logger.error("GPTQ requires VLLM to be available")
                return torch.matmul(input_tensor, weight)
            
            # 从压缩权重中提取GPTQ组件
            qweight = compressed_weight.data.get('qweight')
            qzeros = compressed_weight.data.get('qzeros')
            scales = compressed_weight.data.get('scales')
            g_idx = compressed_weight.data.get('g_idx')
            
            if qweight is None or qzeros is None or scales is None:
                return torch.matmul(input_tensor, weight)
            
            # 使用VLLM的GPTQ方法
            quant_method = self.quantization_methods['gptq_int4']
            
            # 创建临时层用于量化计算
            temp_layer = nn.Linear(qweight.shape[1], qweight.shape[0] * 8, bias=False)  # 4-bit packing
            temp_layer.qweight = nn.Parameter(qweight)
            temp_layer.qzeros = nn.Parameter(qzeros)
            temp_layer.scales = nn.Parameter(scales)
            if g_idx is not None:
                temp_layer.g_idx = nn.Parameter(g_idx)
            
            # 应用量化
            result = quant_method.apply(temp_layer, input_tensor)
            return result
            
        except Exception as e:
            logger.error(f"Error in GPTQ quantization: {e}")
            return torch.matmul(input_tensor, weight)
    
    def _apply_awq_quantization(self, compressed_weight: Any, input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """应用AWQ量化"""
        try:
            if not VLLM_AVAILABLE:
                logger.error("AWQ requires VLLM to be available")
                return torch.matmul(input_tensor, weight)
            
            # 从压缩权重中提取AWQ组件
            qweight = compressed_weight.data.get('qweight')
            qzeros = compressed_weight.data.get('qzeros')
            scales = compressed_weight.data.get('scales')
            qweight_scale = compressed_weight.data.get('qweight_scale')
            
            if qweight is None or qzeros is None or scales is None:
                return torch.matmul(input_tensor, weight)
            
            # 使用VLLM的AWQ方法
            quant_method = self.quantization_methods['awq_int4']
            
            # 创建临时层用于量化计算
            temp_layer = nn.Linear(qweight.shape[1], qweight.shape[0] * 8, bias=False)  # 4-bit packing
            temp_layer.qweight = nn.Parameter(qweight)
            temp_layer.qzeros = nn.Parameter(qzeros)
            temp_layer.scales = nn.Parameter(scales)
            if qweight_scale is not None:
                temp_layer.qweight_scale = nn.Parameter(qweight_scale)
            
            # 应用量化
            result = quant_method.apply(temp_layer, input_tensor)
            return result
            
        except Exception as e:
            logger.error(f"Error in AWQ quantization: {e}")
            return torch.matmul(input_tensor, weight)
    
    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        """前向传播 - 支持混合精度专家量化"""
        # 使用父类的路由逻辑
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            use_grouped_topk=self.use_grouped_topk,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            correction_bias=self.correction_bias,
            custom_routing_function=self.custom_routing_function,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        
        # 检查是否有专家量化配置
        if not self.expert_quant_configs:
            # 没有量化配置，使用原始EPMoE逻辑
            return super().forward(hidden_states, router_logits)
        
        # 应用混合精度专家量化
        return self._forward_with_mixed_precision(hidden_states, topk_weights, topk_ids)
    
    def _forward_with_mixed_precision(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
        """使用混合精度的前向传播"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # 初始化输出
        output = torch.zeros_like(hidden_states)
        
        # 重塑输入以匹配topk_weights和topk_ids的格式
        # topk_weights和topk_ids的形状是 [num_tokens, top_k]
        num_tokens = batch_size * seq_len
        hidden_states_reshaped = hidden_states.view(num_tokens, hidden_size)
        
        # 处理每个token的专家选择
        for i in range(num_tokens):
            # 获取当前token的专家权重和ID
            token_expert_weights = topk_weights[i]  # shape: [top_k]
            token_expert_ids = topk_ids[i]  # shape: [top_k]
            
            # 处理每个选中的专家
            for j in range(self.top_k):
                expert_weight = token_expert_weights[j]
                expert_id = token_expert_ids[j].item()
                
                if expert_weight == 0:
                    continue
                
                # 检查专家是否在当前分区
                if expert_id < self.start_expert_id or expert_id > self.end_expert_id:
                    continue
                
                # 获取专家在分区内的索引
                local_expert_id = expert_id - self.start_expert_id
                
                # 获取token的隐藏状态
                token_hidden = hidden_states_reshaped[i:i+1, :]  # shape: [1, hidden_size]
                
                # 应用专家量化
                expert_output = self._apply_expert_quantization(
                    expert_id, "w13_weight", 
                    self.w13_weight[local_expert_id], 
                    token_hidden
                )
                
                # 应用激活函数
                if self.activation == "silu":
                    expert_output = torch.nn.functional.silu(expert_output)
                elif self.activation == "gelu":
                    expert_output = torch.nn.functional.gelu(expert_output)
                
                # 应用down projection
                expert_output = self._apply_expert_quantization(
                    expert_id, "w2_weight", 
                    self.w2_weight[local_expert_id], 
                    expert_output
                )
                
                # 加权累加到输出
                output.view(num_tokens, hidden_size)[i:i+1, :] += expert_weight * expert_output
        
        return output


def replace_epmoe_with_mixed_precision(model: nn.Module, mixed_precision_loader) -> nn.Module:
    """将模型中的EPMoE模块替换为混合精度EPMoE模块"""
    logger.info("Replacing EPMoE modules with mixed precision EPMoE modules...")
    
    replaced_count = 0
    
    for name, module in model.named_modules():
        # 检查是否是EPMoE模块
        if isinstance(module, EPMoE) and not isinstance(module, MixedPrecisionEPMoE):
            logger.info(f"Found EPMoE module: {name}")
            
            # 创建混合精度EPMoE模块
            # 从权重形状推断hidden_size
            hidden_size = module.w13_weight.shape[-1]  # w13_weight: [num_experts, 2*intermediate_size, hidden_size]
            intermediate_size = module.intermediate_size
            
            mixed_epmoe = MixedPrecisionEPMoE(
                num_experts=module.num_experts,
                top_k=module.top_k,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                layer_id=module.layer_id,
                params_dtype=module.w13_weight.dtype,
                renormalize=module.renormalize,
                use_grouped_topk=module.use_grouped_topk,
                num_expert_group=module.num_expert_group,
                num_fused_shared_experts=0,
                topk_group=module.topk_group,
                quant_config=getattr(module, 'quant_config', None),
                tp_size=module.tp_size,
                prefix=getattr(module, 'prefix', ''),
                correction_bias=getattr(module, 'correction_bias', None),
                custom_routing_function=getattr(module, 'custom_routing_function', None),
                activation=getattr(module, 'activation', 'silu'),
                routed_scaling_factor=getattr(module, 'routed_scaling_factor', None),
                use_per_token_if_dynamic=getattr(module, 'use_per_token_if_dynamic', True),
            )
            
            # 复制原始权重
            mixed_epmoe.w13_weight.data = module.w13_weight.data.clone()
            mixed_epmoe.w2_weight.data = module.w2_weight.data.clone()
            if hasattr(module, 'w2_input_scale'):
                mixed_epmoe.w2_input_scale.data = module.w2_input_scale.data.clone()
            if hasattr(module, 'w2_weight_scale'):
                mixed_epmoe.w2_weight_scale.data = module.w2_weight_scale.data.clone()
            
            # 设置专家量化配置
            expert_quant_configs = {}
            for weight_name, precision in mixed_precision_loader.mixed_precision_config.weight_mapping.items():
                if "experts" in weight_name and "w13_weight" in weight_name:
                    # 提取专家ID
                    parts = weight_name.split('.')
                    for i, part in enumerate(parts):
                        if part == "experts" and i + 1 < len(parts):
                            expert_id = int(parts[i + 1])
                            if expert_id not in expert_quant_configs:
                                expert_quant_configs[expert_id] = ExpertQuantizationConfig(
                                    w13_precision=precision,
                                    w2_precision="fp16"  # 默认FP16
                                )
                            else:
                                expert_quant_configs[expert_id].w13_precision = precision
                            break
                elif "experts" in weight_name and "w2_weight" in weight_name:
                    # 提取专家ID
                    parts = weight_name.split('.')
                    for i, part in enumerate(parts):
                        if part == "experts" and i + 1 < len(parts):
                            expert_id = int(parts[i + 1])
                            if expert_id not in expert_quant_configs:
                                expert_quant_configs[expert_id] = ExpertQuantizationConfig(
                                    w13_precision="fp16",  # 默认FP16
                                    w2_precision=precision
                                )
                            else:
                                expert_quant_configs[expert_id].w2_precision = precision
                            break
            
            # 设置量化配置
            for expert_id, config in expert_quant_configs.items():
                mixed_epmoe.set_expert_quantization_config(expert_id, config)
            
            # 替换模块
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent_module = model
                for part in parent_name.split('.'):
                    if hasattr(parent_module, part):
                        parent_module = getattr(parent_module, part)
                    else:
                        logger.error(f"Parent module {parent_name} not found")
                        break
                else:
                    setattr(parent_module, child_name, mixed_epmoe)
                    replaced_count += 1
            else:
                setattr(model, child_name, mixed_epmoe)
                replaced_count += 1
    
    logger.info(f"EPMoE module replacement completed: {replaced_count} modules replaced")
    return model
