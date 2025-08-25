#!/usr/bin/env python3
"""
混合精度线性层
使用SGLang的量化kernel，避免de-quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any

# SGLang量化支持导入 - 使用实际可用的类
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.linear import LinearBase, LinearMethodBase

# 尝试导入vllm的量化方法（如果可用）
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

logger = logging.getLogger(__name__)


class MixedPrecisionLinear(LinearBase):
    """混合精度线性层 - 使用SGLang的量化kernel"""
    
    def __init__(self, input_size: int, output_size: int, bias: bool = True, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__(input_size, output_size, bias, device, dtype)
        self.compressed_weight = None
        self.weight_format = None
        self.quantization_method = None
        self._init_quantization_methods()

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
        if VLLM_AVAILABLE:
            self.quantization_methods['gptq_int4'] = GPTQLinearMethod()
            self.quantization_methods['awq_int4'] = AWQLinearMethod()
        else:
            logger.warning("VLLM not available, GPTQ and AWQ quantization methods will not work")
            # 添加占位符
            self.quantization_methods['gptq_int4'] = GPTQLinearMethod()
            self.quantization_methods['awq_int4'] = AWQLinearMethod()

    def set_compressed_weight(self, compressed_weight):
        """设置压缩权重"""
        self.compressed_weight = compressed_weight
        self.weight_format = compressed_weight.format.value
        if self.weight_format in self.quantization_methods:
            self.quantization_method = self.quantization_methods[self.weight_format]
        else:
            logger.warning(f"No quantization method found for format: {self.weight_format}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播 - 使用SGLang的量化kernel"""
        if self.compressed_weight is None:
            return F.linear(input, self.weight, self.bias)
        
        if self.quantization_method is not None:
            return self._forward_with_quantization(input)
        else:
            logger.warning(f"Using fallback linear for format: {self.weight_format}")
            return F.linear(input, self.weight, self.bias)

    def _forward_with_quantization(self, input: torch.Tensor) -> torch.Tensor:
        """使用量化kernel进行前向传播"""
        try:
            if self.weight_format == 'fp8':
                return self._forward_fp8(input)
            elif self.weight_format == 'gptq_int4':
                return self._forward_gptq(input)
            elif self.weight_format == 'awq_int4':
                return self._forward_awq(input)
            else:
                logger.warning(f"Unsupported weight format: {self.weight_format}")
                return F.linear(input, self.weight, self.bias)
        except Exception as e:
            logger.error(f"Error in quantized forward pass: {e}")
            return F.linear(input, self.weight, self.bias)

    def _forward_fp8(self, input: torch.Tensor) -> torch.Tensor:
        """FP8量化前向传播"""
        try:
            # 从压缩权重中提取FP8数据
            weight_data = self.compressed_weight.data.get('weight')
            scale_inv = self.compressed_weight.data.get('scale_inv')
            
            if weight_data is None:
                logger.error("FP8 weight data not found")
                return F.linear(input, self.weight, self.bias)
            
            # 设置临时参数（SGLang的LinearMethod需要这些作为layer的属性）
            original_weight = self.weight
            original_scale_inv = getattr(self, 'weight_scale_inv', None)
            
            self.weight = nn.Parameter(weight_data)
            if scale_inv is not None:
                self.weight_scale_inv = nn.Parameter(scale_inv)
            
            # 调用SGLang的FP8线性方法
            result = self.quantization_method.apply(self, input)
            
            # 恢复原始参数
            self.weight = original_weight
            if original_scale_inv is not None:
                self.weight_scale_inv = original_scale_inv
            elif hasattr(self, 'weight_scale_inv'):
                delattr(self, 'weight_scale_inv')
            
            return result
            
        except Exception as e:
            logger.error(f"Error in FP8 forward pass: {e}")
            return F.linear(input, self.weight, self.bias)

    def _forward_gptq(self, input: torch.Tensor) -> torch.Tensor:
        """GPTQ量化前向传播"""
        try:
            if not VLLM_AVAILABLE:
                logger.error("GPTQ requires VLLM to be available")
                return F.linear(input, self.weight, self.bias)
            
            # 从压缩权重中提取GPTQ组件
            qweight = self.compressed_weight.data.get('qweight')
            qzeros = self.compressed_weight.data.get('qzeros')
            scales = self.compressed_weight.data.get('scales')
            g_idx = self.compressed_weight.data.get('g_idx')
            
            if qweight is None or qzeros is None or scales is None:
                logger.error("GPTQ components not found")
                return F.linear(input, self.weight, self.bias)
            
            # 设置临时参数
            original_weight = self.weight
            original_qweight = getattr(self, 'qweight', None)
            original_qzeros = getattr(self, 'qzeros', None)
            original_scales = getattr(self, 'scales', None)
            original_g_idx = getattr(self, 'g_idx', None)
            
            self.qweight = nn.Parameter(qweight)
            self.qzeros = nn.Parameter(qzeros)
            self.scales = nn.Parameter(scales)
            if g_idx is not None:
                self.g_idx = nn.Parameter(g_idx)
            
            # 调用VLLM的GPTQ线性方法
            result = self.quantization_method.apply(self, input)
            
            # 恢复原始参数
            self.weight = original_weight
            if original_qweight is not None:
                self.qweight = original_qweight
            else:
                delattr(self, 'qweight')
            
            if original_qzeros is not None:
                self.qzeros = original_qzeros
            else:
                delattr(self, 'qzeros')
            
            if original_scales is not None:
                self.scales = original_scales
            else:
                delattr(self, 'scales')
            
            if original_g_idx is not None:
                self.g_idx = original_g_idx
            elif hasattr(self, 'g_idx'):
                delattr(self, 'g_idx')
            
            return result
            
        except Exception as e:
            logger.error(f"Error in GPTQ forward pass: {e}")
            return F.linear(input, self.weight, self.bias)

    def _forward_awq(self, input: torch.Tensor) -> torch.Tensor:
        """AWQ量化前向传播"""
        try:
            if not VLLM_AVAILABLE:
                logger.error("AWQ requires VLLM to be available")
                return F.linear(input, self.weight, self.bias)
            
            # 从压缩权重中提取AWQ组件
            qweight = self.compressed_weight.data.get('qweight')
            qzeros = self.compressed_weight.data.get('qzeros')
            scales = self.compressed_weight.data.get('scales')
            qweight_scale = self.compressed_weight.data.get('qweight_scale')
            
            if qweight is None or qzeros is None or scales is None:
                logger.error("AWQ components not found")
                return F.linear(input, self.weight, self.bias)
            
            # 设置临时参数
            original_weight = self.weight
            original_qweight = getattr(self, 'qweight', None)
            original_qzeros = getattr(self, 'qzeros', None)
            original_scales = getattr(self, 'scales', None)
            original_qweight_scale = getattr(self, 'qweight_scale', None)
            
            self.qweight = nn.Parameter(qweight)
            self.qzeros = nn.Parameter(qzeros)
            self.scales = nn.Parameter(scales)
            if qweight_scale is not None:
                self.qweight_scale = nn.Parameter(qweight_scale)
            
            # 调用VLLM的AWQ线性方法
            result = self.quantization_method.apply(self, input)
            
            # 恢复原始参数
            self.weight = original_weight
            if original_qweight is not None:
                self.qweight = original_qweight
            else:
                delattr(self, 'qweight')
            
            if original_qzeros is not None:
                self.qzeros = original_qzeros
            else:
                delattr(self, 'qzeros')
            
            if original_scales is not None:
                self.scales = original_scales
            else:
                delattr(self, 'scales')
            
            if original_qweight_scale is not None:
                self.qweight_scale = original_qweight_scale
            elif hasattr(self, 'qweight_scale'):
                delattr(self, 'qweight_scale')
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AWQ forward pass: {e}")
            return F.linear(input, self.weight, self.bias)

    def get_memory_usage(self) -> int:
        """获取内存使用量（字节）"""
        if self.compressed_weight:
            return self.compressed_weight.get_memory_usage()
        return self.weight.numel() * self.weight.element_size()

    def get_compression_ratio(self) -> float:
        """获取压缩比"""
        if self.compressed_weight:
            original_size = self.compressed_weight.original_shape[0] * self.compressed_weight.original_shape[1] * 2  # FP16
            compressed_size = self.compressed_weight.compressed_size
            return original_size / compressed_size
        return 1.0


def replace_linear_with_mixed_precision(model: nn.Module, mixed_precision_loader, use_cache: bool = True) -> nn.Module:
    """将模型中的线性层替换为混合精度线性层"""
    logger.info("Replacing linear layers with mixed precision layers...")
    
    replaced_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 检查是否有对应的混合精度权重
            weight_name = name + ".weight"
            if weight_name in mixed_precision_loader.mixed_precision_config.weight_mapping:
                precision = mixed_precision_loader.mixed_precision_config.weight_mapping[weight_name]
                
                # 创建混合精度线性层
                mixed_layer = MixedPrecisionLinear(
                    input_size=module.in_features,
                    output_size=module.out_features,
                    bias=module.bias is not None,
                    device=module.weight.device,
                    dtype=module.weight.dtype
                )
                
                # 加载压缩权重
                if use_cache and weight_name in mixed_precision_loader.compressed_weights:
                    compressed_weight = mixed_precision_loader.compressed_weights[weight_name]
                else:
                    compressed_weight = mixed_precision_loader.load_weight(weight_name, precision)
                
                if compressed_weight:
                    mixed_layer.set_compressed_weight(compressed_weight)
                    logger.info(f"Replaced {name} with mixed precision layer (precision: {precision})")
                    replaced_count += 1
                else:
                    logger.warning(f"Failed to load compressed weight for {name}")
                
                # 替换模块 - 使用正确的父模块查找方法
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    # 找到父模块
                    parent_module = model
                    for part in parent_name.split('.'):
                        if hasattr(parent_module, part):
                            parent_module = getattr(parent_module, part)
                        else:
                            logger.error(f"Parent module {parent_name} not found")
                            break
                    else:
                        # 成功找到父模块，替换子模块
                        setattr(parent_module, child_name, mixed_layer)
                else:
                    # 根级别的线性层
                    setattr(model, child_name, mixed_layer)
    
    logger.info(f"Mixed precision layer replacement completed: {replaced_count} layers replaced")
    return model
