#!/usr/bin/env python3
"""
混合精度线性层 - 使用SGLang现成的量化kernel
避免de-quantization，直接使用压缩权重进行推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

# SGLang量化支持导入
from sglang.srt.layers.quantization import (
    Fp8LinearMethod, GPTQLinearMethod, AWQLinearMethod,
    QuantizationConfig
)
from sglang.srt.layers.linear import LinearBase, LinearMethodBase

logger = logging.getLogger(__name__)


class MixedPrecisionLinear(LinearBase):
    """混合精度线性层 - 使用SGLang的量化kernel"""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(input_size, output_size, bias, device, dtype)
        
        # 存储压缩权重
        self.compressed_weight = None
        self.weight_format = None
        self.quantization_method = None
        
        # 初始化量化方法
        self._init_quantization_methods()
    
    def _init_quantization_methods(self):
        """初始化量化方法"""
        self.quantization_methods = {
            'fp8': Fp8LinearMethod(),
            'gptq_int4': GPTQLinearMethod(),
            'awq_int4': AWQLinearMethod(),
        }
    
    def set_compressed_weight(self, compressed_weight):
        """设置压缩权重"""
        self.compressed_weight = compressed_weight
        self.weight_format = compressed_weight.format.value
        
        # 根据权重格式选择量化方法
        if self.weight_format in self.quantization_methods:
            self.quantization_method = self.quantization_methods[self.weight_format]
            logger.debug(f"Set quantization method for {self.weight_format}")
        else:
            logger.warning(f"No quantization method found for format: {self.weight_format}")
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播 - 使用SGLang的量化kernel"""
        if self.compressed_weight is None:
            # 如果没有压缩权重，使用标准线性层
            return F.linear(input, self.weight, self.bias)
        
        # 使用SGLang的量化kernel
        if self.quantization_method is not None:
            return self._forward_with_quantization(input)
        else:
            # 回退到标准线性层
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
            # 回退到标准线性层
            return F.linear(input, self.weight, self.bias)
    
    def _forward_fp8(self, input: torch.Tensor) -> torch.Tensor:
        """FP8量化前向传播"""
        try:
            # 获取FP8权重和缩放因子
            weight_data = self.compressed_weight.data
            weight = weight_data['weight']
            scale_inv = weight_data.get('scale_inv', None)
            
            # 使用SGLang的FP8 kernel
            if hasattr(self.quantization_method, 'apply'):
                # 创建临时的权重参数
                temp_weight = nn.Parameter(weight)
                if scale_inv is not None:
                    temp_scale_inv = nn.Parameter(scale_inv)
                    setattr(self, '_temp_scale_inv', temp_scale_inv)
                
                # 应用量化方法
                result = self.quantization_method.apply(self, input)
                
                # 清理临时参数
                delattr(self, '_temp_scale_inv')
                return result
            else:
                # 直接使用FP8权重
                return F.linear(input, weight, self.bias)
                
        except Exception as e:
            logger.error(f"Error in FP8 forward pass: {e}")
            return F.linear(input, self.weight, self.bias)
    
    def _forward_gptq(self, input: torch.Tensor) -> torch.Tensor:
        """GPTQ量化前向传播"""
        try:
            # 获取GPTQ组件
            weight_data = self.compressed_weight.data
            qweight = weight_data['qweight']
            qzeros = weight_data['qzeros']
            scales = weight_data['scales']
            g_idx = weight_data.get('g_idx', None)
            
            # 使用SGLang的GPTQ kernel
            if hasattr(self.quantization_method, 'apply'):
                # 创建临时的权重参数
                temp_qweight = nn.Parameter(qweight)
                temp_qzeros = nn.Parameter(qzeros)
                temp_scales = nn.Parameter(scales)
                
                setattr(self, 'qweight', temp_qweight)
                setattr(self, 'qzeros', temp_qzeros)
                setattr(self, 'scales', temp_scales)
                
                if g_idx is not None:
                    temp_g_idx = nn.Parameter(g_idx)
                    setattr(self, 'g_idx', temp_g_idx)
                
                # 应用量化方法
                result = self.quantization_method.apply(self, input)
                
                # 清理临时参数
                delattr(self, 'qweight')
                delattr(self, 'qzeros')
                delattr(self, 'scales')
                if g_idx is not None:
                    delattr(self, 'g_idx')
                
                return result
            else:
                # 直接使用GPTQ权重（需要de-quantization）
                logger.warning("GPTQ kernel not available, using de-quantization")
                weight = self._dequantize_gptq(qweight, qzeros, scales, g_idx)
                return F.linear(input, weight, self.bias)
                
        except Exception as e:
            logger.error(f"Error in GPTQ forward pass: {e}")
            return F.linear(input, self.weight, self.bias)
    
    def _forward_awq(self, input: torch.Tensor) -> torch.Tensor:
        """AWQ量化前向传播"""
        try:
            # 获取AWQ组件
            weight_data = self.compressed_weight.data
            qweight = weight_data['qweight']
            qzeros = weight_data['qzeros']
            scales = weight_data['scales']
            qweight_scale = weight_data.get('qweight_scale', None)
            
            # 使用SGLang的AWQ kernel
            if hasattr(self.quantization_method, 'apply'):
                # 创建临时的权重参数
                temp_qweight = nn.Parameter(qweight)
                temp_qzeros = nn.Parameter(qzeros)
                temp_scales = nn.Parameter(scales)
                
                setattr(self, 'qweight', temp_qweight)
                setattr(self, 'qzeros', temp_qzeros)
                setattr(self, 'scales', temp_scales)
                
                if qweight_scale is not None:
                    temp_qweight_scale = nn.Parameter(qweight_scale)
                    setattr(self, 'qweight_scale', temp_qweight_scale)
                
                # 应用量化方法
                result = self.quantization_method.apply(self, input)
                
                # 清理临时参数
                delattr(self, 'qweight')
                delattr(self, 'qzeros')
                delattr(self, 'scales')
                if qweight_scale is not None:
                    delattr(self, 'qweight_scale')
                
                return result
            else:
                # 直接使用AWQ权重（需要de-quantization）
                logger.warning("AWQ kernel not available, using de-quantization")
                weight = self._dequantize_awq(qweight, qzeros, scales, qweight_scale)
                return F.linear(input, weight, self.bias)
                
        except Exception as e:
            logger.error(f"Error in AWQ forward pass: {e}")
            return F.linear(input, self.weight, self.bias)
    
    def _dequantize_gptq(self, qweight, qzeros, scales, g_idx=None):
        """GPTQ反量化（仅作为fallback）"""
        try:
            # 简单的GPTQ反量化实现
            pack = 32 // 4  # 4-bit packing
            oc_pack, ic = qweight.shape
            oc = oc_pack * pack
            
            # 解包qweight
            unpacked = torch.zeros(oc, ic, dtype=torch.int32, device=qweight.device)
            for i in range(8):
                shift = i * 4
                mask = 0xF
                unpacked[i::8, :] = (qweight >> shift) & mask
            
            # 反量化
            weight = scales * (unpacked.float() - qzeros)
            return weight.t()
            
        except Exception as e:
            logger.error(f"Error in GPTQ dequantization: {e}")
            return torch.zeros(ic, oc, device=qweight.device, dtype=torch.float16)
    
    def _dequantize_awq(self, qweight, qzeros, scales, qweight_scale=None):
        """AWQ反量化（仅作为fallback）"""
        try:
            # 简单的AWQ反量化实现
            pack = 32 // 4  # 4-bit packing
            oc_pack, ic = qweight.shape
            oc = oc_pack * pack
            
            # 解包qweight
            unpacked = torch.zeros(oc, ic, dtype=torch.int32, device=qweight.device)
            for i in range(8):
                shift = i * 4
                mask = 0xF
                unpacked[i::8, :] = (qweight >> shift) & mask
            
            # 反量化
            weight = scales * (unpacked.float() - qzeros)
            if qweight_scale is not None:
                weight = weight * qweight_scale
            return weight.t()
            
        except Exception as e:
            logger.error(f"Error in AWQ dequantization: {e}")
            return torch.zeros(ic, oc, device=qweight.device, dtype=torch.float16)
    
    def get_memory_usage(self) -> int:
        """获取内存使用量"""
        if self.compressed_weight is not None:
            return self.compressed_weight.get_memory_usage()
        else:
            return self.weight.numel() * self.weight.element_size()
    
    def get_compression_ratio(self) -> float:
        """获取压缩比"""
        if self.compressed_weight is not None:
            original_size = self.compressed_weight.original_shape[0] * self.compressed_weight.original_shape[1] * 2  # FP16
            compressed_size = self.compressed_weight.get_memory_usage()
            return original_size / compressed_size
        else:
            return 1.0
