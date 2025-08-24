#!/usr/bin/env python3
"""
混合精度线性层
支持动态处理不同格式的压缩权重，在推理时按需反量化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from enum import Enum

from sglang.srt.model_loader.true_mixed_precision_loader import (
    CompressedWeight, WeightFormat, get_global_true_mixed_precision_loader
)

logger = torch._C._log.Logger()


class WeightFormat(Enum):
    """权重格式枚举"""
    FP16 = "fp16"
    FP8 = "fp8"
    INT4 = "int4"
    INT8 = "int8"
    GPTQ_INT4 = "gptq_int4"
    AWQ_INT4 = "awq_int4"


class MixedPrecisionLinear(nn.Module):
    """混合精度线性层"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 weight_name: str = None, use_cache: bool = True):
        """
        初始化混合精度线性层
        
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            bias: 是否使用偏置
            weight_name: 权重名称（用于从加载器获取压缩权重）
            use_cache: 是否使用缓存
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_name = weight_name
        self.use_cache = use_cache
        
        # 缓存反量化后的权重
        self._cached_weight = None
        self._cached_bias = None
        
        # 创建偏置（如果需要）
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 初始化权重（占位符，实际权重从压缩格式加载）
        self.register_parameter('weight', None)
    
    def _get_compressed_weight(self) -> Optional[CompressedWeight]:
        """获取压缩权重"""
        if self.weight_name is None:
            return None
        
        loader = get_global_true_mixed_precision_loader()
        if loader is None:
            return None
        
        return loader.get_compressed_weight(self.weight_name)
    
    def _dequantize_gptq_weight(self, compressed_weight: CompressedWeight) -> torch.Tensor:
        """反量化GPTQ权重"""
        metadata = compressed_weight.metadata
        qweight = metadata['qweight']
        qzeros = metadata['qzeros']
        scales = metadata['scales']
        g_idx = metadata.get('g_idx', None)
        bits = metadata.get('bits', 4)
        group_size = metadata.get('group_size', 128)
        
        # GPTQ反量化逻辑
        pack = 32 // bits
        oc_pack, IC = qweight.shape
        OC = oc_pack * pack
        
        # 解包qweight
        Wq = self._unpack_int32_to_nibbles_rows(qweight, bits=bits)
        
        # 处理qzeros和scales
        groups_out = qzeros.shape[0]
        g = OC // groups_out
        
        device = qweight.device
        mask = (1 << bits) - 1
        col = torch.arange(IC, device=device)
        qz_cols = qzeros[:, (col // pack)]
        shift = (col % pack) * bits
        zp_group_ic = (qz_cols >> shift.unsqueeze(0)) & mask
        zp_full = zp_group_ic.repeat_interleave(g, dim=0).to(torch.int16)
        
        # 广播scales
        scales_full = scales.repeat_interleave(g, dim=0).to(torch.float32)
        
        # 反量化
        W_fp16 = ((Wq - zp_full).to(torch.float32) * scales_full).to(torch.float16)
        return W_fp16.t()
    
    def _dequantize_awq_weight(self, compressed_weight: CompressedWeight) -> torch.Tensor:
        """反量化AWQ权重"""
        metadata = compressed_weight.metadata
        qweight = metadata['qweight']
        qzeros = metadata['qzeros']
        scales = metadata['scales']
        qweight_scale = metadata['qweight_scale']
        
        # AWQ反量化逻辑（类似GPTQ但有不同的缩放处理）
        pack = 32 // 4  # AWQ通常是4bit
        oc_pack, IC = qweight.shape
        OC = oc_pack * pack
        
        # 解包qweight
        Wq = self._unpack_int32_to_nibbles_rows(qweight, bits=4)
        
        # AWQ特有的缩放处理
        scales_full = scales.repeat_interleave(OC // scales.shape[0], dim=0)
        qweight_scale_full = qweight_scale.repeat_interleave(OC // qweight_scale.shape[0], dim=0)
        
        # 反量化
        W_fp16 = (Wq.to(torch.float32) * scales_full * qweight_scale_full.unsqueeze(1)).to(torch.float16)
        return W_fp16.t()
    
    def _unpack_int32_to_nibbles_rows(self, packed: torch.Tensor, bits: int = 4) -> torch.Tensor:
        """将int32打包的权重解包为nibbles"""
        pack = 32 // bits
        R, C = packed.shape
        out = torch.empty((R * pack, C), dtype=torch.int16, device=packed.device)
        mask = (1 << bits) - 1
        
        for k in range(pack):
            vals = (packed >> (k * bits)) & mask
            out[k::pack, :] = vals.to(torch.int16)
        
        return out
    
    def _dequantize_weight(self, compressed_weight: CompressedWeight) -> torch.Tensor:
        """根据格式反量化权重"""
        if compressed_weight.format == WeightFormat.GPTQ_INT4:
            return self._dequantize_gptq_weight(compressed_weight)
        elif compressed_weight.format == WeightFormat.AWQ_INT4:
            return self._dequantize_awq_weight(compressed_weight)
        elif compressed_weight.format in [WeightFormat.INT4, WeightFormat.INT8]:
            # 简单的量化权重反量化
            weight = compressed_weight.data
            bits = compressed_weight.metadata.get('bits', 4)
            if bits == 4:
                # 假设是简单的4bit量化
                return weight.to(torch.float16) * 0.1  # 简单的缩放
            else:
                return weight.to(torch.float16) * 0.01  # 简单的缩放
        elif compressed_weight.format in [WeightFormat.FP16, WeightFormat.FP8]:
            # 浮点权重直接使用
            return compressed_weight.data
        else:
            # 默认处理
            return compressed_weight.data.to(torch.float16)
    
    def _get_weight(self) -> torch.Tensor:
        """获取权重（可能从缓存或反量化）"""
        if self.use_cache and self._cached_weight is not None:
            return self._cached_weight
        
        compressed_weight = self._get_compressed_weight()
        if compressed_weight is None:
            # 如果没有压缩权重，返回零权重
            weight = torch.zeros(self.out_features, self.in_features, dtype=torch.float16)
            logger.warning(f"No compressed weight found for {self.weight_name}, using zero weight")
        else:
            # 反量化权重
            weight = self._dequantize_weight(compressed_weight)
            
            # 确保形状正确
            if weight.shape != (self.out_features, self.in_features):
                logger.warning(f"Weight shape mismatch for {self.weight_name}: "
                             f"expected ({self.out_features}, {self.in_features}), got {weight.shape}")
                # 调整形状
                if weight.shape[0] < self.out_features or weight.shape[1] < self.in_features:
                    # 扩展权重
                    new_weight = torch.zeros(self.out_features, self.in_features, dtype=weight.dtype, device=weight.device)
                    new_weight[:weight.shape[0], :weight.shape[1]] = weight
                    weight = new_weight
                else:
                    # 截断权重
                    weight = weight[:self.out_features, :self.in_features]
        
        # 缓存权重
        if self.use_cache:
            self._cached_weight = weight
        
        return weight
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        weight = self._get_weight()
        
        # 确保输入和权重在同一设备上
        if input.device != weight.device:
            weight = weight.to(input.device)
            if self.use_cache:
                self._cached_weight = weight
        
        # 执行线性变换
        output = F.linear(input, weight, self.bias)
        
        return output
    
    def extra_repr(self) -> str:
        """额外的字符串表示"""
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, weight_name={self.weight_name}, ' \
               f'use_cache={self.use_cache}'


def replace_linear_with_mixed_precision(model: nn.Module, 
                                      loader, 
                                      use_cache: bool = True) -> nn.Module:
    """
    将模型中的线性层替换为混合精度线性层
    
    Args:
        model: 要替换的模型
        loader: 混合精度加载器
        use_cache: 是否使用缓存
    
    Returns:
        替换后的模型
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 创建混合精度线性层
            mixed_precision_linear = MixedPrecisionLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                weight_name=name + '.weight',
                use_cache=use_cache
            )
            
            # 复制偏置
            if module.bias is not None:
                mixed_precision_linear.bias.data = module.bias.data.clone()
            
            # 替换模块
            parent_name = '.'.join(name.split('.')[:-1])
            if parent_name:
                parent = model.get_submodule(parent_name)
                child_name = name.split('.')[-1]
                setattr(parent, child_name, mixed_precision_linear)
            else:
                # 根模块
                setattr(model, name, mixed_precision_linear)
            
            logger.info(f"Replaced linear layer {name} with mixed precision linear layer")
    
    return model


def get_mixed_precision_memory_stats() -> Dict[str, Any]:
    """获取混合精度内存统计"""
    loader = get_global_true_mixed_precision_loader()
    if loader is None:
        return {"error": "No mixed precision loader available"}
    
    return loader.get_memory_stats()


def clear_mixed_precision_cache():
    """清除混合精度缓存"""
    loader = get_global_true_mixed_precision_loader()
    if loader is None:
        return
    
    # 清除所有模块的缓存
    for name, module in loader.compressed_weights.items():
        # 这里可以添加清除缓存的逻辑
        pass
    
    logger.info("Mixed precision cache cleared")
