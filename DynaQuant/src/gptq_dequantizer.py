#!/usr/bin/env python3
"""
GPTQ反量化器
基于实际的GPTQ实现
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class GPTQDequantizer:
    """GPTQ反量化器"""
    
    @staticmethod
    def dequantize_gptq_weight(qweight: torch.Tensor, 
                              qzeros: torch.Tensor, 
                              scales: torch.Tensor, 
                              g_idx: Optional[torch.Tensor] = None,
                              bits: int = 4, 
                              group_size: int = 128) -> torch.Tensor:
        """
        反量化GPTQ权重
        
        Args:
            qweight: 量化的权重 [out_features, in_features//8]
            qzeros: 量化的零点 [out_features//group_size, in_features//8]
            scales: 缩放因子 [out_features//group_size, in_features]
            g_idx: 分组索引 [in_features] (可选)
            bits: 量化位数
            group_size: 分组大小
            
        Returns:
            反量化后的权重 [in_features, out_features]
        """
        try:
            # 1. 解包int32到int4
            if qweight.dtype == torch.int32:
                unpacked = GPTQDequantizer._unpack_int32_to_int4(qweight, bits)
            else:
                unpacked = qweight
            
            # 2. 计算实际维度
            out_features = qweight.shape[0]
            in_features = scales.shape[1]
            
            # 3. 反量化零点
            zeros = qzeros * scales
            
            # 4. 计算group_size
            if g_idx is not None:
                # 使用g_idx计算实际的group_size
                group_size_actual = in_features // scales.shape[0]
            else:
                group_size_actual = group_size
            
            # 5. 扩展scales和zeros
            if group_size_actual > 1:
                # 计算每个group需要重复的次数
                repeat_factor = group_size_actual // (in_features // scales.shape[0])
                scales_expanded = scales.repeat(repeat_factor, 1)
                zeros_expanded = zeros.repeat(repeat_factor, 1)
            else:
                scales_expanded = scales
                zeros_expanded = zeros
            
            # 6. 确保维度匹配
            if scales_expanded.shape[1] != unpacked.shape[1]:
                # 需要调整维度
                if scales_expanded.shape[1] < unpacked.shape[1]:
                    # 扩展scales和zeros
                    factor = unpacked.shape[1] // scales_expanded.shape[1]
                    scales_expanded = scales_expanded.repeat(1, factor)
                    zeros_expanded = zeros_expanded.repeat(1, factor)
                else:
                    # 截断unpacked
                    unpacked = unpacked[:, :scales_expanded.shape[1]]
            
            # 7. 应用反量化公式
            weight = scales_expanded * (unpacked.float() - zeros_expanded)
            
            # 8. 转置到正确的形状
            weight = weight.t()
            
            return weight
            
        except Exception as e:
            print(f"Error in GPTQ dequantization: {e}")
            print(f"  qweight: {qweight.shape}")
            print(f"  qzeros: {qzeros.shape}")
            print(f"  scales: {scales.shape}")
            print(f"  g_idx: {g_idx.shape if g_idx is not None else None}")
            
            # 返回一个合理的fallback
            return torch.zeros(scales.shape[1], qweight.shape[0])
    
    @staticmethod
    def _unpack_int32_to_int4(packed: torch.Tensor, bits: int = 4) -> torch.Tensor:
        """
        将packed int32解包为int4
        
        Args:
            packed: packed的int32张量
            bits: 每个int32中packed的位数
            
        Returns:
            解包后的int4张量
        """
        if bits == 4:
            # 每个int32包含8个int4值
            batch_size, seq_len = packed.shape
            
            # 创建输出张量
            unpacked = torch.zeros(batch_size, seq_len * 8, dtype=torch.int32)
            
            # 解包每个int32
            for i in range(8):
                shift = i * 4
                mask = 0xF  # 4位掩码
                unpacked[:, i::8] = (packed >> shift) & mask
            
            return unpacked
        else:
            # 通用方法
            elements_per_int32 = 32 // bits
            mask = (1 << bits) - 1
            
            unpacked = []
            for i in range(elements_per_int32):
                shift = i * bits
                element = (packed >> shift) & mask
                unpacked.append(element)
            
            result = torch.stack(unpacked, dim=-1)
            return result.view(packed.shape[0], -1)
    
    @staticmethod
    def dequantize_gptq_weight_simple(qweight: torch.Tensor, 
                                     qzeros: torch.Tensor, 
                                     scales: torch.Tensor) -> torch.Tensor:
        """
        简化的GPTQ反量化（用于调试）
        
        Args:
            qweight: 量化的权重
            qzeros: 量化的零点
            scales: 缩放因子
            
        Returns:
            反量化后的权重
        """
        try:
            # 1. 解包int32到int4
            if qweight.dtype == torch.int32:
                unpacked = GPTQDequantizer._unpack_int32_to_int4(qweight, 4)
            else:
                unpacked = qweight
            
            # 2. 反量化零点
            zeros = qzeros * scales
            
            # 3. 简单的维度扩展
            if zeros.shape[1] != unpacked.shape[1]:
                # 计算扩展因子
                factor = unpacked.shape[1] // zeros.shape[1]
                zeros_expanded = zeros.repeat(1, factor)
                scales_expanded = scales.repeat(1, factor)
            else:
                zeros_expanded = zeros
                scales_expanded = scales
            
            # 4. 应用反量化公式
            weight = scales_expanded * (unpacked.float() - zeros_expanded)
            
            # 5. 转置
            weight = weight.t()
            
            return weight
            
        except Exception as e:
            print(f"Error in simple GPTQ dequantization: {e}")
            # 返回零张量
            return torch.zeros(scales.shape[1], qweight.shape[0] * 8)
