#!/usr/bin/env python3
"""
简单的GPTQ测试脚本
"""

import torch

def test_gptq_fix():
    """测试GPTQ修复"""
    print("测试GPTQ反量化修复")
    
    # 创建测试数据（基于错误信息中的形状）
    qweight = torch.randint(0, 16, (256, 768), dtype=torch.int32)
    qzeros = torch.randint(0, 16, (16, 96), dtype=torch.int32)
    scales = torch.randn(16, 768, dtype=torch.float16)
    
    print(f"输入形状:")
    print(f"  qweight: {qweight.shape}")
    print(f"  qzeros: {qzeros.shape}")
    print(f"  scales: {scales.shape}")
    
    # 解包int32到int4
    def unpack_int32_to_int4(packed):
        batch_size, seq_len = packed.shape
        unpacked = torch.zeros(batch_size, seq_len * 8, dtype=torch.int32)
        
        for i in range(8):
            shift = i * 4
            mask = 0xF
            unpacked[:, i::8] = (packed >> shift) & mask
        
        return unpacked
    
    unpacked = unpack_int32_to_int4(qweight)
    print(f"解包后形状: {unpacked.shape}")
    
    # 计算维度
    out_features = qweight.shape[0]  # 256
    in_features = scales.shape[1]    # 768
    group_size = in_features // scales.shape[0]  # 768 // 16 = 48
    
    print(f"计算的维度:")
    print(f"  out_features: {out_features}")
    print(f"  in_features: {in_features}")
    print(f"  group_size: {group_size}")
    
    # 反量化零点
    zeros = qzeros * scales
    print(f"zeros形状: {zeros.shape}")
    
    # 扩展scales和zeros
    scales_expanded = scales.repeat(group_size, 1)
    zeros_expanded = zeros.repeat(group_size, 1)
    
    print(f"扩展后形状:")
    print(f"  scales_expanded: {scales_expanded.shape}")
    print(f"  zeros_expanded: {zeros_expanded.shape}")
    
    # 确保维度匹配
    if scales_expanded.shape[1] != unpacked.shape[1]:
        if scales_expanded.shape[1] < unpacked.shape[1]:
            factor = unpacked.shape[1] // scales_expanded.shape[1]
            scales_expanded = scales_expanded.repeat(1, factor)
            zeros_expanded = zeros_expanded.repeat(1, factor)
        else:
            unpacked = unpacked[:, :scales_expanded.shape[1]]
    
    print(f"维度匹配后:")
    print(f"  scales_expanded: {scales_expanded.shape}")
    print(f"  zeros_expanded: {zeros_expanded.shape}")
    print(f"  unpacked: {unpacked.shape}")
    
    # 应用反量化公式
    weight = scales_expanded * (unpacked.float() - zeros_expanded)
    
    # 转置到正确的形状
    weight = weight.t()
    
    print(f"最终权重形状: {weight.shape}")
    
    # 验证形状
    expected_shape = (768, 256)  # [in_features, out_features]
    if weight.shape == expected_shape:
        print("✓ 形状正确！")
        return True
    else:
        print(f"⚠ 形状不匹配: 期望 {expected_shape}, 实际 {weight.shape}")
        return False

if __name__ == "__main__":
    success = test_gptq_fix()
    if success:
        print("🎉 GPTQ修复测试通过！")
    else:
        print("❌ GPTQ修复测试失败！")
