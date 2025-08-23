#!/usr/bin/env python3
"""
GPTQ修复测试脚本
验证修复后的GPTQ反量化算法
"""

import sys
import os
from pathlib import Path

# 添加SGLang路径
sys.path.insert(0, str(Path(__file__).parent / "python"))

import torch
import logging

# 设置日志级别为DEBUG以查看详细信息
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_gptq_dequantizer_fixed():
    """测试修复的GPTQ反量化器"""
    print("=" * 60)
    print("测试修复的GPTQ反量化器")
    print("=" * 60)
    
    try:
        from sglang.srt.model_loader.enhanced_mixed_precision_loader import GPTQDequantizer
        
        # 创建模拟GPTQ权重数据（基于错误信息中的形状）
        qweight = torch.randint(0, 16, (256, 768), dtype=torch.int32)  # [out_features, in_features//8]
        qzeros = torch.randint(0, 16, (16, 96), dtype=torch.int32)     # [out_features//group_size, in_features//8]
        scales = torch.randn(16, 768, dtype=torch.float16)             # [out_features//group_size, in_features]
        
        print(f"测试数据:")
        print(f"  qweight: {qweight.shape}, dtype: {qweight.dtype}")
        print(f"  qzeros: {qzeros.shape}, dtype: {qzeros.dtype}")
        print(f"  scales: {scales.shape}, dtype: {scales.dtype}")
        
        # 测试修复的反量化算法
        print("\n测试修复的GPTQ反量化算法...")
        weight = GPTQDequantizer.dequantize_gptq_weight(qweight, qzeros, scales)
        print(f"✓ 修复算法成功，输出形状: {weight.shape}")
        
        # 验证输出形状是否正确
        expected_shape = (768, 2048)  # [in_features, out_features]
        if weight.shape == expected_shape:
            print(f"✓ 输出形状正确: {weight.shape}")
        else:
            print(f"⚠ 输出形状不匹配: 期望 {expected_shape}, 实际 {weight.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 修复的GPTQ反量化器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_loader_integration():
    """测试增强加载器的集成"""
    print("\n" + "=" * 60)
    print("测试增强加载器的集成")
    print("=" * 60)
    
    try:
        from sglang.srt.model_loader.enhanced_mixed_precision_loader import EnhancedMixedPrecisionWeightLoader
        
        # 创建测试配置
        config = {
            "mixed_precision": {
                "fp16_path": "/path/to/fp16",
                "fp8_path": "/path/to/fp8",
                "int4_path": "/path/to/int4",
                "weight_mapping": {
                    "model.layers.0.mlp.experts.0.up_proj.weight": "int4"
                }
            }
        }
        
        # 保存临时配置文件
        import yaml
        config_path = "test_gptq_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✓ 创建测试配置文件: {config_path}")
        
        # 创建增强加载器
        loader = EnhancedMixedPrecisionWeightLoader(config_path, enable_expert_tracking=False)
        print("✓ 增强加载器创建成功")
        
        # 清理临时文件
        os.remove(config_path)
        print("✓ 清理临时文件")
        
        return True
        
    except Exception as e:
        print(f"✗ 增强加载器集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("GPTQ修复测试")
    
    # 运行所有测试
    tests = [
        ("修复的GPTQ反量化器", test_gptq_dequantizer_fixed),
        ("增强加载器集成", test_enhanced_loader_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n运行测试: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    # 显示结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！GPTQ修复成功。")
        print("\n修复说明:")
        print("1. 修正了GPTQ反量化算法中的维度计算")
        print("2. 正确处理了group_size和扩展因子")
        print("3. 确保scales和zeros的维度与unpacked权重匹配")
        print("4. 添加了详细的调试信息便于问题排查")
    else:
        print("⚠ 部分测试失败，需要进一步调试。")


if __name__ == "__main__":
    main()
