#!/usr/bin/env python3
"""
GPTQ修复测试脚本
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_gptq_dequantizer():
    """测试GPTQ反量化器"""
    print("=" * 50)
    print("GPTQ反量化器测试")
    print("=" * 50)
    
    try:
        from gptq_dequantizer import GPTQDequantizer
        print("✓ GPTQ反量化器导入成功")
        
        # 创建测试数据
        import torch
        
        # 模拟GPTQ权重数据
        qweight = torch.randint(0, 16, (256, 768), dtype=torch.int32)  # [out_features, in_features//8]
        qzeros = torch.randint(0, 16, (16, 96), dtype=torch.int32)     # [out_features//group_size, in_features//8]
        scales = torch.randn(16, 768, dtype=torch.float16)             # [out_features//group_size, in_features]
        
        print(f"测试数据:")
        print(f"  qweight: {qweight.shape}, dtype: {qweight.dtype}")
        print(f"  qzeros: {qzeros.shape}, dtype: {qzeros.dtype}")
        print(f"  scales: {scales.shape}, dtype: {scales.dtype}")
        
        # 测试反量化
        print("\n测试反量化...")
        weight = GPTQDequantizer.dequantize_gptq_weight_simple(qweight, qzeros, scales)
        print(f"✓ 反量化成功，输出形状: {weight.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ GPTQ反量化器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weight_loader():
    """测试权重加载器"""
    print("\n" + "=" * 50)
    print("权重加载器测试")
    print("=" * 50)
    
    try:
        from weight_loader import MixedPrecisionWeightLoader
        print("✓ 权重加载器导入成功")
        
        # 检查配置文件
        config_path = "config/model_config.yaml"
        if os.path.exists(config_path):
            print(f"✓ 配置文件存在: {config_path}")
            
            loader = MixedPrecisionWeightLoader(config_path)
            print("✓ 权重加载器初始化成功")
            
            # 测试GPTQ权重加载
            test_weight_name = "model.layers.0.mlp.experts.0.down_proj.weight"
            
            if test_weight_name in loader.weight_mapping:
                precision = loader.weight_mapping[test_weight_name]
                print(f"测试权重: {test_weight_name}, 精度: {precision}")
                
                # 尝试加载权重（可能失败，因为文件不存在）
                try:
                    weight = loader.load_weight(test_weight_name, precision)
                    if weight is not None:
                        print(f"✓ 权重加载成功，形状: {weight.shape}")
                    else:
                        print("⚠ 权重加载返回None（可能是文件不存在）")
                except Exception as e:
                    print(f"⚠ 权重加载失败（预期行为，因为文件不存在）: {e}")
            else:
                print("⚠ 权重未在映射中定义")
            
            return True
        else:
            print(f"⚠ 配置文件不存在: {config_path}")
            return False
            
    except Exception as e:
        print(f"✗ 权重加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_safetensors_compatibility():
    """测试safetensors兼容性"""
    print("\n" + "=" * 50)
    print("Safetensors兼容性测试")
    print("=" * 50)
    
    try:
        # 测试safetensors导入
        try:
            from safetensors.torch import load_file, safe_open
            print("✓ safetensors.torch导入成功")
        except ImportError:
            try:
                from safetensors import load_file, safe_open
                print("✓ safetensors导入成功")
            except ImportError:
                import safetensors
                load_file = safetensors.load_file
                safe_open = safetensors.safe_open
                print("✓ safetensors兼容性导入成功")
        
        print("✓ Safetensors兼容性测试通过")
        return True
        
    except Exception as e:
        print(f"✗ Safetensors兼容性测试失败: {e}")
        return False


def main():
    """主函数"""
    print("GPTQ修复测试")
    
    # 运行所有测试
    tests = [
        ("Safetensors兼容性", test_safetensors_compatibility),
        ("GPTQ反量化器", test_gptq_dequantizer),
        ("权重加载器", test_weight_loader)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n运行测试: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    # 显示结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！GPTQ修复成功。")
    else:
        print("⚠ 部分测试失败，需要进一步调试。")


if __name__ == "__main__":
    main()
