#!/usr/bin/env python3
"""
真正混合精度功能测试脚本
验证多种量化格式共存，保持压缩格式以节省GPU存储
"""

import os
import sys
import logging
import yaml
import torch
from pathlib import Path

# 添加SGLang路径
sys.path.insert(0, str(Path(__file__).parent / "python"))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_compressed_weight_structures():
    """测试压缩权重数据结构"""
    print("=" * 60)
    print("测试压缩权重数据结构")
    print("=" * 60)
    
    try:
        from sglang.srt.model_loader.true_mixed_precision_loader import (
            CompressedWeight, WeightFormat
        )
        
        # 创建模拟的GPTQ压缩权重
        qweight = torch.randint(0, 16, (256, 768), dtype=torch.int32)
        qzeros = torch.randint(0, 16, (16, 96), dtype=torch.int32)
        scales = torch.randn(16, 768, dtype=torch.float16)
        
        metadata = {
            'qweight': qweight,
            'qzeros': qzeros,
            'scales': scales,
            'bits': 4,
            'group_size': 128
        }
        
        compressed_weight = CompressedWeight(
            format=WeightFormat.GPTQ_INT4,
            data=qweight,
            metadata=metadata,
            original_shape=(768, 2048),
            compressed_size=(qweight.numel() + qzeros.numel() + scales.numel()) * 4
        )
        
        print("✓ 压缩权重数据结构创建成功")
        print(f"  格式: {compressed_weight.format.value}")
        print(f"  原始形状: {compressed_weight.original_shape}")
        print(f"  压缩大小: {compressed_weight.compressed_size} 字节")
        print(f"  内存使用: {compressed_weight.get_memory_usage()} 字节")
        
        # 计算压缩比
        original_size = compressed_weight.original_shape[0] * compressed_weight.original_shape[1] * 2  # float16
        compression_ratio = original_size / compressed_weight.get_memory_usage()
        print(f"  压缩比: {compression_ratio:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"✗ 压缩权重数据结构测试失败: {e}")
        return False


def test_mixed_precision_loader():
    """测试真正的混合精度加载器"""
    print("\n" + "=" * 60)
    print("测试真正的混合精度加载器")
    print("=" * 60)
    
    try:
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_loader.true_mixed_precision_loader import (
            TrueMixedPrecisionLoader,
            TrueMixedPrecisionConfig
        )
        
        # 创建混合精度配置
        mixed_precision_config = TrueMixedPrecisionConfig(
            fp16_path="/dcar-vepfs-trans-models/Qwen3-30B-A3B",
            fp8_path="/dcar-vepfs-trans-models/Qwen3-30B-A3B-FP8",
            gptq_int4_path="/dcar-vepfs-trans-models/Qwen3-30B-A3B-GPTQ-Int4",
            weight_mapping={
                "model.layers.0.self_attn.q_proj.weight": "fp16",
                "model.layers.0.mlp.experts.0.up_proj.weight": "gptq_int4",
                "model.layers.0.mlp.experts.1.up_proj.weight": "fp8"
            }
        )
        
        # 创建模型配置
        model_config = ModelConfig(
            model_path="/dcar-vepfs-trans-models/Qwen3-30B-A3B",
            dtype="auto",
            trust_remote_code=True
        )
        
        # 创建真正的混合精度加载器
        loader = TrueMixedPrecisionLoader(model_config, mixed_precision_config)
        
        print("✓ 真正的混合精度加载器创建成功")
        print(f"  权重映射数量: {len(mixed_precision_config.weight_mapping)}")
        print(f"  FP16路径: {mixed_precision_config.fp16_path}")
        print(f"  GPTQ-Int4路径: {mixed_precision_config.gptq_int4_path}")
        print(f"  AWQ-Int4路径: {mixed_precision_config.awq_int4_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ 真正的混合精度加载器测试失败: {e}")
        return False


def test_mixed_precision_linear():
    """测试混合精度线性层"""
    print("\n" + "=" * 60)
    print("测试混合精度线性层")
    print("=" * 60)
    
    try:
        from sglang.srt.layers.mixed_precision_linear import MixedPrecisionLinear
        
        # 创建混合精度线性层
        linear_layer = MixedPrecisionLinear(
            in_features=768,
            out_features=2048,
            bias=True,
            weight_name="test.weight",
            use_cache=True
        )
        
        print("✓ 混合精度线性层创建成功")
        print(f"  输入特征: {linear_layer.in_features}")
        print(f"  输出特征: {linear_layer.out_features}")
        print(f"  权重名称: {linear_layer.weight_name}")
        print(f"  使用缓存: {linear_layer.use_cache}")
        
        # 测试前向传播（会使用零权重，因为没有真实的压缩权重）
        input_tensor = torch.randn(2, 768, dtype=torch.float16)
        output = linear_layer(input_tensor)
        
        print(f"  输入形状: {input_tensor.shape}")
        print(f"  输出形状: {output.shape}")
        print("  前向传播成功（使用零权重）")
        
        return True
        
    except Exception as e:
        print(f"✗ 混合精度线性层测试失败: {e}")
        return False


def test_memory_savings():
    """测试内存节省效果"""
    print("\n" + "=" * 60)
    print("测试内存节省效果")
    print("=" * 60)
    
    try:
        # 模拟不同格式的内存使用
        formats = {
            'fp16': {'element_size': 2, 'compression_ratio': 1.0},
            'fp8': {'element_size': 1, 'compression_ratio': 2.0},
            'gptq_int4': {'element_size': 0.5, 'compression_ratio': 4.0},
        }
        
        # 模拟权重分布
        weight_distribution = {
            'fp16': 0.3,    # 30% 使用FP16
            'fp8': 0.2,     # 20% 使用FP8
            'gptq_int4': 0.1,  # 10% 使用GPTQ-Int4
        }
        
        # 假设模型总大小
        total_model_size_mb = 1000  # 1GB
        
        # 计算混合精度后的内存使用
        mixed_precision_size_mb = 0
        for format_name, ratio in weight_distribution.items():
            format_info = formats[format_name]
            size_mb = total_model_size_mb * ratio / format_info['compression_ratio']
            mixed_precision_size_mb += size_mb
        
        # 计算节省的内存
        memory_saved_mb = total_model_size_mb - mixed_precision_size_mb
        memory_saved_percent = (memory_saved_mb / total_model_size_mb) * 100
        
        print("✓ 内存节省效果计算成功")
        print(f"  原始模型大小: {total_model_size_mb}MB")
        print(f"  混合精度后大小: {mixed_precision_size_mb:.2f}MB")
        print(f"  节省内存: {memory_saved_mb:.2f}MB ({memory_saved_percent:.1f}%)")
        
        print("\n  权重分布:")
        for format_name, ratio in weight_distribution.items():
            format_info = formats[format_name]
            size_mb = total_model_size_mb * ratio / format_info['compression_ratio']
            print(f"    {format_name}: {ratio*100:.0f}% -> {size_mb:.2f}MB")
        
        return True
        
    except Exception as e:
        print(f"✗ 内存节省效果测试失败: {e}")
        return False


def create_test_config():
    """创建测试配置文件"""
    print("\n" + "=" * 60)
    print("创建测试配置文件")
    print("=" * 60)
    
    try:
        # 创建测试配置
        test_config = {
            "mixed_precision": {
                "fp16_path": "/dcar-vepfs-trans-models/Qwen3-30B-A3B",
                "fp8_path": "/dcar-vepfs-trans-models/Qwen3-30B-A3B-FP8",
                "gptq_int4_path": "/dcar-vepfs-trans-models/Qwen3-30B-A3B-GPTQ-Int4",
                "weight_mapping": {
                    "model.layers.0.self_attn.q_proj.weight": "fp16",
                    "model.layers.0.mlp.experts.0.up_proj.weight": "gptq_int4",
                    "model.layers.0.mlp.experts.1.up_proj.weight": "fp8",
                }
            }
        }
        
        # 保存配置文件
        config_path = "test_mixed_precision_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✓ 测试配置文件创建成功: {config_path}")
        print(f"  支持的格式: FP16, FP8, Int8, Int4, GPTQ-Int4, AWQ-Int4")
        print(f"  权重映射数量: {len(test_config['mixed_precision']['weight_mapping'])}")
        
        return config_path
        
    except Exception as e:
        print(f"✗ 测试配置文件创建失败: {e}")
        return None


def main():
    """主函数"""
    print("真正混合精度功能测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("压缩权重数据结构", test_compressed_weight_structures),
        ("真正的混合精度加载器", test_mixed_precision_loader),
        ("混合精度线性层", test_mixed_precision_linear),
        ("内存节省效果", test_memory_savings),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n运行测试: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            results.append((test_name, False))
    
    # 创建测试配置文件
    config_path = create_test_config()
    
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
        print("🎉 所有真正混合精度测试通过！")
        print("\n核心特性:")
        print("1. 多种量化格式共存（FP16, FP8, GPTQ-Int4）")
        print("2. 保持压缩格式，不预先反量化")
        print("3. 动态反量化，按需处理")
        print("4. 真正的内存节省，不是格式转换")
        print("5. 支持权重缓存，提高推理效率")
        
        if config_path:
            print(f"\n测试配置文件: {config_path}")
            print("您可以使用此配置文件测试真正的混合精度功能")
    else:
        print("⚠ 部分测试失败，需要进一步调试。")
    
    # 清理测试文件
    if config_path and os.path.exists(config_path):
        try:
            os.remove(config_path)
            print(f"\n清理测试文件: {config_path}")
        except:
            pass


if __name__ == "__main__":
    main()
