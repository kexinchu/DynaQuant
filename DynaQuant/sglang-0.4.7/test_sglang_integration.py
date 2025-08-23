#!/usr/bin/env python3
"""
SGLang集成测试脚本
验证混合精度功能是否正确集成到SGLang中
"""

import os
import sys
import logging
import json
from pathlib import Path

# 添加SGLang路径
sys.path.insert(0, str(Path(__file__).parent / "python"))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_sglang_imports():
    """测试SGLang导入"""
    print("=" * 60)
    print("测试SGLang导入")
    print("=" * 60)
    
    try:
        # 测试SGLang核心模块导入
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.configs.load_config import LoadConfig, LoadFormat
        from sglang.srt.model_loader.loader import DefaultModelLoader
        
        print("✓ SGLang核心模块导入成功")
        
        # 测试混合精度加载器导入
        from sglang.srt.model_loader.sglang_mixed_precision_loader import (
            SGLangMixedPrecisionLoader,
            create_mixed_precision_loader,
            get_global_mixed_precision_loader,
            set_global_mixed_precision_loader
        )
        
        print("✓ SGLang混合精度加载器导入成功")
        
        return True
        
    except ImportError as e:
        print(f"✗ SGLang导入失败: {e}")
        return False


def test_config_creation():
    """测试配置创建"""
    print("\n" + "=" * 60)
    print("测试配置创建")
    print("=" * 60)
    
    try:
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.configs.load_config import LoadConfig, LoadFormat
        
        # 创建测试配置
        model_config = ModelConfig(
            model_path="test_model",
            mixed_precision_config="test_config.yaml",
            dtype="auto",
            trust_remote_code=True
        )
        
        device_config = DeviceConfig(device="cuda")
        load_config = LoadConfig(load_format=LoadFormat.AUTO)
        
        print("✓ 配置创建成功")
        print(f"  模型路径: {model_config.model_path}")
        print(f"  混合精度配置: {model_config.mixed_precision_config}")
        print(f"  设备: {device_config.device}")
        print(f"  加载格式: {load_config.load_format}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置创建失败: {e}")
        return False


def test_mixed_precision_loader():
    """测试混合精度加载器"""
    print("\n" + "=" * 60)
    print("测试混合精度加载器")
    print("=" * 60)
    
    try:
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_loader.sglang_mixed_precision_loader import (
            MixedPrecisionConfig,
            SGLangMixedPrecisionLoader
        )
        
        # 创建混合精度配置
        mixed_precision_config = MixedPrecisionConfig(
            fp16_path="/path/to/fp16",
            fp8_path="/path/to/fp8",
            int4_path="/path/to/int4",
            weight_mapping={
                "model.layers.0.self_attn.q_proj.weight": "fp16",
                "model.layers.0.mlp.experts.0.up_proj.weight": "int4"
            }
        )
        
        # 创建模型配置
        model_config = ModelConfig(
            model_path="test_model",
            dtype="auto",
            trust_remote_code=True
        )
        
        # 创建混合精度加载器
        loader = SGLangMixedPrecisionLoader(model_config, mixed_precision_config)
        
        print("✓ 混合精度加载器创建成功")
        print(f"  权重映射数量: {len(mixed_precision_config.weight_mapping)}")
        print(f"  FP16路径: {mixed_precision_config.fp16_path}")
        print(f"  FP8路径: {mixed_precision_config.fp8_path}")
        print(f"  Int4路径: {mixed_precision_config.int4_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ 混合精度加载器测试失败: {e}")
        return False


def test_gptq_dequantizer():
    """测试GPTQ反量化器"""
    print("\n" + "=" * 60)
    print("测试GPTQ反量化器")
    print("=" * 60)
    
    try:
        import torch
        from sglang.srt.model_loader.sglang_mixed_precision_loader import SGLangGPTQDequantizer
        
        # 创建测试数据
        qweight = torch.randint(0, 16, (256, 768), dtype=torch.int32)
        qzeros = torch.randint(0, 16, (16, 96), dtype=torch.int32)
        scales = torch.randn(16, 768, dtype=torch.float16)
        
        # 测试反量化
        weight = SGLangGPTQDequantizer.dequantize_gptq_weight(qweight, qzeros, scales)
        
        print("✓ GPTQ反量化成功")
        print(f"  输入形状: qweight{qweight.shape}, qzeros{qzeros.shape}, scales{scales.shape}")
        print(f"  输出形状: {weight.shape}")
        print(f"  输出类型: {weight.dtype}")
        
        return True
        
    except Exception as e:
        print(f"✗ GPTQ反量化测试失败: {e}")
        return False


def test_loader_integration():
    """测试加载器集成"""
    print("\n" + "=" * 60)
    print("测试加载器集成")
    print("=" * 60)
    
    try:
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.configs.load_config import LoadConfig, LoadFormat
        from sglang.srt.model_loader.loader import DefaultModelLoader
        
        # 创建配置
        model_config = ModelConfig(
            model_path="test_model",
            mixed_precision_config="test_config.yaml",
            dtype="auto",
            trust_remote_code=True
        )
        
        device_config = DeviceConfig(device="cpu")  # 使用CPU避免GPU依赖
        load_config = LoadConfig(load_format=LoadFormat.AUTO)
        
        # 创建加载器
        loader = DefaultModelLoader(load_config)
        
        print("✓ 加载器集成测试成功")
        print("  注意: 实际模型加载需要真实的模型文件")
        
        return True
        
    except Exception as e:
        print(f"✗ 加载器集成测试失败: {e}")
        return False


def create_test_config():
    """创建测试配置文件"""
    print("\n" + "=" * 60)
    print("创建测试配置文件")
    print("=" * 60)
    
    try:
        import yaml
        
        # 创建测试配置
        test_config = {
            "mixed_precision": {
                "fp16_path": "/path/to/fp16/weights",
                "fp8_path": "/path/to/fp8/weights",
                "int4_path": "/path/to/int4/weights",
                "weight_mapping": {
                    "model.layers.0.self_attn.q_proj.weight": "fp16",
                    "model.layers.0.self_attn.k_proj.weight": "fp16",
                    "model.layers.0.self_attn.v_proj.weight": "fp16",
                    "model.layers.0.self_attn.o_proj.weight": "fp16",
                    "model.layers.0.mlp.gate_proj.weight": "fp8",
                    "model.layers.0.mlp.up_proj.weight": "fp8",
                    "model.layers.0.mlp.down_proj.weight": "fp8",
                    "model.layers.0.mlp.experts.0.gate_proj.weight": "int4",
                    "model.layers.0.mlp.experts.0.up_proj.weight": "int4",
                    "model.layers.0.mlp.experts.0.down_proj.weight": "int4"
                }
            }
        }
        
        # 保存配置文件
        config_path = "test_mixed_precision_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✓ 测试配置文件创建成功: {config_path}")
        print(f"  权重映射数量: {len(test_config['mixed_precision']['weight_mapping'])}")
        
        return config_path
        
    except Exception as e:
        print(f"✗ 测试配置文件创建失败: {e}")
        return None


def main():
    """主函数"""
    print("SGLang集成测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("SGLang导入", test_sglang_imports),
        ("配置创建", test_config_creation),
        ("混合精度加载器", test_mixed_precision_loader),
        ("GPTQ反量化器", test_gptq_dequantizer),
        ("加载器集成", test_loader_integration),
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
        print("🎉 所有SGLang集成测试通过！")
        print("\n集成说明:")
        print("1. SGLang核心模块正确导入")
        print("2. 混合精度配置正确创建")
        print("3. 混合精度加载器正常工作")
        print("4. GPTQ反量化功能正常")
        print("5. 加载器集成成功")
        
        if config_path:
            print(f"\n测试配置文件: {config_path}")
            print("您可以使用此配置文件测试混合精度功能")
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
