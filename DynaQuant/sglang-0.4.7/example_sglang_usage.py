#!/usr/bin/env python3
"""
SGLang混合精度集成使用示例
展示如何真正使用SGLang的API加载混合精度模型
"""

import os
import sys
import logging
import yaml
from pathlib import Path

# 添加SGLang路径
sys.path.insert(0, str(Path(__file__).parent / "python"))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mixed_precision_config():
    """创建混合精度配置文件"""
    config = {
        "mixed_precision": {
            "fp16_path": "/path/to/fp16/weights",
            "fp8_path": "/path/to/fp8/weights", 
            "int4_path": "/path/to/int4/weights",
            "weight_mapping": {
                # 注意力层使用FP16
                "model.layers.0.self_attn.q_proj.weight": "fp16",
                "model.layers.0.self_attn.k_proj.weight": "fp16",
                "model.layers.0.self_attn.v_proj.weight": "fp16",
                "model.layers.0.self_attn.o_proj.weight": "fp16",
                
                # MLP层使用FP8
                "model.layers.0.mlp.gate_proj.weight": "fp8",
                "model.layers.0.mlp.up_proj.weight": "fp8",
                "model.layers.0.mlp.down_proj.weight": "fp8",
                
                # 专家层使用Int4
                "model.layers.0.mlp.experts.0.gate_proj.weight": "int4",
                "model.layers.0.mlp.experts.0.up_proj.weight": "int4",
                "model.layers.0.mlp.experts.0.down_proj.weight": "int4"
            }
        }
    }
    
    config_path = "example_mixed_precision_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"创建混合精度配置文件: {config_path}")
    return config_path


def example_sglang_mixed_precision_loading():
    """示例：使用SGLang API加载混合精度模型"""
    print("=" * 60)
    print("SGLang混合精度加载示例")
    print("=" * 60)
    
    try:
        # 导入SGLang模块
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.configs.load_config import LoadConfig, LoadFormat
        from sglang.srt.model_loader.loader import DefaultModelLoader
        from sglang.srt.model_loader.sglang_mixed_precision_loader import (
            get_global_mixed_precision_loader
        )
        
        print("✓ SGLang模块导入成功")
        
        # 创建混合精度配置文件
        config_path = create_mixed_precision_config()
        
        # 创建SGLang配置
        model_config = ModelConfig(
            model_path="/path/to/your/model",  # 替换为实际模型路径
            mixed_precision_config=config_path,
            dtype="auto",
            trust_remote_code=True
        )
        
        device_config = DeviceConfig(device="cuda")
        load_config = LoadConfig(load_format=LoadFormat.AUTO)
        
        print("✓ SGLang配置创建成功")
        print(f"  模型路径: {model_config.model_path}")
        print(f"  混合精度配置: {model_config.mixed_precision_config}")
        print(f"  设备: {device_config.device}")
        print(f"  数据类型: {model_config.dtype}")
        
        # 创建模型加载器
        loader = DefaultModelLoader(load_config)
        print("✓ 模型加载器创建成功")
        
        # 注意：这里不会实际加载模型，因为需要真实的模型文件
        # 但会演示配置和加载器的创建过程
        print("\n📝 使用说明:")
        print("1. 将模型路径替换为实际的模型路径")
        print("2. 确保混合精度权重文件存在于配置的路径中")
        print("3. 运行以下代码加载模型:")
        print("   model = loader.load_model(model_config=model_config, device_config=device_config)")
        
        # 演示如何获取混合精度加载器信息
        print("\n🔍 混合精度加载器信息:")
        print("   - 当模型加载时，会自动创建SGLangMixedPrecisionLoader")
        print("   - 可以通过get_global_mixed_precision_loader()获取加载器实例")
        print("   - 加载器会处理不同精度的权重加载和GPTQ反量化")
        
        return True
        
    except ImportError as e:
        print(f"✗ SGLang模块导入失败: {e}")
        print("请确保SGLang已正确安装")
        return False
    except Exception as e:
        print(f"✗ 示例执行失败: {e}")
        return False


def example_mixed_precision_loader_usage():
    """示例：直接使用混合精度加载器"""
    print("\n" + "=" * 60)
    print("混合精度加载器直接使用示例")
    print("=" * 60)
    
    try:
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_loader.sglang_mixed_precision_loader import (
            SGLangMixedPrecisionLoader,
            MixedPrecisionConfig,
            create_mixed_precision_loader
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
            model_path="/path/to/your/model",
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
        
        print("\n📝 使用说明:")
        print("1. 创建MixedPrecisionConfig配置混合精度参数")
        print("2. 创建SGLangMixedPrecisionLoader实例")
        print("3. 调用load_model_weights()加载权重到模型")
        print("4. 使用SGLang的推理引擎进行推理")
        
        return True
        
    except Exception as e:
        print(f"✗ 混合精度加载器示例失败: {e}")
        return False


def example_gptq_dequantization():
    """示例：GPTQ反量化"""
    print("\n" + "=" * 60)
    print("GPTQ反量化示例")
    print("=" * 60)
    
    try:
        import torch
        from sglang.srt.model_loader.sglang_mixed_precision_loader import SGLangGPTQDequantizer
        
        # 创建模拟GPTQ数据
        qweight = torch.randint(0, 16, (256, 768), dtype=torch.int32)
        qzeros = torch.randint(0, 16, (16, 96), dtype=torch.int32)
        scales = torch.randn(16, 768, dtype=torch.float16)
        
        # 执行反量化
        weight = SGLangGPTQDequantizer.dequantize_gptq_weight(qweight, qzeros, scales)
        
        print("✓ GPTQ反量化成功")
        print(f"  输入形状: qweight{qweight.shape}, qzeros{qzeros.shape}, scales{scales.shape}")
        print(f"  输出形状: {weight.shape}")
        print(f"  输出类型: {weight.dtype}")
        
        print("\n📝 使用说明:")
        print("1. GPTQ反量化会自动在加载Int4权重时执行")
        print("2. 支持标准的GPTQ格式：qweight, qzeros, scales, g_idx")
        print("3. 自动处理维度匹配和设备转换")
        
        return True
        
    except Exception as e:
        print(f"✗ GPTQ反量化示例失败: {e}")
        return False


def main():
    """主函数"""
    print("SGLang混合精度集成使用示例")
    print("=" * 60)
    
    # 运行示例
    examples = [
        ("SGLang混合精度加载", example_sglang_mixed_precision_loading),
        ("混合精度加载器使用", example_mixed_precision_loader_usage),
        ("GPTQ反量化", example_gptq_dequantization),
    ]
    
    results = []
    for example_name, example_func in examples:
        print(f"\n运行示例: {example_name}")
        try:
            result = example_func()
            results.append((example_name, result))
        except Exception as e:
            print(f"✗ 示例异常: {e}")
            results.append((example_name, False))
    
    # 显示结果
    print("\n" + "=" * 60)
    print("示例执行结果")
    print("=" * 60)
    
    for example_name, result in results:
        status = "✓ 成功" if result else "✗ 失败"
        print(f"{example_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 个示例成功")
    
    if passed == total:
        print("🎉 所有示例执行成功！")
        print("\n下一步:")
        print("1. 准备真实的模型文件和混合精度权重")
        print("2. 修改配置文件中的路径")
        print("3. 使用SGLang API加载和推理模型")
        print("4. 享受SGLang的高性能混合精度推理！")
    else:
        print("⚠ 部分示例失败，请检查环境配置。")
    
    # 清理临时文件
    config_path = "example_mixed_precision_config.yaml"
    if os.path.exists(config_path):
        try:
            os.remove(config_path)
            print(f"\n清理临时文件: {config_path}")
        except:
            pass


if __name__ == "__main__":
    main()
