#!/usr/bin/env python3
"""
Safetensors兼容性测试脚本
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_safetensors_import():
    """测试safetensors导入"""
    print("=" * 50)
    print("Safetensors兼容性测试")
    print("=" * 50)
    
    print("1. 测试safetensors导入...")
    
    # 测试不同的导入方式
    import_methods = [
        ("from safetensors.torch import load_file, safe_open", 
         lambda: __import__('safetensors.torch').torch.load_file),
        ("from safetensors import load_file, safe_open", 
         lambda: __import__('safetensors').load_file),
        ("import safetensors", 
         lambda: __import__('safetensors').torch.load_file)
    ]
    
    for method_name, import_func in import_methods:
        try:
            print(f"   尝试: {method_name}")
            func = import_func()
            print(f"   ✓ 成功: {method_name}")
            break
        except Exception as e:
            print(f"   ✗ 失败: {method_name} - {e}")
    
    print("\n2. 测试权重加载器导入...")
    
    try:
        from weight_loader import MixedPrecisionWeightLoader
        print("   ✓ 权重加载器导入成功")
    except Exception as e:
        print(f"   ✗ 权重加载器导入失败: {e}")
        return False
    
    print("\n3. 测试配置文件...")
    
    config_path = "config/model_config.yaml"
    if os.path.exists(config_path):
        print(f"   ✓ 配置文件存在: {config_path}")
        
        try:
            loader = MixedPrecisionWeightLoader(config_path)
            print("   ✓ 权重加载器初始化成功")
            print(f"   精度路径: {loader.precision_paths}")
        except Exception as e:
            print(f"   ✗ 权重加载器初始化失败: {e}")
            return False
    else:
        print(f"   ⚠ 配置文件不存在: {config_path}")
        print("   创建测试配置...")
        
        # 创建测试配置
        test_config = {
            'model': {
                'mixed_precision': {
                    'fp16_path': '/tmp/test_fp16',
                    'fp8_path': '/tmp/test_fp8',
                    'int4_path': '/tmp/test_int4',
                    'weight_mapping': {
                        'model.layers.0.mlp.experts.0.down_proj.weight': 'int4'
                    }
                }
            }
        }
        
        import yaml
        os.makedirs('config', exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        print(f"   ✓ 测试配置已创建: {config_path}")
    
    print("\n" + "=" * 50)
    print("Safetensors兼容性测试完成")
    print("=" * 50)
    
    return True


def test_gptq_loading():
    """测试GPTQ权重加载"""
    print("\n" + "=" * 50)
    print("GPTQ权重加载测试")
    print("=" * 50)
    
    try:
        from weight_loader import MixedPrecisionWeightLoader
        
        config_path = "config/model_config.yaml"
        if not os.path.exists(config_path):
            print("配置文件不存在，跳过GPTQ测试")
            return
        
        loader = MixedPrecisionWeightLoader(config_path)
        
        # 测试权重名称
        test_weight_name = "model.layers.0.mlp.experts.0.down_proj.weight"
        
        print(f"测试权重: {test_weight_name}")
        
        # 检查权重映射
        if test_weight_name in loader.weight_mapping:
            precision = loader.weight_mapping[test_weight_name]
            print(f"精度类型: {precision}")
            
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
        
    except Exception as e:
        print(f"GPTQ测试失败: {e}")


def main():
    """主函数"""
    print("Safetensors兼容性测试")
    
    # 运行测试
    success = test_safetensors_import()
    
    if success:
        test_gptq_loading()
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
