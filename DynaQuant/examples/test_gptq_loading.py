#!/usr/bin/env python3
"""
GPTQ权重加载测试脚本
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from weight_loader import MixedPrecisionWeightLoader


def test_gptq_weight_loading():
    """测试GPTQ权重加载"""
    print("=" * 60)
    print("GPTQ权重加载测试")
    print("=" * 60)
    
    # 配置文件路径
    config_path = "config/model_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return
    
    try:
        # 创建权重加载器
        loader = MixedPrecisionWeightLoader(config_path)
        
        print("1. 权重加载器初始化成功")
        print(f"   配置文件: {config_path}")
        print(f"   精度路径: {loader.precision_paths}")
        
        # 测试GPTQ权重加载
        print("\n2. 测试GPTQ权重加载")
        
        # 测试权重名称
        test_weight_names = [
            "model.layers.0.mlp.experts.0.down_proj.weight",
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.0.up_proj.weight"
        ]
        
        for weight_name in test_weight_names:
            print(f"\n   测试权重: {weight_name}")
            
            # 检查权重映射
            if weight_name in loader.weight_mapping:
                precision = loader.weight_mapping[weight_name]
                print(f"   精度类型: {precision}")
                
                # 尝试加载权重
                weight = loader.load_weight(weight_name, precision)
                if weight is not None:
                    print(f"   ✓ 加载成功，形状: {weight.shape}, 类型: {weight.dtype}")
                else:
                    print(f"   ✗ 加载失败")
            else:
                print(f"   ⚠ 权重未在映射中定义")
        
        # 测试权重文件检测
        print("\n3. 测试权重文件检测")
        
        for precision, path in loader.precision_paths.items():
            print(f"\n   精度: {precision}")
            print(f"   路径: {path}")
            
            if os.path.exists(path):
                print(f"   ✓ 路径存在")
                
                # 检查是否有safetensors文件
                safetensors_file = os.path.join(path, "model.safetensors")
                if os.path.exists(safetensors_file):
                    print(f"   ✓ 找到safetensors文件")
                    
                    # 尝试加载文件头信息
                    try:
                        # 兼容性导入safetensors
                        try:
                            from safetensors.torch import safe_open
                        except ImportError:
                            try:
                                from safetensors import safe_open
                            except ImportError:
                                import safetensors
                                safe_open = safetensors.safe_open
                        
                        metadata = safe_open(safetensors_file, framework="pt").metadata()
                        print(f"   ✓ 文件元数据: {len(metadata)} 个键")
                        
                        # 显示前几个键
                        keys = list(metadata.keys())[:5]
                        print(f"   前5个键: {keys}")
                        
                        # 检查是否包含GPTQ键
                        gptq_keys = [key for key in metadata.keys() if any(suffix in key for suffix in ['qweight', 'qzeros', 'scales'])]
                        if gptq_keys:
                            print(f"   ✓ 检测到GPTQ格式，包含 {len(gptq_keys)} 个GPTQ键")
                            print(f"   GPTQ键示例: {gptq_keys[:3]}")
                        else:
                            print(f"   ⚠ 未检测到GPTQ格式")
                            
                    except Exception as e:
                        print(f"   ✗ 读取文件失败: {e}")
                        print(f"   错误详情: {type(e).__name__}: {e}")
                else:
                    print(f"   ⚠ 未找到safetensors文件")
            else:
                print(f"   ✗ 路径不存在")
        
        print("\n" + "=" * 60)
        print("GPTQ权重加载测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_gptq_components():
    """测试GPTQ组件解析"""
    print("\n" + "=" * 60)
    print("GPTQ组件解析测试")
    print("=" * 60)
    
    config_path = "config/model_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return
    
    try:
        loader = MixedPrecisionWeightLoader(config_path)
        
        # 测试权重名称
        test_weight_name = "model.layers.0.mlp.experts.0.down_proj.weight"
        
        print(f"测试权重: {test_weight_name}")
        
        # 获取GPTQ组件名称
        base_name = test_weight_name.replace('.weight', '')
        qweight_name = f"{base_name}.qweight"
        qzeros_name = f"{base_name}.qzeros"
        scales_name = f"{base_name}.scales"
        g_idx_name = f"{base_name}.g_idx"
        
        print(f"基础名称: {base_name}")
        print(f"qweight: {qweight_name}")
        print(f"qzeros: {qzeros_name}")
        print(f"scales: {scales_name}")
        print(f"g_idx: {g_idx_name}")
        
        # 测试int4路径
        int4_path = loader.precision_paths['int4']
        safetensors_file = os.path.join(int4_path, "model.safetensors")
        
        if os.path.exists(safetensors_file):
            print(f"\n检查文件: {safetensors_file}")
            
            try:
                # 兼容性导入safetensors
                try:
                    from safetensors.torch import safe_open
                except ImportError:
                    try:
                        from safetensors import safe_open
                    except ImportError:
                        import safetensors
                        safe_open = safetensors.safe_open
                
                with safe_open(safetensors_file, framework="pt") as f:
                    # 检查GPTQ组件是否存在
                    components = {
                        'qweight': qweight_name in f.keys(),
                        'qzeros': qzeros_name in f.keys(),
                        'scales': scales_name in f.keys(),
                        'g_idx': g_idx_name in f.keys()
                    }
                    
                    print("GPTQ组件检查结果:")
                    for component, exists in components.items():
                        status = "✓" if exists else "✗"
                        print(f"  {status} {component}: {exists}")
                    
                    # 如果所有必需组件都存在，尝试加载
                    if components['qweight'] and components['qzeros'] and components['scales']:
                        print("\n尝试加载GPTQ组件...")
                        
                        # 加载组件
                        qweight = f.get_tensor(qweight_name)
                        qzeros = f.get_tensor(qzeros_name)
                        scales = f.get_tensor(scales_name)
                        g_idx = f.get_tensor(g_idx_name) if components['g_idx'] else None
                        
                        print(f"qweight shape: {qweight.shape}, dtype: {qweight.dtype}")
                        print(f"qzeros shape: {qzeros.shape}, dtype: {qzeros.dtype}")
                        print(f"scales shape: {scales.shape}, dtype: {scales.dtype}")
                        if g_idx is not None:
                            print(f"g_idx shape: {g_idx.shape}, dtype: {g_idx.dtype}")
                        
                        # 尝试反量化
                        try:
                            dequantized = loader._dequantize_gptq_weight(qweight, qzeros, scales, g_idx)
                            print(f"✓ 反量化成功，形状: {dequantized.shape}, 类型: {dequantized.dtype}")
                        except Exception as e:
                            print(f"✗ 反量化失败: {e}")
                    else:
                        print("缺少必需的GPTQ组件")
                        
            except Exception as e:
                print(f"读取文件失败: {e}")
                print(f"错误详情: {type(e).__name__}: {e}")
        else:
            print(f"文件不存在: {safetensors_file}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("GPTQ权重加载功能测试")
    
    # 运行测试
    test_gptq_weight_loading()
    test_gptq_components()


if __name__ == "__main__":
    main()
