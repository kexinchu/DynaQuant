#!/usr/bin/env python3
"""
示例：使用Safetensors索引文件分析模型
"""

import os
import sys
import json
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from safetensors_index_analyzer import SafetensorsIndexAnalyzer


def analyze_qwen_model():
    """分析Qwen模型示例"""
    print("=" * 60)
    print("Qwen模型分析示例")
    print("=" * 60)
    
    # 假设的模型路径
    model_path = "/dcar-vepfs-trans-models/Qwen3-235B-A22B"
    
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        print("请修改model_path为实际的模型路径")
        return
    
    try:
        # 创建分析器
        analyzer = SafetensorsIndexAnalyzer(model_path)
        
        # 打印摘要信息
        analyzer.print_summary()
        
        # 分析特定层的权重
        print("\n" + "=" * 40)
        print("特定层权重分析")
        print("=" * 40)
        
        # 分析第0层的权重
        layer_0_weights = analyzer.get_layer_weights(0)
        print(f"第0层权重数量: {len(layer_0_weights)}")
        for weight in layer_0_weights[:5]:  # 只显示前5个
            print(f"  {weight}")
        if len(layer_0_weights) > 5:
            print(f"  ... 还有 {len(layer_0_weights) - 5} 个权重")
        
        # 分析注意力层权重
        attention_weights = analyzer.get_attention_weights()
        print(f"\n注意力层权重数量: {len(attention_weights)}")
        for weight in attention_weights[:3]:  # 只显示前3个
            print(f"  {weight}")
        
        # 分析MLP层权重
        mlp_weights = analyzer.get_mlp_weights()
        print(f"\nMLP层权重数量: {len(mlp_weights)}")
        for weight in mlp_weights[:3]:  # 只显示前3个
            print(f"  {weight}")
        
        # 创建混合精度映射
        print("\n" + "=" * 40)
        print("创建混合精度映射")
        print("=" * 40)
        
        weight_mapping = analyzer.create_mixed_precision_mapping(
            attention_precision="fp16",
            mlp_precision="fp8",
            expert_precision="int4",
            embedding_precision="fp16",
            norm_precision="fp16",
            lm_head_precision="fp16"
        )
        
        # 统计各精度的权重数量
        precision_counts = {}
        for weight_name, precision in weight_mapping.items():
            precision_counts[precision] = precision_counts.get(precision, 0) + 1
        
        print("混合精度映射统计:")
        for precision, count in precision_counts.items():
            print(f"  {precision}: {count} 个权重")
        
        # 导出配置
        config_path = "config/qwen_mixed_precision_config.yaml"
        analyzer.export_mixed_precision_config(
            config_path,
            attention_precision="fp16",
            mlp_precision="fp8",
            expert_precision="int4",
            embedding_precision="fp16",
            norm_precision="fp16",
            lm_head_precision="fp16"
        )
        
        print(f"\n混合精度配置已导出到: {config_path}")
        
    except Exception as e:
        print(f"分析失败: {e}")


def analyze_custom_model(model_path: str):
    """分析自定义模型"""
    print("=" * 60)
    print(f"自定义模型分析: {model_path}")
    print("=" * 60)
    
    try:
        # 创建分析器
        analyzer = SafetensorsIndexAnalyzer(model_path)
        
        # 打印摘要信息
        analyzer.print_summary()
        
        # 导出权重列表
        weights_path = "weights_list.json"
        analyzer.export_weight_list(weights_path)
        
        # 导出混合精度配置
        config_path = "mixed_precision_config.yaml"
        analyzer.export_mixed_precision_config(config_path)
        
        print(f"\n分析完成！")
        print(f"权重列表: {weights_path}")
        print(f"混合精度配置: {config_path}")
        
    except Exception as e:
        print(f"分析失败: {e}")


def main():
    """主函数"""
    print("Safetensors索引文件分析示例")
    print("=" * 60)
    
    # 示例1：分析Qwen模型
    analyze_qwen_model()
    
    # 示例2：分析自定义模型（如果提供了路径）
    if len(sys.argv) > 1:
        custom_model_path = sys.argv[1]
        print("\n" + "=" * 60)
        analyze_custom_model(custom_model_path)
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
