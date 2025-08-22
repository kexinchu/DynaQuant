#!/usr/bin/env python3
"""
SGLang Safetensors索引文件分析工具
用于分析和处理model.safetensors.index.json文件
"""

import os
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict


class SGLangSafetensorsIndexAnalyzer:
    """SGLang Safetensors索引文件分析器"""
    
    def __init__(self, model_path: str):
        """
        初始化分析器
        
        Args:
            model_path: 模型路径
        """
        self.model_path = Path(model_path)
        self.index_file = self.model_path / "model.safetensors.index.json"
        
        if not self.index_file.exists():
            raise FileNotFoundError(f"索引文件不存在: {self.index_file}")
        
        # 加载索引数据
        with open(self.index_file, 'r', encoding='utf-8') as f:
            self.index_data = json.load(f)
    
    def get_weight_map(self) -> Dict[str, str]:
        """获取权重映射"""
        return self.index_data.get("weight_map", {})
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取元数据"""
        return self.index_data.get("metadata", {})
    
    def get_total_size(self) -> int:
        """获取总大小"""
        return self.index_data.get("total_size", 0)
    
    def analyze_weight_distribution(self) -> Dict[str, List[str]]:
        """分析权重分布"""
        weight_map = self.get_weight_map()
        file_distribution = defaultdict(list)
        
        for weight_name, file_name in weight_map.items():
            file_distribution[file_name].append(weight_name)
        
        return dict(file_distribution)
    
    def get_file_sizes(self) -> Dict[str, int]:
        """获取文件大小信息"""
        file_sizes = {}
        weight_map = self.get_weight_map()
        
        for weight_name, file_name in weight_map.items():
            file_path = self.model_path / file_name
            if file_path.exists():
                file_sizes[file_name] = file_path.stat().st_size
            else:
                file_sizes[file_name] = 0
        
        return file_sizes
    
    def find_weights_by_pattern(self, pattern: str) -> List[str]:
        """根据模式查找权重"""
        weight_map = self.get_weight_map()
        matching_weights = []
        
        for weight_name in weight_map.keys():
            if pattern.lower() in weight_name.lower():
                matching_weights.append(weight_name)
        
        return matching_weights
    
    def get_layer_weights(self, layer_num: int) -> List[str]:
        """获取指定层的权重"""
        pattern = f"model.layers.{layer_num}."
        return self.find_weights_by_pattern(pattern)
    
    def get_attention_weights(self) -> List[str]:
        """获取注意力层权重"""
        return self.find_weights_by_pattern("self_attn")
    
    def get_mlp_weights(self) -> List[str]:
        """获取MLP层权重"""
        return self.find_weights_by_pattern("mlp")
    
    def get_embedding_weights(self) -> List[str]:
        """获取嵌入层权重"""
        return self.find_weights_by_pattern("embed")
    
    def get_norm_weights(self) -> List[str]:
        """获取归一化层权重"""
        return self.find_weights_by_pattern("norm")
    
    def get_lm_head_weights(self) -> List[str]:
        """获取语言模型头权重"""
        return self.find_weights_by_pattern("lm_head")
    
    def create_mixed_precision_mapping(self, 
                                     attention_precision: str = "fp16",
                                     mlp_precision: str = "fp8",
                                     expert_precision: str = "int4",
                                     embedding_precision: str = "fp16",
                                     norm_precision: str = "fp16",
                                     lm_head_precision: str = "fp16") -> Dict[str, str]:
        """
        创建混合精度映射
        
        Args:
            attention_precision: 注意力层精度
            mlp_precision: MLP层精度
            expert_precision: 专家层精度
            embedding_precision: 嵌入层精度
            norm_precision: 归一化层精度
            lm_head_precision: 语言模型头精度
            
        Returns:
            权重映射字典
        """
        weight_mapping = {}
        weight_map = self.get_weight_map()
        
        for weight_name in weight_map.keys():
            if "self_attn" in weight_name:
                weight_mapping[weight_name] = attention_precision
            elif "mlp" in weight_name and "experts" in weight_name:
                weight_mapping[weight_name] = expert_precision
            elif "mlp" in weight_name:
                weight_mapping[weight_name] = mlp_precision
            elif "embed" in weight_name:
                weight_mapping[weight_name] = embedding_precision
            elif "norm" in weight_name:
                weight_mapping[weight_name] = norm_precision
            elif "lm_head" in weight_name:
                weight_mapping[weight_name] = lm_head_precision
            else:
                # 默认使用FP16
                weight_mapping[weight_name] = "fp16"
        
        return weight_mapping
    
    def print_summary(self):
        """打印摘要信息"""
        print("=" * 60)
        print("SGLang Safetensors索引文件分析")
        print("=" * 60)
        print(f"模型路径: {self.model_path}")
        print(f"索引文件: {self.index_file}")
        
        weight_map = self.get_weight_map()
        metadata = self.get_metadata()
        total_size = self.get_total_size()
        
        print(f"\n基本信息:")
        print(f"  总权重数量: {len(weight_map)}")
        print(f"  总大小: {total_size / (1024**3):.2f} GB")
        print(f"  元数据: {metadata}")
        
        # 分析文件分布
        file_distribution = self.analyze_weight_distribution()
        print(f"\n文件分布:")
        for file_name, weights in file_distribution.items():
            print(f"  {file_name}: {len(weights)} 个权重")
        
        # 分析权重类型分布
        attention_weights = self.get_attention_weights()
        mlp_weights = self.get_mlp_weights()
        embedding_weights = self.get_embedding_weights()
        norm_weights = self.get_norm_weights()
        lm_head_weights = self.get_lm_head_weights()
        
        print(f"\n权重类型分布:")
        print(f"  注意力层权重: {len(attention_weights)}")
        print(f"  MLP层权重: {len(mlp_weights)}")
        print(f"  嵌入层权重: {len(embedding_weights)}")
        print(f"  归一化层权重: {len(norm_weights)}")
        print(f"  语言模型头权重: {len(lm_head_weights)}")
        
        # 分析层数
        layer_weights = defaultdict(list)
        for weight_name in weight_map.keys():
            if "model.layers." in weight_name:
                parts = weight_name.split(".")
                if len(parts) >= 3:
                    try:
                        layer_num = int(parts[2])
                        layer_weights[layer_num].append(weight_name)
                    except ValueError:
                        pass
        
        if layer_weights:
            print(f"\n层数分析:")
            print(f"  总层数: {len(layer_weights)}")
            print(f"  层范围: {min(layer_weights.keys())} - {max(layer_weights.keys())}")
        
        print("=" * 60)
    
    def export_mixed_precision_config(self, output_path: str, 
                                    attention_precision: str = "fp16",
                                    mlp_precision: str = "fp8",
                                    expert_precision: str = "int4",
                                    embedding_precision: str = "fp16",
                                    norm_precision: str = "fp16",
                                    lm_head_precision: str = "fp16"):
        """
        导出混合精度配置
        
        Args:
            output_path: 输出路径
            attention_precision: 注意力层精度
            mlp_precision: MLP层精度
            expert_precision: 专家层精度
            embedding_precision: 嵌入层精度
            norm_precision: 归一化层精度
            lm_head_precision: 语言模型头精度
        """
        weight_mapping = self.create_mixed_precision_mapping(
            attention_precision, mlp_precision, expert_precision,
            embedding_precision, norm_precision, lm_head_precision
        )
        
        config = {
            "mixed_precision": {
                "fp16_path": str(self.model_path),
                "fp8_path": str(self.model_path),
                "int4_path": str(self.model_path),
                "weight_mapping": weight_mapping
            },
            "inference": {
                "max_seq_length": 4096,
                "max_batch_size": 32,
                "dtype": "bfloat16",
                "device_map": "auto",
                "load_in_8bit": False,
                "load_in_4bit": False
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"混合精度配置已导出到: {output_path}")
    
    def export_weight_list(self, output_path: str):
        """导出权重列表"""
        weight_map = self.get_weight_map()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(weight_map, f, indent=2, ensure_ascii=False)
        
        print(f"权重列表已导出到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SGLang Safetensors索引文件分析工具")
    parser.add_argument("--model_path", required=True, help="模型路径")
    parser.add_argument("--export_config", help="导出混合精度配置文件路径")
    parser.add_argument("--export_weights", help="导出权重列表文件路径")
    parser.add_argument("--attention_precision", default="fp16", help="注意力层精度")
    parser.add_argument("--mlp_precision", default="fp8", help="MLP层精度")
    parser.add_argument("--expert_precision", default="int4", help="专家层精度")
    parser.add_argument("--embedding_precision", default="fp16", help="嵌入层精度")
    parser.add_argument("--norm_precision", default="fp16", help="归一化层精度")
    parser.add_argument("--lm_head_precision", default="fp16", help="语言模型头精度")
    
    args = parser.parse_args()
    
    try:
        # 创建分析器
        analyzer = SGLangSafetensorsIndexAnalyzer(args.model_path)
        
        # 打印摘要
        analyzer.print_summary()
        
        # 导出配置
        if args.export_config:
            analyzer.export_mixed_precision_config(
                args.export_config,
                args.attention_precision,
                args.mlp_precision,
                args.expert_precision,
                args.embedding_precision,
                args.norm_precision,
                args.lm_head_precision
            )
        
        # 导出权重列表
        if args.export_weights:
            analyzer.export_weight_list(args.export_weights)
    
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
