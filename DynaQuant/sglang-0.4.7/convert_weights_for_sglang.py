#!/usr/bin/env python3
"""
SGLang权重转换工具
将单一权重文件转换为不同精度格式，用于混合精度部署
"""

import os
import torch
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import safetensors
from transformers import AutoConfig, AutoModelForCausalLM
import numpy as np
import shutil


class SGLangWeightConverter:
    """SGLang专用权重转换器"""
    
    def __init__(self, model_path: str, output_dir: str):
        """
        初始化权重转换器
        
        Args:
            model_path: 原始模型路径
            output_dir: 输出目录
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建不同精度的输出目录
        self.fp16_dir = self.output_dir / "fp16"
        self.fp8_dir = self.output_dir / "fp8"
        self.int4_dir = self.output_dir / "int4"
        
        for dir_path in [self.fp16_dir, self.fp8_dir, self.int4_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def load_original_weights(self) -> Dict[str, torch.Tensor]:
        """加载原始权重"""
        print(f"加载原始权重: {self.model_path}")
        
        # 尝试加载safetensors格式
        safetensors_path = os.path.join(self.model_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            weights = safetensors.torch.load_file(safetensors_path)
            print(f"从safetensors加载权重，共{len(weights)}个参数")
            return weights
        
        # 尝试加载PyTorch格式
        pytorch_path = os.path.join(self.model_path, "pytorch_model.bin")
        if os.path.exists(pytorch_path):
            weights = torch.load(pytorch_path, map_location='cpu')
            print(f"从PyTorch加载权重，共{len(weights)}个参数")
            return weights
        
        # 尝试加载分片权重
        weights = {}
        for i in range(100):  # 假设最多100个分片
            shard_path = os.path.join(self.model_path, f"pytorch_model-{i:05d}-of-*.bin")
            if os.path.exists(shard_path.replace("*", "00001")):  # 简化检查
                shard_weights = torch.load(shard_path.replace("*", "00001"), map_location='cpu')
                weights.update(shard_weights)
                print(f"加载分片 {i}")
            else:
                break
        
        if weights:
            print(f"从分片权重加载，共{len(weights)}个参数")
            return weights
        
        raise FileNotFoundError(f"在{self.model_path}中未找到权重文件")
    
    def convert_to_fp16(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """转换为FP16格式"""
        print("转换为FP16格式...")
        fp16_weights = {}
        
        for name, weight in weights.items():
            if weight.dtype in [torch.float32, torch.float64]:
                fp16_weights[name] = weight.half()
            else:
                fp16_weights[name] = weight
        
        return fp16_weights
    
    def convert_to_fp8(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """转换为FP8格式"""
        print("转换为FP8格式...")
        fp8_weights = {}
        
        for name, weight in weights.items():
            if weight.dtype in [torch.float32, torch.float64, torch.float16]:
                # 转换为FP8 (使用float8_e4m3fn格式)
                if hasattr(torch, 'float8_e4m3fn'):
                    fp8_weights[name] = weight.to(torch.float8_e4m3fn)
                else:
                    # 如果没有FP8支持，使用FP16
                    fp8_weights[name] = weight.half()
            else:
                fp8_weights[name] = weight
        
        return fp8_weights
    
    def convert_to_int4(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """转换为Int4格式"""
        print("转换为Int4格式...")
        int4_weights = {}
        
        for name, weight in weights.items():
            if weight.dtype in [torch.float32, torch.float64, torch.float16]:
                # 使用bitsandbytes进行4位量化
                try:
                    import bitsandbytes as bnb
                    # 将权重转换为4位量化格式
                    int4_weight = bnb.nn.Int4Params.from_pretrained(
                        weight, 
                        requires_grad=False
                    )
                    int4_weights[name] = int4_weight
                except ImportError:
                    print("警告: bitsandbytes未安装，使用FP16替代Int4")
                    int4_weights[name] = weight.half()
            else:
                int4_weights[name] = weight
        
        return int4_weights
    
    def save_weights(self, weights: Dict[str, torch.Tensor], output_path: Path, format: str = "safetensors"):
        """保存权重"""
        if format == "safetensors":
            safetensors.torch.save_file(weights, output_path / "model.safetensors")
        else:
            torch.save(weights, output_path / "pytorch_model.bin")
        
        print(f"权重已保存到: {output_path}")
    
    def copy_model_files(self, source_dir: str, target_dir: Path):
        """复制模型配置文件"""
        source_path = Path(source_dir)
        target_path = target_dir
        
        # 复制配置文件
        config_files = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        for file_name in config_files:
            source_file = source_path / file_name
            if source_file.exists():
                shutil.copy2(source_file, target_path / file_name)
                print(f"复制配置文件: {file_name}")
    
    def create_weight_mapping_config(self, weights: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """创建权重映射配置"""
        print("创建权重映射配置...")
        
        # 分析权重类型，自动分配精度
        weight_mapping = {}
        
        for name, weight in weights.items():
            # 根据权重名称和大小自动分配精度
            if "attention" in name.lower() or "attn" in name.lower():
                # 注意力层使用FP16
                weight_mapping[name] = "fp16"
            elif "mlp" in name.lower() and "experts" in name.lower():
                # 专家层使用Int4
                weight_mapping[name] = "int4"
            elif "mlp" in name.lower():
                # 普通MLP层使用FP8
                weight_mapping[name] = "fp8"
            elif "embed" in name.lower() or "norm" in name.lower():
                # 嵌入和归一化层使用FP16
                weight_mapping[name] = "fp16"
            else:
                # 默认使用FP16
                weight_mapping[name] = "fp16"
        
        return weight_mapping
    
    def convert_weights(self, create_config: bool = True):
        """转换权重"""
        print("开始权重转换...")
        
        # 加载原始权重
        original_weights = self.load_original_weights()
        
        # 转换为不同精度
        fp16_weights = self.convert_to_fp16(original_weights)
        fp8_weights = self.convert_to_fp8(original_weights)
        int4_weights = self.convert_to_int4(original_weights)
        
        # 保存权重
        self.save_weights(fp16_weights, self.fp16_dir, "safetensors")
        self.save_weights(fp8_weights, self.fp8_dir, "safetensors")
        self.save_weights(int4_weights, self.int4_dir, "safetensors")
        
        # 复制模型配置文件到每个精度目录
        self.copy_model_files(self.model_path, self.fp16_dir)
        self.copy_model_files(self.model_path, self.fp8_dir)
        self.copy_model_files(self.model_path, self.int4_dir)
        
        # 创建权重映射配置
        if create_config:
            weight_mapping = self.create_weight_mapping_config(original_weights)
            
            config = {
                "mixed_precision": {
                    "fp16_path": str(self.fp16_dir),
                    "fp8_path": str(self.fp8_dir),
                    "int4_path": str(self.int4_dir),
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
            
            config_path = self.output_dir / "mixed_precision_config.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            print(f"权重映射配置已保存到: {config_path}")
        
        print("权重转换完成!")
        
        # 显示统计信息
        print("\n转换统计:")
        print(f"FP16权重数量: {len(fp16_weights)}")
        print(f"FP8权重数量: {len(fp8_weights)}")
        print(f"Int4权重数量: {len(int4_weights)}")
        
        # 计算文件大小
        fp16_size = sum(w.numel() * 2 for w in fp16_weights.values() if w.dtype == torch.float16)
        fp8_size = sum(w.numel() * 1 for w in fp8_weights.values() if hasattr(w, 'dtype') and w.dtype == torch.float8_e4m3fn)
        int4_size = sum(w.numel() * 0.5 for w in int4_weights.values())
        
        print(f"FP16总大小: {fp16_size / 1024**3:.2f} GB")
        print(f"FP8总大小: {fp8_size / 1024**3:.2f} GB")
        print(f"Int4总大小: {int4_size / 1024**3:.2f} GB")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SGLang权重转换工具")
    parser.add_argument("--model_path", required=True, help="原始模型路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--create_config", action="store_true", help="创建权重映射配置")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SGLang权重转换工具")
    print("=" * 60)
    print(f"原始模型路径: {args.model_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"创建配置: {args.create_config}")
    print("=" * 60)
    
    # 创建转换器并执行转换
    converter = SGLangWeightConverter(args.model_path, args.output_dir)
    converter.convert_weights(args.create_config)


if __name__ == "__main__":
    main()
