import os
import torch
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import safetensors
from transformers import AutoConfig
import bitsandbytes as bnb


class MixedPrecisionWeightLoader:
    """混合精度权重加载器"""
    
    def __init__(self, config_path: str):
        """
        初始化权重加载器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.mixed_precision_config = self.config['model']['mixed_precision']
        self.weight_mapping = self.mixed_precision_config['weight_mapping']
        
        # 精度路径映射
        self.precision_paths = {
            'fp16': self.mixed_precision_config['fp16_path'],
            'fp8': self.mixed_precision_config['fp8_path'],
            'int4': self.mixed_precision_config['int4_path']
        }
        
        # 缓存已加载的权重文件
        self.weight_cache = {}
        
    def _load_safetensors_file(self, file_path: str) -> Dict[str, torch.Tensor]:
        """加载safetensors文件"""
        if file_path in self.weight_cache:
            return self.weight_cache[file_path]
        
        if os.path.exists(file_path):
            weights = safetensors.torch.load_file(file_path)
            self.weight_cache[file_path] = weights
            return weights
        else:
            raise FileNotFoundError(f"Weight file not found: {file_path}")
    
    def _load_pytorch_file(self, file_path: str) -> Dict[str, torch.Tensor]:
        """加载PyTorch权重文件"""
        if file_path in self.weight_cache:
            return self.weight_cache[file_path]
        
        if os.path.exists(file_path):
            weights = torch.load(file_path, map_location='cpu')
            self.weight_cache[file_path] = weights
            return weights
        else:
            raise FileNotFoundError(f"Weight file not found: {file_path}")
    
    def _find_weight_file(self, weight_name: str, precision: str) -> Optional[str]:
        """查找权重文件路径"""
        precision_path = self.precision_paths[precision]
        
        # 尝试不同的文件扩展名
        possible_files = [
            f"{precision_path}/{weight_name}.safetensors",
            f"{precision_path}/{weight_name}.bin",
            f"{precision_path}/pytorch_model.bin",
            f"{precision_path}/model.safetensors"
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                return file_path
        
        return None
    
    def load_weight(self, weight_name: str) -> Optional[torch.Tensor]:
        """
        根据配置加载指定权重
        
        Args:
            weight_name: 权重名称
            
        Returns:
            加载的权重张量
        """
        if weight_name not in self.weight_mapping:
            # 默认使用FP16
            precision = 'fp16'
        else:
            precision = self.weight_mapping[weight_name]
        
        # 查找权重文件
        weight_file = self._find_weight_file(weight_name, precision)
        if weight_file is None:
            print(f"Warning: Weight file not found for {weight_name} with precision {precision}")
            return None
        
        # 加载权重文件
        if weight_file.endswith('.safetensors'):
            weights = self._load_safetensors_file(weight_file)
        else:
            weights = self._load_pytorch_file(weight_file)
        
        # 提取指定权重
        if weight_name in weights:
            weight = weights[weight_name]
            
            # 根据精度进行相应的处理
            if precision == 'int4':
                # Int4权重需要特殊处理
                weight = self._process_int4_weight(weight)
            elif precision == 'fp8':
                # FP8权重处理
                weight = self._process_fp8_weight(weight)
            elif precision == 'fp16':
                # FP16权重处理
                weight = weight.half()
            
            return weight
        else:
            print(f"Warning: Weight {weight_name} not found in file {weight_file}")
            return None
    
    def _process_int4_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """处理Int4权重"""
        # 这里可以根据具体的Int4格式进行调整
        # 假设权重已经是正确的Int4格式
        return weight
    
    def _process_fp8_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """处理FP8权重"""
        # 转换为FP8格式
        if weight.dtype == torch.float16:
            # 从FP16转换为FP8
            return weight.to(torch.float8_e4m3fn)
        elif weight.dtype == torch.float32:
            # 从FP32转换为FP8
            return weight.to(torch.float8_e4m3fn)
        else:
            return weight
    
    def load_model_weights(self, model) -> None:
        """
        为模型加载混合精度权重
        
        Args:
            model: 要加载权重的模型
        """
        print("Loading mixed precision weights...")
        
        # 获取模型状态字典
        state_dict = model.state_dict()
        
        # 遍历模型权重
        for name, param in model.named_parameters():
            if name in state_dict:
                weight = self.load_weight(name)
                if weight is not None:
                    # 确保权重形状匹配
                    if weight.shape == param.shape:
                        param.data = weight.to(param.device)
                        print(f"Loaded {name} with shape {weight.shape}")
                    else:
                        print(f"Warning: Shape mismatch for {name}: expected {param.shape}, got {weight.shape}")
        
        print("Mixed precision weights loading completed!")
    
    def get_weight_info(self) -> Dict[str, Any]:
        """获取权重信息"""
        info = {
            'precision_paths': self.precision_paths,
            'weight_mapping': self.weight_mapping,
            'cached_files': list(self.weight_cache.keys())
        }
        return info
