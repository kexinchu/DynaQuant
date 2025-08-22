# Adapted from the original weight loader with mixed precision support

import os
import torch
import yaml
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import safetensors
from transformers import AutoConfig
import bitsandbytes as bnb

from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)


class MixedPrecisionWeightLoader:
    """混合精度权重加载器 - 基于sglang架构"""
    
    def __init__(self, config_path: str):
        """
        初始化混合精度权重加载器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.mixed_precision_config = self.config.get('mixed_precision', {})
        self.weight_mapping = self.mixed_precision_config.get('weight_mapping', {})
        
        # 精度路径映射
        self.precision_paths = {
            'fp16': self.mixed_precision_config.get('fp16_path', ''),
            'fp8': self.mixed_precision_config.get('fp8_path', ''),
            'int4': self.mixed_precision_config.get('int4_path', '')
        }
        
        # 缓存已加载的权重文件
        self.weight_cache = {}
        
        logger.info(f"Mixed precision loader initialized with {len(self.weight_mapping)} weight mappings")
    
    def _load_safetensors_file(self, file_path: str) -> Dict[str, torch.Tensor]:
        """加载safetensors文件"""
        if file_path in self.weight_cache:
            return self.weight_cache[file_path]
        
        if os.path.exists(file_path):
            weights = safetensors.torch.load_file(file_path)
            self.weight_cache[file_path] = weights
            logger.debug(f"Loaded safetensors file: {file_path}")
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
            logger.debug(f"Loaded PyTorch file: {file_path}")
            return weights
        else:
            raise FileNotFoundError(f"Weight file not found: {file_path}")
    
    def _find_weight_file(self, weight_name: str, precision: str) -> Optional[str]:
        """查找权重文件路径"""
        precision_path = self.precision_paths[precision]
        if not precision_path:
            return None
        
        # 首先尝试使用safetensors索引文件
        index_file = os.path.join(precision_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            weight_file = self._find_weight_from_index(weight_name, index_file, precision_path)
            if weight_file:
                return weight_file
        
        # 尝试不同的文件扩展名和路径
        possible_files = [
            f"{precision_path}/{weight_name}.safetensors",
            f"{precision_path}/{weight_name}.bin",
            f"{precision_path}/pytorch_model.bin",
            f"{precision_path}/model.safetensors",
            f"{precision_path}/pytorch_model-00001-of-00001.bin",
            f"{precision_path}/model-00001-of-00001.safetensors"
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                return file_path
        
        return None
    
    def _find_weight_from_index(self, weight_name: str, index_file: str, precision_path: str) -> Optional[str]:
        """从safetensors索引文件中查找权重文件"""
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # 检查权重是否在索引中
            if weight_name in index_data.get("weight_map", {}):
                weight_file = index_data["weight_map"][weight_name]
                full_path = os.path.join(precision_path, weight_file)
                if os.path.exists(full_path):
                    logger.debug(f"Found weight {weight_name} in {full_path}")
                    return full_path
            
            return None
        except Exception as e:
            logger.warning(f"Failed to read index file {index_file}: {e}")
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
            logger.warning(f"Weight file not found for {weight_name} with precision {precision}")
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
                weight = self._process_int4_weight(weight)
            elif precision == 'fp8':
                weight = self._process_fp8_weight(weight)
            elif precision == 'fp16':
                weight = self._process_fp16_weight(weight)
            
            logger.debug(f"Loaded {weight_name} with precision {precision}, shape: {weight.shape}")
            return weight
        else:
            logger.warning(f"Weight {weight_name} not found in file {weight_file}")
            return None
    
    def _process_fp16_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """处理FP16权重"""
        if weight.dtype in [torch.float32, torch.float64]:
            return weight.half()
        return weight
    
    def _process_fp8_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """处理FP8权重"""
        if weight.dtype in [torch.float32, torch.float64, torch.float16]:
            # 转换为FP8格式
            if hasattr(torch, 'float8_e4m3fn'):
                return weight.to(torch.float8_e4m3fn)
            else:
                # 如果没有FP8支持，使用FP16
                logger.warning("FP8 not supported, falling back to FP16")
                return weight.half()
        return weight
    
    def _process_int4_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """处理Int4权重"""
        if weight.dtype in [torch.float32, torch.float64, torch.float16]:
            try:
                # 使用bitsandbytes进行4位量化
                int4_weight = bnb.nn.Int4Params.from_pretrained(
                    weight, 
                    requires_grad=False
                )
                return int4_weight
            except Exception as e:
                logger.warning(f"Int4 conversion failed: {e}, falling back to FP16")
                return weight.half()
        return weight
    
    def load_model_weights(self, model: torch.nn.Module) -> None:
        """
        为模型加载混合精度权重
        
        Args:
            model: 要加载权重的模型
        """
        logger.info("Loading mixed precision weights...")
        
        loaded_count = 0
        failed_count = 0
        
        # 遍历模型权重
        for name, param in model.named_parameters():
            weight = self.load_weight(name)
            if weight is not None:
                # 确保权重形状匹配
                if weight.shape == param.shape:
                    param.data = weight.to(param.device)
                    loaded_count += 1
                    logger.debug(f"Loaded {name} with shape {weight.shape}")
                else:
                    logger.warning(f"Shape mismatch for {name}: expected {param.shape}, got {weight.shape}")
                    failed_count += 1
            else:
                failed_count += 1
        
        logger.info(f"Mixed precision weights loading completed: {loaded_count} loaded, {failed_count} failed")
    
    def get_weight_info(self) -> Dict[str, Any]:
        """获取权重信息"""
        info = {
            'precision_paths': self.precision_paths,
            'weight_mapping': self.weight_mapping,
            'cached_files': list(self.weight_cache.keys())
        }
        return info


def create_mixed_precision_loader(config_path: str) -> Optional[MixedPrecisionWeightLoader]:
    """
    创建混合精度权重加载器
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        混合精度权重加载器实例，如果配置不存在则返回None
    """
    if not os.path.exists(config_path):
        logger.info(f"Mixed precision config not found: {config_path}")
        return None
    
    try:
        return MixedPrecisionWeightLoader(config_path)
    except Exception as e:
        logger.error(f"Failed to create mixed precision loader: {e}")
        return None


def load_mixed_precision_weights(model: torch.nn.Module, config_path: str) -> bool:
    """
    加载混合精度权重到模型
    
    Args:
        model: 要加载权重的模型
        config_path: 配置文件路径
        
    Returns:
        是否成功加载
    """
    loader = create_mixed_precision_loader(config_path)
    if loader is None:
        return False
    
    try:
        loader.load_model_weights(model)
        return True
    except Exception as e:
        logger.error(f"Failed to load mixed precision weights: {e}")
        return False
