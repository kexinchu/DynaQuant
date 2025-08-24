#!/usr/bin/env python3
"""
真正的混合精度权重加载器
保持权重的压缩格式，多种量化方式共存以节省GPU存储
"""

import os
import torch
import yaml
import logging
import json
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# 兼容性处理safetensors导入
try:
    from safetensors.torch import load_file, safe_open
except ImportError:
    try:
        from safetensors import load_file, safe_open
    except ImportError:
        import safetensors
        load_file = safetensors.load_file
        safe_open = safetensors.safe_open

# SGLang核心导入
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.model_loader.loader import ModelLoader
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)


class WeightFormat(Enum):
    """权重格式枚举"""
    FP16 = "fp16"
    FP8 = "fp8"
    INT4 = "int4"
    INT8 = "int8"
    GPTQ_INT4 = "gptq_int4"
    AWQ_INT4 = "awq_int4"


@dataclass
class CompressedWeight:
    """压缩权重数据结构"""
    format: WeightFormat
    data: Any  # 原始压缩数据
    metadata: Dict[str, Any]  # 元数据（如scales, zeros等）
    original_shape: Tuple[int, ...]  # 原始形状
    compressed_size: int  # 压缩后大小（字节）
    
    def get_memory_usage(self) -> int:
        """获取内存使用量（字节）"""
        if isinstance(self.data, torch.Tensor):
            return self.data.numel() * self.data.element_size()
        return self.compressed_size


@dataclass
class TrueMixedPrecisionConfig:
    """真正的混合精度配置"""
    fp16_path: str = ""
    fp8_path: str = ""
    int4_path: str = ""
    int8_path: str = ""
    gptq_int4_path: str = ""
    awq_int4_path: str = ""
    weight_mapping: Dict[str, str] = None
    
    def __post_init__(self):
        if self.weight_mapping is None:
            self.weight_mapping = {}


class TrueMixedPrecisionLoader(ModelLoader):
    """真正的混合精度权重加载器"""
    
    def __init__(self, config: ModelConfig, mixed_precision_config: TrueMixedPrecisionConfig):
        """
        初始化真正的混合精度加载器
        
        Args:
            config: SGLang模型配置
            mixed_precision_config: 混合精度配置
        """
        super().__init__(config)
        self.mixed_precision_config = mixed_precision_config
        self.weight_cache = {}  # 缓存已加载的权重文件
        self.compressed_weights = {}  # 存储压缩权重
        self.memory_saved = 0  # 节省的内存（字节）
        
        logger.info(f"True mixed precision loader initialized with {len(mixed_precision_config.weight_mapping)} weight mappings")
    
    def _load_safetensors_file(self, file_path: str) -> Dict[str, torch.Tensor]:
        """加载safetensors文件"""
        if file_path in self.weight_cache:
            return self.weight_cache[file_path]
        
        if os.path.exists(file_path):
            weights = load_file(file_path)
            self.weight_cache[file_path] = weights
            logger.debug(f"Loaded safetensors file: {file_path}")
            return weights
        else:
            raise FileNotFoundError(f"Weight file not found: {file_path}")
    
    def _find_weight_file(self, weight_name: str, precision: str) -> Optional[str]:
        """查找权重文件路径"""
        if precision == 'fp16':
            precision_path = self.mixed_precision_config.fp16_path
        elif precision == 'fp8':
            precision_path = self.mixed_precision_config.fp8_path
        elif precision == 'int4':
            precision_path = self.mixed_precision_config.int4_path
        elif precision == 'int8':
            precision_path = self.mixed_precision_config.int8_path
        elif precision == 'gptq_int4':
            precision_path = self.mixed_precision_config.gptq_int4_path
        elif precision == 'awq_int4':
            precision_path = self.mixed_precision_config.awq_int4_path
        else:
            return None
        
        if not precision_path:
            return None
        
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
    
    def _is_gptq_weight(self, weights: Dict[str, torch.Tensor], weight_name: str) -> bool:
        """检查是否是GPTQ权重"""
        base_name = weight_name.replace('.weight', '')
        gptq_components = [
            f"{base_name}.qweight",
            f"{base_name}.qzeros", 
            f"{base_name}.scales"
        ]
        
        return all(comp in weights for comp in gptq_components)
    
    def _is_awq_weight(self, weights: Dict[str, torch.Tensor], weight_name: str) -> bool:
        """检查是否是AWQ权重"""
        base_name = weight_name.replace('.weight', '')
        awq_components = [
            f"{base_name}.qweight",
            f"{base_name}.qzeros",
            f"{base_name}.scales",
            f"{base_name}.qweight_scale"
        ]
        
        return all(comp in weights for comp in awq_components)
    
    def _load_compressed_weight(self, weight_name: str, precision: str) -> Optional[CompressedWeight]:
        """加载压缩权重，保持压缩格式"""
        try:
            # 查找权重文件
            weight_file = self._find_weight_file(weight_name, precision)
            if not weight_file:
                logger.warning(f"Weight file not found for {weight_name} with precision {precision}")
                return None
            
            # 加载权重文件
            if weight_file.endswith('.safetensors'):
                weights = self._load_safetensors_file(weight_file)
            else:
                weights = torch.load(weight_file, map_location='cpu')
            
            # 根据精度类型处理权重
            if precision == 'gptq_int4' and self._is_gptq_weight(weights, weight_name):
                return self._load_gptq_compressed_weight(weights, weight_name)
            elif precision == 'awq_int4' and self._is_awq_weight(weights, weight_name):
                return self._load_awq_compressed_weight(weights, weight_name)
            elif precision in ['int4', 'int8']:
                return self._load_quantized_weight(weights, weight_name, precision)
            elif precision in ['fp16', 'fp8']:
                return self._load_float_weight(weights, weight_name, precision)
            else:
                # 直接加载权重
                if weight_name in weights:
                    weight = weights[weight_name]
                    return CompressedWeight(
                        format=WeightFormat(precision.upper()),
                        data=weight,
                        metadata={},
                        original_shape=weight.shape,
                        compressed_size=weight.numel() * weight.element_size()
                    )
                else:
                    logger.warning(f"Weight {weight_name} not found in file {weight_file}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error loading compressed weight {weight_name} with precision {precision}: {e}")
            return None
    
    def _load_gptq_compressed_weight(self, weights: Dict[str, torch.Tensor], weight_name: str) -> CompressedWeight:
        """加载GPTQ压缩权重，保持压缩格式"""
        base_name = weight_name.replace('.weight', '')
        
        qweight = weights[f"{base_name}.qweight"]
        qzeros = weights[f"{base_name}.qzeros"]
        scales = weights[f"{base_name}.scales"]
        
        # g_idx是可选的
        g_idx = None
        g_idx_name = f"{base_name}.g_idx"
        if g_idx_name in weights:
            g_idx = weights[g_idx_name]
        
        # 计算压缩大小
        compressed_size = (qweight.numel() + qzeros.numel() + scales.numel()) * 4  # 假设都是float32/int32
        
        # 估算原始大小（假设是float16）
        original_shape = (scales.shape[1], qweight.shape[1] * 8)  # 8 = 32/4 bits
        original_size = original_shape[0] * original_shape[1] * 2  # float16 = 2 bytes
        
        metadata = {
            'qweight': qweight,
            'qzeros': qzeros,
            'scales': scales,
            'g_idx': g_idx,
            'bits': 4,
            'group_size': 128
        }
        
        return CompressedWeight(
            format=WeightFormat.GPTQ_INT4,
            data=qweight,  # 主要数据
            metadata=metadata,
            original_shape=original_shape,
            compressed_size=compressed_size
        )
    
    def _load_awq_compressed_weight(self, weights: Dict[str, torch.Tensor], weight_name: str) -> CompressedWeight:
        """加载AWQ压缩权重，保持压缩格式"""
        base_name = weight_name.replace('.weight', '')
        
        qweight = weights[f"{base_name}.qweight"]
        qzeros = weights[f"{base_name}.qzeros"]
        scales = weights[f"{base_name}.scales"]
        qweight_scale = weights[f"{base_name}.qweight_scale"]
        
        # 计算压缩大小
        compressed_size = (qweight.numel() + qzeros.numel() + scales.numel() + qweight_scale.numel()) * 4
        
        # 估算原始大小
        original_shape = (scales.shape[1], qweight.shape[1] * 8)
        original_size = original_shape[0] * original_shape[1] * 2
        
        metadata = {
            'qweight': qweight,
            'qzeros': qzeros,
            'scales': scales,
            'qweight_scale': qweight_scale,
            'bits': 4,
            'group_size': 128
        }
        
        return CompressedWeight(
            format=WeightFormat.AWQ_INT4,
            data=qweight,
            metadata=metadata,
            original_shape=original_shape,
            compressed_size=compressed_size
        )
    
    def _load_quantized_weight(self, weights: Dict[str, torch.Tensor], weight_name: str, precision: str) -> CompressedWeight:
        """加载量化权重，保持压缩格式"""
        if weight_name in weights:
            weight = weights[weight_name]
            
            # 根据精度确定格式
            if precision == 'int4':
                format_type = WeightFormat.INT4
                # 假设int4权重是压缩格式
                compressed_size = weight.numel() * 0.5  # int4 = 0.5 bytes per element
            else:  # int8
                format_type = WeightFormat.INT8
                compressed_size = weight.numel() * 1  # int8 = 1 byte per element
            
            return CompressedWeight(
                format=format_type,
                data=weight,
                metadata={'bits': 4 if precision == 'int4' else 8},
                original_shape=weight.shape,
                compressed_size=compressed_size
            )
        else:
            raise ValueError(f"Weight {weight_name} not found")
    
    def _load_float_weight(self, weights: Dict[str, torch.Tensor], weight_name: str, precision: str) -> CompressedWeight:
        """加载浮点权重"""
        if weight_name in weights:
            weight = weights[weight_name]
            
            # 转换到指定精度
            if precision == 'fp16':
                weight = weight.half()
                format_type = WeightFormat.FP16
                element_size = 2
            else:  # fp8
                if hasattr(torch, 'float8_e4m3fn'):
                    weight = weight.to(torch.float8_e4m3fn)
                    format_type = WeightFormat.FP8
                    element_size = 1
                else:
                    weight = weight.half()
                    format_type = WeightFormat.FP16
                    element_size = 2
            
            compressed_size = weight.numel() * element_size
            
            return CompressedWeight(
                format=format_type,
                data=weight,
                metadata={},
                original_shape=weight.shape,
                compressed_size=compressed_size
            )
        else:
            raise ValueError(f"Weight {weight_name} not found")
    
    def load_model_weights(self, model: torch.nn.Module) -> Dict[str, Any]:
        """加载模型权重，保持压缩格式"""
        stats = {
            'loaded': 0,
            'skipped': 0,
            'errors': 0,
            'details': [],
            'memory_saved_mb': 0,
            'compressed_weights': {}
        }
        
        # 获取模型设备
        model_device = next(model.parameters()).device
        logger.info(f"Model device: {model_device}")
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight_name = name + '.weight'
                
                if weight_name in self.mixed_precision_config.weight_mapping:
                    precision = self.mixed_precision_config.weight_mapping[weight_name]
                    compressed_weight = self._load_compressed_weight(weight_name, precision)
                    
                    if compressed_weight is not None:
                        try:
                            # 存储压缩权重
                            self.compressed_weights[weight_name] = compressed_weight
                            
                            # 计算内存节省
                            original_size = compressed_weight.original_shape[0] * compressed_weight.original_shape[1] * 2  # float16
                            saved_bytes = original_size - compressed_weight.get_memory_usage()
                            self.memory_saved += saved_bytes
                            
                            stats['loaded'] += 1
                            stats['details'].append({
                                'name': weight_name,
                                'precision': precision,
                                'format': compressed_weight.format.value,
                                'status': 'loaded',
                                'original_shape': list(compressed_weight.original_shape),
                                'compressed_size_mb': compressed_weight.get_memory_usage() / (1024 * 1024),
                                'memory_saved_mb': saved_bytes / (1024 * 1024)
                            })
                            
                            logger.info(f"Loaded compressed weight: {weight_name}, format: {compressed_weight.format.value}, "
                                      f"compressed size: {compressed_weight.get_memory_usage() / (1024 * 1024):.2f}MB, "
                                      f"saved: {saved_bytes / (1024 * 1024):.2f}MB")
                            
                        except Exception as e:
                            logger.error(f"Error setting compressed weight {weight_name}: {e}")
                            stats['errors'] += 1
                            stats['details'].append({
                                'name': weight_name,
                                'precision': precision,
                                'status': 'error',
                                'error': str(e)
                            })
                    else:
                        stats['skipped'] += 1
                        stats['details'].append({
                            'name': weight_name,
                            'precision': precision,
                            'status': 'not_found'
                        })
        
        stats['memory_saved_mb'] = self.memory_saved / (1024 * 1024)
        stats['compressed_weights'] = {name: weight.format.value for name, weight in self.compressed_weights.items()}
        
        logger.info(f"Model weights loaded: {stats['loaded']} loaded, {stats['skipped']} skipped, {stats['errors']} errors")
        logger.info(f"Total memory saved: {stats['memory_saved_mb']:.2f}MB")
        
        return stats
    
    def get_compressed_weight(self, weight_name: str) -> Optional[CompressedWeight]:
        """获取压缩权重"""
        return self.compressed_weights.get(weight_name)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计"""
        total_compressed_size = sum(weight.get_memory_usage() for weight in self.compressed_weights.values())
        total_original_size = sum(
            weight.original_shape[0] * weight.original_shape[1] * 2 
            for weight in self.compressed_weights.values()
        )
        
        return {
            'total_compressed_size_mb': total_compressed_size / (1024 * 1024),
            'total_original_size_mb': total_original_size / (1024 * 1024),
            'memory_saved_mb': self.memory_saved / (1024 * 1024),
            'compression_ratio': total_original_size / total_compressed_size if total_compressed_size > 0 else 1.0,
            'compressed_weights_count': len(self.compressed_weights)
        }


def create_true_mixed_precision_loader(config: ModelConfig, mixed_precision_config_path: str) -> TrueMixedPrecisionLoader:
    """创建真正的混合精度加载器"""
    # 加载混合精度配置
    with open(mixed_precision_config_path, 'r', encoding='utf-8') as f:
        mixed_precision_data = yaml.safe_load(f)
    
    mixed_precision_config = TrueMixedPrecisionConfig(
        fp16_path=mixed_precision_data.get('mixed_precision', {}).get('fp16_path', ''),
        fp8_path=mixed_precision_data.get('mixed_precision', {}).get('fp8_path', ''),
        int4_path=mixed_precision_data.get('mixed_precision', {}).get('int4_path', ''),
        int8_path=mixed_precision_data.get('mixed_precision', {}).get('int8_path', ''),
        gptq_int4_path=mixed_precision_data.get('mixed_precision', {}).get('gptq_int4_path', ''),
        awq_int4_path=mixed_precision_data.get('mixed_precision', {}).get('awq_int4_path', ''),
        weight_mapping=mixed_precision_data.get('mixed_precision', {}).get('weight_mapping', {})
    )
    
    return TrueMixedPrecisionLoader(config, mixed_precision_config)


# 全局真正的混合精度加载器实例
_global_true_mixed_precision_loader: Optional[TrueMixedPrecisionLoader] = None


def get_global_true_mixed_precision_loader() -> Optional[TrueMixedPrecisionLoader]:
    """获取全局真正的混合精度加载器"""
    return _global_true_mixed_precision_loader


def set_global_true_mixed_precision_loader(loader: TrueMixedPrecisionLoader):
    """设置全局真正的混合精度加载器"""
    global _global_true_mixed_precision_loader
    _global_true_mixed_precision_loader = loader
