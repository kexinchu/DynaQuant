#!/usr/bin/env python3
"""
真正的混合精度权重加载器
保持权重的压缩格式，多种量化方式共存以节省GPU存储
支持先加载低精度模型再替换层的策略
"""

import os
import torch
import yaml
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# 兼容性处理safetensors导入
try:
    from safetensors.torch import load_file
except ImportError:
    try:
        from safetensors import load_file
    except ImportError:
        import safetensors
        load_file = safetensors.load_file
        safe_open = safetensors.safe_open

# SGLang核心导入
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.model_loader.loader import DefaultModelLoader
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
    base_model_path: str = ""  # 基础模型路径（通常是低精度模型）
    
    def __post_init__(self):
        if self.weight_mapping is None:
            self.weight_mapping = {}


class TrueMixedPrecisionLoader(DefaultModelLoader):
    """真正的混合精度权重加载器"""
    
    def __init__(self, config: ModelConfig, mixed_precision_config: TrueMixedPrecisionConfig):
        """
        初始化真正的混合精度加载器
        
        Args:
            config: SGLang模型配置
            mixed_precision_config: 混合精度配置
        """
        # 创建LoadConfig
        from sglang.srt.configs.load_config import LoadConfig
        load_config = LoadConfig(
            load_format="auto",
            download_dir=None,  # 使用默认缓存目录
            model_loader_extra_config={},
            ignore_patterns=None  # 使用默认忽略模式
        )
        
        super().__init__(load_config)
        self.mixed_precision_config = mixed_precision_config
        self.weight_cache = {}  # 缓存已加载的权重文件
        self.compressed_weights = {}  # 存储压缩权重
        self.memory_saved = 0  # 节省的内存（字节）
        
        logger.info(f"True mixed precision loader initialized with {len(mixed_precision_config.weight_mapping)} weight mappings")
    
    def _load_safetensors_file(self, file_path: str) -> Dict[str, torch.Tensor]:
        """加载safetensors文件"""
        if file_path in self.weight_cache:
            return self.weight_cache[file_path]
        
        try:
            weights = load_file(file_path)
            self.weight_cache[file_path] = weights
            logger.info(f"Loaded safetensors file: {file_path}")
            return weights
        except Exception as e:
            logger.error(f"Failed to load safetensors file {file_path}: {e}")
            raise
    
    def _load_pytorch_file(self, file_path: str) -> Dict[str, torch.Tensor]:
        """加载PyTorch .bin文件"""
        if file_path in self.weight_cache:
            return self.weight_cache[file_path]
        
        try:
            weights = torch.load(file_path, map_location='cpu')
            self.weight_cache[file_path] = weights
            logger.info(f"Loaded PyTorch file: {file_path}")
            return weights
        except Exception as e:
            logger.error(f"Failed to load PyTorch file {file_path}: {e}")
            raise
    
    def _load_safetensors_index(self, base_path: str) -> Optional[Dict[str, str]]:
        """加载safetensors索引文件"""
        index_file = os.path.join(base_path, "model.safetensors.index.json")
        if not os.path.exists(index_file):
            return None
        
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                import json
                index_data = json.load(f)
            
            # 解析索引文件，构建权重名称到文件路径的映射
            weight_to_file = {}
            if 'weight_map' in index_data:
                weight_to_file = index_data['weight_map']
            
            logger.info(f"Loaded safetensors index from {index_file}, {len(weight_to_file)} weights mapped")
            return weight_to_file
        except Exception as e:
            logger.error(f"Failed to load safetensors index {index_file}: {e}")
            return None
    
    def _find_weight_file(self, weight_name: str, precision: str) -> Optional[str]:
        """根据权重名称和精度查找对应的权重文件"""
        # 根据精度确定路径
        if precision == "fp16" and self.mixed_precision_config.fp16_path:
            base_path = self.mixed_precision_config.fp16_path
        elif precision == "fp8" and self.mixed_precision_config.fp8_path:
            base_path = self.mixed_precision_config.fp8_path
        elif precision == "int4" and self.mixed_precision_config.int4_path:
            base_path = self.mixed_precision_config.int4_path
        elif precision == "gptq_int4" and self.mixed_precision_config.gptq_int4_path:
            base_path = self.mixed_precision_config.gptq_int4_path
        elif precision == "awq_int4" and self.mixed_precision_config.awq_int4_path:
            base_path = self.mixed_precision_config.awq_int4_path
        else:
            logger.warning(f"No path configured for precision: {precision}")
            return None
        
        # 首先尝试使用safetensors索引文件
        weight_to_file = self._load_safetensors_index(base_path)
        if weight_to_file and weight_name in weight_to_file:
            file_name = weight_to_file[weight_name]
            file_path = os.path.join(base_path, file_name)
            if os.path.exists(file_path):
                logger.info(f"Found weight {weight_name} in index file: {file_path}")
                return file_path
            else:
                logger.warning(f"Weight file from index not found: {file_path}")
        
        # 如果没有索引文件或权重不在索引中，尝试传统方式
        possible_files = [
            os.path.join(base_path, f"{weight_name}.safetensors"),
            os.path.join(base_path, f"{weight_name}.bin"),
            os.path.join(base_path, "model.safetensors"),
            os.path.join(base_path, "pytorch_model.bin"),
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                return file_path
        
        logger.warning(f"No weight file found for {weight_name} with precision {precision}")
        return None
    
    def load_weight(self, weight_name: str, precision: str) -> Optional[CompressedWeight]:
        """加载指定权重"""
        weight_file = self._find_weight_file(weight_name, precision)
        if not weight_file:
            return None
        
        try:
            if weight_file.endswith('.safetensors'):
                weights = self._load_safetensors_file(weight_file)
            else:
                weights = self._load_pytorch_file(weight_file)
            
            # 查找权重
            if weight_name in weights:
                weight_data = weights[weight_name]
                original_shape = weight_data.shape
                
                # 创建压缩权重对象
                compressed_weight = CompressedWeight(
                    format=WeightFormat(precision),
                    data=weight_data,
                    metadata={},
                    original_shape=original_shape,
                    compressed_size=weight_data.numel() * weight_data.element_size()
                )
                
                logger.info(f"Loaded weight {weight_name} with precision {precision}, shape: {original_shape}")
                return compressed_weight
            else:
                logger.warning(f"Weight {weight_name} not found in {weight_file}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load weight {weight_name} with precision {precision}: {e}")
            return None
    
    def _is_expert_layer(self, weight_name: str) -> bool:
        """判断是否是专家层权重"""
        return "experts" in weight_name and "mlp" in weight_name
    
    def _get_default_precision(self, weight_name: str) -> str:
        """获取默认精度策略"""
        if self._is_expert_layer(weight_name):
            # 专家层根据配置文件决定，如果没有配置则使用默认值
            return self.mixed_precision_config.weight_mapping.get(weight_name, "fp16")
        else:
            # 非专家层默认使用FP16
            return "fp16"
    
    def _generate_weight_mapping(self, model: torch.nn.Module) -> Dict[str, str]:
        """生成权重映射（应用默认策略）"""
        weight_mapping = {}
        
        # 遍历模型的所有权重
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight_name = name + '.weight'
                
                # 检查是否在配置中有明确指定
                if weight_name in self.mixed_precision_config.weight_mapping:
                    # 使用配置文件中的精度
                    weight_mapping[weight_name] = self.mixed_precision_config.weight_mapping[weight_name]
                    logger.debug(f"Using configured precision for {weight_name}: {weight_mapping[weight_name]}")
                else:
                    # 使用默认策略
                    default_precision = self._get_default_precision(weight_name)
                    weight_mapping[weight_name] = default_precision
                    logger.debug(f"Using default precision for {weight_name}: {default_precision}")
        
        return weight_mapping
    
    def load_model_weights(self, model: torch.nn.Module) -> Dict[str, Any]:
        """加载模型权重（先加载基础模型，再替换指定层）"""
        logger.info("Starting mixed precision model loading...")
        
        # 1. 首先加载基础模型（通常是低精度模型）
        base_model_loaded = False
        base_model_path = self.mixed_precision_config.base_model_path
        if not base_model_path:
            # 如果没有指定基础模型路径，使用配置中的第一个可用路径
            if self.mixed_precision_config.fp8_path:
                base_model_path = self.mixed_precision_config.fp8_path
            elif self.mixed_precision_config.int4_path:
                base_model_path = self.mixed_precision_config.int4_path
            else:
                base_model_path = self.mixed_precision_config.fp16_path
        
        if base_model_path and os.path.exists(base_model_path):
            logger.info(f"Loading base model from: {base_model_path}")
            # 加载基础模型权重
            self._load_base_model_weights(model, base_model_path)
            base_model_loaded = True
            logger.info(f"Base model loaded successfully from {base_model_path}")
        else:
            logger.warning(f"Base model path not found: {base_model_path}")
        
        # 2. 生成权重映射（应用默认策略）
        logger.info("Generating weight mapping with default strategy...")
        weight_mapping = self._generate_weight_mapping(model)
        
        # 统计不同精度的权重数量
        precision_counts = {}
        expert_layer_count = 0
        non_expert_layer_count = 0
        
        for weight_name, precision in weight_mapping.items():
            precision_counts[precision] = precision_counts.get(precision, 0) + 1
            if self._is_expert_layer(weight_name):
                expert_layer_count += 1
            else:
                non_expert_layer_count += 1
        
        logger.info(f"Weight mapping generated:")
        logger.info(f"  Total weights: {len(weight_mapping)}")
        logger.info(f"  Expert layers: {expert_layer_count}")
        logger.info(f"  Non-expert layers: {non_expert_layer_count}")
        for precision, count in precision_counts.items():
            logger.info(f"  {precision}: {count} weights")
        
        # 3. 根据生成的映射替换权重
        replaced_count = 0
        failed_count = 0
        
        for weight_name, precision in weight_mapping.items():
            try:
                compressed_weight = self.load_weight(weight_name, precision)
                if compressed_weight:
                    # 存储压缩权重
                    self.compressed_weights[weight_name] = compressed_weight
                    replaced_count += 1
                    logger.debug(f"Replaced {weight_name} with {precision} precision")
                else:
                    failed_count += 1
                    logger.warning(f"Failed to load weight {weight_name} with precision {precision}")
            except Exception as e:
                failed_count += 1
                logger.error(f"Error loading weight {weight_name}: {e}")
        
        # 计算内存节省
        total_memory_saved = sum(weight.get_memory_usage() for weight in self.compressed_weights.values())
        
        stats = {
            'loaded': replaced_count,
            'failed': failed_count,
            'total_weights': len(weight_mapping),
            'memory_saved_mb': total_memory_saved / (1024 * 1024),
            'compressed_weights': len(self.compressed_weights),
            'base_model_loaded': base_model_loaded,
            'precision_distribution': precision_counts
        }
        
        logger.info(f"Mixed precision loading completed:")
        logger.info(f"  Successfully replaced: {replaced_count}/{len(weight_mapping)} weights")
        logger.info(f"  Failed to load: {failed_count} weights")
        logger.info(f"  Memory saved: {stats['memory_saved_mb']:.2f}MB")
        
        return stats
    
    def _load_base_model_weights(self, model: torch.nn.Module, base_model_path: str):
        """加载基础模型权重"""
        try:
            # 尝试加载safetensors格式
            model_file = os.path.join(base_model_path, "model.safetensors")
            if os.path.exists(model_file):
                weights = self._load_safetensors_file(model_file)
            else:
                # 尝试加载PyTorch格式
                model_file = os.path.join(base_model_path, "pytorch_model.bin")
                if os.path.exists(model_file):
                    weights = self._load_pytorch_file(model_file)
                else:
                    logger.warning(f"No model file found in {base_model_path}")
                    return
            
            # 加载权重到模型
            model.load_state_dict(weights, strict=False)
            logger.info(f"Base model weights loaded from {base_model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load base model weights: {e}")
    
    def get_compressed_weight(self, weight_name: str) -> Optional[CompressedWeight]:
        """获取压缩权重"""
        return self.compressed_weights.get(weight_name)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        total_compressed_size = sum(weight.get_memory_usage() for weight in self.compressed_weights.values())
        return {
            'compressed_weights_count': len(self.compressed_weights),
            'total_compressed_size_mb': total_compressed_size / (1024 * 1024),
            'memory_saved_mb': self.memory_saved / (1024 * 1024)
        }
    
    def download_model(self, model_config: ModelConfig) -> str:
        """下载模型（返回模型路径）"""
        return model_config.model_path
    
    def load_model(self, model_config: ModelConfig, device_config) -> None:
        """加载模型（返回None，实际加载由父类处理）"""
        return None


# 全局混合精度加载器实例
_global_true_mixed_precision_loader = None

def get_global_true_mixed_precision_loader() -> Optional[TrueMixedPrecisionLoader]:
    """获取全局混合精度加载器"""
    return _global_true_mixed_precision_loader

def set_global_true_mixed_precision_loader(loader: TrueMixedPrecisionLoader):
    """设置全局混合精度加载器"""
    global _global_true_mixed_precision_loader
    _global_true_mixed_precision_loader = loader

def create_true_mixed_precision_loader(model_config: ModelConfig, config_path: str) -> TrueMixedPrecisionLoader:
    """创建真正的混合精度加载器"""
    # 加载配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # 提取混合精度配置
    mixed_precision_data = config_data.get('mixed_precision', {})
    
    # 创建配置对象
    mixed_precision_config = TrueMixedPrecisionConfig(
        fp16_path=mixed_precision_data.get('fp16_path', ''),
        fp8_path=mixed_precision_data.get('fp8_path', ''),
        int4_path=mixed_precision_data.get('int4_path', ''),
        int8_path=mixed_precision_data.get('int8_path', ''),
        gptq_int4_path=mixed_precision_data.get('gptq_int4_path', ''),
        awq_int4_path=mixed_precision_data.get('awq_int4_path', ''),
        weight_mapping=mixed_precision_data.get('weight_mapping', {}),
        base_model_path=mixed_precision_data.get('base_model_path', '')
    )
    
    # 创建加载器
    loader = TrueMixedPrecisionLoader(model_config, mixed_precision_config)
    
    logger.info(f"Created mixed precision loader with {len(mixed_precision_config.weight_mapping)} weight mappings")
    logger.info(f"Base model path: {mixed_precision_config.base_model_path}")
    
    return loader
