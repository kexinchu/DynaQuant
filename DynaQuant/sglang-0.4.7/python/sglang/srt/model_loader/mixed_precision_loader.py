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
from typing import Dict, Any, Optional, Tuple, List
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
            # logger.info(f"Loaded safetensors file: {file_path}")
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
            # logger.info(f"Loaded PyTorch file: {file_path}")
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
            
            # logger.info(f"Loaded safetensors index from {index_file}, {len(weight_to_file)} weights mapped")
            return weight_to_file
        except Exception as e:
            logger.error(f"Failed to load safetensors index {index_file}: {e}")
            return None
    
    def _normalize_weight_name(self, weight_name: str, precision: str) -> List[str]:
        """标准化权重名称，处理不同量化精度的差异"""
        normalized_names = [weight_name]
        
        # 处理qkv_proj -> q_proj, k_proj, v_proj的转换
        # SGLang的QKVParallelLinear期望qkv_proj.weight，但权重文件可能是分离的
        if "qkv_proj" in weight_name:
            # 正确提取基础名称，避免双点号问题
            base_name = weight_name.replace(".qkv_proj.weight", "")
            normalized_names = [
                base_name + ".q_proj.weight",
                base_name + ".k_proj.weight", 
                base_name + ".v_proj.weight"
            ]
            # 同时保留原始的qkv_proj.weight名称，以防权重文件中有直接的qkv_proj权重
            normalized_names.append(weight_name)
        
        # 处理不同精度的权重名称差异
        if precision in ["fp8", "int8"]:
            # FP8/Int8模型可能有weight_scale_inv后缀
            additional_names = []
            for name in normalized_names:
                if not name.endswith("_scale_inv"):
                    additional_names.append(name + "_scale_inv")
            normalized_names.extend(additional_names)
        
        # 处理GPTQ-Int4的权重名称差异
        elif precision in ["gptq_int4", "awq_int4"]:
            # GPTQ/AWQ模型使用分离的量化组件
            gptq_names = []
            for name in normalized_names:
                if name.endswith(".weight"):
                    base_name = name.replace(".weight", "")
                    gptq_names.extend([
                        base_name + ".qweight",
                        base_name + ".qzeros", 
                        base_name + ".scales",
                        base_name + ".g_idx"  # 可选组件
                    ])
            if gptq_names:
                normalized_names = gptq_names
        
        return normalized_names
    
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
        if not weight_to_file:
            # 如果没有索引文件，尝试传统方式
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
        
        # 标准化权重名称，处理不同精度的差异
        normalized_names = self._normalize_weight_name(weight_name, precision)
        
        # 对于GPTQ-Int4，需要确保所有组件都在同一个文件中
        if precision in ["gptq_int4", "awq_int4"]:
            return self._find_gptq_weight_file(weight_name, normalized_names, weight_to_file, base_path)
        
        # 尝试所有标准化的权重名称
        for normalized_name in normalized_names:
            if normalized_name in weight_to_file:
                file_name = weight_to_file[normalized_name]
                file_path = os.path.join(base_path, file_name)
                if os.path.exists(file_path):
                    # if normalized_name != weight_name:
                    #     logger.info(f"Found normalized weight {normalized_name} for {weight_name} in index file: {file_path}")
                    # else:
                    #     logger.info(f"Found weight {weight_name} in index file: {file_path}")
                    return file_path
                else:
                    logger.warning(f"Weight file from index not found: {file_path}")
        
        # 如果找不到权重文件，输出调试信息
        logger.warning(f"No weight file found for {weight_name} with precision {precision}")
        logger.info(f"Tried normalized names: {normalized_names}")
        
        # 输出索引文件中的权重名称样本（用于调试）
        logger.info(f"Available weights in index file (first 10):")
        for i, (name, file) in enumerate(list(weight_to_file.items())[:10]):
            logger.info(f"  {i+1}. {name} -> {file}")
        
        return None
    
    def _find_gptq_weight_file(self, weight_name: str, normalized_names: List[str], weight_to_file: Dict[str, str], base_path: str) -> Optional[str]:
        """查找GPTQ权重文件，确保所有组件都在同一个文件中"""
        # 对于GPTQ，我们需要找到包含所有必要组件的文件
        base_name = weight_name.replace(".weight", "")
        required_components = [
            base_name + ".qweight",
            base_name + ".qzeros",
            base_name + ".scales"
        ]
        
        # 检查每个文件，找到包含所有必要组件的文件
        file_components = {}
        for component in required_components:
            if component in weight_to_file:
                file_name = weight_to_file[component]
                if file_name not in file_components:
                    file_components[file_name] = []
                file_components[file_name].append(component)
        
        # 找到包含所有必要组件的文件
        for file_name, components in file_components.items():
            if len(components) >= 3:  # 至少需要qweight, qzeros, scales
                file_path = os.path.join(base_path, file_name)
                if os.path.exists(file_path):
                    logger.info(f"Found GPTQ weight file {file_path} with components: {components}")
                    return file_path
        
        logger.warning(f"No GPTQ weight file found with all required components for {weight_name}")
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
            
            # 处理qkv_proj的特殊情况
            # SGLang的QKVParallelLinear期望qkv_proj.weight，但权重文件可能是分离的q_proj, k_proj, v_proj
            if "qkv_proj" in weight_name:
                return self._load_qkv_weight_for_sglang(weight_name, weights, precision)
            
            # 处理GPTQ-Int4的特殊情况
            if precision in ["gptq_int4", "awq_int4"]:
                return self._load_gptq_weight(weight_name, weights, precision)
            
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
                
                # logger.info(f"Loaded weight {weight_name} with precision {precision}, shape: {original_shape}")
                return compressed_weight
            else:
                logger.warning(f"Weight {weight_name} not found in {weight_file}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load weight {weight_name} with precision {precision}: {e}")
            return None
    
    def _load_qkv_weight_for_sglang(self, weight_name: str, weights: Dict[str, torch.Tensor], precision: str) -> Optional[CompressedWeight]:
        """为SGLang加载qkv权重，处理分离的q_proj, k_proj, v_proj合并为qkv_proj"""
        # 正确提取基础名称，避免双点号问题
        base_name = weight_name.replace(".qkv_proj.weight", "")
        q_name = base_name + ".q_proj.weight"
        k_name = base_name + ".k_proj.weight"
        v_name = base_name + ".v_proj.weight"
        
        # 检查是否存在分离的权重
        if q_name in weights and k_name in weights and v_name in weights:
            q_weight = weights[q_name]
            k_weight = weights[k_name]
            v_weight = weights[v_name]
            
            # 合并qkv权重 - 按照SGLang的QKVParallelLinear期望的格式
            # QKV权重在最后一个维度上连接
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=-1)
            original_shape = qkv_weight.shape
            
            # 创建压缩权重对象
            compressed_weight = CompressedWeight(
                format=WeightFormat(precision),
                data=qkv_weight,
                metadata={
                    'q_weight': q_weight,
                    'k_weight': k_weight,
                    'v_weight': v_weight,
                    'is_merged': True,
                    'q_size': q_weight.shape[-1],
                    'kv_size': k_weight.shape[-1]
                },
                original_shape=original_shape,
                compressed_size=qkv_weight.numel() * qkv_weight.element_size()
            )
            
            # logger.info(f"Merged qkv weight {weight_name} from {q_name}, {k_name}, {v_name}, shape: {original_shape}")
            return compressed_weight
        else:
            # 如果找不到分离的权重，检查是否有直接的qkv_proj权重
            if weight_name in weights:
                weight_data = weights[weight_name]
                original_shape = weight_data.shape
                
                compressed_weight = CompressedWeight(
                    format=WeightFormat(precision),
                    data=weight_data,
                    metadata={},
                    original_shape=original_shape,
                    compressed_size=weight_data.numel() * weight_data.element_size()
                )
                
                # logger.info(f"Found direct qkv weight {weight_name}, shape: {original_shape}")
                return compressed_weight
            else:
                logger.warning(f"Neither separate qkv weights nor direct qkv weight found for {weight_name}")
                logger.warning(f"Tried: {q_name}, {k_name}, {v_name}, {weight_name}")
                return None
    
    def _load_gptq_weight(self, weight_name: str, weights: Dict[str, torch.Tensor], precision: str) -> Optional[CompressedWeight]:
        """加载GPTQ权重，处理分离的量化组件"""
        base_name = weight_name.replace(".weight", "")
        
        # 检查是否存在GPTQ组件
        qweight_name = base_name + ".qweight"
        qzeros_name = base_name + ".qzeros"
        scales_name = base_name + ".scales"
        g_idx_name = base_name + ".g_idx"
        
        if qweight_name in weights and qzeros_name in weights and scales_name in weights:
            qweight = weights[qweight_name]
            qzeros = weights[qzeros_name]
            scales = weights[scales_name]
            g_idx = weights.get(g_idx_name, None)  # g_idx是可选的
            
            # 计算压缩大小
            compressed_size = (qweight.numel() + qzeros.numel() + scales.numel()) * 4  # 假设都是float32/int32
            if g_idx is not None:
                compressed_size += g_idx.numel() * 4
            
            # 估算原始形状（从scales推断）
            original_shape = (scales.shape[1], qweight.shape[1] * 8)  # 8 = 32/4 bits
            
            # 创建压缩权重对象
            compressed_weight = CompressedWeight(
                format=WeightFormat(precision),
                data=qweight,  # 主要数据
                metadata={
                    'qweight': qweight,
                    'qzeros': qzeros,
                    'scales': scales,
                    'g_idx': g_idx,
                    'bits': 4,
                    'group_size': 128,
                    'is_gptq': True
                },
                original_shape=original_shape,
                compressed_size=compressed_size
            )
            
            logger.info(f"Loaded GPTQ weight {weight_name}, shape: {original_shape}, compressed size: {compressed_size}")
            return compressed_weight
        else:
            logger.warning(f"GPTQ components not found for {weight_name}: {qweight_name}, {qzeros_name}, {scales_name}")
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
        """加载模型权重（使用精确的层级初始化）"""
        logger.info("Starting mixed precision model loading with selective layer initialization...")
        
        # 1. 验证模型结构兼容性
        if not self._validate_model_compatibility(model):
            logger.error("Model structure compatibility check failed")
            return {'loaded': 0, 'failed': 1, 'base_model_loaded': False}
        
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
        
        # 4. 根据生成的映射替换权重
        replaced_count = 0
        failed_count = 0
        
        for weight_name, precision in weight_mapping.items():
            try:
                # 验证权重是否存在于模型中
                if not self._weight_exists_in_model(model, weight_name):
                    logger.warning(f"Weight {weight_name} does not exist in model, skipping")
                    failed_count += 1
                    continue
                
                compressed_weight = self.load_weight(weight_name, precision)
                if compressed_weight:
                    # 验证权重形状兼容性
                    if self._validate_weight_compatibility(model, weight_name, compressed_weight):
                        # 存储压缩权重
                        self.compressed_weights[weight_name] = compressed_weight
                        replaced_count += 1
                        logger.debug(f"Replaced {weight_name} with {precision} precision")
                    else:
                        failed_count += 1
                        logger.warning(f"Weight shape incompatible for {weight_name}")
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
            'base_model_loaded': True,  # 现在使用精确的层级初始化
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
    
    def _validate_model_compatibility(self, model: torch.nn.Module) -> bool:
        """验证模型结构兼容性"""
        logger.info("Validating model structure compatibility...")
        
        try:
            # 检查模型是否有必要的属性
            if not hasattr(model, 'named_modules'):
                logger.error("Model does not have named_modules method")
                return False
            
            # 检查模型是否有权重参数
            has_weights = False
            for name, module in model.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    has_weights = True
                    break
            
            if not has_weights:
                logger.error("Model does not have any weight parameters")
                return False
            
            logger.info("Model structure compatibility check passed")
            return True
            
        except Exception as e:
            logger.error(f"Model compatibility check failed: {e}")
            return False
    
    def _weight_exists_in_model(self, model: torch.nn.Module, weight_name: str) -> bool:
        """检查权重是否存在于模型中"""
        try:
            # 解析权重名称
            module_names = weight_name.split('.')
            if module_names[-1] != 'weight':
                return False
            
            # 获取模块名称
            module_name = '.'.join(module_names[:-1])
            
            # 检查模块是否存在
            module = model
            for name in module_name.split('.'):
                if hasattr(module, name):
                    module = getattr(module, name)
                else:
                    return False
            
            # 检查是否有weight属性
            return hasattr(module, 'weight') and module.weight is not None
            
        except Exception as e:
            logger.debug(f"Error checking weight existence for {weight_name}: {e}")
            return False
    
    def _validate_weight_compatibility(self, model: torch.nn.Module, weight_name: str, compressed_weight: CompressedWeight) -> bool:
        """验证权重形状兼容性"""
        try:
            # 获取模型中的权重形状
            module_names = weight_name.split('.')
            module_name = '.'.join(module_names[:-1])
            
            module = model
            for name in module_name.split('.'):
                if hasattr(module, name):
                    module = getattr(module, name)
                else:
                    return False
            
            if not hasattr(module, 'weight') or module.weight is None:
                return False
            
            model_weight_shape = module.weight.shape
            compressed_weight_shape = compressed_weight.original_shape
            
            # 检查形状是否兼容
            if model_weight_shape != compressed_weight_shape:
                logger.warning(f"Weight shape mismatch for {weight_name}: "
                             f"model has {model_weight_shape}, compressed has {compressed_weight_shape}")
                return False
            
            logger.debug(f"Weight shape compatible for {weight_name}: {model_weight_shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating weight compatibility for {weight_name}: {e}")
            return False
    
    def _get_model_weight_info(self, model: torch.nn.Module) -> Dict[str, Any]:
        """获取模型权重信息"""
        weight_info = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight_name = name + '.weight'
                weight_info[weight_name] = {
                    'shape': module.weight.shape,
                    'dtype': module.weight.dtype,
                    'device': module.weight.device
                }
        
        return weight_info
    
    def initialize_specific_layers(self, model: torch.nn.Module, base_model_path: str):
        """精确初始化特定层，而不是重新初始化整个模型"""
        logger.info("Starting selective layer initialization...")
        
        try:
            # 1. 获取需要初始化的层列表
            layers_to_initialize = self._get_layers_to_initialize()
            logger.info(f"Found {len(layers_to_initialize)} layers to initialize")
            
            # 2. 加载基础模型的权重
            base_weights = self._load_base_model_weights_dict(base_model_path)
            if not base_weights:
                logger.warning("Failed to load base model weights for layer initialization")
                return
            
            # 3. 精确初始化每个需要的层
            initialized_count = 0
            for layer_name in layers_to_initialize:
                if self._initialize_single_layer(model, layer_name, base_weights):
                    initialized_count += 1
                    logger.debug(f"Initialized layer: {layer_name}")
                else:
                    logger.warning(f"Failed to initialize layer: {layer_name}")
            
            logger.info(f"Selective layer initialization completed: {initialized_count}/{len(layers_to_initialize)} layers initialized")
            
        except Exception as e:
            logger.error(f"Error during selective layer initialization: {e}")
    
    def _get_layers_to_initialize(self) -> List[str]:
        """获取需要初始化的层列表"""
        layers_to_initialize = set()
        
        # 从权重映射中提取层名称
        for weight_name in self.mixed_precision_config.weight_mapping.keys():
            # 解析权重名称，提取层名称
            # 例如: "model.layers.12.self_attn.q_proj.weight" -> "model.layers.12.self_attn"
            layer_name = self._extract_layer_name_from_weight(weight_name)
            if layer_name:
                layers_to_initialize.add(layer_name)
        
        # 添加qkv_proj相关的层（如果存在）
        qkv_layers = set()
        for layer_name in list(layers_to_initialize):
            if "self_attn" in layer_name:
                # 检查是否有qkv_proj相关的权重
                qkv_weight_name = layer_name + ".qkv_proj.weight"
                if qkv_weight_name in self.mixed_precision_config.weight_mapping:
                    qkv_layers.add(layer_name)
        
        layers_to_initialize.update(qkv_layers)
        
        logger.info(f"Layers to initialize: {list(layers_to_initialize)}")
        return list(layers_to_initialize)
    
    def _extract_layer_name_from_weight(self, weight_name: str) -> Optional[str]:
        """从权重名称中提取层名称"""
        # 移除权重后缀
        if weight_name.endswith('.weight'):
            layer_name = weight_name[:-7]  # 移除 '.weight'
        else:
            layer_name = weight_name
        
        # 处理特殊情况
        if '.q_proj.' in layer_name or '.k_proj.' in layer_name or '.v_proj.' in layer_name:
            # 对于分离的q_proj, k_proj, v_proj，提取到self_attn层
            parts = layer_name.split('.')
            for i, part in enumerate(parts):
                if part == 'self_attn':
                    layer_name = '.'.join(parts[:i+1])
                    break
        
        return layer_name
    
    def _initialize_single_layer(self, model: torch.nn.Module, layer_name: str, base_weights: Dict[str, torch.Tensor]) -> bool:
        """初始化单个层"""
        try:
            # 1. 获取层模块
            layer_module = self._get_module_by_name(model, layer_name)
            if layer_module is None:
                logger.warning(f"Layer module not found: {layer_name}")
                return False
            
            # 2. 收集该层的所有权重
            layer_weights = {}
            for weight_name, weight_tensor in base_weights.items():
                if weight_name.startswith(layer_name + '.'):
                    layer_weights[weight_name] = weight_tensor
            
            if not layer_weights:
                logger.warning(f"No weights found for layer: {layer_name}")
                return False
            
            # 3. 初始化层的权重
            self._initialize_layer_weights(layer_module, layer_weights, layer_name)
            
            logger.debug(f"Successfully initialized layer: {layer_name} with {len(layer_weights)} weights")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing layer {layer_name}: {e}")
            return False
    
    def _get_module_by_name(self, model: torch.nn.Module, module_name: str) -> Optional[torch.nn.Module]:
        """根据名称获取模块"""
        try:
            module = model
            for name in module_name.split('.'):
                if hasattr(module, name):
                    module = getattr(module, name)
                else:
                    return None
            return module
        except Exception as e:
            logger.error(f"Error getting module {module_name}: {e}")
            return None
    
    def _initialize_layer_weights(self, layer_module: torch.nn.Module, layer_weights: Dict[str, torch.Tensor], layer_name: str):
        """初始化层的权重"""
        try:
            # 对于不同类型的层，采用不同的初始化策略
            if "self_attn" in layer_name:
                self._initialize_attention_layer(layer_module, layer_weights, layer_name)
            elif "mlp" in layer_name:
                self._initialize_mlp_layer(layer_module, layer_weights, layer_name)
            else:
                # 通用初始化
                self._initialize_generic_layer(layer_module, layer_weights, layer_name)
                
        except Exception as e:
            logger.error(f"Error initializing layer weights for {layer_name}: {e}")
    
    def _initialize_attention_layer(self, layer_module: torch.nn.Module, layer_weights: Dict[str, torch.Tensor], layer_name: str):
        """初始化注意力层"""
        logger.debug(f"Initializing attention layer: {layer_name}")
        
        # 检查是否有qkv_proj权重
        qkv_weight_name = layer_name + ".qkv_proj.weight"
        if qkv_weight_name in layer_weights:
            # 处理qkv_proj权重
            qkv_weight = layer_weights[qkv_weight_name]
            if hasattr(layer_module, 'qkv_proj'):
                layer_module.qkv_proj.weight.data = qkv_weight
                logger.debug(f"Set qkv_proj weight for {layer_name}")
        
        # 处理分离的q_proj, k_proj, v_proj权重
        for proj_name in ['q_proj', 'k_proj', 'v_proj']:
            weight_name = layer_name + "." + proj_name + ".weight"
            if weight_name in layer_weights:
                weight = layer_weights[weight_name]
                if hasattr(layer_module, proj_name):
                    getattr(layer_module, proj_name).weight.data = weight
                    logger.debug(f"Set {proj_name} weight for {layer_name}")
        
        # 处理o_proj权重
        o_weight_name = layer_name + ".o_proj.weight"
        if o_weight_name in layer_weights and hasattr(layer_module, 'o_proj'):
            layer_module.o_proj.weight.data = layer_weights[o_weight_name]
            logger.debug(f"Set o_proj weight for {layer_name}")
    
    def _initialize_mlp_layer(self, layer_module: torch.nn.Module, layer_weights: Dict[str, torch.Tensor], layer_name: str):
        """初始化MLP层"""
        logger.debug(f"Initializing MLP layer: {layer_name}")
        
        # 处理MLP的各个投影层
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            weight_name = layer_name + "." + proj_name + ".weight"
            if weight_name in layer_weights:
                weight = layer_weights[weight_name]
                if hasattr(layer_module, proj_name):
                    getattr(layer_module, proj_name).weight.data = weight
                    logger.debug(f"Set {proj_name} weight for {layer_name}")
    
    def _initialize_generic_layer(self, layer_module: torch.nn.Module, layer_weights: Dict[str, torch.Tensor], layer_name: str):
        """通用层初始化"""
        logger.debug(f"Initializing generic layer: {layer_name}")
        
        # 直接设置权重
        for weight_name, weight_tensor in layer_weights.items():
            # 提取权重名称（移除层前缀）
            if weight_name.startswith(layer_name + '.'):
                param_name = weight_name[len(layer_name) + 1:]  # 移除层名和点号
                if hasattr(layer_module, param_name):
                    getattr(layer_module, param_name).data = weight_tensor
                    logger.debug(f"Set {param_name} for {layer_name}")
    
    def _load_base_model_weights_dict(self, base_model_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """加载基础模型权重字典"""
        try:
            # 首先尝试加载safetensors索引文件
            index_file = os.path.join(base_model_path, "model.safetensors.index.json")
            if os.path.exists(index_file):
                # logger.info(f"Loading safetensors index from {index_file}")
                return self._load_safetensors_index_weights(base_model_path, index_file)
            
            # 尝试加载单个safetensors文件
            model_file = os.path.join(base_model_path, "model.safetensors")
            if os.path.exists(model_file):
                # logger.info(f"Loading single safetensors file: {model_file}")
                return self._load_safetensors_file(model_file)
            
            # 尝试加载PyTorch格式
            model_file = os.path.join(base_model_path, "pytorch_model.bin")
            if os.path.exists(model_file):
                # logger.info(f"Loading PyTorch model file: {model_file}")
                return self._load_pytorch_file(model_file)
            
            # 尝试加载分片的PyTorch文件
            pytorch_files = [f for f in os.listdir(base_model_path) if f.startswith("pytorch_model-") and f.endswith(".bin")]
            if pytorch_files:
                # logger.info(f"Loading PyTorch sharded files: {len(pytorch_files)} files")
                return self._load_pytorch_sharded_files(base_model_path, pytorch_files)
            
            logger.warning(f"No model file found in {base_model_path}")
            # logger.info(f"Available files in {base_model_path}:")
            try:
                for f in os.listdir(base_model_path)[:10]:  # 只显示前10个文件
                    logger.info(f"  - {f}")
            except Exception as e:
                logger.error(f"Error listing directory: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load base model weights: {e}")
            return None
    
    def _load_safetensors_index_weights(self, base_model_path: str, index_file: str) -> Optional[Dict[str, torch.Tensor]]:
        """从safetensors索引文件加载权重"""
        try:
            import json
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            weight_to_file = index_data.get('weight_map', {})
            if not weight_to_file:
                logger.warning("No weight_map found in index file")
                return None
            
            all_weights = {}
            loaded_files = set()
            
            for weight_name, file_name in weight_to_file.items():
                file_path = os.path.join(base_model_path, file_name)
                
                if file_name not in loaded_files:
                    if os.path.exists(file_path):
                        logger.debug(f"Loading safetensors file: {file_path}")
                        file_weights = self._load_safetensors_file(file_path)
                        if file_weights:
                            all_weights.update(file_weights)
                            loaded_files.add(file_name)
                        else:
                            logger.warning(f"Failed to load file: {file_path}")
                    else:
                        logger.warning(f"File not found: {file_path}")
            
            # logger.info(f"Loaded {len(all_weights)} weights from {len(loaded_files)} files")
            return all_weights
            
        except Exception as e:
            logger.error(f"Error loading safetensors index: {e}")
            return None
    
    def _load_pytorch_sharded_files(self, base_model_path: str, pytorch_files: List[str]) -> Optional[Dict[str, torch.Tensor]]:
        """加载分片的PyTorch文件"""
        try:
            all_weights = {}
            
            for file_name in sorted(pytorch_files):  # 确保按顺序加载
                file_path = os.path.join(base_model_path, file_name)
                if os.path.exists(file_path):
                    logger.debug(f"Loading PyTorch file: {file_path}")
                    file_weights = self._load_pytorch_file(file_path)
                    if file_weights:
                        all_weights.update(file_weights)
                    else:
                        logger.warning(f"Failed to load file: {file_path}")
                else:
                    logger.warning(f"File not found: {file_path}")
            
            # logger.info(f"Loaded {len(all_weights)} weights from {len(pytorch_files)} PyTorch files")
            return all_weights
            
        except Exception as e:
            logger.error(f"Error loading PyTorch sharded files: {e}")
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
