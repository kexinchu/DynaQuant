#!/usr/bin/env python3
"""
真正的混合精度权重加载器
最大化复用SGLang的现有功能，避免重复造轮子
使用SGLang现成的量化支持，保持权重的压缩格式，多种量化方式共存以节省GPU存储
"""

import os
import torch
import yaml
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import torch.nn as nn

# SGLang核心导入 - 复用现有功能
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import (
    safetensors_weights_iterator, pt_weights_iterator
)
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.load_config import LoadConfig

# SGLang量化支持导入 - 复用现有量化功能
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.gptq import GPTQConfig
from sglang.srt.layers.quantization.awq import AWQConfig
from sglang.srt.layers.quantization.blockwise_int8 import BlockInt8Config
from sglang.srt.layers.linear import LinearBase, LinearMethodBase

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
    """真正的混合精度权重加载器 - 最大化复用SGLang现有功能"""
    
    def __init__(self, config: ModelConfig, mixed_precision_config: TrueMixedPrecisionConfig):
        """
        初始化真正的混合精度加载器
        
        Args:
            config: SGLang模型配置
            mixed_precision_config: 混合精度配置
        """
        # 复用SGLang的LoadConfig
        load_config = LoadConfig(
            load_format="auto",
            download_dir=None,
            model_loader_extra_config={},
            ignore_patterns=None
        )
        
        # 调用父类初始化，复用SGLang的现有功能
        super().__init__(load_config)
        self.mixed_precision_config = mixed_precision_config
        self.weight_cache = {}  # 复用SGLang的缓存机制
        self.compressed_weights = {}  # 存储压缩权重
        self.memory_saved = 0  # 节省的内存（字节）
        
        # 初始化量化配置 - 复用SGLang的量化支持
        self._init_quantization_configs()
        
        logger.info(f"True mixed precision loader initialized with {len(mixed_precision_config.weight_mapping)} weight mappings")
        logger.info("Using SGLang native quantization support - no de-quantization")
    
    def _init_quantization_configs(self):
        """初始化量化配置 - 复用SGLang的量化配置"""
        # 复用SGLang的量化配置，使用正确的构造函数参数
        self.quantization_configs = {
            "fp8": Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                ignored_layers=None,
                weight_block_size=None
            ),
            "gptq_int4": GPTQConfig(
                weight_bits=4,
                group_size=128,
                desc_act=True,
                lm_head_quantized=False,
                dynamic={}
            ),
            "awq_int4": AWQConfig(
                weight_bits=4,
                group_size=128,
                zero_point=True,
                modules_to_not_convert=None
            ),
            "int8": BlockInt8Config(
                is_checkpoint_int8_serialized=True,
                activation_scheme="dynamic",
                ignored_layers=None,
                weight_block_size=[128, 128]
            )
        }
        
        logger.info("Initialized quantization configs for FP8, GPTQ-Int4, AWQ-Int4, and Int8")
    
    def _get_quantization_config(self, precision: str) -> Optional[QuantizationConfig]:
        """获取量化配置"""
        return self.quantization_configs.get(precision)
    
    def _find_weight_file(self, weight_name: str, precision: str) -> Optional[str]:
        """根据权重名称和精度查找对应的权重文件 - 复用SGLang的权重查找逻辑"""
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
        
        # 复用SGLang的权重文件查找逻辑
        try:
            # 创建临时的ModelConfig
            temp_model_config = ModelConfig(
                model_path=base_path,
                dtype=torch.float16,
                trust_remote_code=True
            )
            
            # 使用SGLang的权重迭代器来查找权重文件
            source = DefaultModelLoader.Source.init_new(temp_model_config, None)
            
            # 使用SGLang的权重迭代器查找权重
            for name, weight in self._get_weights_iterator(source):
                if name == weight_name:
                    # 找到权重后，返回对应的文件路径
                    # 这里需要从权重迭代器中获取文件路径信息
                    # 由于SGLang的权重迭代器不直接提供文件路径，我们需要其他方法
                    return self._get_weight_file_path(base_path, weight_name)
            
            logger.warning(f"Weight {weight_name} not found in any files for precision {precision}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding weight file for {weight_name} with precision {precision}: {e}")
            return None
    
    def _get_weight_file_path(self, base_path: str, weight_name: str) -> Optional[str]:
        """获取权重文件路径"""
        try:
            # 检查safetensors索引文件
            index_file = os.path.join(base_path, "model.safetensors.index.json")
            if os.path.exists(index_file):
                import json
                with open(index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                
                weight_map = index_data.get('weight_map', {})
                if weight_name in weight_map:
                    file_name = weight_map[weight_name]
                    file_path = os.path.join(base_path, file_name)
                    if os.path.exists(file_path):
                        return file_path
            
            # 检查单个safetensors文件
            model_file = os.path.join(base_path, "model.safetensors")
            if os.path.exists(model_file):
                return model_file
            
            # 检查PyTorch文件
            model_file = os.path.join(base_path, "pytorch_model.bin")
            if os.path.exists(model_file):
                return model_file
            
            # 检查分片的PyTorch文件
            import glob
            pytorch_files = glob.glob(os.path.join(base_path, "pytorch_model-*.bin"))
            if pytorch_files:
                # 返回第一个文件，实际使用时需要检查权重是否在该文件中
                return pytorch_files[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting weight file path for {weight_name}: {e}")
            return None
    
    def load_weight(self, weight_name: str, precision: str) -> Optional[CompressedWeight]:
        """加载指定权重 - 复用SGLang的权重加载逻辑"""
        weight_file = self._find_weight_file(weight_name, precision)
        if not weight_file:
            return None
        
        try:
            # 对于需要多个组件的量化权重，我们需要加载整个文件
            if precision in ["gptq_int4", "awq_int4"]:
                return self._load_quantized_weight_from_file(weight_name, weight_file, precision)
            
            # 对于单个权重文件，使用原有的逻辑
            if weight_file.endswith('.safetensors'):
                # 复用SGLang的safetensors加载
                for name, weight in safetensors_weights_iterator(weight_file):
                    if name == weight_name:
                        return self._process_weight(weight_name, weight, precision)
            else:
                # 复用SGLang的PyTorch加载
                for name, weight in pt_weights_iterator(weight_file):
                    if name == weight_name:
                        return self._process_weight(weight_name, weight, precision)
            
            logger.warning(f"Weight {weight_name} not found in {weight_file}")
            return None
                
        except Exception as e:
            logger.error(f"Failed to load weight {weight_name} with precision {precision}: {e}")
            return None
    
    def _load_quantized_weight_from_file(self, weight_name: str, weight_file: str, precision: str) -> Optional[CompressedWeight]:
        """从文件中加载量化权重（需要多个组件）"""
        try:
            # 加载整个文件的所有权重
            weights = {}
            if weight_file.endswith('.safetensors'):
                for name, weight in safetensors_weights_iterator(weight_file):
                    weights[name] = weight
            else:
                for name, weight in pt_weights_iterator(weight_file):
                    weights[name] = weight
            
            # 使用量化权重加载方法
            return self._load_quantized_weight(weight_name, weights, precision)
            
        except Exception as e:
            logger.error(f"Failed to load quantized weight from file {weight_file}: {e}")
            return None
    
    def _process_weight(self, weight_name: str, weight: torch.Tensor, precision: str) -> Optional[CompressedWeight]:
        """处理权重 - 复用SGLang的权重处理逻辑"""
        try:
            # 处理qkv_proj的特殊情况
            if "qkv_proj" in weight_name:
                return self._load_qkv_weight_for_sglang(weight_name, {weight_name: weight}, precision)
            
            # 处理量化权重的特殊情况
            if precision in ["gptq_int4", "awq_int4", "fp8"]:
                # 对于量化权重，我们需要完整的权重字典来查找相关组件
                # 这里我们只能处理单个权重，所以对于需要多个组件的量化方法，
                # 应该在load_weight方法中处理，而不是在这里
                if precision == "fp8":
                    # FP8权重可能只有单个权重文件
                    return self._load_fp8_weight_compressed(weight_name, {weight_name: weight})
                else:
                    # GPTQ和AWQ需要多个组件，这里无法处理
                    logger.warning(f"Cannot process {precision} weight {weight_name} in single weight mode")
                    return None
            
            # 复用SGLang的权重形状验证逻辑
            original_shape = weight.shape
            
            # 复用SGLang的张量并行分片逻辑
            if self._is_tensor_parallel_sharding(original_shape, original_shape, weight_name):
                logger.info(f"Applying tensor parallel sharding for {weight_name}")
                sharded_weight = self._shard_weight_for_tensor_parallel(weight, weight_name, original_shape)
                weight = sharded_weight
                original_shape = weight.shape
            
            # 创建压缩权重对象
            compressed_weight = CompressedWeight(
                format=WeightFormat(precision),
                data=weight,
                metadata={},
                original_shape=original_shape,
                compressed_size=weight.numel() * weight.element_size()
            )
            
            return compressed_weight
            
        except Exception as e:
            logger.error(f"Error processing weight {weight_name}: {e}")
            return None
    
    def _load_quantized_weight(self, weight_name: str, weights: Dict[str, torch.Tensor], precision: str) -> Optional[CompressedWeight]:
        """加载量化权重 - 保持压缩格式，不进行de-quantization"""
        try:
            # 获取量化配置
            quant_config = self._get_quantization_config(precision)
            if not quant_config:
                logger.error(f"No quantization config found for precision: {precision}")
                return None
            
            # 根据精度类型处理
            if precision == "gptq_int4":
                return self._load_gptq_weight_compressed(weight_name, weights)
            elif precision == "awq_int4":
                return self._load_awq_weight_compressed(weight_name, weights)
            elif precision == "fp8":
                return self._load_fp8_weight_compressed(weight_name, weights)
            else:
                logger.warning(f"Unsupported precision for compressed loading: {precision}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load quantized weight {weight_name} with precision {precision}: {e}")
            return None
    
    def _load_gptq_weight_compressed(self, weight_name: str, weights: Dict[str, torch.Tensor]) -> Optional[CompressedWeight]:
        """加载GPTQ权重 - 保持压缩格式"""
        base_name = weight_name.replace(".weight", "")
        
        # 查找GPTQ组件
        qweight_name = base_name + ".qweight"
        qzeros_name = base_name + ".qzeros"
        scales_name = base_name + ".scales"
        g_idx_name = base_name + ".g_idx"
        
        if qweight_name in weights and qzeros_name in weights and scales_name in weights:
            qweight = weights[qweight_name]
            qzeros = weights[qzeros_name]
            scales = weights[scales_name]
            g_idx = weights.get(g_idx_name, None)
            
            # 计算原始形状
            pack = 32 // 4  # 4-bit packing
            oc_pack, ic = qweight.shape
            oc = oc_pack * pack
            
            original_shape = (oc, ic)
            
            # 创建压缩权重对象 - 保持压缩格式
            compressed_weight = CompressedWeight(
                format=WeightFormat.GPTQ_INT4,
                data={
                    'qweight': qweight,
                    'qzeros': qzeros,
                    'scales': scales,
                    'g_idx': g_idx
                },
                metadata={
                    'bits': 4,
                    'group_size': 128,
                    'pack': pack
                },
                original_shape=original_shape,
                compressed_size=qweight.numel() * qweight.element_size() + 
                              qzeros.numel() * qzeros.element_size() + 
                              scales.numel() * scales.element_size()
            )
            
            logger.info(f"Loaded compressed GPTQ weight {weight_name}, shape: {original_shape}")
            return compressed_weight
        else:
            logger.warning(f"GPTQ components not found for {weight_name}")
            return None
    
    def _load_awq_weight_compressed(self, weight_name: str, weights: Dict[str, torch.Tensor]) -> Optional[CompressedWeight]:
        """加载AWQ权重 - 保持压缩格式"""
        base_name = weight_name.replace(".weight", "")
        
        # 查找AWQ组件
        qweight_name = base_name + ".qweight"
        qzeros_name = base_name + ".qzeros"
        scales_name = base_name + ".scales"
        qweight_scale_name = base_name + ".qweight_scale"
        
        if qweight_name in weights and qzeros_name in weights and scales_name in weights:
            qweight = weights[qweight_name]
            qzeros = weights[qzeros_name]
            scales = weights[scales_name]
            qweight_scale = weights.get(qweight_scale_name, None)
            
            # 计算原始形状
            pack = 32 // 4  # 4-bit packing
            oc_pack, ic = qweight.shape
            oc = oc_pack * pack
            
            original_shape = (oc, ic)
            
            # 创建压缩权重对象 - 保持压缩格式
            compressed_weight = CompressedWeight(
                format=WeightFormat.AWQ_INT4,
                data={
                    'qweight': qweight,
                    'qzeros': qzeros,
                    'scales': scales,
                    'qweight_scale': qweight_scale
                },
                metadata={
                    'bits': 4,
                    'group_size': 128,
                    'pack': pack
                },
                original_shape=original_shape,
                compressed_size=qweight.numel() * qweight.element_size() + 
                              qzeros.numel() * qzeros.element_size() + 
                              scales.numel() * scales.element_size()
            )
            
            logger.info(f"Loaded compressed AWQ weight {weight_name}, shape: {original_shape}")
            return compressed_weight
        else:
            logger.warning(f"AWQ components not found for {weight_name}")
            return None
    
    def _load_fp8_weight_compressed(self, weight_name: str, weights: Dict[str, torch.Tensor]) -> Optional[CompressedWeight]:
        """加载FP8权重 - 保持压缩格式"""
        # 检查权重和scale_inv是否存在
        weight_data = weights.get(weight_name)
        scale_name = weight_name + "_scale_inv"
        scale_data = weights.get(scale_name, None)
        
        if weight_data is not None:
            original_shape = weight_data.shape
            
            # 创建压缩权重对象 - 保持压缩格式
            compressed_weight = CompressedWeight(
                format=WeightFormat.FP8,
                data={
                    'weight': weight_data,
                    'scale_inv': scale_data
                },
                metadata={
                    'bits': 8,
                    'dtype': 'fp8'
                },
                original_shape=original_shape,
                compressed_size=weight_data.numel() * weight_data.element_size()
            )
            
            logger.info(f"Loaded compressed FP8 weight {weight_name}, shape: {original_shape}")
            return compressed_weight
        else:
            logger.warning(f"FP8 weight {weight_name} not found")
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
            
            # 检查权重形状兼容性
            logger.debug(f"QKV weights shapes - q: {q_weight.shape}, k: {k_weight.shape}, v: {v_weight.shape}")
            
            # 验证权重形状是否兼容
            q_input_size = q_weight.shape[0]
            k_input_size = k_weight.shape[0]
            v_input_size = v_weight.shape[0]
            
            logger.debug(f"QKV input sizes - q: {q_input_size}, k: {k_input_size}, v: {v_input_size}")
            
            # 检查是否所有输入维度都相同（标准情况）
            if q_input_size == k_input_size == v_input_size:
                logger.debug("Standard QKV weights - all input dimensions match")
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=-1)
            else:
                # 检查是否是分片权重
                logger.warning(f"QKV weights appear to be sharded or have different input dimensions")
                logger.warning(f"This might be due to tensor parallelism or special weight processing")
                
                # 首先检查是否是GQA模型
                if self._is_gqa_model(q_weight, k_weight, v_weight):
                    logger.info("Detected GQA model - handling grouped-query attention")
                    qkv_weight = self._handle_gqa_qkv_weights(q_weight, k_weight, v_weight)
                    logger.debug("Successfully handled GQA QKV weights")
                # 然后检查是否是张量并行分片
                elif self._is_tensor_parallel_qkv_sharding(q_weight, k_weight, v_weight):
                    logger.info("Detected tensor parallel QKV sharding - performing proper sharding")
                    qkv_weight = self._shard_qkv_weights_for_tensor_parallel(q_weight, k_weight, v_weight)
                    logger.debug("Successfully sharded QKV weights for tensor parallel")
                else:
                    # Fallback to padding strategy if neither GQA nor known TP sharding
                    max_input_size = max(q_input_size, k_input_size, v_input_size)
                    
                    # Pad smaller weights with zeros along dim=0
                    if q_input_size < max_input_size:
                        padding = torch.zeros(max_input_size - q_input_size, q_weight.shape[1], 
                                            dtype=q_weight.dtype, device=q_weight.device)
                        q_weight = torch.cat([q_weight, padding], dim=0)
                    
                    if k_input_size < max_input_size:
                        padding = torch.zeros(max_input_size - k_input_size, k_weight.shape[1], 
                                            dtype=k_weight.dtype, device=k_weight.device)
                        k_weight = torch.cat([k_weight, padding], dim=0)
                    
                    if v_input_size < max_input_size:
                        padding = torch.zeros(max_input_size - v_input_size, v_weight.shape[1], 
                                            dtype=v_weight.dtype, device=v_weight.device)
                        v_weight = torch.cat([v_weight, padding], dim=0)
                    
                    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=-1)
                    logger.warning(f"Padded QKV weights to max input size {max_input_size}")
            
            # 创建压缩权重对象
            compressed_weight = CompressedWeight(
                format=WeightFormat(precision),
                data=qkv_weight,
                metadata={},
                original_shape=qkv_weight.shape,
                compressed_size=qkv_weight.numel() * qkv_weight.element_size()
            )
            
            logger.info(f"Loaded QKV weight {weight_name}, shape: {qkv_weight.shape}")
            return compressed_weight
        else:
            logger.warning(f"QKV components not found for {weight_name}")
            return None
    
    def _is_gqa_model(self, q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor) -> bool:
        """检查是否是GQA (Grouped-Query Attention) 模型"""
        try:
            q_input_size = q_weight.shape[0]
            k_input_size = k_weight.shape[0]
            v_input_size = v_weight.shape[0]
            
            if (q_input_size > k_input_size and q_input_size > v_input_size and 
                k_input_size == v_input_size and q_input_size % k_input_size == 0):
                
                gqa_ratio = q_input_size // k_input_size
                logger.debug(f"Detected potential GQA model with ratio {gqa_ratio}:1")
                
                if gqa_ratio in [2, 4, 8, 16]:
                    logger.info(f"Detected GQA model with {gqa_ratio}:1 ratio")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking GQA model: {e}")
            return False
    
    def _handle_gqa_qkv_weights(self, q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor) -> torch.Tensor:
        """处理GQA (Grouped-Query Attention) 模型的QKV权重"""
        try:
            q_input_size = q_weight.shape[0]
            k_input_size = k_weight.shape[0]
            v_input_size = v_weight.shape[0]
            
            gqa_ratio = q_input_size // k_input_size
            logger.info(f"Handling GQA model with {gqa_ratio}:1 ratio")
            
            # Get current tensor parallel rank (assuming tp_size=4 from user's script)
            tp_rank = 0
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    tp_rank = dist.get_rank() % 4  # tp_size=4
                    logger.info(f"Current tensor parallel rank: {tp_rank}")
            except Exception as e:
                logger.warning(f"Could not get tensor parallel rank: {e}, using rank 0")
            
            # GQA sharding strategy for tp_size=4
            # Each rank handles 1/4 of the query heads
            q_shard_size = q_input_size // 4  # e.g., 4096 // 4 = 1024
            start_idx = tp_rank * q_shard_size
            end_idx = start_idx + q_shard_size
            q_shard = q_weight[start_idx:end_idx, :]
            
            # k_proj and v_proj also need sharding for tp_size=4
            # Each rank handles 1/4 of the KV heads
            k_shard_size = k_input_size // 4  # e.g., 512 // 4 = 128
            k_start_idx = tp_rank * k_shard_size
            k_end_idx = k_start_idx + k_shard_size
            k_shard = k_weight[k_start_idx:k_end_idx, :]
            
            v_shard_size = v_input_size // 4  # e.g., 512 // 4 = 128
            v_start_idx = tp_rank * v_shard_size
            v_end_idx = v_start_idx + v_shard_size
            v_shard = v_weight[v_start_idx:v_end_idx, :]
            
            logger.debug(f"GQA QKV shard shapes - q: {q_shard.shape}, k: {k_shard.shape}, v: {v_shard.shape}")
            
            qkv_shard = torch.cat([q_shard, k_shard, v_shard], dim=-1)
            logger.info(f"Successfully created GQA QKV shard for rank {tp_rank}, shape: {qkv_shard.shape}")
            return qkv_shard
        except Exception as e:
            logger.error(f"Error handling GQA QKV weights: {e}")
            logger.warning("Falling back to original QKV weights")
            return q_weight
    
    def _is_tensor_parallel_qkv_sharding(self, q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor) -> bool:
        """检查是否是张量并行QKV分片的情况"""
        try:
            q_input_size = q_weight.shape[0]
            k_input_size = k_weight.shape[0]
            v_input_size = v_weight.shape[0]
            
            # 检查是否是典型的张量并行分片模式
            if (q_input_size == k_input_size == v_input_size):
                # 标准情况，不是分片
                return False
            
            # 检查是否是张量并行分片
            if (q_input_size > k_input_size and q_input_size > v_input_size and 
                k_input_size == v_input_size):
                
                # 计算分片大小
                tp_size = q_input_size // k_input_size
                
                # 检查是否是合理的分片大小
                if tp_size in [2, 4, 8, 16] and q_input_size % tp_size == 0:
                    logger.info(f"Detected tensor parallel QKV sharding with tp_size={tp_size}")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking tensor parallel QKV sharding: {e}")
            return False
    
    def _shard_qkv_weights_for_tensor_parallel(self, q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor) -> torch.Tensor:
        """为张量并行分片QKV权重"""
        try:
            q_input_size = q_weight.shape[0]
            k_input_size = k_weight.shape[0]
            v_input_size = v_weight.shape[0]
            
            # 计算张量并行大小
            if q_input_size > k_input_size and q_input_size > v_input_size:
                tp_size = q_input_size // k_input_size
            else:
                # 默认使用4-way张量并行
                tp_size = 4
                logger.warning(f"Could not determine tp_size, using default {tp_size}")
            
            # 获取当前张量并行rank
            tp_rank = 0
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    tp_rank = dist.get_rank() % tp_size
            except Exception as e:
                logger.warning(f"Could not get tensor parallel rank: {e}, using rank 0")
            
            # 分片Q权重
            q_shard_size = q_input_size // tp_size
            q_start_idx = tp_rank * q_shard_size
            q_end_idx = q_start_idx + q_shard_size
            q_shard = q_weight[q_start_idx:q_end_idx, :]
            
            # 分片K权重
            k_shard_size = k_input_size // tp_size
            k_start_idx = tp_rank * k_shard_size
            k_end_idx = k_start_idx + k_shard_size
            k_shard = k_weight[k_start_idx:k_end_idx, :]
            
            # 分片V权重
            v_shard_size = v_input_size // tp_size
            v_start_idx = tp_rank * v_shard_size
            v_end_idx = v_start_idx + v_shard_size
            v_shard = v_weight[v_start_idx:v_end_idx, :]
            
            # 合并分片
            qkv_shard = torch.cat([q_shard, k_shard, v_shard], dim=-1)
            
            logger.info(f"Successfully sharded QKV weights for tensor parallel rank {tp_rank}, shape: {qkv_shard.shape}")
            return qkv_shard
            
        except Exception as e:
            logger.error(f"Error sharding QKV weights for tensor parallel: {e}")
            return q_weight
    
    def _is_tensor_parallel_sharding(self, model_shape: torch.Size, compressed_shape: torch.Size, weight_name: str) -> bool:
        """检查是否是张量并行分片的情况"""
        try:
            # 复用SGLang的张量并行检测逻辑
            if model_shape == compressed_shape:
                return False
            
            # 检查是否是典型的张量并行分片模式
            if "embed_tokens" in weight_name:
                # 嵌入层：vocab_size被分片
                if len(model_shape) == 2 and len(compressed_shape) == 2:
                    if model_shape[1] == compressed_shape[1]:  # hidden_size相同
                        return model_shape[0] > compressed_shape[0] and model_shape[0] % compressed_shape[0] == 0
            
            elif any(proj in weight_name for proj in ["o_proj", "gate_proj", "up_proj", "down_proj"]):
                # 线性层：输出维度被分片
                if len(model_shape) == 2 and len(compressed_shape) == 2:
                    if model_shape[0] == compressed_shape[0]:  # 输入维度相同
                        return model_shape[1] > compressed_shape[1] and model_shape[1] % compressed_shape[1] == 0
                    elif model_shape[1] == compressed_shape[1]:  # 输出维度相同
                        return model_shape[0] > compressed_shape[0] and model_shape[0] % compressed_shape[0] == 0
            
            return False
        except Exception as e:
            logger.error(f"Error checking tensor parallel sharding: {e}")
            return False
    
    def _shard_weight_for_tensor_parallel(self, weight: torch.Tensor, weight_name: str, target_shape: torch.Size) -> torch.Tensor:
        """为张量并行分片权重"""
        try:
            current_shape = weight.shape
            
            # 获取当前张量并行rank
            tp_rank = 0
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    tp_rank = dist.get_rank() % 4  # 假设tp_size=4
            except Exception as e:
                logger.warning(f"Could not get tensor parallel rank: {e}, using rank 0")
            
            # 根据权重类型进行分片
            if "embed_tokens" in weight_name:
                # 嵌入层：vocab_size被分片
                if len(current_shape) == 2 and len(target_shape) == 2:
                    if current_shape[1] == target_shape[1]:  # hidden_size相同
                        vocab_size_full = current_shape[0]
                        vocab_size_shard = target_shape[0]
                        tp_size = vocab_size_full // vocab_size_shard
                        
                        # 计算当前rank对应的vocab范围
                        start_idx = tp_rank * vocab_size_shard
                        end_idx = start_idx + vocab_size_shard
                        
                        # 分片嵌入权重
                        sharded_weight = weight[start_idx:end_idx, :]
                        logger.info(f"Sharded embedding weight for rank {tp_rank}: {current_shape} -> {sharded_weight.shape}")
                        return sharded_weight
            
            elif any(proj in weight_name for proj in ["o_proj", "gate_proj", "up_proj", "down_proj"]):
                # 线性层：输出维度被分片
                if len(current_shape) == 2 and len(target_shape) == 2:
                    if current_shape[0] == target_shape[0]:  # 输入维度相同
                        output_size_full = current_shape[1]
                        output_size_shard = target_shape[1]
                        tp_size = output_size_full // output_size_shard
                        
                        # 计算当前rank对应的输出维度范围
                        start_idx = tp_rank * output_size_shard
                        end_idx = start_idx + output_size_shard
                        
                        # 分片线性层权重
                        sharded_weight = weight[:, start_idx:end_idx]
                        logger.info(f"Sharded linear weight for rank {tp_rank}: {current_shape} -> {sharded_weight.shape}")
                        return sharded_weight
                    
                    elif current_shape[1] == target_shape[1]:  # 输出维度相同
                        input_size_full = current_shape[0]
                        input_size_shard = target_shape[0]
                        tp_size = input_size_full // input_size_shard
                        
                        # 计算当前rank对应的输入维度范围
                        start_idx = tp_rank * input_size_shard
                        end_idx = start_idx + input_size_shard
                        
                        # 分片线性层权重
                        sharded_weight = weight[start_idx:end_idx, :]
                        logger.info(f"Sharded linear weight for rank {tp_rank}: {current_shape} -> {sharded_weight.shape}")
                        return sharded_weight
            
            # 如果无法分片，返回原始权重
            logger.warning(f"Could not shard weight {weight_name}, returning original")
            return weight
            
        except Exception as e:
            logger.error(f"Error sharding weight {weight_name} for tensor parallel: {e}")
            return weight


    def initialize_specific_layers(self, model: torch.nn.Module, base_model_path: str):
        """初始化特定层 - 使用基础模型路径"""
        logger.info("Initializing specific layers from base model...")
        
        try:
            # 创建临时的ModelConfig用于基础模型
            temp_model_config = ModelConfig(
                model_path=base_model_path,
                dtype=torch.float16,
                trust_remote_code=True
            )
            
            # 使用SGLang的权重迭代器加载基础模型权重
            source = DefaultModelLoader.Source.init_new(temp_model_config, None)
            
            # 获取需要初始化的层列表
            layers_to_initialize = set()
            for weight_name in self.mixed_precision_config.weight_mapping.keys():
                # 提取层名称（去掉.weight后缀）
                layer_name = weight_name.replace('.weight', '')
                layers_to_initialize.add(layer_name)
            
            logger.info(f"Found {len(layers_to_initialize)} layers to initialize")
            
            # 加载基础模型权重并初始化特定层
            initialized_count = 0
            for name, weight in self._get_weights_iterator(source):
                # 检查是否是需要的层
                layer_name = name.replace('.weight', '')
                if layer_name in layers_to_initialize:
                    # 初始化该层的权重
                    if self._initialize_layer_weight(model, name, weight):
                        initialized_count += 1
                        logger.debug(f"Initialized layer: {layer_name}")
            
            logger.info(f"Successfully initialized {initialized_count} layers from base model")
            
        except Exception as e:
            logger.error(f"Error initializing specific layers: {e}")
    
    def _initialize_layer_weight(self, model: torch.nn.Module, weight_name: str, weight: torch.Tensor) -> bool:
        """初始化单个层的权重"""
        try:
            # 根据权重名称找到对应的模块
            module_names = weight_name.split('.')
            current_module = model
            
            # 遍历模块路径
            for i, module_name in enumerate(module_names[:-1]):  # 除了最后一个（权重名称）
                if hasattr(current_module, module_name):
                    current_module = getattr(current_module, module_name)
                else:
                    # 检查是否是数字模块名（可能是专家编号）
                    if module_name.isdigit():
                        # 尝试查找数字索引的模块
                        try:
                            expert_id = int(module_name)
                            # 检查是否是ModuleList或ModuleDict
                            if isinstance(current_module, (nn.ModuleList, nn.ModuleDict)):
                                if expert_id < len(current_module):
                                    current_module = current_module[expert_id]
                                else:
                                    logger.debug(f"Expert {expert_id} not found in module list/dict, skipping")
                                    return False
                            else:
                                logger.debug(f"Module {module_name} (expert number) not found in {current_module}, skipping")
                                return False
                        except (ValueError, IndexError):
                            logger.debug(f"Could not access expert {module_name}, skipping")
                            return False
                    else:
                        logger.warning(f"Module {module_name} not found in {current_module}")
                        return False
            
            # 设置权重
            weight_param_name = module_names[-1]  # 权重参数名称
            if hasattr(current_module, weight_param_name):
                weight_param = getattr(current_module, weight_param_name)
                if isinstance(weight_param, nn.Parameter):
                    weight_param.data = weight
                    return True
                else:
                    logger.warning(f"{weight_param_name} is not a Parameter in {current_module}")
                    return False
            else:
                logger.warning(f"Weight {weight_param_name} not found in {current_module}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing layer weight {weight_name}: {e}")
            return False
    
    def load_model_weights(self, model: torch.nn.Module) -> Dict[str, Any]:
        """加载模型权重 - 返回加载统计信息"""
        logger.info("Loading mixed precision model weights...")
        
        stats = {
            'loaded': 0,
            'total': len(self.mixed_precision_config.weight_mapping),
            'memory_saved_mb': 0.0,
            'base_model_loaded': False
        }
        
        try:
            # 加载混合精度权重
            for weight_name, precision in self.mixed_precision_config.weight_mapping.items():
                compressed_weight = self.load_weight(weight_name, precision)
                if compressed_weight:
                    # 存储压缩权重供后续使用
                    self.compressed_weights[weight_name] = compressed_weight
                    stats['loaded'] += 1
                    
                    # 计算内存节省
                    original_size = compressed_weight.original_shape[0] * compressed_weight.original_shape[1] * 2  # FP16
                    compressed_size = compressed_weight.compressed_size
                    memory_saved = original_size - compressed_size
                    stats['memory_saved_mb'] += memory_saved / (1024 * 1024)  # 转换为MB
                    
                    logger.debug(f"Loaded {weight_name} with precision {precision}")
                else:
                    logger.warning(f"Failed to load {weight_name} with precision {precision}")
            
            logger.info(f"Mixed precision weights loaded: {stats['loaded']}/{stats['total']}")
            logger.info(f"Memory saved: {stats['memory_saved_mb']:.2f}MB")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            return stats


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
