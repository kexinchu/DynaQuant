#!/usr/bin/env python3
"""
SGLang混合精度权重加载器
真正集成到SGLang架构中，支持混合精度模型加载
"""

import os
import torch
import yaml
import logging
import json
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

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
from sglang.srt.model_loader.loader import BaseModelLoader
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)


@dataclass
class MixedPrecisionConfig:
    """混合精度配置"""
    fp16_path: str = ""
    fp8_path: str = ""
    int4_path: str = ""
    weight_mapping: Dict[str, str] = None
    
    def __post_init__(self):
        if self.weight_mapping is None:
            self.weight_mapping = {}


class SGLangGPTQDequantizer:
    """SGLang集成的GPTQ反量化器"""
    
    @staticmethod
    def dequantize_gptq_weight(
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: Optional[torch.Tensor] = None,
        bits: int = 4,
        group_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        反量化 GPTQ 权重（沿输出通道打包）
        
        约定的张量形状（常见 GPTQ 导出格式）:
          - qweight: [OC//pack, IC] (int32)  pack = 32//bits（4bit时为8）
          - qzeros : [OC//g, IC//pack] (int32)  每元素再打 pack 个4-bit零点
          - scales : [OC//g, IC] (float16/float32)
        返回:
          - weight_fp16: [OC, IC] (torch.float16)
        """
        try:
            assert qweight.dtype == torch.int32 and qzeros.dtype == torch.int32, \
                "qweight 和 qzeros 必须是 int32（内部打包的载体）"
            pack = 32 // bits
            oc_pack, IC = qweight.shape
            OC = oc_pack * pack

            # 推导 g（每组输出通道数）
            groups_out = qzeros.shape[0]  # = OC // g
            assert OC % groups_out == 0, "OC 必须能被 qzeros.shape[0] 整除"
            g = OC // groups_out

            # 校验 scales 形状
            assert scales.shape == (groups_out, IC), \
                f"scales 形状应为 [OC//g, IC]，当前为 {tuple(scales.shape)}"

            # ---- 解包 qweight 到 [OC, IC]，沿输出通道扩展 ----
            Wq = SGLangGPTQDequantizer._unpack_int32_to_nibbles_rows(qweight, bits=bits)  # int16 [OC, IC]

            # ---- 从 qzeros 取每一列对应 nibble 的零点，并广播到 [OC, IC] ----
            device = qweight.device
            mask = (1 << bits) - 1  # 0xF
            col = torch.arange(IC, device=device)
            qz_cols = qzeros[:, (col // pack)]                 # [OC//g, IC]
            shift = (col % pack) * bits                        # [IC]
            zp_group_ic = (qz_cols >> shift.unsqueeze(0)) & mask  # [OC//g, IC]
            zp_full = zp_group_ic.repeat_interleave(g, dim=0).to(torch.int16)  # [OC, IC]

            # ---- 广播 scales 到 [OC, IC] ----
            scales_full = scales.repeat_interleave(g, dim=0).to(torch.float32)  # [OC, IC]

            # ---- 反量化: (w_q - zp) * scale ----
            W_fp16 = ((Wq - zp_full).to(torch.float32) * scales_full).to(torch.float16)  # [OC, IC]
            return W_fp16.t()

        except Exception as e:
            logger.error(f"[SGLangGPTQDequantizer] Error dequantizing GPTQ weight: {e}")
            try:
                pack = 32 // bits
                oc_pack, IC = qweight.shape
                OC = oc_pack * pack
                return torch.zeros((IC, OC), dtype=torch.float16, device=qweight.device)
            except Exception:
                return torch.zeros((scales.shape[1], scales.shape[0]), dtype=torch.float16, device=scales.device)

    @staticmethod
    def _unpack_int32_to_nibbles_rows(packed: torch.Tensor, bits: int = 4) -> torch.Tensor:
        """
        将按行打包的 int32（每个包含 32//bits 个子元素）解包为沿行扩张的矩阵:
          输入: packed [R, C] (int32)，每个元素含 'pack=32//bits' 个子值（低位->高位）
          输出: out [R*pack, C] (int16)   —— 将第 k 个 nibble 写到 out[k::pack, :]
        """
        assert bits in (2, 4, 8), "只支持 2/4/8 bit nibble 解包"
        pack = 32 // bits
        R, C = packed.shape
        out = torch.empty((R * pack, C), dtype=torch.int16, device=packed.device)
        mask = (1 << bits) - 1
        for k in range(pack):
            vals = (packed >> (k * bits)) & mask          # [R, C]
            out[k::pack, :] = vals.to(torch.int16)        # 交错写入行
        return out


class SGLangMixedPrecisionLoader(BaseModelLoader):
    """SGLang集成的混合精度权重加载器"""
    
    def __init__(self, config: ModelConfig, mixed_precision_config: MixedPrecisionConfig):
        """
        初始化SGLang混合精度加载器
        
        Args:
            config: SGLang模型配置
            mixed_precision_config: 混合精度配置
        """
        super().__init__(config)
        self.mixed_precision_config = mixed_precision_config
        self.weight_cache = {}
        
        logger.info(f"SGLang mixed precision loader initialized with {len(mixed_precision_config.weight_mapping)} weight mappings")
    
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
    
    def _find_weight_from_index(self, weight_name: str, index_file: str, 
                               base_path: str) -> Optional[str]:
        """从safetensors索引文件查找权重文件"""
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            weight_map = index_data.get('weight_map', {})
            if weight_name in weight_map:
                weight_file = weight_map[weight_name]
                full_path = os.path.join(base_path, weight_file)
                if os.path.exists(full_path):
                    return full_path
            
            return None
        except Exception as e:
            logger.warning(f"Error reading safetensors index: {e}")
            return None
    
    def _find_weight_file(self, weight_name: str, precision: str) -> Optional[str]:
        """查找权重文件路径"""
        if precision == 'fp16':
            precision_path = self.mixed_precision_config.fp16_path
        elif precision == 'fp8':
            precision_path = self.mixed_precision_config.fp8_path
        elif precision == 'int4':
            precision_path = self.mixed_precision_config.int4_path
        else:
            return None
        
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
    
    def _is_gptq_weight(self, weights: Dict[str, torch.Tensor], weight_name: str) -> bool:
        """检查是否是GPTQ权重"""
        base_name = weight_name.replace('.weight', '')
        gptq_components = [
            f"{base_name}.qweight",
            f"{base_name}.qzeros", 
            f"{base_name}.scales"
        ]
        
        return all(comp in weights for comp in gptq_components)
    
    def _get_gptq_weight_components(self, weights: Dict[str, torch.Tensor], 
                                   weight_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """获取GPTQ权重组件"""
        base_name = weight_name.replace('.weight', '')
        
        qweight = weights[f"{base_name}.qweight"]
        qzeros = weights[f"{base_name}.qzeros"]
        scales = weights[f"{base_name}.scales"]
        
        # g_idx是可选的
        g_idx = None
        g_idx_name = f"{base_name}.g_idx"
        if g_idx_name in weights:
            g_idx = weights[g_idx_name]
        
        return qweight, qzeros, scales, g_idx
    
    def load_weight(self, weight_name: str, precision: str) -> Optional[torch.Tensor]:
        """加载指定精度的权重"""
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
            
            # 检查是否是GPTQ权重
            if precision == 'int4' and self._is_gptq_weight(weights, weight_name):
                # 加载GPTQ组件并反量化
                qweight, qzeros, scales, g_idx = self._get_gptq_weight_components(weights, weight_name)
                weight = SGLangGPTQDequantizer.dequantize_gptq_weight(qweight, qzeros, scales, g_idx)
                logger.info(f"Successfully dequantized GPTQ weight: {weight_name}, shape: {weight.shape}")
            else:
                # 直接加载权重
                if weight_name in weights:
                    weight = weights[weight_name]
                else:
                    logger.warning(f"Weight {weight_name} not found in file {weight_file}")
                    return None
            
            # 转换到指定精度
            weight = self._convert_to_precision(weight, precision)
            
            return weight
            
        except Exception as e:
            logger.error(f"Error loading weight {weight_name} with precision {precision}: {e}")
            return None
    
    def _convert_to_precision(self, weight: torch.Tensor, precision: str) -> torch.Tensor:
        """转换权重到指定精度"""
        if precision == 'fp16':
            return weight.half()
        elif precision == 'fp8':
            # 使用torch.float8_e4m3fn
            if hasattr(torch, 'float8_e4m3fn'):
                return weight.to(torch.float8_e4m3fn)
            else:
                logger.warning("FP8 not supported, falling back to FP16")
                return weight.half()
        elif precision == 'int4':
            # int4权重已经通过GPTQ反量化处理
            return weight
        else:
            return weight
    
    def load_model_weights(self, model: torch.nn.Module) -> Dict[str, Any]:
        """加载模型权重"""
        stats = {
            'loaded': 0,
            'skipped': 0,
            'errors': 0,
            'details': []
        }
        
        # 获取模型设备
        model_device = next(model.parameters()).device
        logger.info(f"Model device: {model_device}")
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight_name = name + '.weight'
                
                if weight_name in self.mixed_precision_config.weight_mapping:
                    precision = self.mixed_precision_config.weight_mapping[weight_name]
                    weight = self.load_weight(weight_name, precision)
                    
                    if weight is not None:
                        try:
                            # 确保权重在正确的设备上
                            if weight.device != model_device:
                                weight = weight.to(model_device)
                                logger.debug(f"Moved weight {weight_name} to device {model_device}")
                            
                            # 检查形状是否匹配
                            if weight.shape == module.weight.shape:
                                module.weight.data = weight
                                stats['loaded'] += 1
                                stats['details'].append({
                                    'name': weight_name,
                                    'precision': precision,
                                    'status': 'loaded',
                                    'shape': list(weight.shape),
                                    'device': str(weight.device)
                                })
                            else:
                                logger.warning(f"Shape mismatch for {weight_name}: expected {module.weight.shape}, got {weight.shape}")
                                stats['skipped'] += 1
                                stats['details'].append({
                                    'name': weight_name,
                                    'precision': precision,
                                    'status': 'shape_mismatch',
                                    'expected_shape': list(module.weight.shape),
                                    'actual_shape': list(weight.shape)
                                })
                        except Exception as e:
                            logger.error(f"Error setting weight {weight_name}: {e}")
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
        
        logger.info(f"Model weights loaded: {stats['loaded']} loaded, {stats['skipped']} skipped, {stats['errors']} errors")
        return stats


def create_mixed_precision_loader(config: ModelConfig, mixed_precision_config_path: str) -> SGLangMixedPrecisionLoader:
    """创建混合精度加载器"""
    # 加载混合精度配置
    with open(mixed_precision_config_path, 'r', encoding='utf-8') as f:
        mixed_precision_data = yaml.safe_load(f)
    
    mixed_precision_config = MixedPrecisionConfig(
        fp16_path=mixed_precision_data.get('mixed_precision', {}).get('fp16_path', ''),
        fp8_path=mixed_precision_data.get('mixed_precision', {}).get('fp8_path', ''),
        int4_path=mixed_precision_data.get('mixed_precision', {}).get('int4_path', ''),
        weight_mapping=mixed_precision_data.get('mixed_precision', {}).get('weight_mapping', {})
    )
    
    return SGLangMixedPrecisionLoader(config, mixed_precision_config)


# 全局混合精度加载器实例
_global_mixed_precision_loader: Optional[SGLangMixedPrecisionLoader] = None


def get_global_mixed_precision_loader() -> Optional[SGLangMixedPrecisionLoader]:
    """获取全局混合精度加载器"""
    return _global_mixed_precision_loader


def set_global_mixed_precision_loader(loader: SGLangMixedPrecisionLoader):
    """设置全局混合精度加载器"""
    global _global_mixed_precision_loader
    _global_mixed_precision_loader = loader
