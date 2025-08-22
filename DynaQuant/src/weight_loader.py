import os
import torch
import torch.nn as nn
import yaml
import json
from typing import Dict, Any, Optional, Tuple
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
        
        # 首先尝试使用safetensors索引文件
        index_file = os.path.join(precision_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            weight_file = self._find_weight_from_index(weight_name, index_file, precision_path)
            if weight_file:
                return weight_file
        # may not exist model.safetensors.index.json; in Qwen3-30B-A3B
        weight_file = os.path.join(precision_path, "model.safetensors")
        if os.path.exists(weight_file):
            return weight_file
        
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
                    return full_path
            
            return None
        except Exception as e:
            print(f"Warning: Failed to read index file {index_file}: {e}")
            return None
    
    def load_weight(self, weight_name: str, precision: str) -> Optional[torch.Tensor]:
        """
        根据配置加载指定权重
        
        Args:
            weight_name: 权重名称
            precision: 精度类型
            
        Returns:
            加载的权重张量
        """        
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
        
        print(f"Loading weight: {weight_name} with precision: {precision}")

        # 检查是否是GPTQ权重
        if precision == 'int4' and self._is_gptq_format(weights):
            return self._load_gptq_weight(weight_name, weights)
        
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
    
    def _is_gptq_format(self, weights: Dict[str, torch.Tensor]) -> bool:
        """检查权重文件是否是GPTQ格式"""
        # 检查是否包含GPTQ特有的键
        gptq_keys = ['qweight', 'qzeros', 'scales']
        return any(any(key in weight_name for key in gptq_keys) for weight_name in weights.keys())
    
    def _load_gptq_weight(self, weight_name: str, weights: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        加载GPTQ格式的权重
        
        Args:
            weight_name: 权重名称
            weights: 权重字典
            
        Returns:
            反量化后的权重张量
        """
        # 获取GPTQ组件
        components = self._get_gptq_weight_components(weight_name, weights)
        if components is None:
            return None
        
        qweight, qzeros, scales, g_idx = components
        
        # 反量化权重
        dequantized_weight = self._dequantize_gptq_weight(qweight, qzeros, scales, g_idx)
        
        print(f"Successfully dequantized GPTQ weight: {weight_name}, shape: {dequantized_weight.shape}")
        return dequantized_weight
    
    def _process_int4_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """处理Int4权重"""
        # 这里可以根据具体的Int4格式进行调整
        # 假设权重已经是正确的Int4格式
        return weight
    
    def _is_gptq_weight(self, weight_name: str) -> bool:
        """检查是否是GPTQ权重"""
        gptq_suffixes = ['qweight', 'qzeros', 'scales', 'g_idx']
        return any(suffix in weight_name for suffix in gptq_suffixes)
    
    def _get_gptq_weight_components(self, weight_name: str, weights: Dict[str, torch.Tensor]) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        获取GPTQ权重的所有组件
        
        Args:
            weight_name: 权重名称（如 'model.layers.0.mlp.experts.0.down_proj.weight'）
            weights: 权重字典
            
        Returns:
            (qweight, qzeros, scales, g_idx) 或 None
        """
        # 从权重名称中提取基础名称
        base_name = weight_name.replace('.weight', '')
        
        # 构建GPTQ组件名称
        qweight_name = f"{base_name}.qweight"
        qzeros_name = f"{base_name}.qzeros"
        scales_name = f"{base_name}.scales"
        g_idx_name = f"{base_name}.g_idx"
        
        # 检查所有必需的组件是否存在
        if (qweight_name in weights and 
            qzeros_name in weights and 
            scales_name in weights):
            
            qweight = weights[qweight_name]
            qzeros = weights[qzeros_name]
            scales = weights[scales_name]
            g_idx = weights.get(g_idx_name, None)  # g_idx是可选的
            
            return qweight, qzeros, scales, g_idx
        else:
            print(f"Warning: GPTQ components not found for {weight_name}")
            print(f"  Looking for: {qweight_name}, {qzeros_name}, {scales_name}")
            print(f"  Available keys: {list(weights.keys())[:10]}...")  # 只显示前10个键
            return None
    
    def _dequantize_gptq_weight(self, qweight: torch.Tensor, qzeros: torch.Tensor, 
                               scales: torch.Tensor, g_idx: Optional[torch.Tensor] = None,
                               bits: int = 4, group_size: int = 128) -> torch.Tensor:
        """
        反量化GPTQ权重
        
        Args:
            qweight: 量化的权重
            qzeros: 量化的零点
            scales: 缩放因子
            g_idx: 分组索引（可选）
            bits: 量化位数
            group_size: 分组大小
            
        Returns:
            反量化后的权重
        """
        try:
            # 解包int32到int4
            if qweight.dtype == torch.int32:
                unpacked = self._unpack_int32_to_int4_optimized(qweight, bits)
            else:
                unpacked = qweight
            
            # 确保维度匹配
            if len(unpacked.shape) == 2 and len(scales.shape) == 2:
                # 如果scales是2D，需要扩展维度
                scales_expanded = scales.unsqueeze(-1)
                qzeros_expanded = qzeros.unsqueeze(-1)
            else:
                scales_expanded = scales
                qzeros_expanded = qzeros
            
            # 应用GPTQ反量化公式
            # weight = scale * (qweight - qzero)
            weight = scales_expanded * (unpacked.float() - qzeros_expanded)
            
            return weight
            
        except Exception as e:
            print(f"Error dequantizing GPTQ weight: {e}")
            print(f"  qweight shape: {qweight.shape}, dtype: {qweight.dtype}")
            print(f"  qzeros shape: {qzeros.shape}, dtype: {qzeros.dtype}")
            print(f"  scales shape: {scales.shape}, dtype: {scales.dtype}")
            # 返回一个零张量作为fallback
            return torch.zeros(scales.shape[0], unpacked.shape[1] if len(unpacked.shape) > 1 else 1)
    
    def _unpack_int32_to_int4(self, packed: torch.Tensor, bits: int = 4) -> torch.Tensor:
        """
        将packed int32解包为int4
        
        Args:
            packed: packed的int32张量
            bits: 每个int32中packed的位数
            
        Returns:
            解包后的int4张量
        """
        # 每个int32包含8个int4值
        elements_per_int32 = 32 // bits
        
        # 创建掩码
        mask = (1 << bits) - 1
        
        # 解包
        unpacked = []
        for i in range(elements_per_int32):
            shift = i * bits
            element = (packed >> shift) & mask
            unpacked.append(element)
        
        # 连接所有元素
        result = torch.stack(unpacked, dim=-1)
        return result.view(packed.shape[0], -1)
    
    def _unpack_int32_to_int4_optimized(self, packed: torch.Tensor, bits: int = 4) -> torch.Tensor:
        """
        优化的int32到int4解包函数
        
        Args:
            packed: packed的int32张量
            bits: 每个int32中packed的位数
            
        Returns:
            解包后的int4张量
        """
        # 使用更高效的方法
        if bits == 4:
            # 对于4位量化，每个int32包含8个int4值
            # 使用位操作进行解包
            unpacked = torch.zeros(packed.shape[0], packed.shape[1] * 8, dtype=torch.int32)
            
            for i in range(8):
                shift = i * 4
                mask = 0xF  # 4位掩码
                unpacked[:, i::8] = (packed >> shift) & mask
            
            return unpacked
        else:
            # 通用方法
            return self._unpack_int32_to_int4(packed, bits)
    
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
        
        # 统计信息
        loaded_count = 0
        skipped_count = 0
        error_count = 0
        
        # 遍历模型权重
        for name, param in model.named_parameters():
            # 只处理专家层权重
            if "expert" not in name:
                continue
                
            if name in state_dict:
                if name not in self.weight_mapping:
                    # 默认量化粒度-跳过
                    skipped_count += 1
                    continue
                else:
                    precision = self.weight_mapping[name]
                
                try:
                    weight = self.load_weight(name, precision)
                    if weight is not None:
                        # 确保权重形状匹配
                        if weight.shape == param.shape:
                            param.data = weight.to(param.device)
                            loaded_count += 1
                            print(f"✓ Loaded {name} with shape {weight.shape} (precision: {precision})")
                        else:
                            print(f"⚠ Warning: Shape mismatch for {name}: expected {param.shape}, got {weight.shape}")
                            error_count += 1
                    else:
                        print(f"⚠ Warning: Failed to load weight {name}")
                        error_count += 1
                except Exception as e:
                    print(f"✗ Error loading {name}: {e}")
                    error_count += 1
        
        print(f"Mixed precision weights loading completed!")
        print(f"  Loaded: {loaded_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Errors: {error_count}")
    
    def get_weight_info(self) -> Dict[str, Any]:
        """获取权重信息"""
        info = {
            'precision_paths': self.precision_paths,
            'weight_mapping': self.weight_mapping,
            'cached_files': list(self.weight_cache.keys())
        }
        return info
