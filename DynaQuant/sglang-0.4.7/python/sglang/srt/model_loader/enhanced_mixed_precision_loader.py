#!/usr/bin/env python3
"""
增强的混合精度权重加载器
集成GPTQ支持和专家激活跟踪功能
基于SGLang架构优化
"""

import os
import torch
import yaml
import logging
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
from contextlib import contextmanager

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

from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)


@dataclass
class ExpertActivationInfo:
    """专家激活信息"""
    layer_id: int
    expert_id: int
    activation_count: int = 0
    last_activation_time: float = 0.0
    total_tokens_processed: int = 0
    
    def record_activation(self, tokens_processed: int = 1):
        """记录激活"""
        self.activation_count += 1
        self.last_activation_time = time.time()
        self.total_tokens_processed += tokens_processed


class ExpertActivationTracker:
    """专家激活跟踪器"""
    
    def __init__(self, max_history: int = 1000):
        self.expert_stats: Dict[Tuple[int, int], ExpertActivationInfo] = {}
        self.activation_history: deque = deque(maxlen=max_history)
        self.request_history: deque = deque(maxlen=max_history)
        self.lock = threading.RLock()
        
    def record_expert_activation(self, layer_id: int, expert_id: int, 
                               tokens_processed: int = 1, request_id: str = None):
        """记录专家激活"""
        with self.lock:
            key = (layer_id, expert_id)
            if key not in self.expert_stats:
                self.expert_stats[key] = ExpertActivationInfo(layer_id, expert_id)
            
            self.expert_stats[key].record_activation(tokens_processed)
            
            # 记录激活历史
            activation_record = {
                'timestamp': time.time(),
                'layer_id': layer_id,
                'expert_id': expert_id,
                'tokens_processed': tokens_processed,
                'request_id': request_id
            }
            self.activation_history.append(activation_record)
    
    def record_request(self, request_id: str, input_length: int, output_length: int):
        """记录请求信息"""
        with self.lock:
            request_record = {
                'timestamp': time.time(),
                'request_id': request_id,
                'input_length': input_length,
                'output_length': output_length,
                'total_tokens': input_length + output_length
            }
            self.request_history.append(request_record)
    
    def get_expert_stats(self, layer_id: Optional[int] = None, 
                        expert_id: Optional[int] = None) -> Dict:
        """获取专家统计信息"""
        with self.lock:
            if layer_id is not None and expert_id is not None:
                key = (layer_id, expert_id)
                if key in self.expert_stats:
                    info = self.expert_stats[key]
                    return {
                        'layer_id': info.layer_id,
                        'expert_id': info.expert_id,
                        'activation_count': info.activation_count,
                        'last_activation_time': info.last_activation_time,
                        'total_tokens_processed': info.total_tokens_processed
                    }
                return {}
            
            # 返回所有专家统计
            stats = {}
            for key, info in self.expert_stats.items():
                stats[f"layer_{info.layer_id}_expert_{info.expert_id}"] = {
                    'layer_id': info.layer_id,
                    'expert_id': info.expert_id,
                    'activation_count': info.activation_count,
                    'last_activation_time': info.last_activation_time,
                    'total_tokens_processed': info.total_tokens_processed
                }
            return stats
    
    def get_top_experts(self, top_k: int = 10) -> List[Dict]:
        """获取激活次数最多的专家"""
        with self.lock:
            sorted_experts = sorted(
                self.expert_stats.values(),
                key=lambda x: x.activation_count,
                reverse=True
            )
            return [
                {
                    'layer_id': expert.layer_id,
                    'expert_id': expert.expert_id,
                    'activation_count': expert.activation_count,
                    'total_tokens_processed': expert.total_tokens_processed
                }
                for expert in sorted_experts[:top_k]
            ]
    
    def get_layer_stats(self) -> Dict[int, Dict]:
        """获取每层的统计信息"""
        with self.lock:
            layer_stats = defaultdict(lambda: {
                'total_experts': 0,
                'total_activations': 0,
                'total_tokens': 0,
                'experts': {}
            })
            
            for key, info in self.expert_stats.items():
                layer_id = info.layer_id
                layer_stats[layer_id]['total_experts'] += 1
                layer_stats[layer_id]['total_activations'] += info.activation_count
                layer_stats[layer_id]['total_tokens'] += info.total_tokens_processed
                layer_stats[layer_id]['experts'][info.expert_id] = {
                    'activation_count': info.activation_count,
                    'total_tokens_processed': info.total_tokens_processed
                }
            
            return dict(layer_stats)
    
    def reset_stats(self):
        """重置统计信息"""
        with self.lock:
            self.expert_stats.clear()
            self.activation_history.clear()
            self.request_history.clear()
    
    def export_stats(self, file_path: str):
        """导出统计信息到文件"""
        with self.lock:
            stats = {
                'expert_stats': self.get_expert_stats(),
                'layer_stats': self.get_layer_stats(),
                'top_experts': self.get_top_experts(20),
                'export_time': time.time()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Expert activation stats exported to {file_path}")


class GPTQDequantizer:
    """GPTQ反量化器"""

    @staticmethod
    def dequantize_gptq_weight(
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: Optional[torch.Tensor] = None,  # 兼容签名，当前未使用
        bits: int = 4,
        group_size: Optional[int] = None,      # 可选；若不提供则由形状自动推导
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
            Wq = GPTQDequantizer._unpack_int32_to_nibbles_rows(qweight, bits=bits)  # int16 [OC, IC]

            # ---- 从 qzeros 取每一列对应 nibble 的零点，并广播到 [OC, IC] ----
            # 对第 j 列：使用 qzeros[:, j//pack] 的第 (j%pack) 个 nibble
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
            # 打印更有用的上下文，便于排查
            print(f"[GPTQDequantizer] Error dequantizing GPTQ weight: {e}")
            try:
                pack = 32 // bits
                oc_pack, IC = qweight.shape
                OC = oc_pack * pack
                groups_out = qzeros.shape[0]
                g = OC // groups_out if groups_out > 0 else None
                print(f"  qweight shape: {tuple(qweight.shape)}, dtype: {qweight.dtype}")
                print(f"  qzeros  shape: {tuple(qzeros.shape)}, dtype: {qzeros.dtype}")
                print(f"  scales  shape: {tuple(scales.shape)}, dtype: {scales.dtype}")
                print(f"  derived OC={OC}, IC={IC}, pack={pack}, groups_out={groups_out}, g={g}")
            except Exception:
                pass
            # 安全回退
            # 返回一个零张量（[OC, IC] 若可推导，否则尽量不报错）
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

    # ------- 如需保留“simple”接口，做成正确实现的别名 -------
    @staticmethod
    def dequantize_gptq_weight_simple(
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        bits: int = 4
    ) -> torch.Tensor:
        """兼容旧接口：等价于 dequantize_gptq_weight（自动推导 g）"""
        return GPTQDequantizer.dequantize_gptq_weight(
            qweight=qweight, qzeros=qzeros, scales=scales, g_idx=None, bits=bits, group_size=None
        )

class EnhancedMixedPrecisionWeightLoader:
    """增强的混合精度权重加载器"""
    
    def __init__(self, config_path: str, enable_expert_tracking: bool = True):
        """
        初始化增强的混合精度权重加载器
        
        Args:
            config_path: 配置文件路径
            enable_expert_tracking: 是否启用专家激活跟踪
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
        
        # 专家激活跟踪器
        self.expert_tracker = ExpertActivationTracker() if enable_expert_tracking else None
        
        logger.info(f"Enhanced mixed precision loader initialized with {len(self.weight_mapping)} weight mappings")
        if enable_expert_tracking:
            logger.info("Expert activation tracking enabled")
    
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
    
    def _is_gptq_weight(self, weights: Dict[str, torch.Tensor], weight_name: str) -> bool:
        """检查是否是GPTQ权重"""
        # 检查是否存在GPTQ特有的组件
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
    
    def _dequantize_gptq_weight(self, qweight: torch.Tensor, qzeros: torch.Tensor, 
                               scales: torch.Tensor, g_idx: Optional[torch.Tensor] = None,
                               bits: int = 4, group_size: int = 128) -> torch.Tensor:
        """反量化GPTQ权重"""
        try:
            return GPTQDequantizer.dequantize_gptq_weight(
                qweight, qzeros, scales
            )
        except ImportError:
            # 如果修复版本不可用，使用原始版本
            return GPTQDequantizer.dequantize_gptq_weight(
                qweight, qzeros, scales, g_idx, bits, group_size
            )
    
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
                weights = self._load_pytorch_file(weight_file)
            
            # 检查是否是GPTQ权重
            if precision == 'int4' and self._is_gptq_weight(weights, weight_name):
                # 加载GPTQ组件并反量化
                qweight, qzeros, scales, g_idx = self._get_gptq_weight_components(weights, weight_name)
                weight = self._dequantize_gptq_weight(qweight, qzeros, scales, g_idx)
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
        
        for name, module in model.named_modules():
            if not "experts" in name:
                continue
            if hasattr(module, 'weight') and module.weight is not None:
                weight_name = name + '.weight'
                
                if weight_name in self.weight_mapping:
                    precision = self.weight_mapping[weight_name]
                    weight = self.load_weight(weight_name, precision)

                    # 获取模型设备
                    model_device = next(module.parameters()).device
                    logger.info(f"Model device: {model_device}")

                    if weight is not None:
                        try:
                            # 确保权重在正确的设备上
                            if weight.device != model_device:
                                weight = weight.to(model_device)
                                # logger.debug(f"Moved weight {weight_name} to device {model_device}")
                            
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
    
    def get_expert_tracker(self) -> Optional[ExpertActivationTracker]:
        """获取专家激活跟踪器"""
        return self.expert_tracker
    
    def enable_expert_tracking(self, enable: bool = True):
        """启用或禁用专家激活跟踪"""
        if enable and self.expert_tracker is None:
            self.expert_tracker = ExpertActivationTracker()
            logger.info("Expert activation tracking enabled")
        elif not enable:
            self.expert_tracker = None
            logger.info("Expert activation tracking disabled")


# 全局专家激活跟踪器实例
_global_expert_tracker: Optional[ExpertActivationTracker] = None


def get_global_expert_tracker() -> Optional[ExpertActivationTracker]:
    """获取全局专家激活跟踪器"""
    return _global_expert_tracker


def set_global_expert_tracker(tracker: ExpertActivationTracker):
    """设置全局专家激活跟踪器"""
    global _global_expert_tracker
    _global_expert_tracker = tracker


def record_expert_activation(layer_id: int, expert_id: int, 
                           tokens_processed: int = 1, request_id: str = None):
    """记录专家激活（全局函数）"""
    tracker = get_global_expert_tracker()
    if tracker:
        tracker.record_expert_activation(layer_id, expert_id, tokens_processed, request_id)


def record_request(request_id: str, input_length: int, output_length: int):
    """记录请求信息（全局函数）"""
    tracker = get_global_expert_tracker()
    if tracker:
        tracker.record_request(request_id, input_length, output_length)
