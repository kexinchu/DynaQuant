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
import numpy as np
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
from sglang.srt.model_loader.mixed_precision_quantizer import (
    MixedPrecisionQuantizer,
    ExpertQuantizationManager,
    init_global_quantization_system,
    get_global_quantization_manager
)

logger = logging.getLogger(__name__)


@dataclass
class ExpertActivationSummary:
    """专家激活摘要信息"""
    activity: int = 0  # 实际激活/启动次数  or 激活次数
    tokens: int = 0    # 处理的总token数量
    hot_cold_score: float = 0.0  # hot-cold分数
    last_activation_time: float = 0.0  # 最后激活时间


class EfficientActivationTracker:
    """高效的激活跟踪器，使用字典结构替代列表"""
    
    def __init__(self):
        # 结构: activation_info[layer_id][expert_id] = ExpertActivationSummary
        self.activation_info: Dict[int, Dict[int, ExpertActivationSummary]] = defaultdict(lambda: defaultdict(ExpertActivationSummary))
        self.lock = threading.RLock()
    
    def record_activation(self, layer_id: int, expert_id: int, 
                         tokens_processed: int = 1, activation_strength: float = 1.0):
        """记录激活"""
        with self.lock:
            summary = self.activation_info[layer_id][expert_id]
            summary.activity += 1
            summary.tokens += tokens_processed
            summary.last_activation_time = time.time()
            # hot_cold_score 将在导出时计算
        # here is called 
    
    def get_activation_info(self, layer_id: int = None, expert_id: int = None) -> Dict:
        """获取激活信息"""
        print(f"🔍 [EXPERT_TRACKING] 记录激活 in EfficientActivationTracker: {self.activation_info}")
        with self.lock:
            if layer_id is not None and expert_id is not None:
                # 返回特定expert的信息
                if layer_id in self.activation_info and expert_id in self.activation_info[layer_id]:
                    summary = self.activation_info[layer_id][expert_id]
                    return {
                        'layer_id': layer_id,
                        'expert_id': expert_id,
                        'activity': summary.activity,
                        'tokens': summary.tokens,
                        'hot_cold_score': summary.hot_cold_score,
                        'last_activation_time': summary.last_activation_time
                    }
                return {}
            
            # 返回所有信息
            result = {}
            for layer_id, experts in self.activation_info.items():
                print(f"🔍 [EXPERT_TRACKING] 获取所有信息 in EfficientActivationTracker: {layer_id}, {experts}")
                result[f'layer_{layer_id}'] = {}
                for expert_id, summary in experts.items():
                    result[f'layer_{layer_id}'][f'expert_{expert_id}'] = {
                        'layer_id': layer_id,
                        'expert_id': expert_id,
                        'activity': summary.activity,
                        'tokens': summary.tokens,
                        'hot_cold_score': summary.hot_cold_score,
                        'last_activation_time': summary.last_activation_time
                    }
            return result
    
    def calculate_hot_cold_scores(self):
        """计算所有expert的hot-cold分数"""
        with self.lock:
            for layer_id, experts in self.activation_info.items():
                if not experts:
                    print(f"🔍 [EXPERT_TRACKING] 没有专家 in EfficientActivationTracker: {layer_id}")
                    continue
                
                # 找到该层激活次数最多的expert
                max_activity = max(summary.activity for summary in experts.values())
                
                for expert_id, summary in experts.items():
                    if max_activity == 0:
                        summary.hot_cold_score = 0.0
                    else:
                        summary.hot_cold_score = summary.activity / max_activity
    
    def get_top_experts(self, top_k: int = 10) -> List[Dict]:
        """获取激活次数最多的expert"""
        with self.lock:
            all_experts = []
            for layer_id, experts in self.activation_info.items():
                for expert_id, summary in experts.items():
                    all_experts.append({
                        'layer_id': layer_id,
                        'expert_id': expert_id,
                        'activity': summary.activity,
                        'tokens': summary.tokens,
                        'hot_cold_score': summary.hot_cold_score
                    })
            
            # 按激活次数排序
            all_experts.sort(key=lambda x: x['activity'], reverse=True)
            return all_experts[:top_k]
    
    def get_hot_cold_scores(self) -> Dict[str, Dict]:
        """获取所有expert的hot-cold分数"""
        with self.lock:
            scores = {}
            for layer_id, experts in self.activation_info.items():
                for expert_id, summary in experts.items():
                    scores[f"layer_{layer_id}_expert_{expert_id}"] = {
                        'layer_id': layer_id,
                        'expert_id': expert_id,
                        'hot_cold_score': round(summary.hot_cold_score, 4),
                        'activity': summary.activity,
                        'tokens': summary.tokens
                    }
            return scores
    
    def reset(self):
        """重置所有数据"""
        print(f"🔍 [EXPERT_TRACKING] 重置统计信息 in EfficientActivationTracker")
        with self.lock:
            self.activation_info.clear()
    
    def get_summary_stats(self) -> Dict:
        """获取摘要统计"""
        with self.lock:
            total_experts = 0
            total_activity = 0
            total_tokens = 0
            layers_count = len(self.activation_info)
            
            for layer_id, experts in self.activation_info.items():
                total_experts += len(experts)
                for summary in experts.values():
                    total_activity += summary.activity
                    total_tokens += summary.tokens
            
            return {
                'total_layers': layers_count,
                'total_experts': total_experts,
                'total_activity': total_activity,
                'total_tokens': total_tokens
            }


# 移除ExpertActivationInfo类，使用EfficientActivationTracker中的ExpertActivationSummary


class ExpertActivationTracker:
    """专家激活跟踪器 - 优化版本，使用高效数据结构"""
    
    def __init__(self, max_history: int = 1000, time_window: int = 300):
        # 只使用高效跟踪器，移除冗余的expert_stats
        self.efficient_tracker = EfficientActivationTracker()
        self.request_history: deque = deque(maxlen=max_history)
        self.lock = threading.RLock()
        
        # 多进程支持
        self.process_id = os.getpid()
        self.rank = 0  # 默认rank，在多进程环境下会被正确设置
        
        # 时间窗口控制（默认5分钟）
        self.time_window = time_window
        self.last_analysis_time = time.time()
        self.activation_buffer = []
        
    def record_expert_activation(self, layer_id: int, expert_id: int, 
                               tokens_processed: int = 1, request_id: str = None, 
                               activation_strength: float = 1.0):
        """记录专家激活 - 优化版本，只使用高效跟踪器"""
        # 直接使用高效跟踪器，避免双重记录
        self.efficient_tracker.record_activation(layer_id, expert_id, tokens_processed, activation_strength)
        
        # 添加到激活缓冲区
        with self.lock:
            self.activation_buffer.append({
                'timestamp': time.time(),
                'layer_id': layer_id,
                'expert_id': expert_id,
                'activation_strength': activation_strength
            })
            
            # 检查是否需要分析（基于时间窗口）
            self._check_time_window_analysis()
    
    def record_expert_activation_batch(self, layer_id: int, expert_id: int, 
                                     tokens_processed: int = 1, request_id: str = None, 
                                     activation_strength: float = 1.0):
        """批量记录专家激活 - 优化版本，只使用高效跟踪器"""
        # 直接使用高效跟踪器，避免双重记录
        self.efficient_tracker.record_activation(layer_id, expert_id, tokens_processed, activation_strength)
    
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
        """获取专家统计信息 - 使用高效跟踪器"""
        return self.efficient_tracker.get_activation_info(layer_id, expert_id)
    
    def get_top_experts(self, top_k: int = 10) -> List[Dict]:
        """获取激活次数最多的专家 - 使用高效跟踪器"""
        return self.efficient_tracker.get_top_experts(top_k)
    
    def get_hot_cold_scores(self) -> Dict[str, Dict]:
        """获取所有专家的hot-cold分数 - 使用高效跟踪器"""
        return self.efficient_tracker.get_hot_cold_scores()
    
    def get_top_hot_experts(self, top_k: int = 20) -> List[Dict]:
        """获取最hot的专家（按hot-cold分数排序） - 使用高效跟踪器"""
        # 先计算hot-cold分数，然后获取top experts
        self.efficient_tracker.calculate_hot_cold_scores()
        return self.efficient_tracker.get_top_experts(top_k)
    
    def get_layer_stats(self) -> Dict[int, Dict]:
        """获取每层的统计信息 - 使用高效跟踪器"""
        # 从高效跟踪器获取数据并转换为层统计格式
        activation_info = self.efficient_tracker.get_activation_info()
        layer_stats = {}
        
        for layer_key, layer_data in activation_info.items():
            print(f"🔍 [EXPERT_TRACKING] 获取每层的统计信息(get_layer_stats) in ExpertActivationTracker: {layer_key}, {layer_data}")
            if layer_key.startswith('layer_'):
                layer_id = int(layer_key.split('_')[1])
                if layer_id not in layer_stats:
                    layer_stats[layer_id] = {
                        'total_experts': 0,
                        'total_activations': 0,
                        'total_tokens': 0,
                        'experts': {}
                    }
                
                for expert_key, expert_data in layer_data.items():
                    if expert_key.startswith('expert_'):
                        expert_id = int(expert_key.split('_')[1])
                        layer_stats[layer_id]['total_experts'] += 1
                        layer_stats[layer_id]['total_activations'] += expert_data['activity']
                        layer_stats[layer_id]['total_tokens'] += expert_data['tokens']
                        layer_stats[layer_id]['experts'][expert_id] = {
                            'activation_count': expert_data['activity'],
                            'total_tokens_processed': expert_data['tokens'],
                            'hot_cold_score': expert_data['hot_cold_score']
                        }
        
        return layer_stats
    
    def reset_stats(self):
        """重置统计信息"""
        print(f"🔍 [EXPERT_TRACKING] 重置统计信息 in ExpertActivationTracker")
        with self.lock:
            self.efficient_tracker.reset()
            self.request_history.clear()
    
    def export_stats(self, file_path: str):
        """导出统计信息到文件"""
        with self.lock:
            stats = {
                'expert_stats': self.get_expert_stats(),
                'layer_stats': self.get_layer_stats(),
                'top_experts': self.get_top_experts(20),
                'hot_cold_scores': self.get_hot_cold_scores(),
                'top_hot_experts': self.get_top_hot_experts(20),
                'export_time': time.time()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Expert activation stats exported to {file_path}")
    
    def update_all_hot_cold_scores(self):
        """更新所有expert的hot-cold分数（仅在导出时调用）"""
        print(f"🔍 [EXPERT_TRACKING] 开始更新所有expert的hot-cold分数")
        
        # 使用高效跟踪器计算hot-cold分数
        self.efficient_tracker.calculate_hot_cold_scores()
        
        print(f"✅ [EXPERT_TRACKING] 完成所有expert的hot-cold分数更新")
    
    def export_hot_cold_report(self, file_path: str):
        """导出专门的hot-cold报告"""
        # 在导出前先更新所有expert的hot-cold分数
        self.update_all_hot_cold_scores()
        
        with self.lock:
            # 使用高效跟踪器获取数据
            summary_stats = self.efficient_tracker.get_summary_stats()
            
            report = {
                'export_time': time.time(),
                'process_id': self.process_id,
                'rank': self.rank,
                'summary': summary_stats,
                'hot_cold_scores': self.efficient_tracker.get_hot_cold_scores(),
                'top_hot_experts': self.efficient_tracker.get_top_experts(50),
                'activation_info': self.efficient_tracker.get_activation_info()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Hot-cold report exported to {file_path}")
    
    def get_expert_stats_by_layer(self) -> Dict[str, Dict[str, Any]]:
        """获取按层分组的专家统计数据 - 使用高效跟踪器"""
        print(f"🔍 [EXPERT_TRACKING] 获取按层分组的专家统计数据(get_expert_stats_by_layer) in ExpertActivationTracker")
        return self.get_layer_stats()
    
    def _check_time_window_analysis(self):
        """检查时间窗口分析"""
        current_time = time.time()
        
        # 检查是否到了分析时间
        if current_time - self.last_analysis_time >= self.time_window:
            self._perform_time_window_analysis()
    
    def _perform_time_window_analysis(self):
        """执行时间窗口分析"""
        try:
            logger.info(f"📊 执行时间窗口分析 (窗口大小: {self.time_window}秒)")
            
            # 更新hot-cold分数
            self.efficient_tracker.calculate_hot_cold_scores()
            
            # 获取统计信息
            expert_stats = self.get_expert_stats()
            layer_stats = self.get_layer_stats()
            top_experts = self.get_top_experts(10)
            
            logger.info(f"  - 统计层数: {len(expert_stats)}")
            logger.info(f"  - 层统计: {len(layer_stats)}")
            logger.info(f"  - Top专家: {len(top_experts)}")
            
            # 显示前5个最活跃的专家
            if top_experts:
                logger.info("🔥 最活跃的专家:")
                for i, expert in enumerate(top_experts[:5]):
                    logger.info(f"    {i+1}. 层{expert['layer_id']} 专家{expert['expert_id']}: "
                              f"激活{expert['activity']}次, 热度{expert['hot_cold_score']:.3f}")
            
            # 更新分析时间
            self.last_analysis_time = time.time()
            
            # 清空激活缓冲区
            self.activation_buffer.clear()
            
            logger.info("✅ 时间窗口分析完成")
            
        except Exception as e:
            logger.error(f"时间窗口分析失败: {e}")
    
    def get_time_window_stats(self) -> Dict[str, Any]:
        """获取时间窗口统计信息"""
        with self.lock:
            current_time = time.time()
            window_start = current_time - self.time_window
            
            # 过滤时间窗口内的激活记录
            window_activations = [
                act for act in self.activation_buffer 
                if act['timestamp'] >= window_start
            ]
            
            return {
                'time_window': self.time_window,
                'window_start': window_start,
                'current_time': current_time,
                'total_activations': len(window_activations),
                'buffer_size': len(self.activation_buffer),
                'last_analysis_time': self.last_analysis_time,
                'time_since_last_analysis': current_time - self.last_analysis_time
            }
    
    def force_time_window_analysis(self):
        """强制执行时间窗口分析"""
        self._perform_time_window_analysis()
    
    def set_time_window(self, time_window: int):
        """设置时间窗口大小"""
        with self.lock:
            self.time_window = time_window
            logger.info(f"时间窗口设置为: {time_window}秒")
    
    def aggregate_from_other_processes(self, other_trackers: List['ExpertActivationTracker']):
        """从其他进程聚合统计数据 - 简化版本"""
        # 简化聚合逻辑，只聚合请求历史
        with self.lock:
            for other_tracker in other_trackers:
                with other_tracker.lock:
                    # 聚合请求历史
                    self.request_history.extend(other_tracker.request_history)
            
            logger.info(f"聚合完成，当前统计: {len(self.efficient_tracker.activation_info)} 个expert")


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
    
    def __init__(self, config_path: str, enable_expert_tracking: bool = True, enable_quantization: bool = True):
        """
        初始化增强的混合精度权重加载器
        
        Args:
            config_path: 配置文件路径
            enable_expert_tracking: 是否启用专家激活跟踪
            enable_quantization: 是否启用混合精度量化
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
        
        # 混合精度量化系统
        self.quantization_manager = None
        if enable_quantization:
            self.quantization_manager = init_global_quantization_system(config_path)
            if self.expert_tracker:
                self.quantization_manager.set_expert_tracker(self.expert_tracker)
            self.quantization_manager.enable_quantization(True)
        
        logger.info(f"Enhanced mixed precision loader initialized with {len(self.weight_mapping)} weight mappings")
        if enable_expert_tracking:
            logger.info("Expert activation tracking enabled")
        if enable_quantization:
            logger.info("Mixed precision quantization enabled")
    
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
    
    def enable_quantization(self, enable: bool = True):
        """启用或禁用混合精度量化"""
        if self.quantization_manager:
            self.quantization_manager.enable_quantization(enable)
            logger.info(f"Mixed precision quantization {'enabled' if enable else 'disabled'}")
        else:
            logger.warning("Quantization manager not available")
    
    def quantize_model_weights(self, model: torch.nn.Module) -> Dict[str, Any]:
        """量化模型权重"""
        if self.quantization_manager:
            return self.quantization_manager.quantize_expert_weights(model)
        else:
            logger.warning("Quantization manager not available")
            return {'quantized': 0, 'skipped': 0, 'errors': 1, 'details': [{'error': 'Quantization manager not available'}]}
    
    def update_quantization_profiles(self):
        """更新量化配置档案"""
        if self.quantization_manager:
            self.quantization_manager.update_expert_profiles_from_tracker()
        else:
            logger.warning("Quantization manager not available")
    
    def export_quantization_report(self, file_path: str):
        """导出量化报告"""
        if self.quantization_manager:
            self.quantization_manager.export_quantization_report(file_path)
        else:
            logger.warning("Quantization manager not available")
    
    def get_quantization_manager(self):
        """获取量化管理器"""
        return self.quantization_manager


# 全局专家激活跟踪器实例
_global_expert_tracker: Optional[ExpertActivationTracker] = None


def get_global_expert_tracker() -> Optional[ExpertActivationTracker]:
    """获取全局专家激活跟踪器"""
    return _global_expert_tracker


def set_global_expert_tracker(tracker: ExpertActivationTracker):
    """设置全局专家激活跟踪器"""
    global _global_expert_tracker
    _global_expert_tracker = tracker


def init_global_expert_tracker() -> ExpertActivationTracker:
    """初始化全局专家激活跟踪器"""
    global _global_expert_tracker
    if _global_expert_tracker is None:
        _global_expert_tracker = ExpertActivationTracker()
        logger.info("Global expert tracker initialized")
    return _global_expert_tracker


def record_expert_activation(layer_id: int, expert_id: int, 
                           tokens_processed: int = 1, request_id: str = None, 
                           activation_strength: float = 1.0):
    """记录专家激活（全局函数）"""
    tracker = get_global_expert_tracker()
    if tracker:
        tracker.record_expert_activation(layer_id, expert_id, tokens_processed, request_id, activation_strength)


def record_request(request_id: str, input_length: int, output_length: int):
    """记录请求信息（全局函数）"""
    tracker = get_global_expert_tracker()
    if tracker:
        tracker.record_request(request_id, input_length, output_length)
