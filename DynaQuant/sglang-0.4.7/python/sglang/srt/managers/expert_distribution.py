# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import os
import time
import threading
from abc import ABC
from collections import deque, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import einops
import torch
import torch.distributed

from sglang.srt.managers.expert_location import ExpertLocationMetadata
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from .hot_migration_swapper import HotMigrationSwapper
from .non_expert_fp16_initializer import NonExpertFP16Initializer

# 全局热迁移交换器实例
_global_hot_migration_swapper = None
_global_non_expert_fp16_initializer = None

def init_global_hot_migration_swapper(fp16_path: str, fp8_path: str, gptq_int4_path: str, max_concurrent_swaps: int = 1) -> HotMigrationSwapper:
    """初始化全局热迁移交换器"""
    global _global_hot_migration_swapper
    if _global_hot_migration_swapper is None:
        _global_hot_migration_swapper = HotMigrationSwapper(
            fp16_path=fp16_path,
            fp8_path=fp8_path,
            gptq_int4_path=gptq_int4_path,
            max_concurrent_swaps=max_concurrent_swaps
        )
        logger.info("Global hot migration swapper initialized")
    return _global_hot_migration_swapper

def get_global_hot_migration_swapper() -> Optional[HotMigrationSwapper]:
    """获取全局热迁移交换器"""
    return _global_hot_migration_swapper

def init_global_non_expert_fp16_initializer(fp16_path: str) -> NonExpertFP16Initializer:
    """初始化全局非Expert层FP16初始化器"""
    global _global_non_expert_fp16_initializer
    if _global_non_expert_fp16_initializer is None:
        _global_non_expert_fp16_initializer = NonExpertFP16Initializer(fp16_path=fp16_path)
        logger.info("Global non-expert FP16 initializer initialized")
    return _global_non_expert_fp16_initializer

def get_global_non_expert_fp16_initializer() -> Optional[NonExpertFP16Initializer]:
    """获取全局非Expert层FP16初始化器"""
    return _global_non_expert_fp16_initializer
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import Withable, get_bool_env_var
from sglang.srt.model_loader.enhanced_mixed_precision_loader import get_global_expert_tracker
from sglang.srt.managers.dynamic_quantization_manager import (
    init_global_quantization_manager,
    get_global_quantization_manager,
)
# 全局热迁移交换器函数在下面定义

logger = logging.getLogger(__name__)

# --------------------------------------- Entrypoint -----------------------------------------

_OutputMode = Literal["file", "object"]


class ExpertDistributionRecorder(ABC):
    """Global expert distribution recording"""

    @staticmethod
    def init_new(
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ):
        if server_args.expert_distribution_recorder_mode is not None:
            return _ExpertDistributionRecorderReal(
                server_args, expert_location_metadata, rank
            )
        else:
            return _ExpertDistributionRecorderNoop()

    @contextmanager
    def with_current_layer(self, layer_idx):
        yield

    @contextmanager
    def with_debug_name(self, debug_name):
        yield

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int, forward_batch: ForwardBatch):
        yield

    def on_select_experts(self, topk_ids: torch.Tensor, layer_idx: int = None):
        # 调用原有的hook
        self._on_hook("on_select_experts", topk_ids=topk_ids, layer_idx=layer_idx)
        
        # 新增：我们的expert tracking hook
        try:
            # 检查是否有全局expert tracker
            if hasattr(self, '_expert_tracker_hook_enabled') and self._expert_tracker_hook_enabled:
                self._record_expert_activations(topk_ids, layer_idx)
        except Exception as e:
            # 静默处理错误，不影响原有功能
            logger.debug(f"Expert tracking hook error: {e}")
            pass
    
    def _record_expert_activations(self, topk_ids: torch.Tensor, layer_idx: int = None):
        """记录expert激活信息"""
        try:
            # 获取当前层信息
            if layer_idx is None:
                layer_idx = getattr(self._current_layer_idx, 'value', None)
                if layer_idx is None:
                    # 尝试从上下文推断层索引
                    layer_idx = getattr(self, '_inferred_layer_idx', None)
                    if layer_idx is None:
                        # 如果仍然无法获取层索引，记录警告并跳过
                        logger.warning("无法获取层索引，跳过expert激活记录")
                        return
            
            # 获取全局expert tracker（使用顶部已导入的函数）
            tracker = get_global_expert_tracker()
            if tracker is None:
                return
            
            # 统计每个expert的激活情况
            if topk_ids is not None and topk_ids.numel() > 0:
                # 获取激活的expert IDs
                active_experts = topk_ids.flatten().tolist()
                
                # 计算每个expert的token数量
                # 假设每个token都会激活top-k个expert，所以每个expert处理的token数 = 总token数 / top-k
                total_tokens = topk_ids.shape[0] if len(topk_ids.shape) > 1 else 1
                top_k = topk_ids.shape[1] if len(topk_ids.shape) > 1 else 1
                tokens_per_expert = max(1, total_tokens // top_k) if top_k > 0 else 1
                
                # 批量记录激活的expert（优化版本，减少调用频率）
                # 只记录前10个expert，避免过多调用
                limited_experts = active_experts[:10] if len(active_experts) > 10 else active_experts
                
                for expert_id in limited_experts:
                    if expert_id >= 0:  # 过滤无效ID
                        # 简化映射逻辑，直接使用expert_id，避免复杂的映射计算
                        # 在大多数情况下，expert_id已经是正确的逻辑ID
                        tracker.record_expert_activation_batch(
                            layer_id=layer_idx,
                            expert_id=expert_id,
                            tokens_processed=tokens_per_expert,
                            activation_strength=1.0
                        )
                        
                        # 记录到时间窗口统计
                        if hasattr(self, '_record_expert_activation_to_window'):
                            self._record_expert_activation_to_window(layer_idx, expert_id)
                
                # 检查是否需要分析时间窗口
                current_time = time.time()
                if self._last_analysis_time[layer_idx] is None:
                    self._last_analysis_time[layer_idx] = current_time
                if current_time - self._last_analysis_time[layer_idx] >= self._time_window:
                    self._analyze_time_window(layer_idx)
                        
        except Exception as e:
            logger.debug(f"记录expert激活失败: {e}")
            # 静默处理错误，不影响原有功能
    
    def enable_expert_tracking_hook(self):
        """启用expert tracking hook"""
        self._expert_tracker_hook_enabled = True

    def on_deepep_dispatch_normal(
        self,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        pass

    def on_deepep_dispatch_low_latency(
        self, local_physical_count_of_layer: torch.Tensor
    ):
        pass

    def start_record(self):
        self._on_not_implemented()

    def stop_record(self):
        self._on_not_implemented()

    def dump_record(self, output_mode: _OutputMode = "file"):
        self._on_not_implemented()

    @property
    def recording(self):
        return False

    def _on_not_implemented(self):
        raise Exception(
            "Please set ServerArgs.expert_distribution_recorder_mode to use ExpertDistributionRecorder."
        )


class _ExpertDistributionRecorderNoop(ExpertDistributionRecorder):
    """No-op implementation that supports expert tracking hooks"""
    
    def __init__(self):
        # 初始化必要的属性以支持expert tracking
        self._expert_tracker_hook_enabled = True  # 默认启用expert tracking
        self._current_layer_idx = Withable()
        self._current_forward_pass_id = Withable()
        self._current_debug_name = Withable()
        self._recording = False
        # 初始化空的gatherers以避免属性错误
        self._single_pass_gatherers = {}
        self._accumulator = None
        # 添加必要的属性以避免 KeyError
        self._server_args = None
        self._expert_location_metadata = None
        logger.info("🔍 [DEBUG] _ExpertDistributionRecorderNoop initialized with expert tracking enabled")
    
    def _on_hook(self, hook_name: str, **kwargs):
        """No-op hook implementation that supports expert tracking"""
        # 如果是expert tracking相关的hook，则处理
        if hook_name == "on_select_experts" and hasattr(self, '_expert_tracker_hook_enabled'):
            try:
                if self._expert_tracker_hook_enabled:
                    self._record_expert_activations(kwargs.get('topk_ids'), kwargs.get('layer_idx'))
            except Exception as e:
                # 静默处理错误，不影响原有功能
                pass
    
    def _record_expert_activations(self, topk_ids: torch.Tensor, layer_idx: int = None):
        """记录expert激活信息"""
        try:
            # 获取当前层信息
            if layer_idx is None:
                layer_idx = getattr(self._current_layer_idx, 'value', None)
                if layer_idx is None:
                    # 尝试从上下文推断层索引
                    layer_idx = getattr(self, '_inferred_layer_idx', None)
                    if layer_idx is None:
                        # 如果仍然无法获取层索引，记录警告并跳过
                        logger.warning("无法获取层索引，跳过expert激活记录")
                        return
            
            # 获取全局expert tracker（使用顶部已导入的函数）
            try:
                tracker = get_global_expert_tracker()
                if tracker is None:
                    return
                
                # 统计每个expert的激活情况
                if topk_ids is not None and topk_ids.numel() > 0:
                    # 获取激活的expert IDs
                    active_experts = topk_ids.flatten().tolist()
                    
                    # 计算激活强度（基于top-k权重）
                    activation_strength = 1.0 / len(active_experts) if active_experts else 1.0
                    
                    # 记录每个激活的expert
                    for expert_id in active_experts:
                        print(f"🔍 [EXPERT_TRACKING] 记录每个激活的expert in _ExpertDistributionRecorderNoop, expert_id: {expert_id}, layer_idx: {layer_idx}, activation_strength: {activation_strength}")
                        if expert_id >= 0:  # 过滤无效ID
                            tracker.record_expert_activation(
                                layer_id=layer_idx,
                                expert_id=expert_id,
                                activation_strength=activation_strength
                            )
            except ImportError:
                # 如果enhanced_mixed_precision_loader不可用，静默忽略
                pass
                        
        except Exception as e:
            logger.debug(f"记录expert激活失败: {e}")
            # 静默处理错误，不影响原有功能
    
    def enable_expert_tracking_hook(self):
        """启用expert tracking hook"""
        self._expert_tracker_hook_enabled = True
    
    def with_current_layer(self, layer_idx):
        """设置当前层索引"""
        return self._current_layer_idx.with_value(layer_idx)
    
    def with_forward_pass(self, forward_pass_id: int, forward_batch):
        """设置当前forward pass"""
        return self._current_forward_pass_id.with_value(forward_pass_id)
    
    def with_debug_name(self, debug_name):
        """设置debug名称"""
        return self._current_debug_name.with_value(debug_name)
    
    def _on_forward_pass_start(self, forward_batch):
        """Forward pass 开始（no-op实现）"""
        pass
    
    def _on_forward_pass_end(self, forward_pass_id: int):
        """Forward pass 结束（no-op实现）"""
        # 在 no-op 实现中，我们不需要收集任何数据
        # 但为了保持兼容性，我们可以创建一个空的 single_pass_data
        if hasattr(self, '_accumulator') and self._accumulator is not None:
            # 创建一个空的 global_physical_count 作为占位符
            empty_single_pass_data = {
                "global_physical_count": torch.zeros(
                    (1, 1),  # 最小的形状
                    dtype=torch.int32,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            }
            self._accumulator.append(forward_pass_id, "noop", empty_single_pass_data)
    
    # 添加对基本记录方法的支持，避免抛出异常
    def start_record(self):
        """开始记录（no-op实现）"""
        self._recording = True
    
    def stop_record(self):
        """停止记录（no-op实现）"""
        self._recording = False
    
    def dump_record(self, output_mode: _OutputMode = "file"):
        """导出记录（no-op实现）"""
        # 返回空的结果，避免异常
        return {}
    
    @property
    def recording(self):
        """返回当前记录状态"""
        return self._recording


class _ExpertDistributionRecorderReal(ExpertDistributionRecorder):
    def __init__(
        self,
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ):
        self._server_args = server_args
        self._expert_location_metadata = expert_location_metadata

        self._recording = False
        self._current_forward_pass_id = Withable()
        self._current_layer_idx = Withable()
        self._current_debug_name = Withable()
        self._accumulator = _Accumulator.init_new(
            server_args, expert_location_metadata, rank
        )
        self._single_pass_gatherers = {
            k: _SinglePassGatherer.init_new(server_args, expert_location_metadata, rank)
            for k in self._accumulator.get_single_pass_gatherer_keys()
        }

        # 时间窗口统计功能
        self._time_window = 100  # 5分钟时间窗口
        self._last_analysis_time = defaultdict(time.time)  # layer_id -> time
        self._expert_activation_counts = defaultdict(lambda: defaultdict(int))  # layer_id -> expert_id -> count
        self._expert_activation_lock = threading.RLock()
        
        # 动态量化管理器
        self._quantization_manager = None
        self._init_quantization_manager()
        
        # 参数交换器
        self._parameter_swapper = None
        self._init_parameter_swapper()
        
        # 动态量化配置
        self._quantization_config = {
            'high_threshold': 0.5,    # score > 0.5 使用 fp16
            'medium_threshold': 0.1,  # 0.1 < score <= 0.5 使用 fp8
            # score <= 0.1 使用 gptq-int4
            'precision_mapping': {
                'fp16': 'fp16',
                'fp8': 'fp8',
                'int4': 'gptq_int4'
            }
        }
        
        # Expert量化映射
        self._expert_quantization_map = defaultdict(lambda: defaultdict(str))  # layer_id -> expert_id -> precision

        if server_args.enable_expert_distribution_metrics:
            logger.info(
                "ExpertDistributionRecorder auto start record since enable_expert_distribution_metrics"
            )
            self.start_record()
        
        # 启用我们的expert tracking hook
        self.enable_expert_tracking_hook()
        logger.info("✓ Expert tracking hook已启用")
    
    def _init_quantization_manager(self):
        """初始化量化管理器"""
        try:
            # 从server args获取配置
            server_args = self._server_args
            
            # 获取模型路径配置
            fp16_path = server_args.fp16_model_path or '/dev/shm/Qwen3-235B-A22B'
            fp8_path = server_args.fp8_model_path or '/dev/shm/Qwen3-235B-A22B-FP8'
            gptq_int4_path = server_args.gptq_int4_model_path or '/dev/shm/Qwen3-235B-A22B-GPTQ-Int4'
            
            # 获取阈值配置
            high_threshold = server_args.quantization_high_threshold
            medium_threshold = server_args.quantization_medium_threshold
            
            # 初始化全局量化管理器
            self._quantization_manager = init_global_quantization_manager(
                high_threshold=high_threshold,
                medium_threshold=medium_threshold,
                fp16_path=fp16_path,
                fp8_path=fp8_path,
                gptq_int4_path=gptq_int4_path,
                time_window=self._time_window
            )
            logger.info("Dynamic quantization manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize quantization manager: {e}")
            self._quantization_manager = None
    
    def _init_parameter_swapper(self):
        """初始化参数交换器"""
        try:
            # 从server args获取配置
            server_args = self._server_args
            logger.info(f"🔥 Initializing parameter swapper with enable_dynamic_quantization: {server_args.enable_dynamic_quantization}")
            
            # 获取模型路径配置
            fp16_path = server_args.fp16_model_path or '/dev/shm/Qwen3-235B-A22B'
            fp8_path = server_args.fp8_model_path or '/dev/shm/Qwen3-235B-A22B-FP8'
            gptq_int4_path = server_args.gptq_int4_model_path or '/dev/shm/Qwen3-235B-A22B-GPTQ-Int4'
            
            logger.info(f"Model paths - FP16: {fp16_path}, FP8: {fp8_path}, GPTQ-INT4: {gptq_int4_path}")
            
            # 获取并发交换数量配置
            max_concurrent_swaps = server_args.max_concurrent_swaps
            
            # 初始化全局热迁移交换器
            self._parameter_swapper = init_global_hot_migration_swapper(
                fp16_path=fp16_path,
                fp8_path=fp8_path,
                gptq_int4_path=gptq_int4_path,
                max_concurrent_swaps=max_concurrent_swaps
            )
            logger.info("Hot migration swapper initialized")
            
            # 初始化非expert层FP16初始化器
            self._non_expert_fp16_initializer = init_global_non_expert_fp16_initializer(fp16_path)
            logger.info("Non-expert FP16 initializer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize parameter swapper: {e}")
            import traceback
            traceback.print_exc()
            self._parameter_swapper = None

    def with_current_layer(self, layer_idx):
        return self._current_layer_idx.with_value(layer_idx)

    def with_debug_name(self, debug_name):
        return self._current_debug_name.with_value(debug_name)

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int, forward_batch: ForwardBatch):
        with self._current_forward_pass_id.with_value(forward_pass_id):
            self._on_forward_pass_start(forward_batch)
            try:
                yield
            finally:
                self._on_forward_pass_end(forward_pass_id)

    def _on_forward_pass_start(self, forward_batch: ForwardBatch):
        # 检查并触发热迁移非expert层FP16初始化
        self.check_and_initialize_non_expert_layers_hot()
        
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            gatherer.reset()
            gatherer.on_forward_pass_start(forward_batch)

    def _on_forward_pass_end(self, forward_pass_id: int):
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            single_pass_data = gatherer.collect()
            self._accumulator.append(forward_pass_id, gatherer_key, single_pass_data)

    def on_select_experts(self, topk_ids: torch.Tensor, layer_idx: int = None):
        # 调用原有的hook
        self._on_hook("on_select_experts", topk_ids=topk_ids, layer_idx=layer_idx)
        
        # 新增：我们的expert tracking hook
        try:
            # 检查是否有全局expert tracker
            if hasattr(self, '_expert_tracker_hook_enabled') and self._expert_tracker_hook_enabled:
                self._record_expert_activations(topk_ids, layer_idx)
        except Exception as e:
            # 静默处理错误，不影响原有功能
            logger.debug(f"Expert tracking hook error: {e}")
            pass
    
    def _record_expert_activations(self, topk_ids: torch.Tensor, layer_idx: int = None):
        """记录expert激活信息"""
        try:
            # 获取当前层信息
            if layer_idx is None:
                layer_idx = getattr(self._current_layer_idx, 'value', None)
                if layer_idx is None:
                    # 尝试从上下文推断层索引
                    layer_idx = getattr(self, '_inferred_layer_idx', None)
                    if layer_idx is None:
                        # 如果仍然无法获取层索引，记录警告并跳过
                        logger.warning("无法获取层索引，跳过expert激活记录")
                        return
            
            # 获取全局expert tracker（使用顶部已导入的函数）
            tracker = get_global_expert_tracker()
            if tracker is None:
                return
            
            # 统计每个expert的激活情况
            if topk_ids is not None and topk_ids.numel() > 0:
                # 获取激活的expert IDs
                active_experts = topk_ids.flatten().tolist()
                
                # 计算每个expert的token数量
                # 假设每个token都会激活top-k个expert，所以每个expert处理的token数 = 总token数 / top-k
                total_tokens = topk_ids.shape[0] if len(topk_ids.shape) > 1 else 1
                top_k = topk_ids.shape[1] if len(topk_ids.shape) > 1 else 1
                tokens_per_expert = max(1, total_tokens // top_k) if top_k > 0 else 1
                
                # 批量记录激活的expert（优化版本，减少调用频率）
                # 只记录前10个expert，避免过多调用
                limited_experts = active_experts[:10] if len(active_experts) > 10 else active_experts
                
                for expert_id in limited_experts:
                    if expert_id >= 0:  # 过滤无效ID
                        # 简化映射逻辑，直接使用expert_id，避免复杂的映射计算
                        # 在大多数情况下，expert_id已经是正确的逻辑ID
                        tracker.record_expert_activation_batch(
                            layer_id=layer_idx,
                            expert_id=expert_id,
                            tokens_processed=tokens_per_expert,
                            activation_strength=1.0
                        )
                        
                        # 记录到时间窗口统计
                        if hasattr(self, '_record_expert_activation_to_window'):
                            self._record_expert_activation_to_window(layer_idx, expert_id)
                
                # 检查是否需要分析时间窗口
                current_time = time.time()
                if self._last_analysis_time[layer_idx] is None:
                    self._last_analysis_time[layer_idx] = current_time
                if current_time - self._last_analysis_time[layer_idx] >= self._time_window:
                    self._analyze_time_window(layer_idx)
                        
        except Exception as e:
            logger.debug(f"记录expert激活失败: {e}")
            # 静默处理错误，不影响原有功能
    
    def enable_expert_tracking_hook(self):
        """启用expert tracking hook"""
        self._expert_tracker_hook_enabled = True

    def on_deepep_dispatch_normal(
        self,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        self._on_hook(
            "on_deepep_dispatch_normal",
            local_physical_count_of_layer=local_physical_count_of_layer,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )

    def on_deepep_dispatch_low_latency(
        self, local_physical_count_of_layer: torch.Tensor
    ):
        self._on_hook(
            "on_deepep_dispatch_low_latency",
            local_physical_count_of_layer=local_physical_count_of_layer,
        )

    def _on_hook(self, hook_name: str, **kwargs):
        if not (self._recording or torch.cuda.is_current_stream_capturing()):
            return
        try:
            gatherer = self._single_pass_gatherers[
                self._accumulator.get_single_pass_gatherer_key(
                    self._current_debug_name.value
                )
            ]
            # 从kwargs中移除layer_idx，避免重复传递
            kwargs_without_layer_idx = {k: v for k, v in kwargs.items() if k != 'layer_idx'}
            getattr(gatherer, hook_name)(layer_idx=self._current_layer_idx.value, **kwargs_without_layer_idx)
        except Exception as e:
            logger.warning(f"🔍 [DEBUG] _on_hook failed: {e}")
            # 如果gatherer调用失败，不要影响我们的expert tracking
            pass

    def _reset(self):
        """Reset the expert distribution recorder."""
        logger.info("Resetting ExpertDistributionRecorder...")
        assert (
            self._current_layer_idx.value is None
        ), f"{self._current_layer_idx.value=}"
        for gatherer in self._single_pass_gatherers.values():
            gatherer.reset()
        self._accumulator.reset()

    def start_record(self):
        """Start recording the expert distribution."""
        if self._recording:
            logger.warning(
                "SGLang server is already recording expert ids. Did you forget to dump the expert ids recorded so far by sending requests to the `/stop_expert_distribution_record` and `/dump_expert_distribution_record` endpoints?"
            )
        self._reset()
        self._recording = True

    def stop_record(self):
        """Stop recording the expert distribution."""
        if not self._recording:
            logger.warning(
                "SGLang server has not been recording expert ids. Did you forget to start recording by sending request to the `/start_expert_distribution_record` endpoint?"
            )
        self._recording = False

    def dump_record(self, output_mode: _OutputMode = "file"):
        """Dump the expert distribution record and reset the recorder after dumping."""
        output = self._accumulator.dump(output_mode=output_mode)
        self._reset()
        return output

    @property
    def recording(self):
        return self._recording
    
    def _record_expert_activation_to_window(self, layer_idx: int, expert_id: int):
        """记录expert激活到时间窗口统计"""
        with self._expert_activation_lock:
            self._expert_activation_counts[layer_idx][expert_id] += 1
    
    def _analyze_time_window(self, layer_idx: int):
        """分析时间窗口内的expert激活情况"""
        layer_idx = layer_idx
        current_time = time.time()
        
        with self._expert_activation_lock:
            # 分析当前层expert激活情况
            expert_counts = self._expert_activation_counts[layer_idx]
            max_activations = max(expert_counts.values())
            
            # 计算每个expert的激活分数
            expert_scores = {}
            for expert_id, count in expert_counts.items():
                score = count / max_activations if max_activations > 0 else 0
                expert_scores[expert_id] = score
                
                # 更新量化管理器中的expert分数
                if self._quantization_manager:
                    self._quantization_manager.update_expert_score(layer_idx, expert_id, score)
            
            # 更新expert量化映射
            self._update_expert_quantization_map(layer_idx, expert_scores)
            
            # 执行参数交换 - 使用热迁移
            self._execute_parameter_swaps(layer_idx, expert_scores)
            
            # 清空统计计数器
            self._expert_activation_counts[layer_idx].clear()
            self._last_analysis_time[layer_idx] = current_time
            
        # 生成量化报告
        self.export_quantization_report()
    
    def set_time_window(self, time_window: int):
        """设置时间窗口大小"""
        with self._expert_activation_lock:
            self._time_window = time_window
            logger.info(f"时间窗口设置为: {time_window}秒")
    
    def force_time_window_analysis(self):
        """强制执行时间窗口分析"""
        self._analyze_time_window()
    
    def _determine_quantization_precision(self, score: float) -> str:
        """根据expert激活分数确定量化精度"""
        # 更保守的量化策略，减少量化请求
        if score > self._quantization_config['high_threshold']:
            return "fp16"
        elif score > self._quantization_config['medium_threshold']:
            return "fp8"
        else:
            return "gptq_int4"
    
    def _update_expert_quantization_map(self, layer_idx: int, expert_scores: Dict[int, float]):
        """更新expert量化映射"""
        try:
            for expert_id, score in expert_scores.items():
                precision = self._determine_quantization_precision(score)
                self._expert_quantization_map[layer_idx][expert_id] = precision
        except Exception as e:
            print(f"Error in _update_expert_quantization_map: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def set_quantization_thresholds(self, high_threshold: float = None, medium_threshold: float = None):
        """设置量化阈值"""
        if high_threshold is not None:
            self._quantization_config['high_threshold'] = high_threshold
        if medium_threshold is not None:
            self._quantization_config['medium_threshold'] = medium_threshold
        
        # 同时更新量化管理器
        if self._quantization_manager:
            self._quantization_manager.set_thresholds(high_threshold, medium_threshold)
        
        logger.info(f"Updated quantization thresholds: {self._quantization_config}")
    
    def export_quantization_report(self) -> Dict[str, Any]:
        """导出量化报告"""
        report = {
            'thresholds': self._quantization_config.copy(),
            'expert_quantization_map': dict(self._expert_quantization_map),
            'statistics': {}
        }
        
        # 添加量化管理器统计
        if self._quantization_manager:
            report['statistics'] = self._quantization_manager.get_quantization_stats()
        
        # 添加参数交换器统计
        if self._parameter_swapper:
            report['swap_statistics'] = self._parameter_swapper.get_swap_stats()
        
        return report
    
    def _execute_parameter_swaps(self, layer_idx: int, expert_scores: Dict[int, float]):
        """执行参数交换 - 使用热迁移"""
        try:
            if not self._parameter_swapper:
                logger.warning("No parameter swapper available")
                print(f"No parameter swapper available")
                return
            
            # 检查是否有需要交换的expert
            swap_requests = []
            for expert_id, score in expert_scores.items():
                new_precision = self._determine_quantization_precision(score)
                current_precision = self._expert_quantization_map.get(layer_idx, {}).get(expert_id, 'fp16')
                
                # 只有精度发生变化时才需要交换
                if new_precision != current_precision:
                    swap_requests.append((layer_idx, expert_id, new_precision))
            
            if not swap_requests:
                return
            
            logger.info(f"Safe parameter swap: {len(swap_requests)} experts need precision change")
            
            # 分批处理，避免一次性处理过多expert
            batch_size = 5  # 每批最多处理5个expert
            for i in range(0, len(swap_requests), batch_size):
                batch = swap_requests[i:i + batch_size]
                
                try:
                    # 使用热迁移进行批量交换
                    results = self._parameter_swapper.batch_hot_swap_experts(batch)
                    logger.info(f"Hot migration batch {i//batch_size + 1} results: {results['successful_swaps']}/{results['total_requests']} successful")
                    
                    # 批次间添加延迟，减少系统压力
                    if i + batch_size < len(swap_requests):
                        import time
                        time.sleep(0.05)  # 50ms延迟，热迁移更快
                        
                except Exception as e:
                    logger.error(f"Hot migration batch {i//batch_size + 1} failed: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to execute safe parameter swaps: {e}")
            print(f"Exception in _execute_parameter_swaps_safe: {e}")
            import traceback
            traceback.print_exc()
    
    def set_quantization_thresholds(self, high_threshold: float = None, medium_threshold: float = None):
        """设置量化阈值"""
        with self._expert_activation_lock:
            if high_threshold is not None:
                self._quantization_config['high_threshold'] = high_threshold
            if medium_threshold is not None:
                self._quantization_config['medium_threshold'] = medium_threshold
            logger.info(f"量化阈值已更新: 高阈值={self._quantization_config['high_threshold']}, "
                       f"中阈值={self._quantization_config['medium_threshold']}")
    
    def export_quantization_report(self) -> Dict[str, Any]:
        """导出量化报告"""
        report = {
            'quantization_config': self._quantization_config.copy(),
            'expert_quantization_map': dict(self._expert_quantization_map),
            'statistics': {}
        }
        
        # 统计各精度的expert数量
        precision_stats = defaultdict(int)
        for layer_map in self._expert_quantization_map.values():
            for precision in layer_map.values():
                precision_stats[precision] += 1
        
        report['statistics'] = dict(precision_stats)
        return report
    
    def set_model_runner(self, model_runner):
        """设置ModelRunner实例，用于热迁移参数交换和非expert层FP16初始化"""
        logger.info("🔥 set_model_runner called - starting non-expert FP16 initialization")
        
        if self._parameter_swapper:
            self._parameter_swapper.set_model_runner(model_runner)
            logger.info("ModelRunner instance set for hot migration swapper")
        else:
            logger.warning("No parameter swapper available")
        
        # 保存model_runner引用，稍后在模型加载完成后初始化非expert层
        self._model_runner = model_runner
        self._needs_non_expert_fp16_init = False  # 延迟初始化标记
        self._non_expert_fp16_initialized = False  # 初始化完成标记
        logger.info("ModelRunner reference saved for hot migration non-expert FP16 initialization")
    
    def initialize_non_expert_layers_after_model_load(self):
        """在模型加载完成后初始化非expert层为FP16"""
        if not hasattr(self, '_model_runner') or not self._model_runner:
            logger.warning("No model_runner available for non-expert FP16 initialization")
            return
        
        if not hasattr(self, '_non_expert_fp16_initializer') or not self._non_expert_fp16_initializer:
            logger.warning("No non-expert FP16 initializer available")
            return
        
        if not hasattr(self._model_runner, 'model'):
            logger.warning("ModelRunner has no model attribute, skipping non-expert FP16 initialization")
            return
        
        try:
            logger.info("🎯 Initializing non-expert layers to FP16...")
            logger.info(f"Model type: {type(self._model_runner.model)}")
            
            init_results = self._non_expert_fp16_initializer.initialize_non_expert_layers_fp16(self._model_runner.model)
            
            logger.info(f"✅ Non-expert FP16 initialization completed:")
            logger.info(f"   📊 Total components: {init_results['successful_initializations'] + init_results['failed_initializations']}")
            logger.info(f"   ✅ Successful: {init_results['successful_initializations']}")
            logger.info(f"   ❌ Failed: {init_results['failed_initializations']}")
            logger.info(f"   💾 Memory usage: {init_results['total_memory_usage_mb']:.2f} MB")
            
            # 输出初始化的组件详情
            if init_results['initialized_components']:
                logger.info(f"   🔄 Initialized components:")
                for component in init_results['initialized_components'][:10]:  # 只显示前10个
                    logger.info(f"      {component}")
                if len(init_results['initialized_components']) > 10:
                    logger.info(f"      ... and {len(init_results['initialized_components']) - 10} more")
                    
            # 标记初始化完成
            self._non_expert_fp16_initialized = True
            self._needs_non_expert_fp16_init = False
            
        except Exception as e:
            logger.error(f"Failed to initialize non-expert layers: {e}")
            import traceback
            traceback.print_exc()
    
    def check_and_initialize_non_expert_layers_hot(self):
        """检查并在需要时进行热迁移非expert层FP16初始化"""
        if self._non_expert_fp16_initialized:
            return  # 已经初始化过了
        
        if not hasattr(self, '_needs_non_expert_fp16_init') or not self._needs_non_expert_fp16_init:
            return  # 不需要初始化
        
        # 异步进行初始化，不阻塞推理
        import threading
        def async_init():
            try:
                logger.info("🔥 Hot migration: Starting non-expert FP16 initialization in background...")
                self.initialize_non_expert_layers_after_model_load()
                logger.info("🔥 Hot migration: Non-expert FP16 initialization completed in background")
            except Exception as e:
                logger.error(f"Hot migration initialization failed: {e}")
        
        # 启动后台线程进行初始化
        init_thread = threading.Thread(target=async_init, daemon=True)
        init_thread.start()
        
        # 标记正在初始化，避免重复启动
        self._needs_non_expert_fp16_init = False
    
    def get_expert_quantization_map(self) -> Dict[int, Dict[int, str]]:
        """获取expert量化映射"""
        with self._expert_activation_lock:
            return dict(self._expert_quantization_map)
    
    def export_quantization_report(self) -> Dict[str, Any]:
        """导出量化报告"""
        with self._expert_activation_lock:
            report = {
                'timestamp': time.time(),
                'quantization_config': self._quantization_config.copy(),
                'expert_quantization_map': dict(self._expert_quantization_map),
                'statistics': {}
            }
            
            # 统计各精度的expert数量
            precision_stats = defaultdict(int)
            for layer_map in self._expert_quantization_map.values():
                for precision in layer_map.values():
                    precision_stats[precision] += 1
            
            report['statistics'] = dict(precision_stats)
            return report


# 创建全局expert distribution recorder并启用expert tracking hook
_global_expert_distribution_recorder: Optional[ExpertDistributionRecorder] = (
    _ExpertDistributionRecorderNoop()
)

# 启用expert tracking hook
if _global_expert_distribution_recorder:
    _global_expert_distribution_recorder.enable_expert_tracking_hook()


def get_global_expert_distribution_recorder():
    return _global_expert_distribution_recorder

def get_expert_quantization_map() -> Dict[int, Dict[int, str]]:
    """获取全局expert量化映射"""
    recorder = get_global_expert_distribution_recorder()
    if hasattr(recorder, 'get_expert_quantization_map'):
        return recorder.get_expert_quantization_map()
    return {}

def set_quantization_thresholds(high_threshold: float = None, medium_threshold: float = None):
    """设置全局量化阈值"""
    recorder = get_global_expert_distribution_recorder()
    if hasattr(recorder, 'set_quantization_thresholds'):
        recorder.set_quantization_thresholds(high_threshold, medium_threshold)

def export_quantization_report() -> Dict[str, Any]:
    """导出量化报告"""
    recorder = get_global_expert_distribution_recorder()
    if hasattr(recorder, 'export_quantization_report'):
        return recorder.export_quantization_report()
    return {}


def set_global_expert_distribution_recorder(value):
    global _global_expert_distribution_recorder
    _global_expert_distribution_recorder = value
    # 确保新设置的记录器也启用了expert tracking hook
    if value and hasattr(value, 'enable_expert_tracking_hook'):
        value.enable_expert_tracking_hook()


# --------------------------------------- SinglePassGatherer -----------------------------------------


class _SinglePassGatherer(ABC):
    @staticmethod
    def init_new(
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ) -> "_SinglePassGatherer":
        if server_args.expert_distribution_recorder_mode == "per_token":
            return _DetailSinglePassGatherer(
                server_args, expert_location_metadata, rank
            )

        if server_args.expert_distribution_recorder_mode == "stat_approx":
            if server_args.enable_deepep_moe and (server_args.deepep_mode == "normal"):
                return _DeepepNormalSinglePassGatherer(expert_location_metadata, rank)
            else:
                raise NotImplementedError

        if server_args.enable_deepep_moe:
            if server_args.deepep_mode == "normal":
                return _SelectExpertsSinglePassGatherer(expert_location_metadata, rank)
            elif server_args.deepep_mode == "low_latency":
                return _DeepepLowLatencySinglePassGatherer(
                    expert_location_metadata, rank
                )
            else:
                raise NotImplementedError

        return _SelectExpertsSinglePassGatherer(expert_location_metadata, rank)

    def __init__(self, expert_location_metadata: "ExpertLocationMetadata", rank: int):
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank

    def on_forward_pass_start(self, forward_batch: ForwardBatch):
        pass

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        pass

    def on_deepep_dispatch_low_latency(
        self, layer_idx: int, local_physical_count_of_layer: torch.Tensor
    ):
        pass

    def reset(self):
        raise NotImplementedError

    def collect(self) -> Dict:
        raise NotImplementedError


class _DetailSinglePassGatherer(_SinglePassGatherer):
    # DeepSeek V3 has this value; should generalize later
    _TOP_K_NUM = 8

    def __init__(
        self,
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ):
        super().__init__(expert_location_metadata, rank)
        self._metadata: Optional[Dict[str, Any]] = None
        self._topk_ids_of_layer = torch.zeros(
            (
                expert_location_metadata.num_layers,
                # TODO determine the max number
                server_args.chunked_prefill_size * 8,
                self._TOP_K_NUM,
            ),
            dtype=torch.int32,
            device=server_args.device,
        )
        self._misc_objects: List[Dict[str, Any]] = []
        assert (
            not server_args.enable_two_batch_overlap
        ), "DetailSinglePassGatherer does not support TBO yet"
        # TODO assert shared experts fusion is disabled, o/w data is wrong

    def on_forward_pass_start(self, forward_batch: ForwardBatch):
        assert self._metadata is None
        self._metadata = dict(
            # TODO pr-chain
            # rids=forward_batch.rids,
            input_ids=forward_batch.input_ids.cpu().tolist(),
            positions=forward_batch.positions.cpu().tolist(),
            extend_seq_lens=forward_batch.extend_seq_lens_cpu,
            forward_mode=forward_batch.forward_mode.value,
        )

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        self._topk_ids_of_layer[layer_idx, : topk_ids.shape[0], : topk_ids.shape[1]] = (
            topk_ids
        )

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        self._misc_objects.append(
            dict(
                layer_id=layer_idx,
                num_tokens_per_rank=num_tokens_per_rank.cpu().tolist(),
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank.cpu().tolist(),
                num_tokens_per_expert=num_tokens_per_expert.cpu().tolist(),
            )
        )

    def reset(self):
        self._topk_ids_of_layer[...] = -1
        self._misc_objects.clear()
        self._metadata = None

    def collect(self) -> Dict:
        num_tokens = len(self._metadata["input_ids"])
        return dict(
            **self._metadata,
            topk_ids_of_layer=self._topk_ids_of_layer[:, :num_tokens, :].clone().cpu(),
            misc_objects=self._misc_objects,
        )


class _LayerBasedCpuSinglePassGatherer(_SinglePassGatherer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._objects_of_layer = {}

    def _on_layer_data(self, layer_idx: int, objects: List[int]):
        assert 0 <= layer_idx < self._expert_location_metadata.num_layers
        if layer_idx in self._objects_of_layer:
            self._objects_of_layer[layer_idx] = _list_sum(
                self._objects_of_layer[layer_idx], objects
            )
        else:
            self._objects_of_layer[layer_idx] = objects

    def reset(self):
        self._objects_of_layer.clear()

    def _collect_objects(self, pad_len: int) -> torch.Tensor:
        data = [
            self._objects_of_layer.get(layer_index) or ([0] * pad_len)
            for layer_index in range(self._expert_location_metadata.num_layers)
        ]
        return torch.tensor(data)


def _list_sum(a: List, b: List) -> List:
    return [x + y for x, y in zip(a, b, strict=True)]


class _LayerBasedGpuSinglePassGatherer(_SinglePassGatherer):
    def __init__(self, *args, enable_global_physical_experts: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_global_physical_experts = enable_global_physical_experts
        self._data = torch.zeros(
            (
                self._expert_location_metadata.num_layers,
                (
                    self._expert_location_metadata.num_physical_experts
                    if enable_global_physical_experts
                    else self._expert_location_metadata.num_local_physical_experts
                ),
            ),
            dtype=torch.int,
            device="cuda",
        )

    def reset(self):
        self._data[...] = 0

    def collect(self) -> Dict:
        if self._enable_global_physical_experts:
            global_physical_count = self._data
        else:
            # Can optimize if bottleneck
            global_physical_count = _convert_local_to_global_physical_count(
                self._data,
                rank=self._rank,
                num_local_physical_experts=self._expert_location_metadata.num_local_physical_experts,
                num_physical_experts=self._expert_location_metadata.num_physical_experts,
            )

        return dict(global_physical_count=global_physical_count)


class _SelectExpertsSinglePassGatherer(_LayerBasedGpuSinglePassGatherer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, enable_global_physical_experts=True)

    # can optimize (e.g. fuse / compile)
    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        topk_ids = topk_ids.flatten()
        mask = topk_ids != -1
        self._data[layer_idx, :].scatter_add_(
            dim=0, index=topk_ids.masked_fill(~mask, 0).long(), src=mask.int()
        )


class _DeepepNormalSinglePassGatherer(_LayerBasedCpuSinglePassGatherer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.distributed.get_rank() == 0:
            logger.info(
                "DeepepNormalSinglePassGatherer gathers approximate statistics. "
                "If used with small batch size, consider using expert_distribution_recorder_mode=stat."
            )

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        assert isinstance(local_physical_count_of_layer, list)
        self._on_layer_data(layer_idx, local_physical_count_of_layer)

    def collect(self) -> Dict:
        local_physical_count = super()._collect_objects(
            pad_len=self._expert_location_metadata.num_local_physical_experts
        )
        global_physical_count = _convert_local_to_global_physical_count(
            local_physical_count,
            rank=self._rank,
            num_local_physical_experts=self._expert_location_metadata.num_local_physical_experts,
            num_physical_experts=self._expert_location_metadata.num_physical_experts,
        )
        return dict(global_physical_count=global_physical_count)


class _DeepepLowLatencySinglePassGatherer(_LayerBasedGpuSinglePassGatherer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, enable_global_physical_experts=False)

    def on_deepep_dispatch_low_latency(
        self, layer_idx: int, local_physical_count_of_layer: torch.Tensor
    ):
        # Most naive implementation, can optimize later
        self._data[layer_idx, :] += local_physical_count_of_layer


def _convert_local_to_global_physical_count(
    local_physical_count: torch.Tensor,
    rank: int,
    num_local_physical_experts: int,
    num_physical_experts: int,
) -> torch.Tensor:
    dtype = local_physical_count.dtype
    device = local_physical_count.device
    num_layers, _ = local_physical_count.shape

    ans = torch.zeros((num_layers, num_physical_experts), dtype=dtype, device=device)
    ans[
        :, num_local_physical_experts * rank : num_local_physical_experts * (rank + 1)
    ] = local_physical_count
    return ans


# --------------------------------------- Accumulator -----------------------------------------

_SINGLE_PASS_GATHERER_KEY_PRIMARY = "primary"


class _Accumulator(ABC):
    @staticmethod
    def init_new(
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ) -> "_Accumulator":
        return _Accumulator.get_class(server_args)(
            server_args, expert_location_metadata, rank
        )

    @staticmethod
    def get_class(server_args: ServerArgs) -> Type["_Accumulator"]:
        return {
            "stat": _StatAccumulator,
            "stat_approx": _StatAccumulator,
            "per_pass": _DetailAccumulator,
            "per_token": _DetailAccumulator,
        }[server_args.expert_distribution_recorder_mode]

    def __init__(
        self,
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ):
        self._server_args = server_args
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank

    def get_single_pass_gatherer_keys(self):
        return [_SINGLE_PASS_GATHERER_KEY_PRIMARY]

    def get_single_pass_gatherer_key(self, debug_name: Optional[str]):
        return _SINGLE_PASS_GATHERER_KEY_PRIMARY

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
    ):
        pass

    def reset(self):
        pass

    def dump(self, output_mode: _OutputMode):
        pass


class _UtilizationRateAccumulatorMixin(_Accumulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._enable = self._server_args.enable_expert_distribution_metrics

        if self._enable:
            window_sizes = [10, 100, 1000]
            self._history = _DequeCollection(maxlens=window_sizes)
            self._rank = torch.distributed.get_rank()

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data)
        if self._enable:
            # 检查 global_physical_count 是否存在
            if "global_physical_count" in single_pass_data:
                self._append_utilization_rate(
                    forward_pass_id, single_pass_data["global_physical_count"]
                )
            else:
                # 如果不存在，记录警告并跳过
                logger.debug(f"Missing global_physical_count in single_pass_data for forward_pass_id {forward_pass_id}")

    def reset(self):
        super().reset()
        if self._enable:
            self._history.clear()

    def _append_utilization_rate(
        self, forward_pass_id: int, single_pass_global_physical_count: torch.Tensor
    ):
        gpu_physical_count = compute_gpu_physical_count(
            single_pass_global_physical_count,
            num_gpu=self._expert_location_metadata.ep_size,
        )
        gpu_physical_count = gpu_physical_count.to(self._server_args.device)
        torch.distributed.reduce(
            gpu_physical_count, dst=0, op=torch.distributed.ReduceOp.SUM
        )

        if self._rank == 0:
            utilization_rate_tensor = compute_utilization_rate(gpu_physical_count)
            utilization_rate = torch.mean(utilization_rate_tensor).item()
            self._history.append(utilization_rate)

            gpu_physical_count_sum = gpu_physical_count.sum().item()

            # logger.info(
            #     f"[Expert Balancedness] "
            #     f"forward_pass_id={forward_pass_id} "
            #     f"current_pass_balancedness={utilization_rate:.03f} "
            #     f"{''.join(f'last_{size}_average_balancedness={value:.03f} ' for size, value in self._history.mean().items())} "
            #     f"gpu_physical_count_sum={gpu_physical_count_sum}"
            #     # f"current_pass_per_layer={[round(x, 2) for x in utilization_rate_tensor.cpu().tolist()]}"
            # )


class _DequeCollection:
    def __init__(self, maxlens: List[int]):
        self._dequeues = [deque(maxlen=maxlen) for maxlen in maxlens]

    def append(self, value):
        for d in self._dequeues:
            d.append(value)

    def clear(self):
        for d in self._dequeues:
            d.clear()

    def mean(self) -> Dict[int, float]:
        return {d.maxlen: sum(d) / len(d) for d in self._dequeues}


class _DetailAccumulator(_UtilizationRateAccumulatorMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._records = []

    def get_single_pass_gatherer_keys(self):
        if False:  # TODO `server_args.enable_two_batch_overlap`
            return [_SINGLE_PASS_GATHERER_KEY_PRIMARY, "child_a", "child_b"]
        return super().get_single_pass_gatherer_keys()

    def get_single_pass_gatherer_key(self, debug_name: Optional[str]):
        if False:  # TODO `server_args.enable_two_batch_overlap`
            return debug_name or _SINGLE_PASS_GATHERER_KEY_PRIMARY
        return super().get_single_pass_gatherer_key(debug_name)

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data)

        def _process_object(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().clone()
            return obj

        single_pass_data_processed = {
            k: _process_object(v) for k, v in single_pass_data.items()
        }

        self._records.append(
            dict(
                forward_pass_id=forward_pass_id,
                rank=self._rank,
                gatherer_key=gatherer_key,
                **single_pass_data_processed,
            )
        )

    def reset(self):
        super().reset()
        self._records.clear()

    def dump(self, output_mode: _OutputMode):
        assert output_mode == "file"
        output = dict(
            records=self._records,
            # NOTE: This may change during recording, so here we say it is the "last" one
            last_physical_to_logical_map=self._expert_location_metadata.physical_to_logical_map,
        )
        _dump_to_file(
            f"expert_distribution_recorder_{time.time()}_{self._rank}.pt", output
        )


class _StatAccumulator(_UtilizationRateAccumulatorMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._global_physical_count_of_buffered_step = _Buffer.init_new(
            item_shape=(
                self._expert_location_metadata.num_layers,
                # Cannot use local_physical_count to support select_experts
                self._expert_location_metadata.num_physical_experts,
            ),
            buffer_size=self._server_args.expert_distribution_recorder_buffer_size,
            dtype=torch.int32,
            device=self._server_args.device,
        )
        self._first_dump = True

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data)
        # Can optimize if overhead here is large
        if "global_physical_count" in single_pass_data:
            self._global_physical_count_of_buffered_step.append(
                single_pass_data["global_physical_count"]
            )
        else:
            logger.warning(f"🔍 [DEBUG] Missing global_physical_count in single_pass_data: {list(single_pass_data.keys())}")
            # 创建一个空的tensor作为占位符
            empty_tensor = torch.zeros(
                (self._expert_location_metadata.num_layers, self._expert_location_metadata.num_physical_experts),
                dtype=torch.int32,
                device=self._server_args.device
            )
            self._global_physical_count_of_buffered_step.append(empty_tensor)

    def reset(self):
        super().reset()
        self._global_physical_count_of_buffered_step.reset()

    def dump(self, output_mode: _OutputMode):
        logical_count_of_buffered_step = _convert_global_physical_count_to_logical_count(
            self._global_physical_count_of_buffered_step.get_all(),
            num_layers=self._expert_location_metadata.num_layers,
            num_logical_experts=self._expert_location_metadata.num_logical_experts,
            physical_to_logical_map=self._expert_location_metadata.physical_to_logical_map,
        )

        if self._first_dump:
            self._first_dump = False
            torch.cuda.empty_cache()

        torch.distributed.all_reduce(
            logical_count_of_buffered_step, op=torch.distributed.ReduceOp.SUM
        )

        output = dict(
            rank=self._rank,
            logical_count=logical_count_of_buffered_step,
        )

        if output_mode == "file":
            if self._rank == 0:
                _dump_to_file(f"expert_distribution_recorder_{time.time()}.pt", output)
        elif output_mode == "object":
            return output
        else:
            raise NotImplementedError


def _dump_to_file(name, data):
    save_dir = Path(os.environ.get("SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR", "/tmp"))
    path_output = save_dir / name
    logger.info(f"Write expert distribution to {path_output}")
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(path_output))


class _Buffer:
    @staticmethod
    def init_new(item_shape: Tuple, buffer_size: int, dtype, device):
        if buffer_size < 0:
            return _InfiniteBuffer(item_shape, dtype=dtype, device=device)
        else:
            return _CircularBuffer(item_shape, buffer_size, dtype=dtype, device=device)

    def append(self, value: torch.Tensor):
        raise NotImplementedError

    def get_all(self) -> torch.Tensor:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class _CircularBuffer(_Buffer):
    def __init__(self, item_shape: Tuple, buffer_size: int, dtype, device):
        self._buffer = torch.zeros(
            (buffer_size, *item_shape), dtype=dtype, device=device
        )
        self._curr_index = 0

    def append(self, value: torch.Tensor):
        self._buffer[self._curr_index] = value
        self._curr_index = (self._curr_index + 1) % len(self._buffer)

    def get_all(self) -> torch.Tensor:
        return self._buffer

    def reset(self):
        self._buffer[...] = 0


class _InfiniteBuffer(_Buffer):
    def __init__(self, item_shape: Tuple, dtype, device):
        self._item_shape = item_shape
        self._buffer = torch.zeros((128, *item_shape), dtype=dtype, device=device)
        self._size = 0

    def append(self, value: torch.Tensor):
        curr_buffer_size = len(self._buffer)
        dtype = self._buffer.dtype
        device = self._buffer.device

        if self._size == curr_buffer_size:
            new_buffer = torch.zeros(
                (2 * curr_buffer_size, *self._item_shape), dtype=dtype, device=device
            )
            new_buffer[:curr_buffer_size] = self._buffer
            self._buffer = new_buffer

        self._buffer[self._size] = value
        self._size += 1

    def get_all(self) -> torch.Tensor:
        return self._buffer[: self._size]

    def reset(self):
        self._buffer[...] = 0
        self._size = 0


def _convert_global_physical_count_to_logical_count(
    # (whatever, num_layers, num_physical_experts)
    global_physical_count: torch.Tensor,
    num_layers: int,
    num_logical_experts: int,
    physical_to_logical_map: torch.Tensor,
):
    dim_extra, _, _ = global_physical_count.shape
    dtype = global_physical_count.dtype
    device = global_physical_count.device
    logical_count = torch.zeros(
        (dim_extra, num_layers, num_logical_experts), dtype=dtype, device=device
    )
    logical_count.scatter_add_(
        dim=2,
        index=physical_to_logical_map.unsqueeze(0)
        .expand(dim_extra, -1, -1)
        .to(torch.int64),
        src=global_physical_count,
    )
    return logical_count


def compute_gpu_physical_count(
    physical_count_of_whatever: torch.Tensor,  # (..., num_layer, num_physical_expert)
    num_gpu: int,
):
    """output: gpu_physical_count_of_batch (..., num_layer, num_gpu)"""
    return einops.reduce(
        physical_count_of_whatever,
        "... num_layer (num_gpu num_expert_per_gpu) -> ... num_layer num_gpu",
        "sum",
        num_gpu=num_gpu,
    )


def compute_utilization_rate(
    gpu_physical_count_of_batch: torch.Tensor,  # (..., num_layer, num_gpu)
):
    """output: utilization_rate (..., num_layer)"""
    gpu_physical_count_of_batch = gpu_physical_count_of_batch.float()
    max_gpu_physical_count = einops.reduce(
        gpu_physical_count_of_batch,
        "... num_layer num_gpu -> ... num_layer",
        "max",
    )
    avg_gpu_physical_count = einops.reduce(
        gpu_physical_count_of_batch,
        "... num_layer num_gpu -> ... num_layer",
        "mean",
    )
    return (avg_gpu_physical_count + 1e-5) / (max_gpu_physical_count + 1e-5)


# 添加expert ID映射方法到基类
def _map_physical_to_logical_expert_id(self, physical_expert_id: int, layer_idx: int) -> Optional[int]:
    """将物理expert ID映射到逻辑expert ID"""
    try:
        # 检查是否有expert location metadata
        if hasattr(self, 'expert_location_metadata') and self.expert_location_metadata is not None:
            # 使用expert location metadata进行映射
            if (layer_idx < self.expert_location_metadata.num_layers and 
                physical_expert_id < self.expert_location_metadata.num_physical_experts):
                logical_expert_id = self.expert_location_metadata.physical_to_logical_map[layer_idx, physical_expert_id].item()
                if logical_expert_id >= 0:  # 有效的映射
                    return logical_expert_id
        
        # 如果没有metadata，尝试使用分布式配置进行映射
        try:
            from sglang.srt.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
            from sglang.srt.managers.schedule_batch import global_server_args_dict
            
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            
            # 检查是否启用了 EP (Expert Parallelism)
            if global_server_args_dict.get('enable_ep_moe', False) or global_server_args_dict.get('enable_deepep_moe', False):
                # 在 EP 环境下，每个 rank 只处理部分 expert
                # 需要将物理 expert ID 映射到全局逻辑 expert ID
                num_experts = global_server_args_dict.get('num_experts', 8)
                experts_per_rank = num_experts // tp_size
                global_expert_id = tp_rank * experts_per_rank + physical_expert_id
                return global_expert_id
        except Exception as e:
            logger.debug(f"分布式映射失败: {e}")
        
        # 如果所有映射都失败，返回None表示使用原始ID
        return None
    except Exception as e:
        logger.debug(f"映射expert ID失败: {e}")
        return None

# 将映射方法添加到基类
ExpertDistributionRecorder._map_physical_to_logical_expert_id = _map_physical_to_logical_expert_id

# 添加调试函数
def debug_expert_tracking():
    """调试expert tracking状态"""
    tracker = get_global_expert_tracker()
    print(f"🔍 [DEBUG] Global expert tracker: {tracker}")
    
    recorder = get_global_expert_distribution_recorder()
    print(f"🔍 [DEBUG] Global expert distribution recorder: {recorder}")
    
    if recorder:
        hook_enabled = hasattr(recorder, '_expert_tracker_hook_enabled') and recorder._expert_tracker_hook_enabled
        print(f"🔍 [DEBUG] Expert tracking hook enabled: {hook_enabled}")
        
        if hasattr(recorder, 'expert_location_metadata'):
            print(f"🔍 [DEBUG] Expert location metadata: {recorder.expert_location_metadata}")
    
    if tracker:
        stats = tracker.get_expert_stats()
        print(f"🔍 [DEBUG] Current expert stats count: {len(stats)}")
        for key, stat in list(stats.items())[:3]:  # 显示前3个
            print(f"🔍 [DEBUG] {key}: {stat}")
