#!/usr/bin/env python3
"""
优化的混合精度权重加载器
减少中间精度转换，直接加载对应精度的权重
"""

import os
import torch
import torch.nn as nn
import logging
import json
import yaml
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 导入校准器
from .quantization_calibrator import QuantizationCalibrator, get_global_calibrator


@dataclass
class OptimizedMixedPrecisionConfig:
    """优化的混合精度配置"""
    fp16_model_path: str
    fp8_model_path: str
    gptq_int4_model_path: str
    expert_precision_mapping: Dict[str, str] = None
    calibration_data_path: Optional[str] = None
    enable_calibration: bool = True


class OptimizedMixedPrecisionLoader:
    """优化的混合精度权重加载器"""
    
    def __init__(self, config: OptimizedMixedPrecisionConfig):
        """
        初始化优化的混合精度加载器
        
        Args:
            config: 混合精度配置
        """
        self.config = config
        self.calibrator = get_global_calibrator() if config.enable_calibration else None
        
        # 验证模型路径
        self._validate_model_paths()
        
        # 加载专家精度映射
        self.expert_precision_mapping = config.expert_precision_mapping or {}
        
        logger.info("Optimized mixed precision loader initialized")
    
    def _validate_model_paths(self):
        """验证模型路径"""
        for path_name, path in [
            ("fp16", self.config.fp16_model_path),
            ("fp8", self.config.fp8_model_path),
            ("gptq_int4", self.config.gptq_int4_model_path)
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path_name} model path not found: {path}")
    
    def load_model_weights(self, model: nn.Module) -> Dict[str, Any]:
        """加载模型权重 - 优化的合并流程"""
        stats = {
            'loaded': 0,
            'skipped': 0,
            'errors': 0,
            'details': []
        }
        
        try:
            # 遍历模型的所有模块
            for name, module in model.named_modules():
                if self._should_process_module(name, module):
                    try:
                        # 确定该模块应该使用的精度
                        precision = self._determine_module_precision(name)
                        
                        # 直接从对应精度模型加载权重
                        success = self._load_weight_directly(name, module, precision)
                        
                        if success:
                            stats['loaded'] += 1
                            stats['details'].append({
                                'module': name,
                                'precision': precision,
                                'status': 'loaded'
                            })
                        else:
                            stats['skipped'] += 1
                            stats['details'].append({
                                'module': name,
                                'precision': precision,
                                'status': 'skipped'
                            })
                            
                    except Exception as e:
                        stats['errors'] += 1
                        stats['details'].append({
                            'module': name,
                            'precision': precision if 'precision' in locals() else 'unknown',
                            'status': 'error',
                            'error': str(e)
                        })
                        logger.error(f"Error loading weight for {name}: {e}")
            
            logger.info(f"Mixed precision loading completed: {stats['loaded']} loaded, {stats['skipped']} skipped, {stats['errors']} errors")
            
        except Exception as e:
            logger.error(f"Error in mixed precision loading: {e}")
            stats['errors'] += 1
        
        return stats
    
    def _should_process_module(self, name: str, module: nn.Module) -> bool:
        """判断是否应该处理该模块"""
        # 只处理有权重的线性层和专家层
        return hasattr(module, 'weight') and isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))
    
    def _determine_module_precision(self, name: str) -> str:
        """确定模块应该使用的精度"""
        # 首先检查显式映射
        if name in self.expert_precision_mapping:
            return self.expert_precision_mapping[name]
        
        # 检查是否是专家层
        if self._is_expert_layer(name):
            return self._get_expert_precision(name)
        
        # 检查是否是注意力层
        if self._is_attention_layer(name):
            return "fp16"  # 注意力层保持高精度
        
        # 其他层使用FP16
        return "fp16"
    
    def _is_expert_layer(self, name: str) -> bool:
        """判断是否是专家层"""
        expert_keywords = ["expert", "mlp.experts"]
        return any(keyword in name.lower() for keyword in expert_keywords)
    
    def _is_attention_layer(self, name: str) -> bool:
        """判断是否是注意力层"""
        attention_keywords = ["attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj"]
        return any(keyword in name.lower() for keyword in attention_keywords)
    
    def _get_expert_precision(self, name: str) -> str:
        """根据专家层类型确定精度"""
        # 基于专家子层类型确定量化策略
        if "gate" in name.lower():
            return "fp16"  # W_gate对量化最敏感
        elif "up" in name.lower():
            return "fp8"   # W_up中等敏感
        elif "down" in name.lower():
            return "gptq_int4"  # W_down相对不敏感
        else:
            return "fp8"  # 默认使用FP8
    
    def _load_weight_directly(self, name: str, module: nn.Module, precision: str) -> bool:
        """直接从对应精度模型加载权重"""
        try:
            # 确定权重文件路径
            weight_file_path = self._get_weight_file_path(name, precision)
            if not weight_file_path:
                logger.debug(f"No weight file found for {name} with precision {precision}")
                return False
            
            # 加载权重
            weight = self._load_weight_from_file(weight_file_path, name)
            if weight is None:
                logger.debug(f"Failed to load weight {name} from {weight_file_path}")
                return False
            
            # 应用校准（如果启用）
            if self.calibrator:
                weight = self._apply_calibration(name, weight, precision)
            
            # 直接设置权重，避免中间转换
            module.weight.data = weight
            
            logger.debug(f"Loaded weight {name} with precision {precision}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading weight {name} with precision {precision}: {e}")
            return False
    
    def _get_weight_file_path(self, name: str, precision: str) -> Optional[str]:
        """获取权重文件路径"""
        try:
            # 根据精度确定基础路径
            if precision == "fp16":
                base_path = self.config.fp16_model_path
            elif precision == "fp8":
                base_path = self.config.fp8_model_path
            elif precision == "gptq_int4":
                base_path = self.config.gptq_int4_model_path
            else:
                return None
            
            # 查找权重文件
            for ext in [".safetensors", ".bin", ".pt"]:
                weight_file = os.path.join(base_path, f"{name}{ext}")
                if os.path.exists(weight_file):
                    return weight_file
            
            # 如果没找到具体文件，尝试从索引文件中查找
            index_file = os.path.join(base_path, "model.safetensors.index.json")
            if os.path.exists(index_file):
                return self._find_weight_in_index(index_file, name)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting weight file path for {name}: {e}")
            return None
    
    def _find_weight_in_index(self, index_file: str, name: str) -> Optional[str]:
        """从索引文件中查找权重"""
        try:
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            weight_map = index_data.get("weight_map", {})
            if name in weight_map:
                weight_file = weight_map[name]
                base_path = os.path.dirname(index_file)
                return os.path.join(base_path, weight_file)
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding weight in index: {e}")
            return None
    
    def _load_weight_from_file(self, file_path: str, name: str) -> Optional[torch.Tensor]:
        """从文件中加载权重"""
        try:
            if file_path.endswith(".safetensors"):
                return self._load_from_safetensors(file_path, name)
            elif file_path.endswith(".bin"):
                return self._load_from_bin(file_path, name)
            elif file_path.endswith(".pt"):
                return self._load_from_pt(file_path, name)
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading weight from {file_path}: {e}")
            return None
    
    def _load_from_safetensors(self, file_path: str, name: str) -> Optional[torch.Tensor]:
        """从safetensors文件加载权重"""
        try:
            from safetensors import safe_open
            with safe_open(file_path, framework="pt", device="cpu") as f:
                if name in f.keys():
                    return f.get_tensor(name)
            return None
        except ImportError:
            logger.error("safetensors not available")
            return None
        except Exception as e:
            logger.error(f"Error loading from safetensors: {e}")
            return None
    
    def _load_from_bin(self, file_path: str, name: str) -> Optional[torch.Tensor]:
        """从bin文件加载权重"""
        try:
            state_dict = torch.load(file_path, map_location="cpu")
            return state_dict.get(name)
        except Exception as e:
            logger.error(f"Error loading from bin: {e}")
            return None
    
    def _load_from_pt(self, file_path: str, name: str) -> Optional[torch.Tensor]:
        """从pt文件加载权重"""
        try:
            state_dict = torch.load(file_path, map_location="cpu")
            return state_dict.get(name)
        except Exception as e:
            logger.error(f"Error loading from pt: {e}")
            return None
    
    def _apply_calibration(self, name: str, weight: torch.Tensor, precision: str) -> torch.Tensor:
        """应用校准优化权重"""
        try:
            if not self.calibrator:
                return weight
            
            # 收集权重统计
            self.calibrator.collect_weight_stats(name, weight)
            
            # 获取校准参数
            calibrated_params = self.calibrator.calibrate_quantization_params(name, precision)
            
            # 应用校准（这里主要是记录，实际的量化在模型推理时进行）
            if calibrated_params.calibration_confidence > 0.5:
                logger.debug(f"Applied calibration for {name}: scale={calibrated_params.scale:.4f}")
            
            return weight
            
        except Exception as e:
            logger.error(f"Error applying calibration for {name}: {e}")
            return weight
    
    def save_expert_precision_mapping(self, file_path: str):
        """保存专家精度映射"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.expert_precision_mapping, f, indent=2)
            logger.info(f"Saved expert precision mapping to {file_path}")
        except Exception as e:
            logger.error(f"Error saving expert precision mapping: {e}")
    
    def load_expert_precision_mapping(self, file_path: str):
        """加载专家精度映射"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    self.expert_precision_mapping = json.load(f)
                logger.info(f"Loaded expert precision mapping from {file_path}")
        except Exception as e:
            logger.error(f"Error loading expert precision mapping: {e}")
    
    def generate_loading_report(self, stats: Dict[str, Any]) -> str:
        """生成加载报告"""
        report = f"""
Mixed Precision Loading Report
============================
Total modules processed: {stats['loaded'] + stats['skipped'] + stats['errors']}
Successfully loaded: {stats['loaded']}
Skipped: {stats['skipped']}
Errors: {stats['errors']}

Precision distribution:
"""
        
        # 统计精度分布
        precision_count = {}
        for detail in stats['details']:
            if detail['status'] == 'loaded':
                precision = detail['precision']
                precision_count[precision] = precision_count.get(precision, 0) + 1
        
        for precision, count in precision_count.items():
            report += f"  {precision}: {count}\n"
        
        if stats['errors'] > 0:
            report += "\nErrors:\n"
            for detail in stats['details']:
                if detail['status'] == 'error':
                    report += f"  {detail['module']}: {detail['error']}\n"
        
        return report
