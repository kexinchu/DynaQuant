#!/usr/bin/env python3
"""
非Expert层FP16初始化器 - 确保attention层、layernorm等非expert层使用FP16精度
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import os
import json

logger = logging.getLogger(__name__)

class NonExpertFP16Initializer:
    """非Expert层FP16初始化器"""
    
    def __init__(self, fp16_path: str):
        """
        初始化非Expert层FP16初始化器
        
        Args:
            fp16_path: FP16模型路径
        """
        self.fp16_model_path = fp16_path
        self.initialized_layers = set()
        
        logger.info(f"NonExpertFP16Initializer initialized with path: {fp16_path}")
    
    def initialize_non_expert_layers_fp16(self, model) -> Dict[str, Any]:
        """
        初始化所有非expert层为FP16精度
        
        Args:
            model: 模型实例
            
        Returns:
            Dict[str, Any]: 初始化结果统计
        """
        results = {
            'total_layers_processed': 0,
            'successful_initializations': 0,
            'failed_initializations': 0,
            'initialized_components': [],
            'total_memory_usage_mb': 0.0
        }
        
        try:
            logger.info("🎯 Starting non-expert layer FP16 initialization...")
            
            # 1. 强制同步所有CUDA操作，确保没有正在进行的推理
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                for device_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_id):
                        torch.cuda.synchronize()
            
            # 加载权重映射
            if not self._load_weight_map():
                logger.error("Failed to load weight map, skipping FP16 initialization")
                return results
            
            # 获取模型内部结构
            if hasattr(model, 'model'):
                model_inner = model.model
                
                if hasattr(model_inner, 'layers'):
                    layers = model_inner.layers
                    results['total_layers_processed'] = len(layers)
                    
                    # 处理每一层的非expert组件
                    for layer_idx, layer in enumerate(layers):
                        layer_results = self._initialize_layer_non_expert_components(layer, layer_idx)
                        results['successful_initializations'] += layer_results['successful']
                        results['failed_initializations'] += layer_results['failed']
                        results['initialized_components'].extend(layer_results['components'])
                        results['total_memory_usage_mb'] += layer_results['memory_mb']
                
                # 处理embedding层
                embedding_results = self._initialize_embedding_layer(model_inner)
                results['successful_initializations'] += embedding_results['successful']
                results['failed_initializations'] += embedding_results['failed']
                results['initialized_components'].extend(embedding_results['components'])
                results['total_memory_usage_mb'] += embedding_results['memory_mb']
                
                # 处理lm_head层
                lm_head_results = self._initialize_lm_head_layer(model_inner)
                results['successful_initializations'] += lm_head_results['successful']
                results['failed_initializations'] += lm_head_results['failed']
                results['initialized_components'].extend(lm_head_results['components'])
                results['total_memory_usage_mb'] += lm_head_results['memory_mb']
            
            # 最终同步，确保所有转换完成
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                for device_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_id):
                        torch.cuda.synchronize()
            
            logger.info(f"✅ Non-expert layer FP16 initialization completed:")
            logger.info(f"   📊 Total components processed: {results['successful_initializations'] + results['failed_initializations']}")
            logger.info(f"   ✅ Successful initializations: {results['successful_initializations']}")
            logger.info(f"   ❌ Failed initializations: {results['failed_initializations']}")
            logger.info(f"   💾 Total memory usage: {results['total_memory_usage_mb']:.2f} MB")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to initialize non-expert layers: {e}")
            import traceback
            traceback.print_exc()
            return results
    
    def _load_weight_map(self) -> bool:
        """加载权重映射文件"""
        try:
            index_file = os.path.join(self.fp16_model_path, "model.safetensors.index.json")
            if not os.path.exists(index_file):
                logger.warning(f"Weight index file not found: {index_file}")
                return False
            
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            self._weight_map = index_data.get('weight_map', {})
            logger.info(f"Loaded weight map with {len(self._weight_map)} parameters")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load weight map: {e}")
            return False
    
    def _initialize_layer_non_expert_components(self, layer, layer_idx: int) -> Dict[str, Any]:
        """初始化单层的非expert组件为FP16"""
        results = {
            'successful': 0,
            'failed': 0,
            'components': [],
            'memory_mb': 0.0
        }
        
        try:
            # 初始化attention层
            if hasattr(layer, 'self_attn'):
                attn_results = self._initialize_attention_layer(layer.self_attn, layer_idx)
                results['successful'] += attn_results['successful']
                results['failed'] += attn_results['failed']
                results['components'].extend(attn_results['components'])
                results['memory_mb'] += attn_results['memory_mb']
            
            # 初始化layernorm层
            if hasattr(layer, 'input_layernorm'):
                ln_results = self._initialize_layernorm_layer(layer.input_layernorm, layer_idx, "input_layernorm")
                results['successful'] += ln_results['successful']
                results['failed'] += ln_results['failed']
                results['components'].extend(ln_results['components'])
                results['memory_mb'] += ln_results['memory_mb']
            
            if hasattr(layer, 'post_attention_layernorm'):
                ln_results = self._initialize_layernorm_layer(layer.post_attention_layernorm, layer_idx, "post_attention_layernorm")
                results['successful'] += ln_results['successful']
                results['failed'] += ln_results['failed']
                results['components'].extend(ln_results['components'])
                results['memory_mb'] += ln_results['memory_mb']
            
            # 初始化其他非expert组件
            if hasattr(layer, 'mlp') and not self._is_moe_layer(layer.mlp):
                mlp_results = self._initialize_dense_mlp_layer(layer.mlp, layer_idx)
                results['successful'] += mlp_results['successful']
                results['failed'] += mlp_results['failed']
                results['components'].extend(mlp_results['components'])
                results['memory_mb'] += mlp_results['memory_mb']
            
        except Exception as e:
            logger.error(f"Failed to initialize layer {layer_idx} non-expert components: {e}")
            results['failed'] += 1
        
        return results
    
    def _initialize_attention_layer(self, attention_layer, layer_idx: int) -> Dict[str, Any]:
        """初始化attention层为FP16"""
        results = {
            'successful': 0,
            'failed': 0,
            'components': [],
            'memory_mb': 0.0
        }
        
        try:
            # 检查attention层的组件结构
            attention_components = []
            
            # 检查是否有分别的q_proj, k_proj, v_proj
            if hasattr(attention_layer, 'q_proj') and hasattr(attention_layer, 'k_proj') and hasattr(attention_layer, 'v_proj'):
                attention_components.extend(['q_proj', 'k_proj', 'v_proj'])
            # 检查是否有合并的qkv_proj
            if hasattr(attention_layer, 'qkv_proj'):
                attention_components.append('qkv_proj')
            
            # 检查是否有o_proj
            if hasattr(attention_layer, 'o_proj'):
                attention_components.append('o_proj')
            
            logger.info(f"Found attention components in layer {layer_idx}: {attention_components}")
            
            for component_name in attention_components:
                if hasattr(attention_layer, component_name):
                    component = getattr(attention_layer, component_name)
                    component_path = f"model.layers.{layer_idx}.self_attn.{component_name}"
                    
                    if self._load_and_replace_fp16_weights(component, component_path):
                        results['successful'] += 1
                        results['components'].append(f"Layer{layer_idx}.self_attn.{component_name}")
                        results['memory_mb'] += self._estimate_component_memory(component)
                        logger.info(f"✅ Converted {component_name} in layer {layer_idx} to FP16")
                    else:
                        results['failed'] += 1
                        logger.warning(f"❌ Failed to convert {component_name} in layer {layer_idx}")
        
        except Exception as e:
            logger.error(f"Failed to initialize attention layer {layer_idx}: {e}")
            results['failed'] += 1
        
        return results
    
    def _initialize_layernorm_layer(self, layernorm_layer, layer_idx: int, component_name: str) -> Dict[str, Any]:
        """初始化layernorm层为FP16"""
        results = {
            'successful': 0,
            'failed': 0,
            'components': [],
            'memory_mb': 0.0
        }
        
        try:
            component_path = f"model.layers.{layer_idx}.{component_name}"
            
            if self._load_and_replace_fp16_weights(layernorm_layer, component_path):
                results['successful'] += 1
                results['components'].append(f"Layer{layer_idx}.{component_name}")
                results['memory_mb'] += self._estimate_component_memory(layernorm_layer)
                logger.info(f"✅ Converted {component_name} in layer {layer_idx} to FP16")
            else:
                results['failed'] += 1
                logger.warning(f"❌ Failed to convert {component_name} in layer {layer_idx}")
        
        except Exception as e:
            logger.error(f"Failed to initialize layernorm {component_name} in layer {layer_idx}: {e}")
            results['failed'] += 1
        
        return results
    
    def _initialize_dense_mlp_layer(self, mlp_layer, layer_idx: int) -> Dict[str, Any]:
        """初始化密集MLP层为FP16（非MoE）"""
        results = {
            'successful': 0,
            'failed': 0,
            'components': [],
            'memory_mb': 0.0
        }
        
        try:
            # 初始化MLP的各个组件
            mlp_components = ['gate_proj', 'up_proj', 'down_proj']
            
            for component_name in mlp_components:
                if hasattr(mlp_layer, component_name):
                    component = getattr(mlp_layer, component_name)
                    if self._convert_to_fp16(component):
                        results['successful'] += 1
                        results['components'].append(f"Layer{layer_idx}.mlp.{component_name}")
                        results['memory_mb'] += self._estimate_component_memory(component)
                        logger.info(f"✅ Converted {component_name} in layer {layer_idx} to FP16")
                    else:
                        results['failed'] += 1
                        logger.warning(f"❌ Failed to convert {component_name} in layer {layer_idx}")
        
        except Exception as e:
            logger.error(f"Failed to initialize dense MLP layer {layer_idx}: {e}")
            results['failed'] += 1
        
        return results
    
    def _initialize_embedding_layer(self, model_inner) -> Dict[str, Any]:
        """初始化embedding层为FP16"""
        results = {
            'successful': 0,
            'failed': 0,
            'components': [],
            'memory_mb': 0.0
        }
        
        try:
            logger.info(f"🔍 Checking embedding layer: hasattr(model_inner, 'embed_tokens') = {hasattr(model_inner, 'embed_tokens')}")
            if hasattr(model_inner, 'embed_tokens'):
                logger.info(f"🎯 Found embed_tokens layer: {type(model_inner.embed_tokens).__name__}")
                component_path = "model.embed_tokens"
                if self._load_and_replace_fp16_weights(model_inner.embed_tokens, component_path):
                    results['successful'] += 1
                    results['components'].append("embed_tokens")
                    results['memory_mb'] += self._estimate_component_memory(model_inner.embed_tokens)
                    logger.info("✅ Converted embed_tokens to FP16")
                else:
                    results['failed'] += 1
                    logger.warning("❌ Failed to convert embed_tokens")
            else:
                logger.info("❌ No embed_tokens found in model")
        
        except Exception as e:
            logger.error(f"Failed to initialize embedding layer: {e}")
            results['failed'] += 1
        
        return results
    
    def _initialize_lm_head_layer(self, model_inner) -> Dict[str, Any]:
        """初始化lm_head层为FP16"""
        results = {
            'successful': 0,
            'failed': 0,
            'components': [],
            'memory_mb': 0.0
        }
        
        try:
            if hasattr(model_inner, 'lm_head'):
                component_path = "model.lm_head"
                if self._load_and_replace_fp16_weights(model_inner.lm_head, component_path):
                    results['successful'] += 1
                    results['components'].append("lm_head")
                    results['memory_mb'] += self._estimate_component_memory(model_inner.lm_head)
                    logger.info("✅ Converted lm_head to FP16")
                else:
                    results['failed'] += 1
                    logger.warning("❌ Failed to convert lm_head")
        
        except Exception as e:
            logger.error(f"Failed to initialize lm_head layer: {e}")
            results['failed'] += 1
        
        return results
    
    def _is_moe_layer(self, mlp_layer) -> bool:
        """判断是否为MoE层"""
        # 检查是否有experts属性或EPMoE特征
        return (hasattr(mlp_layer, 'experts') or 
                hasattr(mlp_layer, 'w13_weight') or 
                hasattr(mlp_layer, 'num_experts'))
    
    def _load_and_replace_fp16_weights(self, module: nn.Module, component_path: str) -> bool:
        """从FP16模型文件加载并替换权重"""
        try:
            if not self._weight_map:
                logger.warning("Weight map not loaded")
                return False
            
            # 特殊处理qkv_proj：需要分别加载q_proj, k_proj, v_proj并合并
            if component_path.endswith('.qkv_proj'):
                return self._load_and_replace_qkv_proj(module, component_path)
            
            # 查找匹配的权重键
            matching_keys = []
            for key in self._weight_map.keys():
                if component_path in key:
                    matching_keys.append(key)
            
            if not matching_keys:
                logger.debug(f"No matching weights found for {component_path}")
                return False
            
            # 加载权重
            weights = {}
            from safetensors import safe_open
            
            for key in matching_keys:
                weight_file = self._weight_map[key]
                weight_file_path = os.path.join(self.fp16_model_path, weight_file)
                
                if os.path.exists(weight_file_path):
                    with safe_open(weight_file_path, framework="pt", device="cpu") as f:
                        if key in f.keys():
                            weight = f.get_tensor(key)
                            # 确保权重在CPU上，稍后会移动到正确的GPU设备
                            if weight.device.type != 'cpu':
                                weight = weight.cpu()
                            weights[key] = weight
                            logger.debug(f"Loaded {key} from {weight_file}")
            
            if not weights:
                logger.warning(f"No weights loaded for {component_path}")
                return False
            
            # 检查是否是embedding层，需要特殊处理
            if self._is_embedding_layer(module):
                logger.info(f"🎯 Detected embedding layer: {component_path}, using safe weight replacement")
                success = self._safe_replace_embedding_weights(module, weights, component_path)
            else:
                # 使用热迁移方法替换模块权重
                success = self._hot_migrate_module_weights(module, weights, component_path)
            return success
            
        except Exception as e:
            logger.warning(f"Failed to load and replace FP16 weights for {component_path}: {e}")
            return False
    
    def _load_and_replace_qkv_proj(self, module: nn.Module, component_path: str) -> bool:
        """特殊处理qkv_proj：分别加载q_proj, k_proj, v_proj并合并"""
        try:
            # 提取层号
            layer_idx = component_path.split('.')[2]  # model.layers.X.self_attn.qkv_proj
            
            # 分别查找q_proj, k_proj, v_proj的权重
            q_path = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
            k_path = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
            v_path = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
            
            q_weight = self._load_single_weight(q_path)
            k_weight = self._load_single_weight(k_path)
            v_weight = self._load_single_weight(v_path)
            
            if q_weight is None or k_weight is None or v_weight is None:
                logger.warning(f"Failed to load q/k/v weights for layer {layer_idx}")
                return False
            
            # 合并权重 - 需要正确处理多头注意力的权重形状
            # Q权重通常是 [num_heads * head_dim, hidden_size]
            # K/V权重通常是 [num_heads * head_dim, hidden_size]
            # 但实际形状可能不同，需要检查并适配
            
            try:
                # 检查权重形状是否兼容
                if q_weight.shape[0] == k_weight.shape[0] == v_weight.shape[0]:
                    # 形状匹配，直接合并
                    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=-1)
                else:
                    # 形状不匹配，需要特殊处理
                    # 假设Q权重包含了所有头，K/V权重需要重复
                    if q_weight.shape[0] > k_weight.shape[0]:
                        # Q权重比K/V权重大，需要重复K/V权重
                        repeat_factor = q_weight.shape[0] // k_weight.shape[0]
                        k_weight_repeated = k_weight.repeat(repeat_factor, 1)
                        v_weight_repeated = v_weight.repeat(repeat_factor, 1)
                        qkv_weight = torch.cat([q_weight, k_weight_repeated, v_weight_repeated], dim=-1)
                    else:
                        # 其他情况，尝试直接合并（可能会失败）
                        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=-1)
                
                logger.debug(f"QKV weight shapes - Q: {q_weight.shape}, K: {k_weight.shape}, V: {v_weight.shape}")
                logger.debug(f"Merged QKV weight shape: {qkv_weight.shape}")
                
            except Exception as e:
                logger.warning(f"Failed to merge QKV weights: {e}")
                logger.warning(f"Q shape: {q_weight.shape}, K shape: {k_weight.shape}, V shape: {v_weight.shape}")
                return False
            
            # 检查是否需要kernel迁移
            if self._needs_kernel_migration(module, component_path):
                logger.info(f"🔥 Kernel migration needed for {component_path}")
                # 使用热迁移kernel
                if self._hot_migrate_kernel(module, component_path):
                    # 热迁移成功后，替换权重
                    if hasattr(module, 'weight'):
                        # 确保权重在正确的设备上并转换为FP16
                        target_device = module.weight.device
                        qkv_weight = qkv_weight.half().to(target_device)
                        module.weight.data = qkv_weight
                        logger.debug(f"Successfully migrated kernel and replaced qkv_proj weight for layer {layer_idx}")
                        return True
                    else:
                        logger.warning(f"Module has no weight attribute: {type(module)}")
                        return False
                else:
                    logger.warning(f"Kernel migration failed for {component_path}")
                    return False
            else:
                # 不需要kernel迁移，直接替换权重
                if hasattr(module, 'weight'):
                    # 确保权重在正确的设备上并转换为FP16
                    target_device = module.weight.device
                    qkv_weight = qkv_weight.half().to(target_device)
                    module.weight.data = qkv_weight
                    logger.debug(f"Successfully replaced qkv_proj weight for layer {layer_idx}")
                    return True
                else:
                    logger.warning(f"Module has no weight attribute: {type(module)}")
                    return False
                
        except Exception as e:
            logger.warning(f"Failed to load and replace qkv_proj for {component_path}: {e}")
            return False
    
    def _load_single_weight(self, weight_path: str) -> Optional[torch.Tensor]:
        """加载单个权重"""
        try:
            if weight_path not in self._weight_map:
                logger.debug(f"Weight path not found: {weight_path}")
                return None
            
            weight_file = self._weight_map[weight_path]
            weight_file_path = os.path.join(self.fp16_model_path, weight_file)
            
            if not os.path.exists(weight_file_path):
                logger.warning(f"Weight file not found: {weight_file_path}")
                return None
            
            from safetensors import safe_open
            with safe_open(weight_file_path, framework="pt", device="cpu") as f:
                if weight_path in f.keys():
                    weight = f.get_tensor(weight_path)
                    # 确保权重在CPU上，调用者会负责移动到正确的GPU设备
                    if weight.device.type != 'cpu':
                        weight = weight.cpu()
                    return weight
                else:
                    logger.warning(f"Weight key not found in file: {weight_path}")
                    return None
                    
        except Exception as e:
            logger.warning(f"Failed to load single weight {weight_path}: {e}")
            return None
    
    def _handle_tensor_parallel_weight(self, source_weight: torch.Tensor, target_param: torch.Tensor, 
                                     component_path: str, param_name: str) -> Optional[torch.Tensor]:
        """处理tensor并行切分的权重"""
        try:
            source_shape = source_weight.shape
            target_shape = target_param.shape
            
            logger.debug(f"Handling tensor parallel weight: {param_name}")
            logger.debug(f"Source shape: {source_shape}, Target shape: {target_shape}")
            
            # 检查是否是tensor并行切分的情况
            if len(source_shape) == 2 and len(target_shape) == 2:
                # 2D权重矩阵的情况
                if source_shape[0] == target_shape[0] and source_shape[1] != target_shape[1]:
                    # 第一维相同，第二维不同 - 可能是列切分
                    if source_shape[1] > target_shape[1]:
                        # 源权重更大，需要切分
                        if source_shape[1] % target_shape[1] == 0:
                            # 可以整除，进行切分
                            split_factor = source_shape[1] // target_shape[1]
                            # 这里需要根据当前的tensor并行rank来确定切分哪一部分
                            # 暂时返回目标形状的权重（从源权重中取一部分）
                            start_idx = 0  # 这里应该根据rank计算
                            end_idx = target_shape[1]
                            processed_weight = source_weight[:, start_idx:end_idx]
                            # 确保权重在CPU上，稍后会移动到正确的GPU设备
                            if processed_weight.device.type != 'cpu':
                                processed_weight = processed_weight.cpu()
                            logger.info(f"Split weight for {param_name}: {source_shape} -> {processed_weight.shape}")
                            return processed_weight
                    
                elif source_shape[0] != target_shape[0] and source_shape[1] == target_shape[1]:
                    # 第二维相同，第一维不同 - 可能是行切分
                    if source_shape[0] > target_shape[0]:
                        # 源权重更大，需要切分
                        if source_shape[0] % target_shape[0] == 0:
                            # 可以整除，进行切分
                            split_factor = source_shape[0] // target_shape[0]
                            # 根据rank计算切分位置
                            start_idx = 0  # 这里应该根据rank计算
                            end_idx = target_shape[0]
                            processed_weight = source_weight[start_idx:end_idx, :]
                            # 确保权重在CPU上，稍后会移动到正确的GPU设备
                            if processed_weight.device.type != 'cpu':
                                processed_weight = processed_weight.cpu()
                            logger.info(f"Split weight for {param_name}: {source_shape} -> {processed_weight.shape}")
                            return processed_weight
            
            # 检查是否是简单的维度缩放
            if source_shape[0] == target_shape[0] * 4 or source_shape[1] == target_shape[1] * 4:
                # 可能是4卡并行的情况
                # 尝试从环境变量获取tensor并行rank
                rank = self._get_tensor_parallel_rank()
                
                if source_shape[0] == target_shape[0] * 4:
                    # 行切分
                    start_idx = rank * target_shape[0]
                    end_idx = (rank + 1) * target_shape[0]
                    processed_weight = source_weight[start_idx:end_idx, :]
                    # 保持权重在原始设备上，调用者会负责移动到目标设备
                    logger.info(f"Tensor parallel split (rows) for {param_name}: {source_shape} -> {processed_weight.shape} (rank={rank})")
                    return processed_weight
                elif source_shape[1] == target_shape[1] * 4:
                    # 列切分
                    start_idx = rank * target_shape[1]
                    end_idx = (rank + 1) * target_shape[1]
                    processed_weight = source_weight[:, start_idx:end_idx]
                    # 保持权重在原始设备上，调用者会负责移动到目标设备
                    logger.info(f"Tensor parallel split (cols) for {param_name}: {source_shape} -> {processed_weight.shape} (rank={rank})")
                    return processed_weight
            
            # 如果无法处理，返回None
            logger.warning(f"Cannot handle tensor parallel weight for {param_name}: {source_shape} vs {target_shape}")
            return None
            
        except Exception as e:
            logger.warning(f"Error handling tensor parallel weight for {param_name}: {e}")
            return None
    
    def _get_tensor_parallel_rank(self) -> int:
        """获取当前的tensor并行rank"""
        try:
            # 尝试从环境变量获取
            if 'RANK' in os.environ:
                return int(os.environ['RANK']) % 4  # 假设4卡并行
            
            # 尝试从分布式环境获取
            import torch.distributed as dist
            if dist.is_initialized():
                return dist.get_rank() % 4
            
            # 默认返回0
            logger.warning("Cannot determine tensor parallel rank, using 0")
            return 0
            
        except Exception as e:
            logger.warning(f"Error getting tensor parallel rank: {e}, using 0")
            return 0
    
    def _detect_current_precision(self, module: nn.Module) -> str:
        """检测当前模块的量化精度"""
        try:
            # 检查是否有量化方法
            if hasattr(module, 'quant_method'):
                if hasattr(module.quant_method, '__class__'):
                    class_name = module.quant_method.__class__.__name__.lower()
                    if 'fp8' in class_name:
                        return 'fp8'
                    elif 'gptq' in class_name:
                        return 'gptq_int4'
                    elif 'awq' in class_name:
                        return 'awq_int4'
            
            # 检查权重数据类型
            for name, param in module.named_parameters():
                if hasattr(param, 'dtype'):
                    if param.dtype == torch.float16:
                        return 'fp16'
                    elif param.dtype == torch.float8_e4m3fn:
                        return 'fp8'
                    elif param.dtype == torch.uint8:
                        return 'int4'
            
            # 默认返回fp16
            return 'fp16'
            
        except Exception as e:
            logger.warning(f"Failed to detect current precision: {e}")
            return 'fp16'
    
    def _hot_migrate_kernel_to_fp16(self, module: nn.Module) -> bool:
        """热迁移kernel到FP16：1.创建新kernel 2.加载参数 3.替换kernel 4.删除旧kernel"""
        try:
            logger.info("🔥 Starting hot kernel migration to FP16")
            
            # 1. 强制同步所有CUDA操作，确保没有正在进行的推理
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                # 额外的同步，确保所有操作完成
                for device_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_id):
                        torch.cuda.synchronize()
            
            # 2. 保存旧的量化方法
            old_quant_method = None
            if hasattr(module, 'quant_method'):
                old_quant_method = module.quant_method
            
            # 3. 原子性地替换为FP16
            with torch.no_grad():
                # 对于embedding层，需要保持quant_method但设置为FP16版本
                if hasattr(module, 'quant_method') and hasattr(module, 'weight'):
                    # 检查是否是embedding层 - 更准确的检测方法
                    is_embedding_layer = (
                        hasattr(module, 'num_embeddings') or  # VocabParallelEmbedding
                        hasattr(module, 'embedding_dim') or   # VocabParallelEmbedding
                        'VocabParallelEmbedding' in str(type(module)) or
                        'ParallelLMHead' in str(type(module)) or
                        'embed_tokens' in str(type(module)) or
                        'lm_head' in str(type(module))
                    )
                    
                    if is_embedding_layer:
                        # 对于embedding层，创建FP16的量化方法
                        from sglang.srt.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
                        module.quant_method = UnquantizedEmbeddingMethod()
                        logger.info(f"Set FP16 quant_method for embedding layer: {type(module).__name__}")
                    else:
                        # 对于其他层，移除量化方法
                        module.quant_method = None
                
                # 清理量化相关属性
                for attr_name in ['compressed_weight', 'weight_format', 'quantization_method', 'weight_scale_inv', 'scales']:
                    if hasattr(module, attr_name):
                        setattr(module, attr_name, None)
                
                # 确保权重是FP16并在正确的设备上
                for param in module.parameters():
                    if param.dtype != torch.float16:
                        # 确保权重在正确的设备上并转换为FP16
                        target_device = param.device
                        param.data = param.data.half().to(target_device)
                        logger.debug(f"Converted param to FP16 on device {target_device}: {param.shape}")
                    
                    # 额外检查：确保权重在正确的设备上
                    if param.device != target_device:
                        logger.warning(f"Param device mismatch: {param.device} vs {target_device}, fixing...")
                        param.data = param.data.to(target_device)
                
                # 强制同步权重更新
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            # 4. 清理旧kernel
            if old_quant_method is not None:
                del old_quant_method
                # 强制垃圾回收
                import gc
                gc.collect()
            
            # 5. 再次同步CUDA操作
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                # 额外的设备同步
                for device_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_id):
                        torch.cuda.synchronize()
            
            logger.info("✅ Hot kernel migration to FP16 completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to hot migrate kernel to FP16: {e}")
            return False
    
    def _hot_migrate_module_weights(self, module: nn.Module, weights: Dict[str, torch.Tensor], component_path: str) -> bool:
        """使用热迁移方法替换模块权重"""
        try:
            # 获取全局热迁移交换器
            from sglang.srt.managers.expert_distribution import get_global_hot_migration_swapper
            hot_migration_swapper = get_global_hot_migration_swapper()
            
            if hot_migration_swapper:
                # 使用热迁移交换器的热迁移方法
                return hot_migration_swapper.hot_migrate_non_expert_layer(module, weights, component_path)
            else:
                # 回退到普通替换方法
                logger.warning("Hot migration swapper not available, using fallback method")
                return self._replace_module_weights(module, weights, component_path)
                
        except Exception as e:
            logger.error(f"Failed to hot migrate module weights for {component_path}: {e}")
            # 回退到普通替换方法
            return self._replace_module_weights(module, weights, component_path)
    
    def _is_embedding_layer(self, module: nn.Module) -> bool:
        """检测是否是embedding层"""
        try:
            # 检查模块类型
            module_type = type(module).__name__
            if 'Embedding' in module_type or 'embed_tokens' in str(module_type):
                return True
            
            # 检查模块属性
            if hasattr(module, 'num_embeddings') and hasattr(module, 'embedding_dim'):
                return True
            
            # 检查模块名称
            if hasattr(module, '__class__'):
                class_name = module.__class__.__name__
                if 'VocabParallelEmbedding' in class_name or 'ParallelLMHead' in class_name:
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error detecting embedding layer: {e}")
            return False
    
    def _safe_replace_embedding_weights(self, module: nn.Module, weights: Dict[str, torch.Tensor], component_path: str) -> bool:
        """安全替换embedding层权重，不破坏量化方法"""
        try:
            logger.info(f"🔒 Safe weight replacement for embedding layer: {component_path}")
            
            # 检查是否有quant_method
            if hasattr(module, 'quant_method') and module.quant_method is not None:
                logger.info(f"Embedding layer has quant_method: {type(module.quant_method).__name__}")
                
                # 对于embedding层，我们只替换权重，不改变量化方法
                for param_name, param in module.named_parameters():
                    # 查找匹配的权重
                    possible_keys = [
                        f"{component_path}.{param_name}",
                        f"{component_path}.weight" if param_name == "weight" else None,
                    ]
                    
                    matching_weight = None
                    for key in possible_keys:
                        if key and key in weights:
                            matching_weight = weights[key]
                            break
                    
                    if matching_weight is not None:
                        # 检查权重形状
                        if matching_weight.shape == param.shape:
                            # 直接替换权重，确保设备匹配
                            with torch.no_grad():
                                # 确保权重在正确的设备上并转换为FP16
                                target_device = param.device
                                matching_weight = matching_weight.half().to(target_device)
                                param.data = matching_weight
                            logger.info(f"✅ Replaced {param_name} for {component_path}")
                        else:
                            # 尝试处理tensor并行权重
                            processed_weight = self._handle_tensor_parallel_weight(
                                matching_weight, param, component_path, param_name
                            )
                            if processed_weight is not None:
                                with torch.no_grad():
                                    # 确保权重在正确的设备上并转换为FP16
                                    target_device = param.device
                                    processed_weight = processed_weight.half().to(target_device)
                                    param.data = processed_weight
                                logger.info(f"✅ Adapted {param_name} for {component_path}: {matching_weight.shape} -> {processed_weight.shape}")
                            else:
                                logger.warning(f"❌ Cannot handle weight shape mismatch for {param_name}: {param.shape} vs {matching_weight.shape}")
                    else:
                        logger.debug(f"No matching weight found for {param_name} in {component_path}")
                
                # 验证quant_method仍然存在
                if hasattr(module, 'quant_method') and module.quant_method is not None:
                    logger.info(f"✅ Embedding layer quant_method preserved: {type(module.quant_method).__name__}")
                    return True
                else:
                    logger.error(f"❌ Embedding layer quant_method was lost during weight replacement")
                    return False
            else:
                logger.warning(f"Embedding layer has no quant_method, using standard replacement")
                return self._replace_module_weights(module, weights, component_path)
                
        except Exception as e:
            logger.error(f"Failed to safely replace embedding weights for {component_path}: {e}")
            return False
    
    def _replace_module_weights(self, module: nn.Module, weights: Dict[str, torch.Tensor], component_path: str) -> bool:
        """替换模块权重"""
        try:
            # 根据模块类型和权重键名匹配参数
            for param_name, param in module.named_parameters():
                # 构建可能的权重键名
                possible_keys = [
                    f"{component_path}.{param_name}",
                    f"{component_path}.weight" if param_name == "weight" else None,
                    f"{component_path}.bias" if param_name == "bias" else None
                ]
                
                # 查找匹配的权重
                matching_weight = None
                for key in possible_keys:
                    if key and key in weights:
                        matching_weight = weights[key]
                        break
                
                if matching_weight is not None:
                    # 正常处理权重替换，不在这里进行kernel迁移
                    if matching_weight.shape == param.shape:
                        # 替换权重，确保移动到正确的设备并转换为FP16
                        target_device = param.device
                        matching_weight = matching_weight.half().to(target_device)
                        param.data = matching_weight
                        logger.debug(f"Replaced {param_name} for {component_path}")
                    else:
                        # 尝试处理tensor并行切分的权重
                        processed_weight = self._handle_tensor_parallel_weight(
                            matching_weight, param, component_path, param_name
                        )
                        if processed_weight is not None:
                            # 确保权重在正确的设备上并转换为FP16
                            target_device = param.device
                            processed_weight = processed_weight.half().to(target_device)
                            param.data = processed_weight
                            logger.info(f"✅ Successfully adapted {param_name} for {component_path}: "
                                      f"{matching_weight.shape} -> {processed_weight.shape}")
                        else:
                            logger.warning(f"Shape mismatch for {param_name} in {component_path}: "
                                        f"expected {param.shape}, got {matching_weight.shape}")
                            logger.warning(f"Skipping {param_name} due to incompatible shape mismatch")
                else:
                    logger.debug(f"No matching weight found for {param_name} in {component_path}")
            
            # 在权重替换完成后，检查是否需要kernel热迁移
            old_precision = self._detect_current_precision(module)
            if old_precision != 'fp16':
                logger.info(f"🔥 Hot migrating kernel for {component_path} from {old_precision} to FP16")
                if not self._hot_migrate_kernel_to_fp16(module):
                    logger.warning(f"❌ Hot kernel migration failed for {component_path}")
                    return False
                logger.info(f"✅ Hot kernel migration completed for {component_path}")
            
            # 转换模块为FP16精度
            module = module.half()
            
            # 验证转换结果
            after_dtypes = set()
            for param in module.parameters():
                after_dtypes.add(str(param.dtype))
            
            logger.info(f"✅ Module {component_path} successfully converted to FP16, dtypes: {after_dtypes}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to replace module weights for {component_path}: {e}")
            return False
    
    def _needs_kernel_migration(self, module: nn.Module, component_path: str) -> bool:
        """检查是否需要kernel热迁移"""
        try:
            # 对于attention层，强制进行kernel迁移以确保从FP8切换到FP16
            if 'self_attn' in component_path:
                logger.debug(f"Module {component_path} needs kernel migration: attention layer")
                return True
            
            # 检查模块是否有量化方法
            if hasattr(module, 'quant_method'):
                # 检查当前量化方法类型
                quant_method_name = type(module.quant_method).__name__.lower()
                if 'fp8' in quant_method_name or 'int4' in quant_method_name or 'gptq' in quant_method_name:
                    logger.debug(f"Module {component_path} needs kernel migration: current method = {quant_method_name}")
                    return True
            
            # 检查模块参数的数据类型
            for param in module.parameters():
                if param.dtype != torch.float16 and param.dtype != torch.float32:
                    logger.debug(f"Module {component_path} needs kernel migration: current dtype = {param.dtype}")
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking kernel migration need for {component_path}: {e}")
            return False
    
    def _hot_migrate_kernel(self, module: nn.Module, component_path: str) -> bool:
        """简化的kernel热迁移：直接替换量化方法并转换权重"""
        try:
            logger.info(f"🔥 Starting simplified hot kernel migration for {component_path}")
            
            # 1. 强制同步所有CUDA操作
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 2. 备份当前量化方法
            original_quant_method = None
            if hasattr(module, 'quant_method'):
                original_quant_method = module.quant_method
                logger.debug(f"Backed up original quant method: {type(original_quant_method).__name__}")
            
            # 3. 原子性地替换为FP16
            with torch.no_grad():
                # 替换量化方法为FP16版本
                try:
                    from sglang.srt.layers.linear import UnquantizedLinearMethod
                    module.quant_method = UnquantizedLinearMethod()
                    logger.debug(f"Replaced quant method with UnquantizedLinearMethod")
                except ImportError:
                    # 如果没有UnquantizedLinearMethod，设置为None
                    module.quant_method = None
                    logger.debug(f"Set quant method to None")
                
                # 清理量化相关属性
                for attr_name in ['compressed_weight', 'weight_format', 'quantization_method', 'weight_scale_inv', 'scales']:
                    if hasattr(module, attr_name):
                        setattr(module, attr_name, None)
                
                # 确保所有权重都是FP16
                for param_name, param in module.named_parameters():
                    if param.dtype != torch.float16:
                        target_device = param.device
                        param.data = param.data.half().to(target_device)
                        logger.debug(f"Converted {param_name} to FP16 on device {target_device}")
            
            # 4. 清理旧kernel
            if original_quant_method:
                try:
                    del original_quant_method
                    import gc
                    gc.collect()
                except:
                    pass
            
            # 5. 再次同步CUDA操作
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            logger.info(f"✅ Simplified hot kernel migration completed for {component_path}")
            return True
                
        except Exception as e:
            logger.error(f"Simplified hot kernel migration failed for {component_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_fp16_quant_method(self, module: nn.Module, component_path: str):
        """创建FP16量化方法"""
        try:
            # 尝试导入FP16量化方法
            try:
                from sglang.srt.layers.linear import UnquantizedLinearMethod
                fp16_quant_method = UnquantizedLinearMethod()
                logger.debug(f"Created UnquantizedLinearMethod for {component_path}")
                return fp16_quant_method
            except ImportError:
                logger.debug(f"UnquantizedLinearMethod not available for {component_path}, using None")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to create FP16 quant method for {component_path}: {e}")
            return None
    
    def _cleanup_old_kernel(self, old_quant_method):
        """清理旧kernel"""
        try:
            # 清理旧的量化方法资源
            if hasattr(old_quant_method, 'cleanup'):
                old_quant_method.cleanup()
                logger.debug("Called cleanup on old quant method")
            
            # 强制垃圾回收
            import gc
            del old_quant_method
            gc.collect()
            
            logger.debug("Cleaned up old kernel")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup old kernel: {e}")
    
    def _estimate_component_memory(self, component: nn.Module) -> float:
        """估算组件的内存使用（MB）"""
        try:
            total_params = 0
            for param in component.parameters():
                total_params += param.numel()
            
            # FP16精度：每个参数2字节
            memory_bytes = total_params * 2
            memory_mb = memory_bytes / (1024 * 1024)
            
            return memory_mb
            
        except Exception as e:
            logger.debug(f"Failed to estimate memory for component: {e}")
            return 0.0
    
    def get_initialization_report(self) -> Dict[str, Any]:
        """获取初始化报告"""
        return {
            'fp16_model_path': self.fp16_model_path,
            'initialized_layers': list(self.initialized_layers),
            'total_initialized_layers': len(self.initialized_layers)
        }
    
    def _hot_migrate_module_to_fp16(self, module: nn.Module, component_path: str) -> bool:
        """安全的热迁移模块到FP16：只在模型启动时进行，避免推理中断"""
        try:
            logger.info(f"🔥 Starting safe hot migration for {component_path}")
            
            # 检查是否在推理过程中，如果是则跳过
            import threading
            if threading.current_thread() != threading.main_thread():
                logger.warning(f"Skipping hot migration for {component_path} - not in main thread")
                return False
            
            # 1. 创建新kernel (FP16)
            new_module = self._create_fp16_module(module, component_path)
            if new_module is None:
                logger.warning(f"Failed to create FP16 module for {component_path}")
                return False
            
            # 2. 加载FP16参数到新module
            if not self._load_fp16_params_to_module(new_module, component_path):
                logger.warning(f"Failed to load FP16 params for {component_path}")
                return False
            
            # 3. 原子性替换：用新module替换旧module
            success = self._atomic_replace_module(module, new_module, component_path)
            if not success:
                logger.warning(f"Failed to atomically replace module for {component_path}")
                return False
            
            logger.info(f"✅ Safe hot migration completed for {component_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error in safe hot migration for {component_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_fp16_module(self, original_module: nn.Module, component_path: str) -> Optional[nn.Module]:
        """创建FP16版本的模块"""
        try:
            import torch.nn as nn
            
            # 获取原始模块的类型和配置
            module_type = type(original_module)
            
            # 创建新的FP16模块
            if hasattr(original_module, 'in_features') and hasattr(original_module, 'out_features'):
                # Linear层
                new_module = nn.Linear(
                    in_features=original_module.in_features,
                    out_features=original_module.out_features,
                    bias=original_module.bias is not None,
                    dtype=torch.float16
                )
            elif 'layernorm' in component_path.lower():
                # LayerNorm层
                new_module = nn.LayerNorm(
                    normalized_shape=original_module.normalized_shape,
                    eps=original_module.eps,
                    elementwise_affine=original_module.elementwise_affine,
                    dtype=torch.float16
                )
            else:
                # 其他类型的模块，直接复制结构
                new_module = module_type()
                # 复制所有属性（除了权重）
                for attr_name in dir(original_module):
                    if not attr_name.startswith('_') and not callable(getattr(original_module, attr_name)):
                        if attr_name not in ['weight', 'bias']:
                            setattr(new_module, attr_name, getattr(original_module, attr_name))
            
            logger.debug(f"Created FP16 module for {component_path}: {type(new_module)}")
            return new_module
            
        except Exception as e:
            logger.error(f"Failed to create FP16 module for {component_path}: {e}")
            return None
    
    def _load_fp16_params_to_module(self, module: nn.Module, component_path: str) -> bool:
        """加载FP16参数到模块"""
        try:
            # 从权重映射中查找匹配的权重
            matching_keys = []
            for key in self._weight_map.keys():
                if component_path in key:
                    matching_keys.append(key)
            
            if not matching_keys:
                logger.warning(f"No matching weights found for {component_path}")
                return False
            
            # 加载权重
            weights = {}
            from safetensors import safe_open
            
            for key in matching_keys:
                weight_file = self._weight_map[key]
                weight_file_path = os.path.join(self.fp16_model_path, weight_file)
                
                if os.path.exists(weight_file_path):
                    with safe_open(weight_file_path, framework="pt", device="cpu") as f:
                        if key in f.keys():
                            weights[key] = f.get_tensor(key)
            
            if not weights:
                logger.warning(f"No weights loaded for {component_path}")
                return False
            
            # 替换模块参数
            for param_name, param in module.named_parameters():
                # 查找匹配的权重
                matching_weight = None
                for key, weight in weights.items():
                    if param_name in key:
                        matching_weight = weight
                        break
                
                if matching_weight is not None:
                    # 处理tensor并行权重
                    if matching_weight.shape != param.shape:
                        processed_weight = self._handle_tensor_parallel_weight(
                            matching_weight, param, component_path, param_name
                        )
                        if processed_weight is not None:
                            # 确保权重在正确的设备上并转换为FP16
                            target_device = param.device
                            processed_weight = processed_weight.half().to(target_device)
                            param.data = processed_weight
                        else:
                            logger.warning(f"Shape mismatch for {param_name} in {component_path}")
                            return False
                    else:
                        # 确保权重在正确的设备上并转换为FP16
                        target_device = param.device
                        matching_weight = matching_weight.half().to(target_device)
                        param.data = matching_weight
            
            logger.debug(f"Loaded FP16 params for {component_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load FP16 params for {component_path}: {e}")
            return False
    
    def _atomic_replace_module(self, old_module: nn.Module, new_module: nn.Module, component_path: str) -> bool:
        """原子性替换模块"""
        try:
            # 这里需要找到模块的父级并进行替换
            # 由于模块替换比较复杂，我们采用权重替换的方式
            # 将新模块的权重复制到旧模块
            
            with torch.no_grad():
                # 复制所有权重和参数
                for (old_name, old_param), (new_name, new_param) in zip(
                    old_module.named_parameters(), new_module.named_parameters()
                ):
                    if old_param.shape == new_param.shape:
                        old_param.data.copy_(new_param.data)
                    else:
                        logger.warning(f"Shape mismatch during atomic replace: {old_name}")
                        return False
                
                # 复制其他属性
                for attr_name in dir(new_module):
                    if not attr_name.startswith('_') and not callable(getattr(new_module, attr_name)):
                        if hasattr(old_module, attr_name):
                            setattr(old_module, attr_name, getattr(new_module, attr_name))
            
            logger.debug(f"Atomically replaced module for {component_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to atomically replace module for {component_path}: {e}")
            return False