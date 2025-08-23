#!/usr/bin/env python3
"""
设备问题修复脚本
解决CUDA和CPU设备不匹配的问题
"""

import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def ensure_model_on_device(model: torch.nn.Module, device: str = "cuda") -> torch.nn.Module:
    """确保模型在指定设备上"""
    try:
        # 检查模型当前设备
        current_device = next(model.parameters()).device
        logger.info(f"Model current device: {current_device}")
        
        if str(current_device) != device:
            logger.info(f"Moving model from {current_device} to {device}")
            model = model.to(device)
        
        # 验证所有参数都在正确设备上
        for name, param in model.named_parameters():
            if param.device != torch.device(device):
                logger.warning(f"Parameter {name} is on {param.device}, moving to {device}")
                param.data = param.data.to(device)
        
        return model
        
    except Exception as e:
        logger.error(f"Error ensuring model on device: {e}")
        return model


def fix_tokenizer_device_issues(tokenizer, device: str = "cuda"):
    """修复tokenizer的设备问题"""
    try:
        # 设置pad_token_id
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                logger.info("Set pad_token_id to eos_token_id")
            else:
                # 使用unk_token_id作为pad_token_id
                tokenizer.pad_token_id = tokenizer.unk_token_id
                logger.info("Set pad_token_id to unk_token_id")
        
        # 确保tokenizer有正确的特殊token
        if tokenizer.eos_token is None:
            tokenizer.eos_token = tokenizer.pad_token
            logger.info("Set eos_token to pad_token")
        
        return tokenizer
        
    except Exception as e:
        logger.error(f"Error fixing tokenizer device issues: {e}")
        return tokenizer


def create_proper_attention_mask(input_ids: torch.Tensor, tokenizer, device: str = "cuda") -> torch.Tensor:
    """创建正确的注意力掩码"""
    try:
        # 创建基础注意力掩码
        attention_mask = torch.ones_like(input_ids)
        
        # 如果pad_token_id和eos_token_id相同，需要特殊处理
        if tokenizer.pad_token_id == tokenizer.eos_token_id and tokenizer.pad_token_id is not None:
            # 找到实际的padding位置
            # 假设序列从右到左填充
            for i in range(input_ids.shape[0]):
                seq = input_ids[i]
                # 找到第一个非pad位置
                non_pad_positions = (seq != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                if len(non_pad_positions) > 0:
                    last_non_pad = non_pad_positions[-1]
                    # 将padding位置标记为0
                    attention_mask[i, last_non_pad+1:] = 0
        
        attention_mask = attention_mask.to(device)
        return attention_mask
        
    except Exception as e:
        logger.error(f"Error creating attention mask: {e}")
        # 返回简单的全1掩码
        return torch.ones_like(input_ids).to(device)


def fix_moe_device_issues(model: torch.nn.Module, device: str = "cuda"):
    """修复MoE模块的设备问题"""
    try:
        # 查找所有MoE模块
        moe_modules = []
        for name, module in model.named_modules():
            if hasattr(module, 'experts') or 'moe' in name.lower() or 'expert' in name.lower():
                moe_modules.append((name, module))
        
        logger.info(f"Found {len(moe_modules)} potential MoE modules")
        
        for name, module in moe_modules:
            # 确保模块在正确设备上
            if hasattr(module, 'experts'):
                for i, expert in enumerate(module.experts):
                    if hasattr(expert, 'weight') and expert.weight.device != torch.device(device):
                        logger.info(f"Moving expert {i} in {name} to {device}")
                        expert.to(device)
            
            # 确保模块本身在正确设备上
            if next(module.parameters()).device != torch.device(device):
                logger.info(f"Moving MoE module {name} to {device}")
                module.to(device)
        
        return model
        
    except Exception as e:
        logger.error(f"Error fixing MoE device issues: {e}")
        return model


def validate_model_device_consistency(model: torch.nn.Module, device: str = "cuda") -> Dict[str, Any]:
    """验证模型设备一致性"""
    issues = []
    device_count = {}
    
    try:
        for name, param in model.named_parameters():
            param_device = str(param.device)
            if param_device not in device_count:
                device_count[param_device] = []
            device_count[param_device].append(name)
            
            if param_device != device:
                issues.append(f"Parameter {name} on {param_device}, expected {device}")
        
        # 检查缓冲区
        for name, buffer in model.named_buffers():
            buffer_device = str(buffer.device)
            if buffer_device not in device_count:
                device_count[buffer_device] = []
            device_count[buffer_device].append(f"buffer_{name}")
            
            if buffer_device != device:
                issues.append(f"Buffer {name} on {buffer_device}, expected {device}")
        
        return {
            "device_count": device_count,
            "issues": issues,
            "is_consistent": len(issues) == 0,
            "target_device": device
        }
        
    except Exception as e:
        logger.error(f"Error validating model device consistency: {e}")
        return {
            "error": str(e),
            "is_consistent": False
        }


def comprehensive_device_fix(model: torch.nn.Module, tokenizer, device: str = "cuda") -> Dict[str, Any]:
    """综合设备修复"""
    results = {
        "model_fixed": False,
        "tokenizer_fixed": False,
        "validation": {},
        "issues_found": []
    }
    
    try:
        # 1. 修复模型设备问题
        logger.info("Fixing model device issues...")
        model = ensure_model_on_device(model, device)
        model = fix_moe_device_issues(model, device)
        results["model_fixed"] = True
        
        # 2. 修复tokenizer问题
        logger.info("Fixing tokenizer issues...")
        tokenizer = fix_tokenizer_device_issues(tokenizer, device)
        results["tokenizer_fixed"] = True
        
        # 3. 验证修复结果
        logger.info("Validating device consistency...")
        validation = validate_model_device_consistency(model, device)
        results["validation"] = validation
        
        if not validation["is_consistent"]:
            results["issues_found"] = validation["issues"]
            logger.warning(f"Found {len(validation['issues'])} device consistency issues")
        else:
            logger.info("All device consistency issues resolved")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in comprehensive device fix: {e}")
        results["error"] = str(e)
        return results


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    print("设备问题修复工具")
    print("=" * 50)
    
    # 这里可以添加测试代码
    print("修复工具已准备就绪")
    print("使用方法:")
    print("1. comprehensive_device_fix(model, tokenizer, 'cuda')")
    print("2. ensure_model_on_device(model, 'cuda')")
    print("3. fix_tokenizer_device_issues(tokenizer, 'cuda')")
    print("4. validate_model_device_consistency(model, 'cuda')")
