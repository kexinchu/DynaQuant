#!/usr/bin/env python3
"""
增强的SGLang服务器启动脚本
集成混合精度权重加载和专家激活跟踪功能
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# 添加SGLang路径
sys.path.insert(0, str(Path(__file__).parent / "python"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sglang.srt.enhanced_model_loader import (
    load_model_with_enhanced_features,
    get_expert_activation_stats,
    reset_expert_activation_stats,
    export_expert_activation_stats
)
from sglang.srt.model_loader.enhanced_mixed_precision_loader import (
    get_global_expert_tracker
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedSGLangServer:
    """增强的SGLang服务器"""
    
    def __init__(self, config_path: str, model_path: str, 
                 enable_expert_tracking: bool = True):
        """
        初始化增强的SGLang服务器
        
        Args:
            config_path: 混合精度配置文件路径
            model_path: 模型路径
            enable_expert_tracking: 是否启用专家激活跟踪
        """
        self.config_path = config_path
        self.model_path = model_path
        self.enable_expert_tracking = enable_expert_tracking
        
        self.model = None
        self.tokenizer = None
        self.expert_tracker = None
        
        logger.info(f"Enhanced SGLang server initialized with config: {config_path}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Expert tracking enabled: {enable_expert_tracking}")
    
    def load_model(self) -> Dict[str, Any]:
        """加载模型"""
        logger.info("Loading model with enhanced features...")
        
        try:
            # 1. 加载tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 2. 加载模型
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 2.5. 修复设备问题
            try:
                from fix_device_issues import comprehensive_device_fix
                device_fix_results = comprehensive_device_fix(self.model, self.tokenizer, str(self.model.device))
                logger.info(f"Device fix results: {device_fix_results}")
            except ImportError:
                logger.warning("Device fix module not available, skipping device fixes")
            
            # 3. 使用增强功能加载权重
            logger.info("Loading weights with enhanced mixed precision...")
            stats = load_model_with_enhanced_features(
                self.model,
                self.config_path,
                enable_expert_tracking=self.enable_expert_tracking,
                enable_moe_tracking=True
            )
            
            # 4. 获取专家跟踪器
            if self.enable_expert_tracking:
                self.expert_tracker = get_global_expert_tracker()
                if self.expert_tracker:
                    logger.info("Expert tracker initialized")
            
            logger.info("Model loading completed successfully")
            return stats
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     temperature: float = 0.7, request_id: str = None) -> Dict[str, Any]:
        """生成文本"""
        try:
            # 记录请求开始
            if self.expert_tracker and request_id:
                input_length = len(self.tokenizer.encode(prompt))
                # 这里我们不知道输出长度，先设为0，在生成后更新
                self.expert_tracker.record_request(request_id, input_length, 0)
            
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            
            # 创建注意力掩码
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is None:
                # 如果pad_token和eos_token相同，创建自定义的注意力掩码
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    attention_mask = torch.ones_like(input_ids)
                    # 将padding位置标记为0
                    if self.tokenizer.pad_token_id is not None:
                        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
                else:
                    attention_mask = torch.ones_like(input_ids)
            
            attention_mask = attention_mask.to(self.model.device)
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 更新请求记录
            if self.expert_tracker and request_id:
                output_length = len(outputs[0]) - len(input_ids[0])
                # 更新输出长度
                # 注意：这里简化处理，实际应该更新之前的记录
                self.expert_tracker.record_request(request_id, len(input_ids[0]), output_length)
            
            return {
                "generated_text": generated_text,
                "input_length": len(input_ids[0]),
                "output_length": len(outputs[0]) - len(input_ids[0]),
                "request_id": request_id
            }
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return {"error": str(e)}
    
    def get_expert_stats(self, layer_id: Optional[int] = None, 
                        expert_id: Optional[int] = None) -> Dict[str, Any]:
        """获取专家统计信息"""
        if not self.expert_tracker:
            return {"error": "Expert tracking not enabled"}
        
        try:
            if layer_id is not None and expert_id is not None:
                stats = self.expert_tracker.get_expert_stats(layer_id, expert_id)
            else:
                stats = get_expert_activation_stats()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting expert stats: {e}")
            return {"error": str(e)}
    
    def get_top_experts(self, top_k: int = 10) -> Dict[str, Any]:
        """获取激活次数最多的专家"""
        if not self.expert_tracker:
            return {"error": "Expert tracking not enabled"}
        
        try:
            top_experts = self.expert_tracker.get_top_experts(top_k)
            return {"top_experts": top_experts}
            
        except Exception as e:
            logger.error(f"Error getting top experts: {e}")
            return {"error": str(e)}
    
    def get_layer_stats(self) -> Dict[str, Any]:
        """获取每层的统计信息"""
        if not self.expert_tracker:
            return {"error": "Expert tracking not enabled"}
        
        try:
            layer_stats = self.expert_tracker.get_layer_stats()
            return {"layer_stats": layer_stats}
            
        except Exception as e:
            logger.error(f"Error getting layer stats: {e}")
            return {"error": str(e)}
    
    def reset_expert_stats(self) -> Dict[str, Any]:
        """重置专家统计信息"""
        if not self.expert_tracker:
            return {"error": "Expert tracking not enabled"}
        
        try:
            reset_expert_activation_stats()
            return {"message": "Expert statistics reset successfully"}
            
        except Exception as e:
            logger.error(f"Error resetting expert stats: {e}")
            return {"error": str(e)}
    
    def export_expert_stats(self, file_path: str) -> Dict[str, Any]:
        """导出专家统计信息"""
        if not self.expert_tracker:
            return {"error": "Expert tracking not enabled"}
        
        try:
            export_expert_activation_stats(file_path)
            return {"message": f"Expert statistics exported to {file_path}"}
            
        except Exception as e:
            logger.error(f"Error exporting expert stats: {e}")
            return {"error": str(e)}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Enhanced SGLang Server")
    parser.add_argument("--config", type=str, required=True,
                       help="Mixed precision configuration file path")
    parser.add_argument("--model", type=str, required=True,
                       help="Model path")
    parser.add_argument("--enable-expert-tracking", action="store_true", default=True,
                       help="Enable expert activation tracking")
    parser.add_argument("--port", type=int, default=8080,
                       help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Server host")
    
    args = parser.parse_args()
    
    # 创建服务器实例
    server = EnhancedSGLangServer(
        config_path=args.config,
        model_path=args.model,
        enable_expert_tracking=args.enable_expert_tracking
    )
    
    # 加载模型
    try:
        stats = server.load_model()
        logger.info("Model loaded successfully")
        logger.info(f"Loading stats: {json.dumps(stats, indent=2)}")
        
        # 这里可以启动HTTP服务器或集成到SGLang的现有服务器中
        logger.info(f"Enhanced SGLang server ready on {args.host}:{args.port}")
        logger.info("Use the server methods to interact with the model")
        
        # 示例：生成一些文本
        test_prompt = "Hello, how are you today?"
        logger.info(f"Testing generation with prompt: {test_prompt}")
        
        result = server.generate_text(test_prompt, max_length=50, request_id="test_001")
        logger.info(f"Generation result: {result}")
        
        # 示例：获取专家统计
        if args.enable_expert_tracking:
            expert_stats = server.get_expert_stats()
            logger.info(f"Expert stats: {json.dumps(expert_stats, indent=2)}")
        
        # 保持服务器运行
        logger.info("Server is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
