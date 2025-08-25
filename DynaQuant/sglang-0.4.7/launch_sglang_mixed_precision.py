#!/usr/bin/env python3
"""
支持张量并行(TP=4)和数据并行(DP=2)的混合精度启动脚本
"""

import os
import sys
import argparse
import logging
import torch
import torch.distributed as dist
from pathlib import Path

# 添加SGLang路径
sys.path.insert(0, str(Path(__file__).parent / "python"))

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.distributed.parallel_state import initialize_model_parallel
from sglang.srt.model_parallel import tensor_parallel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_distributed(tp_size: int, dp_size: int, rank: int, world_size: int, 
                    dist_init_addr: str = "127.0.0.1:50000"):
    """初始化分布式环境"""
    logger.info(f"Initializing distributed environment: TP={tp_size}, DP={dp_size}, rank={rank}, world_size={world_size}")
    
    # 设置环境变量
    os.environ["MASTER_ADDR"] = dist_init_addr.split(":")[0]
    os.environ["MASTER_PORT"] = dist_init_addr.split(":")[1]
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank % torch.cuda.device_count())
    
    # 初始化PyTorch分布式
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{dist_init_addr}",
        world_size=world_size,
        rank=rank
    )
    
    # 设置CUDA设备
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    
    logger.info(f"Distributed initialized: rank={rank}, local_rank={local_rank}, device={torch.cuda.current_device()}")
    
    # 初始化模型并行
    initialize_model_parallel(tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=1)
    
    logger.info("Model parallel initialized successfully")


class MixedPrecisionTPDPServer:
    """支持张量并行和数据并行的混合精度服务器"""
    
    def __init__(self, model_path: str, mixed_precision_config_path: str, 
                 tp_size: int = 4, dp_size: int = 2, rank: int = 0, world_size: int = 8,
                 dist_init_addr: str = "127.0.0.1:50000", dtype: str = "auto"):
        """
        初始化混合精度TP/DP服务器
        
        Args:
            model_path: 模型路径
            mixed_precision_config_path: 混合精度配置文件路径
            tp_size: 张量并行大小
            dp_size: 数据并行大小
            rank: 当前进程的rank
            world_size: 总进程数
            dist_init_addr: 分布式初始化地址
            dtype: 数据类型
        """
        self.model_path = model_path
        self.mixed_precision_config_path = mixed_precision_config_path
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.rank = rank
        self.world_size = world_size
        self.dist_init_addr = dist_init_addr
        self.dtype = dtype
        
        # 初始化分布式环境
        init_distributed(tp_size, dp_size, rank, world_size, dist_init_addr)
        
        # SGLang配置
        self.model_config = ModelConfig(
            model_path=model_path,
            mixed_precision_config=mixed_precision_config_path,
            dtype=dtype,
            trust_remote_code=True
        )
        
        self.device_config = DeviceConfig(device=f"cuda:{rank % torch.cuda.device_count()}")
        self.load_config = LoadConfig(load_format=LoadFormat.AUTO)
        
        # 模型加载器
        self.model_loader = DefaultModelLoader(self.load_config)
        
        # 模型实例
        self.model = None
        
        logger.info(f"Mixed precision TP/DP server initialized")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Mixed precision config: {mixed_precision_config_path}")
        logger.info(f"TP size: {tp_size}, DP size: {dp_size}")
        logger.info(f"Rank: {rank}, World size: {world_size}")
        logger.info(f"Device: {self.device_config.device}")
        logger.info(f"Data type: {dtype}")
    
    def load_model(self):
        """加载模型"""
        logger.info("Loading model with mixed precision TP/DP...")
        
        try:
            # 使用SGLang的模型加载器加载模型
            self.model = self.model_loader.load_model(
                model_config=self.model_config,
                device_config=self.device_config
            )
            
            # 应用张量并行
            if self.tp_size > 1:
                logger.info(f"Applying tensor parallelism with TP={self.tp_size}")
                device_mesh = torch.distributed.init_device_mesh(
                    f"cuda:{self.rank % torch.cuda.device_count()}", 
                    (self.tp_size,)
                )
                tensor_parallel(self.model, device_mesh)
                logger.info("Tensor parallelism applied successfully")
            
            logger.info("Model loaded successfully with mixed precision TP/DP")
            return {"status": "success", "model_loaded": True}
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     temperature: float = 0.7):
        """生成文本"""
        try:
            # 这里应该使用SGLang的推理API
            # 由于SGLang的推理需要更复杂的设置，这里提供一个基础实现
            
            # 编码输入
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            
            # 创建注意力掩码
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is None:
                if tokenizer.pad_token_id == tokenizer.eos_token_id:
                    attention_mask = torch.ones_like(input_ids)
                    if tokenizer.pad_token_id is not None:
                        attention_mask[input_ids == tokenizer.pad_token_id] = 0
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
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "generated_text": generated_text,
                "input_length": len(input_ids[0]),
                "output_length": len(outputs[0]),
                "rank": self.rank
            }
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def get_mixed_precision_stats(self):
        """获取混合精度统计"""
        try:
            from sglang.srt.model_loader.mixed_precision_loader import get_global_true_mixed_precision_loader
            from sglang.srt.layers.mixed_precision_linear import get_mixed_precision_memory_stats
            
            mixed_precision_loader = get_global_true_mixed_precision_loader()
            if mixed_precision_loader:
                memory_stats = get_mixed_precision_memory_stats()
                return {
                    "rank": self.rank,
                    "mixed_precision_stats": memory_stats
                }
            else:
                return {
                    "rank": self.rank,
                    "mixed_precision_stats": {"error": "No mixed precision loader available"}
                }
        except Exception as e:
            return {
                "rank": self.rank,
                "mixed_precision_stats": {"error": str(e)}
            }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动混合精度TP/DP服务器")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--mixed-precision-config", type=str, required=True, help="混合精度配置文件路径")
    parser.add_argument("--tp-size", type=int, default=4, help="张量并行大小")
    parser.add_argument("--dp-size", type=int, default=2, help="数据并行大小")
    parser.add_argument("--rank", type=int, default=0, help="当前进程的rank")
    parser.add_argument("--world-size", type=int, default=8, help="总进程数")
    parser.add_argument("--dist-init-addr", type=str, default="127.0.0.1:50000", help="分布式初始化地址")
    parser.add_argument("--dtype", type=str, default="auto", help="数据类型")
    parser.add_argument("--test", action="store_true", help="运行测试")
    
    args = parser.parse_args()
    
    # 验证参数
    if args.tp_size * args.dp_size != args.world_size:
        logger.error(f"TP size ({args.tp_size}) * DP size ({args.dp_size}) must equal world size ({args.world_size})")
        sys.exit(1)
    
    try:
        # 创建服务器
        server = MixedPrecisionTPDPServer(
            model_path=args.model,
            mixed_precision_config_path=args.mixed_precision_config,
            tp_size=args.tp_size,
            dp_size=args.dp_size,
            rank=args.rank,
            world_size=args.world_size,
            dist_init_addr=args.dist_init_addr,
            dtype=args.dtype
        )
        
        # 加载模型
        server.load_model()
        
        # 运行测试
        if args.test:
            logger.info("Running test generation...")
            result = server.generate_text("Hello, how are you?", max_length=50)
            logger.info(f"Test generation result: {result}")
            
            # 获取混合精度统计
            stats = server.get_mixed_precision_stats()
            logger.info(f"Mixed precision stats: {stats}")
        
        logger.info("Server ready for inference")
        
        # 保持服务器运行
        while True:
            try:
                # 这里可以添加HTTP服务器或其他接口
                import time
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down server...")
                break
                
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
    finally:
        # 清理分布式环境
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
