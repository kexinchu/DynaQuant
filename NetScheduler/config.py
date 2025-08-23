#!/usr/bin/env python3
"""
配置文件
集中管理所有测试参数和配置
"""

import os
from typing import Dict, Any

class TestConfig:
    """测试配置类"""
    
    def __init__(self):
        # 基础配置
        self.model_path = "/dev/shm/Qwen3-30B-A3B"  # 模型路径
        self.sglang_path = "sglang-0.4.7"  # sglang路径
        
        # GPU配置
        self.gpu_devices = "0,1,2,3,4,5,6,7"  # 使用的GPU设备
        self.num_gpus = 8  # GPU数量
        
        # 环境变量
        self.env_vars = {
            'SGLANG_DISABLE_MARLIN': '1',
            'SGL_DISABLE_AWQ_MARLIN': '1',
            'SGLANG_DISABLE_SGL_KERNEL': '1',
            'CUDA_VISIBLE_DEVICES': self.gpu_devices
        }
        
        # 服务器配置
        self.server_config = {
            'max_running_requests': 32,
            'max_total_tokens': 40960,
            'dtype': 'bfloat16',
            'trust_remote_code': True,
            'attention_backend': 'torch_native',
            'sampling_backend': 'pytorch',
            'disable_cuda_graph': True,
            'disable_cuda_graph_padding': True,
            'kv_cache_dtype': 'auto',
            'allow_auto_truncate': True,
            'chunked_prefill_size': 16384
        }
        
        # Expert Parallel 配置
        self.ep_config = {
            'tp_size': 4,  # 其他层的TP大小
            'dp_size': 2,  # 其他层的DP大小
            'enable_ep_moe': True,
            'ep_size': 8,  # expert parallel size
            'host': '127.0.0.1',
            'port': 8080
        }
        
        # Tensor Parallel 配置
        self.tp_config = {
            'tp_size': 8,  # 使用TP=8进行expert切分
            'dp_size': 1,  # 不使用DP
            'enable_ep_moe': False,
            'host': '127.0.0.1',
            'port': 8081
        }
        
        # 测试配置
        self.test_config = {
            'max_tokens': 100,
            'temperature': 0.7,
            'timeout': 60,
            'num_benchmark_requests': 10
        }
        
        # 测试prompts
        self.test_prompts = [
            "Hello, this is a test for the pruned Qwen model with Expert Parallel.",
            "Explain the concept of mixture of experts in large language models.",
            "What are the advantages of using expert parallelism in MoE models?",
            "How does routing work in mixture of experts architecture?",
            "Compare tensor parallelism and expert parallelism in MoE models."
        ]
        
        # 基准测试配置
        self.benchmark_config = {
            'sequence_length': 128,
            'qps': 4.0,
            'duration': 10.0,
            'max_devices': 8
        }
    
    def get_ep_server_args(self) -> Dict[str, Any]:
        """获取Expert Parallel服务器参数"""
        args = self.server_config.copy()
        args.update(self.ep_config)
        return args
    
    def get_tp_server_args(self) -> Dict[str, Any]:
        """获取Tensor Parallel服务器参数"""
        args = self.server_config.copy()
        args.update(self.tp_config)
        return args
    
    def get_env_vars(self) -> Dict[str, str]:
        """获取环境变量"""
        return self.env_vars.copy()
    
    def validate_config(self) -> bool:
        """验证配置"""
        # 检查模型路径
        if not os.path.exists(self.model_path):
            print(f"错误: 模型路径不存在: {self.model_path}")
            return False
        
        # 检查sglang路径
        if not os.path.exists(self.sglang_path):
            print(f"警告: sglang路径不存在: {self.sglang_path}")
        
        # 检查GPU数量
        if self.num_gpus < 8:
            print(f"警告: GPU数量少于8个 ({self.num_gpus})")
        
        return True
    
    def print_config(self):
        """打印配置信息"""
        print("=== 测试配置信息 ===")
        print(f"模型路径: {self.model_path}")
        print(f"sglang路径: {self.sglang_path}")
        print(f"GPU设备: {self.gpu_devices}")
        print(f"GPU数量: {self.num_gpus}")
        
        print("\n=== Expert Parallel 配置 ===")
        for key, value in self.ep_config.items():
            print(f"  {key}: {value}")
        
        print("\n=== Tensor Parallel 配置 ===")
        for key, value in self.tp_config.items():
            print(f"  {key}: {value}")
        
        print("\n=== 服务器配置 ===")
        for key, value in self.server_config.items():
            print(f"  {key}: {value}")
        
        print("\n=== 测试配置 ===")
        for key, value in self.test_config.items():
            print(f"  {key}: {value}")

# 全局配置实例
config = TestConfig()

# 配置预设
class ConfigPresets:
    """配置预设"""
    
    @staticmethod
    def high_memory():
        """高内存配置"""
        config.server_config['max_total_tokens'] = 81920
        config.server_config['chunked_prefill_size'] = 32768
        return config
    
    @staticmethod
    def low_memory():
        """低内存配置"""
        config.server_config['max_total_tokens'] = 20480
        config.server_config['chunked_prefill_size'] = 8192
        return config
    
    @staticmethod
    def high_throughput():
        """高吞吐量配置"""
        config.server_config['max_running_requests'] = 64
        config.test_config['max_tokens'] = 50
        return config
    
    @staticmethod
    def low_latency():
        """低延迟配置"""
        config.server_config['max_running_requests'] = 16
        config.test_config['max_tokens'] = 200
        return config

def load_config_from_env():
    """从环境变量加载配置"""
    # 模型路径
    if 'MODEL_PATH' in os.environ:
        config.model_path = os.environ['MODEL_PATH']
    
    # GPU设备
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        config.gpu_devices = os.environ['CUDA_VISIBLE_DEVICES']
        config.num_gpus = len(config.gpu_devices.split(','))
    
    # 端口
    if 'EP_PORT' in os.environ:
        config.ep_config['port'] = int(os.environ['EP_PORT'])
    
    if 'TP_PORT' in os.environ:
        config.tp_config['port'] = int(os.environ['TP_PORT'])

def save_config_to_file(filename: str = "test_config.json"):
    """保存配置到文件"""
    import json
    
    config_dict = {
        'model_path': config.model_path,
        'gpu_devices': config.gpu_devices,
        'num_gpus': config.num_gpus,
        'ep_config': config.ep_config,
        'tp_config': config.tp_config,
        'server_config': config.server_config,
        'test_config': config.test_config,
        'benchmark_config': config.benchmark_config
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"配置已保存到: {filename}")

def load_config_from_file(filename: str = "test_config.json"):
    """从文件加载配置"""
    import json
    
    if not os.path.exists(filename):
        print(f"配置文件不存在: {filename}")
        return False
    
    with open(filename, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    # 更新配置
    config.model_path = config_dict.get('model_path', config.model_path)
    config.gpu_devices = config_dict.get('gpu_devices', config.gpu_devices)
    config.num_gpus = config_dict.get('num_gpus', config.num_gpus)
    config.ep_config.update(config_dict.get('ep_config', {}))
    config.tp_config.update(config_dict.get('tp_config', {}))
    config.server_config.update(config_dict.get('server_config', {}))
    config.test_config.update(config_dict.get('test_config', {}))
    config.benchmark_config.update(config_dict.get('benchmark_config', {}))
    
    print(f"配置已从文件加载: {filename}")
    return True

if __name__ == "__main__":
    # 加载环境变量配置
    load_config_from_env()
    
    # 验证配置
    if config.validate_config():
        print("配置验证通过")
        config.print_config()
        
        # 保存配置到文件
        save_config_to_file()
    else:
        print("配置验证失败")
        exit(1)
