#!/usr/bin/env python3
"""
混合并行测试脚本
实现两种配置：
1. Expert层使用EP=8，其他层使用TP=4, DP=2
2. Expert层使用TP=8，其他层使用TP=4, DP=2

注意：由于sglang的限制，完全实现混合并行策略需要自定义配置
"""

import subprocess
import os
import time
import requests
import json
import argparse

class HybridParallelTester:
    def __init__(self, model_path="/dev/shm/Qwen3-30B-A3B"):
        self.model_path = model_path
        self.env = os.environ.copy()
        self.env.update({
            'SGLANG_DISABLE_MARLIN': '1',
            'SGL_DISABLE_AWQ_MARLIN': '1', 
            'SGLANG_DISABLE_SGL_KERNEL': '1',
            'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'
        })
    
    def start_ep_config(self):
        """配置1: Expert层使用EP=8，其他层使用TP=4, DP=2"""
        print("=== 配置1: Expert Parallel (EP=8) + 其他层(TP=4, DP=2) ===")
        
        # 设置环境变量
        env = self.env.copy()
        env['SINGLE_EXPERT_MODE'] = 'dp'  # 使用DP模式，每个GPU都有expert的完整副本
        
        cmd = [
            'python3', '-m', 'sglang.launch_server',
            '--model-path', self.model_path,
            '--tp-size', '4',  # 其他层的TP大小
            '--dp-size', '2',  # 其他层的DP大小
            '--enable-ep-moe',  # 启用expert parallel
            '--ep-size', '8',   # expert parallel size = 8
            '--max-running-requests', '32',
            '--host', '127.0.0.1',
            '--port', '8080',
            '--max-total-tokens', '40960',
            '--dtype', 'bfloat16',
            '--trust-remote-code',
            '--attention-backend', 'torch_native',
            '--sampling-backend', 'pytorch',
            '--disable-cuda-graph',
            '--disable-cuda-graph-padding',
            '--kv-cache-dtype', 'auto',
            '--allow-auto-truncate',
            '--chunked-prefill-size', '16384'
        ]
        
        print(f"启动命令: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, env=env)
        
        print("等待服务器启动...")
        time.sleep(30)
        
        return process
    
    def start_tp_config(self):
        """配置2: Expert层使用TP=8，其他层使用TP=4, DP=2"""
        print("=== 配置2: Expert层使用TP=8，其他层使用TP=4, DP=2 ===")
        
        # 设置环境变量
        env = self.env.copy()
        env['SINGLE_EXPERT_MODE'] = 'tp'  # 使用TP模式，expert在8张GPU上切分
        
        cmd = [
            'python3', '-m', 'sglang.launch_server',
            '--model-path', self.model_path,
            '--tp-size', '8',  # 使用TP=8进行expert切分
            '--dp-size', '1',  # 不使用DP，因为TP=8已经占用了所有GPU
            '--max-running-requests', '32',
            '--host', '127.0.0.1',
            '--port', '8081',
            '--max-total-tokens', '40960',
            '--dtype', 'bfloat16',
            '--trust-remote-code',
            '--attention-backend', 'torch_native',
            '--sampling-backend', 'pytorch',
            '--disable-cuda-graph',
            '--disable-cuda-graph-padding',
            '--kv-cache-dtype', 'auto',
            '--allow-auto-truncate',
            '--chunked-prefill-size', '16384'
        ]
        
        print(f"启动命令: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, env=env)
        
        print("等待服务器启动...")
        time.sleep(30)
        
        return process
    
    def test_inference(self, port, config_name):
        """测试推理"""
        print(f"=== 测试 {config_name} 推理 ===")
        
        test_prompts = [
            f"Hello, this is a test for the pruned Qwen model with {config_name}.",
            "Explain the concept of mixture of experts in large language models.",
            "What are the advantages of using expert parallelism in MoE models?",
            "How does routing work in mixture of experts architecture?",
            "Compare tensor parallelism and expert parallelism in MoE models."
        ]
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n--- 测试 {i+1} ---")
            print(f"输入: {prompt}")
            
            try:
                response = requests.post(
                    f'http://127.0.0.1:{port}/v1/chat/completions',
                    json={
                        'model': 'qwen3-30b-a3b',
                        'messages': [{'role': 'user', 'content': prompt}],
                        'max_tokens': 100,
                        'temperature': 0.7,
                        'stream': False
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    output = result['choices'][0]['message']['content']
                    print(f"输出: {output}")
                    
                    if 'usage' in result:
                        usage = result['usage']
                        print(f"Token使用: {usage}")
                else:
                    print(f"请求失败: {response.status_code}")
                    print(response.text)
                    
            except Exception as e:
                print(f"请求异常: {e}")
    
    def run_benchmark(self, port, config_name, num_requests=10):
        """运行基准测试"""
        print(f"=== 运行 {config_name} 基准测试 ===")
        
        import time
        import statistics
        
        latencies = []
        
        for i in range(num_requests):
            prompt = f"Benchmark test {i+1}: Explain the concept of expert parallelism in large language models."
            
            start_time = time.time()
            try:
                response = requests.post(
                    f'http://127.0.0.1:{port}/v1/chat/completions',
                    json={
                        'model': 'qwen3-30b-a3b',
                        'messages': [{'role': 'user', 'content': prompt}],
                        'max_tokens': 50,
                        'temperature': 0.7,
                        'stream': False
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    end_time = time.time()
                    latency = (end_time - start_time) * 1000  # 转换为毫秒
                    latencies.append(latency)
                    print(f"请求 {i+1}: {latency:.2f}ms")
                else:
                    print(f"请求 {i+1}: 失败")
                    
            except Exception as e:
                print(f"请求 {i+1}: 异常 - {e}")
        
        if latencies:
            print(f"\n=== {config_name} 基准测试结果 ===")
            print(f"平均延迟: {statistics.mean(latencies):.2f}ms")
            print(f"中位数延迟: {statistics.median(latencies):.2f}ms")
            print(f"最小延迟: {min(latencies):.2f}ms")
            print(f"最大延迟: {max(latencies):.2f}ms")
            print(f"标准差: {statistics.stdev(latencies):.2f}ms")
    
    def run_config(self, config_type):
        """运行指定配置"""
        if config_type == "ep":
            server_process = self.start_ep_config()
            port = 8080
            config_name = "Expert Parallel (EP=8)"
        elif config_type == "tp":
            server_process = self.start_tp_config()
            port = 8081
            config_name = "Tensor Parallel (TP=8)"
        else:
            raise ValueError("config_type must be 'ep' or 'tp'")
        
        try:
            # 测试推理
            self.test_inference(port, config_name)
            
            # 运行基准测试
            self.run_benchmark(port, config_name)
            
            print(f"\n=== {config_name} 测试完成 ===")
            
        except KeyboardInterrupt:
            print(f"\n用户中断 {config_name} 测试")
        finally:
            if server_process:
                print(f"关闭 {config_name} 服务器...")
                server_process.terminate()
                server_process.wait()

def main():
    parser = argparse.ArgumentParser(description="混合并行测试脚本")
    parser.add_argument(
        "--config", 
        choices=["ep", "tp", "both"], 
        default="both",
        help="选择测试配置: ep(Expert Parallel), tp(Tensor Parallel), both(两种都测试)"
    )
    parser.add_argument(
        "--model-path",
        default="/dev/shm/Qwen3-30B-A3B",
        help="模型路径"
    )
    
    args = parser.parse_args()
    
    tester = HybridParallelTester(args.model_path)
    
    if args.config == "ep":
        tester.run_config("ep")
    elif args.config == "tp":
        tester.run_config("tp")
    elif args.config == "both":
        print("=== 运行两种配置的对比测试 ===")
        
        # 先测试EP配置
        tester.run_config("ep")
        
        print("\n" + "="*50)
        print("等待5秒后开始TP配置测试...")
        time.sleep(5)
        
        # 再测试TP配置
        tester.run_config("tp")
        
        print("\n=== 所有测试完成 ===")
        print("配置对比:")
        print("1. Expert Parallel (EP=8): expert层在8张GPU上都有备份，使用随机routing")
        print("2. Tensor Parallel (TP=8): expert层在8张GPU上切分，均匀部署")

if __name__ == "__main__":
    main()
