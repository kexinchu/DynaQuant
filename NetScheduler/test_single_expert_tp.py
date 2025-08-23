#!/usr/bin/env python3
"""
测试方案2: Tensor Parallel (TP) 方式
- experts层使用TP方式，一个expert在8张卡上进行TP=8切分
- 其他层使用TP=4, DP=2的并行方式
"""

import subprocess
import os
import time
import requests
import json

def start_tp_server():
    """启动Tensor Parallel服务器"""
    print("=== 启动 Tensor Parallel 服务器 ===")
    
    # 设置环境变量
    env = os.environ.copy()
    env.update({
        'SGLANG_DISABLE_MARLIN': '1',
        'SGL_DISABLE_AWQ_MARLIN': '1', 
        'SGLANG_DISABLE_SGL_KERNEL': '1',
        'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'
    })
    
    # 启动命令 - 使用TP=8进行expert切分
    cmd = [
        'python3', '-m', 'sglang.launch_server',
        '--model-path', '/dev/shm/Qwen3-30B-A3B',  # 修改为你的模型路径
        '--tp-size', '8',  # 使用TP=8进行expert切分
        '--dp-size', '1',  # 不使用DP，因为TP=8已经占用了所有GPU
        '--max-running-requests', '32',
        '--host', '127.0.0.1',
        '--port', '8081',  # 使用不同端口避免冲突
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
    
    # 启动服务器进程
    process = subprocess.Popen(cmd, env=env)
    
    # 等待服务器启动
    print("等待服务器启动...")
    time.sleep(30)
    
    return process

def test_tp_inference():
    """测试TP推理"""
    print("=== 测试 Tensor Parallel 推理 ===")
    
    # 测试请求
    test_prompts = [
        "Hello, this is a test for the pruned Qwen model with Tensor Parallel.",
        "Explain the concept of tensor parallelism in large language models.",
        "What are the advantages of using tensor parallelism in MoE models?",
        "How does weight sharding work in tensor parallel architecture?"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- 测试 {i+1} ---")
        print(f"输入: {prompt}")
        
        # 发送请求到服务器
        try:
            response = requests.post(
                'http://127.0.0.1:8081/v1/chat/completions',
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
                
                # 显示性能指标
                if 'usage' in result:
                    usage = result['usage']
                    print(f"Token使用: {usage}")
            else:
                print(f"请求失败: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"请求异常: {e}")

def main():
    """主函数"""
    print("开始 Tensor Parallel 测试")
    
    # 启动服务器
    server_process = start_tp_server()
    
    try:
        # 测试推理
        test_tp_inference()
        
        print("\n=== 测试完成 ===")
        print("Tensor Parallel 配置:")
        print("- Expert层: TP=8 (expert在8张GPU上切分)")
        print("- 其他层: TP=8 (所有层都使用TP=8)")
        print("- 切分策略: 均匀部署在8张卡上")
        
    except KeyboardInterrupt:
        print("\n用户中断测试")
    finally:
        # 清理服务器进程
        if server_process:
            print("关闭服务器...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    main()
