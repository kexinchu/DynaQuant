#!/usr/bin/env python3
"""
测试方案1: Expert Parallel (EP) 方式
- experts层使用EP方式，一个expert在8张GPU上都创建备份
- 使用随机routing的方式承接流量
- 其他层使用TP=4, DP=2的并行方式
"""

import subprocess
import os
import time
import requests
import json

def start_ep_server():
    """启动Expert Parallel服务器"""
    print("=== 启动 Expert Parallel 服务器 ===")
    
    # 设置环境变量
    env = os.environ.copy()
    env.update({
        'SGLANG_DISABLE_MARLIN': '1',
        'SGL_DISABLE_AWQ_MARLIN': '1', 
        'SGLANG_DISABLE_SGL_KERNEL': '1',
        'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
        'SINGLE_EXPERT_MODE': 'dp'  # 使用DP模式，每个GPU都有expert的完整副本
    })
    
    # 启动命令
    cmd = [
        'python3', '-m', 'sglang.launch_server',
        '--model-path', '/dev/shm/Qwen3-30B-A3B',  # 修改为你的模型路径
        '--tp-size', '8',  # 其他层使用TP=4
        '--dp-size', '1',  # 其他层使用DP=2
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
    
    # 启动服务器进程
    process = subprocess.Popen(cmd, env=env)
    
    # 等待服务器启动
    print("等待服务器启动...")
    time.sleep(30)
    
    return process

def test_ep_inference():
    """测试EP推理"""
    print("=== 测试 Expert Parallel 推理 ===")
    
    # 测试请求
    test_prompts = [
        "Hello, this is a test for the pruned Qwen model with Expert Parallel.",
        "Explain the concept of mixture of experts in large language models.",
        "What are the advantages of using expert parallelism in MoE models?",
        "How does routing work in mixture of experts architecture?"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- 测试 {i+1} ---")
        print(f"输入: {prompt}")
        
        # 发送请求到服务器
        try:
            response = requests.post(
                'http://127.0.0.1:8080/v1/chat/completions',
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
    print("开始 Expert Parallel 测试")
    
    # 启动服务器
    server_process = start_ep_server()
    
    try:
        # 测试推理
        test_ep_inference()
        
        print("\n=== 测试完成 ===")
        print("Expert Parallel 配置:")
        print("- Expert层: EP=8 (每个expert在8张GPU上都有备份)")
        print("- 其他层: TP=8, DP=1")
        print("- 路由策略: 随机routing")
        
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
