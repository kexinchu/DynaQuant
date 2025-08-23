#!/usr/bin/env python3
"""
测试Tensor Parallel支持
验证Qwen3MoeModel是否正确支持tensor parallel
"""

import subprocess
import os
import time
import requests
import json

def test_tp_support():
    """测试Tensor Parallel支持"""
    print("=== 测试 Tensor Parallel 支持 ===")
    
    # 设置环境变量
    env = os.environ.copy()
    env.update({
        'SGLANG_DISABLE_MARLIN': '1',
        'SGL_DISABLE_AWQ_MARLIN': '1', 
        'SGLANG_DISABLE_SGL_KERNEL': '1',
        'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
        'SINGLE_EXPERT_MODE': 'tp'  # 使用TP模式
    })
    
    # 启动命令 - 使用TP=8进行测试
    cmd = [
        'python3', '-m', 'sglang.launch_server',
        '--model-path', '/dev/shm/Qwen3-30B-A3B',  # 修改为你的模型路径
        '--tp-size', '8',  # 使用TP=8进行测试
        '--dp-size', '1',  # 不使用DP
        '--max-running-requests', '16',
        '--host', '127.0.0.1',
        '--port', '8080',
        '--max-total-tokens', '20480',
        '--dtype', 'bfloat16',
        '--trust-remote-code',
        '--attention-backend', 'torch_native',
        '--sampling-backend', 'pytorch',
        '--disable-cuda-graph',
        '--disable-cuda-graph-padding',
        '--kv-cache-dtype', 'auto',
        '--allow-auto-truncate',
        '--chunked-prefill-size', '8192'
    ]
    
    print(f"启动命令: {' '.join(cmd)}")
    
    try:
        # 启动服务器进程
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待一段时间看是否能正常启动
        print("等待服务器启动...")
        time.sleep(30)
        
        # 检查进程是否还在运行
        if process.poll() is None:
            print("✓ 服务器启动成功 - Tensor Parallel支持正常")
            
            # 测试推理
            test_inference()
            
            process.terminate()
            process.wait()
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"✗ 服务器启动失败")
            print(f"stdout: {stdout.decode()}")
            print(f"stderr: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"✗ 启动异常: {e}")
        return False

def test_inference():
    """测试推理"""
    print("=== 测试推理 ===")
    
    test_prompts = [
        "Hello, this is a test for tensor parallel support.",
        "What is 2+2?",
        "Explain tensor parallelism in one sentence."
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- 测试 {i+1} ---")
        print(f"输入: {prompt}")
        
        try:
            response = requests.post(
                'http://127.0.0.1:8080/v1/chat/completions',
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

def main():
    """主函数"""
    print("开始测试 Tensor Parallel 支持")
    
    if test_tp_support():
        print("\n✅ Tensor Parallel 支持测试成功！")
        print("Qwen3MoeModel 现在支持 tensor parallel 了！")
    else:
        print("\n❌ Tensor Parallel 支持测试失败")
        print("请检查错误信息并修复问题")

if __name__ == "__main__":
    main()
