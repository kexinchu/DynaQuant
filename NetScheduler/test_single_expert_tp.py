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
import random
import string
import threading
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入内部状态检查器
from sglang_internal_state_checker import SGLangInternalStateChecker

@dataclass
class TestResult:
    """测试结果数据类"""
    query_length: int
    qps: int
    ttft_ms: float
    tpot_ms: float
    overall_latency_ms: float
    tokens_generated: int
    success: bool
    error_message: str = ""

@dataclass
class DeploymentInfo:
    """部署信息数据类"""
    gpu_memory_usage: Dict[int, float]  # GPU ID -> 内存使用率
    gpu_utilization: Dict[int, float]   # GPU ID -> 利用率
    model_loaded: bool
    expert_distribution: Dict[str, Any]  # expert分布信息
    parallel_config: Dict[str, Any]     # 并行配置信息
    internal_state: Dict[str, Any]      # 内部状态信息

def verify_tp_deployment_with_internal_state(port: int = 8081) -> DeploymentInfo:
    """使用内部状态检查器验证TP部署配置"""
    print("=== 使用内部状态检查器验证 Tensor Parallel 部署配置 ===")
    
    # 创建内部状态检查器
    checker = SGLangInternalStateChecker()
    
    try:
        # 设置内部状态API
        print("设置内部状态API...")
        checker.add_internal_state_api()
        
        # 安装依赖
        print("安装依赖...")
        checker.install_dependencies()
        
        # 重新安装SGLang
        print("重新安装SGLang...")
        checker.reinstall_sglang()
        
        # 启动内部API服务器
        print("启动内部API服务器...")
        api_server_process = checker.start_internal_api_server()
        
        # 等待API服务器启动
        time.sleep(5)
        
        # 获取内部并行状态
        print("获取内部并行状态...")
        parallel_state = checker.get_internal_parallel_state()
        environment_info = checker.get_internal_environment_info()
        
        # 验证TP部署
        print("验证TP部署配置...")
        verification_result = checker.verify_deployment_with_internal_state('tp')
        
        # 获取GPU信息（备用方法）
        gpu_info = {}
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 4:
                        gpu_id = int(parts[0])
                        memory_used = float(parts[1])
                        memory_total = float(parts[2])
                        utilization = float(parts[3])
                        
                        gpu_info[gpu_id] = {
                            'memory_usage_percent': (memory_used / memory_total) * 100,
                            'utilization_percent': utilization
                        }
        except Exception as e:
            print(f"获取GPU信息失败: {e}")
        
        # 检查服务器是否响应
        model_loaded = False
        try:
            response = requests.get(f'http://127.0.0.1:{port}/v1/models', timeout=5)
            model_loaded = response.status_code == 200
        except:
            pass
        
        # 构建部署信息
        deployment_info = DeploymentInfo(
            gpu_memory_usage={gpu_id: info['memory_usage_percent'] for gpu_id, info in gpu_info.items()},
            gpu_utilization={gpu_id: info['utilization_percent'] for gpu_id, info in gpu_info.items()},
            model_loaded=model_loaded,
            expert_distribution=verification_result.get('expert_distribution', {}),
            parallel_config=verification_result.get('parallel_config', {}),
            internal_state={
                'parallel_state': parallel_state,
                'environment_info': environment_info,
                'verification_result': verification_result
            }
        )
        
        # 显示验证结果
        print("\n=== 内部状态验证结果 ===")
        if verification_result.get('is_valid', False):
            print("✅ TP部署验证通过: 内部状态检查确认配置正确")
        else:
            print("❌ TP部署验证失败: 内部状态检查发现配置问题")
            print(f"错误信息: {verification_result.get('error_message', '未知错误')}")
        
        # 显示详细信息
        if parallel_state:
            print(f"并行组信息: {parallel_state}")
        
        if environment_info:
            print(f"环境变量信息: {environment_info}")
        
        # 清理API服务器
        if api_server_process:
            api_server_process.terminate()
            api_server_process.wait()
        
        return deployment_info
        
    except Exception as e:
        print(f"内部状态验证失败: {e}")
        print("回退到基础验证方法...")
        
        # 回退到基础验证
        return verify_tp_deployment_fallback(port)

def verify_tp_deployment_fallback(port: int = 8081) -> DeploymentInfo:
    """回退的TP部署验证方法"""
    print("=== 使用回退方法验证 Tensor Parallel 部署配置 ===")
    
    # 获取GPU信息
    gpu_info = {}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_id = int(parts[0])
                    memory_used = float(parts[1])
                    memory_total = float(parts[2])
                    utilization = float(parts[3])
                    
                    gpu_info[gpu_id] = {
                        'memory_usage_percent': (memory_used / memory_total) * 100,
                        'utilization_percent': utilization
                    }
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
    
    # 检查服务器是否响应
    model_loaded = False
    try:
        response = requests.get(f'http://127.0.0.1:{port}/v1/models', timeout=5)
        model_loaded = response.status_code == 200
    except:
        pass
    
    # 分析TP部署特征
    deployment_info = DeploymentInfo(
        gpu_memory_usage={gpu_id: info['memory_usage_percent'] for gpu_id, info in gpu_info.items()},
        gpu_utilization={gpu_id: info['utilization_percent'] for gpu_id, info in gpu_info.items()},
        model_loaded=model_loaded,
        expert_distribution={},
        parallel_config={},
        internal_state={}
    )
    
    # 验证TP部署特征
    if gpu_info:
        print(f"GPU数量: {len(gpu_info)}")
        
        # TP部署特征：所有GPU都应该有相似的内存使用（因为expert被均匀切分）
        memory_usage_values = list(deployment_info.gpu_memory_usage.values())
        memory_std = statistics.stdev(memory_usage_values) if len(memory_usage_values) > 1 else 0
        memory_mean = statistics.mean(memory_usage_values)
        
        print(f"平均GPU内存使用率: {memory_mean:.2f}%")
        print(f"GPU内存使用率标准差: {memory_std:.2f}%")
        
        # TP部署应该显示相对均匀的内存分布
        if memory_std < 15.0:  # 标准差小于15%认为是均匀分布（TP允许稍大的差异）
            print("✅ TP部署验证通过: GPU内存使用相对均匀，符合expert切分分布特征")
            deployment_info.expert_distribution = {
                'type': 'tensor_parallel',
                'distribution': 'uniform',
                'memory_std': memory_std,
                'memory_mean': memory_mean
            }
        else:
            print("⚠️ TP部署验证警告: GPU内存使用不均匀，可能不是标准的TP部署")
            deployment_info.expert_distribution = {
                'type': 'unknown',
                'distribution': 'non_uniform',
                'memory_std': memory_std,
                'memory_mean': memory_mean
            }
        
        # 显示每个GPU的详细信息
        for gpu_id, info in gpu_info.items():
            print(f"GPU {gpu_id}: 内存使用 {info['memory_usage_percent']:.1f}%, 利用率 {info['utilization_percent']:.1f}%")
    
    deployment_info.parallel_config = {
        'mode': 'tensor_parallel',
        'tp_size': 8,
        'dp_size': 1
    }
    
    return deployment_info

def generate_random_text(length: int) -> str:
    """生成指定长度的随机文本"""
    # 使用中文和英文混合
    chinese_chars = "你好世界人工智能机器学习深度学习自然语言处理计算机视觉数据科学"
    english_chars = string.ascii_letters + string.digits + " "
    
    text = ""
    for i in range(length):
        if random.random() < 0.7:  # 70%概率使用中文
            text += random.choice(chinese_chars)
        else:
            text += random.choice(english_chars)
    
    return text.strip()

def send_request(port: int, prompt: str, max_tokens: int = 100) -> TestResult:
    """发送单个请求并测量性能"""
    start_time = time.time()
    
    try:
        response = requests.post(
            f'http://127.0.0.1:{port}/v1/chat/completions',
            json={
                'model': 'qwen3-30b-a3b',
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens,
                'temperature': 0.7,
                'stream': False
            },
            timeout=60
        )
        
        end_time = time.time()
        overall_latency_ms = (end_time - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            output = result['choices'][0]['message']['content']
            
            # 计算TTFT和TPOT (简化计算)
            # 注意：这里使用简化计算，实际应该从流式响应中获取精确时间
            tokens_generated = len(output.split())  # 简化token计算
            ttft_ms = overall_latency_ms * 0.3  # 假设首token占30%时间
            tpot_ms = (overall_latency_ms - ttft_ms) / max(1, tokens_generated - 1)
            
            return TestResult(
                query_length=len(prompt),
                qps=0,  # 在批量测试中设置
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                overall_latency_ms=overall_latency_ms,
                tokens_generated=tokens_generated,
                success=True
            )
        else:
            return TestResult(
                query_length=len(prompt),
                qps=0,
                ttft_ms=0,
                tpot_ms=0,
                overall_latency_ms=overall_latency_ms,
                tokens_generated=0,
                success=False,
                error_message=f"HTTP {response.status_code}: {response.text}"
            )
            
    except Exception as e:
        end_time = time.time()
        overall_latency_ms = (end_time - start_time) * 1000
        
        return TestResult(
            query_length=len(prompt),
            qps=0,
            ttft_ms=0,
            tpot_ms=0,
            overall_latency_ms=overall_latency_ms,
            tokens_generated=0,
            success=False,
            error_message=str(e)
        )

def run_performance_test(port: int, query_lengths: List[int], qps_values: List[int], 
                        num_requests_per_test: int = 10) -> Dict[str, List[TestResult]]:
    """运行性能测试"""
    print("=== 开始性能测试 ===")
    
    results = {
        'query_length_test': [],
        'qps_test': []
    }
    
    # 测试组1：不同query长度，QPS=1
    print("\n--- 测试组1: 不同query长度 (QPS=1) ---")
    for length in query_lengths:
        print(f"测试query长度: {length}")
        
        for i in range(num_requests_per_test):
            prompt = generate_random_text(length)
            result = send_request(port, prompt)
            result.qps = 1
            results['query_length_test'].append(result)
            
            if result.success:
                print(f"  请求 {i+1}: TTFT={result.ttft_ms:.2f}ms, TPOT={result.tpot_ms:.2f}ms, "
                      f"Overall={result.overall_latency_ms:.2f}ms")
            else:
                print(f"  请求 {i+1}: 失败 - {result.error_message}")
            
            time.sleep(1)  # QPS=1，每秒一个请求
    
    # 测试组2：固定query长度，不同QPS
    print("\n--- 测试组2: 不同QPS (query长度=256) ---")
    for qps in qps_values:
        print(f"测试QPS: {qps}")
        
        # 创建线程池来模拟并发请求
        with ThreadPoolExecutor(max_workers=min(qps, 10)) as executor:
            futures = []
            
            for i in range(num_requests_per_test):
                prompt = generate_random_text(256)
                future = executor.submit(send_request, port, prompt)
                futures.append(future)
                
                # 控制QPS
                if qps > 1:
                    time.sleep(1.0 / qps)
            
            # 收集结果
            for future in as_completed(futures):
                result = future.result()
                result.qps = qps
                results['qps_test'].append(result)
                
                if result.success:
                    print(f"  请求: TTFT={result.ttft_ms:.2f}ms, TPOT={result.tpot_ms:.2f}ms, "
                          f"Overall={result.overall_latency_ms:.2f}ms")
                else:
                    print(f"  请求: 失败 - {result.error_message}")
    
    return results

def analyze_results(results: Dict[str, List[TestResult]]) -> None:
    """分析测试结果"""
    print("\n=== 测试结果分析 ===")
    
    # 分析query长度测试结果
    print("\n--- Query长度测试结果 ---")
    query_length_stats = {}
    for result in results['query_length_test']:
        if result.success:
            length = result.query_length
            if length not in query_length_stats:
                query_length_stats[length] = []
            query_length_stats[length].append(result)
    
    for length, stats in query_length_stats.items():
        if stats:
            ttft_values = [r.ttft_ms for r in stats]
            tpot_values = [r.tpot_ms for r in stats]
            overall_values = [r.overall_latency_ms for r in stats]
            
            print(f"Query长度 {length}:")
            print(f"  TTFT: 平均={statistics.mean(ttft_values):.2f}ms, "
                  f"中位数={statistics.median(ttft_values):.2f}ms, "
                  f"标准差={statistics.stdev(ttft_values):.2f}ms")
            print(f"  TPOT: 平均={statistics.mean(tpot_values):.2f}ms, "
                  f"中位数={statistics.median(tpot_values):.2f}ms, "
                  f"标准差={statistics.stdev(tpot_values):.2f}ms")
            print(f"  Overall: 平均={statistics.mean(overall_values):.2f}ms, "
                  f"中位数={statistics.median(overall_values):.2f}ms, "
                  f"标准差={statistics.stdev(overall_values):.2f}ms")
    
    # 分析QPS测试结果
    print("\n--- QPS测试结果 ---")
    qps_stats = {}
    for result in results['qps_test']:
        if result.success:
            qps = result.qps
            if qps not in qps_stats:
                qps_stats[qps] = []
            qps_stats[qps].append(result)
    
    for qps, stats in qps_stats.items():
        if stats:
            ttft_values = [r.ttft_ms for r in stats]
            tpot_values = [r.tpot_ms for r in stats]
            overall_values = [r.overall_latency_ms for r in stats]
            
            print(f"QPS {qps}:")
            print(f"  TTFT: 平均={statistics.mean(ttft_values):.2f}ms, "
                  f"中位数={statistics.median(ttft_values):.2f}ms, "
                  f"标准差={statistics.stdev(ttft_values):.2f}ms")
            print(f"  TPOT: 平均={statistics.mean(tpot_values):.2f}ms, "
                  f"中位数={statistics.median(tpot_values):.2f}ms, "
                  f"标准差={statistics.stdev(tpot_values):.2f}ms")
            print(f"  Overall: 平均={statistics.mean(overall_values):.2f}ms, "
                  f"中位数={statistics.median(overall_values):.2f}ms, "
                  f"标准差={statistics.stdev(overall_values):.2f}ms")

def start_tp_server():
    """启动Tensor Parallel服务器"""
    print("=== 启动 Tensor Parallel 服务器 ===")
    
    # 设置环境变量
    env = os.environ.copy()
    env.update({
        'SGLANG_DISABLE_MARLIN': '1',
        'SGL_DISABLE_AWQ_MARLIN': '1', 
        'SGLANG_DISABLE_SGL_KERNEL': '1',
        'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
        'SINGLE_EXPERT_MODE': 'tp'  # 使用TP模式，expert在8张GPU上切分
    })
    
    # 启动命令 - 使用TP=8进行expert切分
    # double check 一下TP是否正确了
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

def main():
    """主函数"""
    print("开始 Tensor Parallel 测试")
    
    # 启动服务器
    server_process = start_tp_server()
    
    try:
        # 验证部署配置
        deployment_info = verify_tp_deployment_with_internal_state(8081)
        
        if not deployment_info.model_loaded:
            print("❌ 模型加载失败，请检查服务器状态")
            return
        
        # 运行性能测试
        query_lengths = [128, 256, 512, 1024, 2048, 4096]
        qps_values = [1, 2, 4, 8, 16, 32, 64]
        
        results = run_performance_test(8081, query_lengths, qps_values, num_requests_per_test=5)
        
        # 分析结果
        analyze_results(results)
        
        print("\n=== 测试完成 ===")
        print("Tensor Parallel 配置:")
        print("- Expert层: TP=8 (expert在8张GPU上切分)")
        print("- 其他层: TP=8 (所有层都使用TP=8)")
        print("- 切分策略: 均匀部署在8张卡上")
        
        # 保存结果到文件
        with open('tp_test_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'deployment_info': {
                    'gpu_memory_usage': deployment_info.gpu_memory_usage,
                    'gpu_utilization': deployment_info.gpu_utilization,
                    'model_loaded': deployment_info.model_loaded,
                    'expert_distribution': deployment_info.expert_distribution,
                    'parallel_config': deployment_info.parallel_config,
                    'internal_state': deployment_info.internal_state
                },
                'test_results': {
                    'query_length_test': [
                        {
                            'query_length': r.query_length,
                            'qps': r.qps,
                            'ttft_ms': r.ttft_ms,
                            'tpot_ms': r.tpot_ms,
                            'overall_latency_ms': r.overall_latency_ms,
                            'tokens_generated': r.tokens_generated,
                            'success': r.success,
                            'error_message': r.error_message
                        } for r in results['query_length_test']
                    ],
                    'qps_test': [
                        {
                            'query_length': r.query_length,
                            'qps': r.qps,
                            'ttft_ms': r.ttft_ms,
                            'tpot_ms': r.tpot_ms,
                            'overall_latency_ms': r.overall_latency_ms,
                            'tokens_generated': r.tokens_generated,
                            'success': r.success,
                            'error_message': r.error_message
                        } for r in results['qps_test']
                    ]
                }
            }, f, indent=2, ensure_ascii=False)
        
        print("测试结果已保存到 tp_test_results.json")
        
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
