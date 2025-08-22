#!/usr/bin/env python3
"""
SGLang混合精度功能测试脚本
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

# 添加sglang路径
sys.path.insert(0, str(Path(__file__).parent / "python"))

import sglang as sgl


def test_health_check(server_url: str):
    """测试健康检查"""
    print("=" * 50)
    print("测试健康检查")
    print("=" * 50)
    
    try:
        response = requests.get(f"{server_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"服务器状态: {data.get('status', 'unknown')}")
            print(f"模型已加载: {data.get('model_loaded', False)}")
            print(f"设备: {data.get('device', 'unknown')}")
            return True
        else:
            print(f"健康检查失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"健康检查异常: {e}")
        return False


def test_sglang_api(server_url: str):
    """测试SGLang API"""
    print("\n" + "=" * 50)
    print("测试SGLang API")
    print("=" * 50)
    
    try:
        # 设置后端
        sgl.set_default_backend(server_url)
        
        # 创建提示
        prompt = "请介绍一下人工智能的发展历史："
        
        # 生成文本
        start_time = time.time()
        response = sgl.generate(
            prompt, 
            max_new_tokens=100, 
            temperature=0.7,
            top_p=0.9
        )
        generation_time = time.time() - start_time
        
        print(f"输入提示: {prompt}")
        print(f"生成文本: {response.text}")
        print(f"生成时间: {generation_time:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"SGLang API测试失败: {e}")
        return False


def test_batch_generation(server_url: str):
    """测试批量生成"""
    print("\n" + "=" * 50)
    print("测试批量生成")
    print("=" * 50)
    
    try:
        # 设置后端
        sgl.set_default_backend(server_url)
        
        # 批量提示
        prompts = [
            "什么是机器学习？",
            "深度学习与传统机器学习有什么区别？",
            "请解释一下神经网络的工作原理。"
        ]
        
        # 批量生成
        start_time = time.time()
        responses = []
        for prompt in prompts:
            response = sgl.generate(
                prompt, 
                max_new_tokens=50, 
                temperature=0.8
            )
            responses.append(response.text)
        batch_time = time.time() - start_time
        
        print(f"批量生成时间: {batch_time:.2f}秒")
        print("\n生成结果:")
        for i, (prompt, text) in enumerate(zip(prompts, responses)):
            print(f"\n{i+1}. 输入: {prompt}")
            print(f"   输出: {text}")
        
        return True
        
    except Exception as e:
        print(f"批量生成测试失败: {e}")
        return False


def test_mixed_precision_info(server_url: str):
    """测试混合精度信息查询"""
    print("\n" + "=" * 50)
    print("测试混合精度信息查询")
    print("=" * 50)
    
    try:
        # 尝试获取模型信息
        response = requests.get(f"{server_url}/model_info")
        if response.status_code == 200:
            data = response.json()
            print("模型信息:")
            print(f"  模型名称: {data.get('model_name', 'unknown')}")
            print(f"  设备: {data.get('device', 'unknown')}")
            print(f"  数据类型: {data.get('dtype', 'unknown')}")
            print(f"  最大序列长度: {data.get('max_seq_length', 'unknown')}")
            print(f"  最大批处理大小: {data.get('max_batch_size', 'unknown')}")
            
            # 检查是否有混合精度信息
            weight_info = data.get('weight_info', {})
            if weight_info:
                print("\n权重信息:")
                print(f"  精度路径: {weight_info.get('precision_paths', {})}")
                print(f"  权重映射数量: {len(weight_info.get('weight_mapping', {}))}")
                print(f"  缓存文件数量: {len(weight_info.get('cached_files', []))}")
            
            return True
        else:
            print(f"模型信息查询失败: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"模型信息查询异常: {e}")
        return False


def test_performance_comparison(server_url: str):
    """测试性能对比"""
    print("\n" + "=" * 50)
    print("测试性能对比")
    print("=" * 50)
    
    try:
        # 设置后端
        sgl.set_default_backend(server_url)
        
        # 测试提示
        test_prompts = [
            "人工智能是什么？",
            "机器学习的基本原理是什么？",
            "深度学习与传统机器学习有什么区别？",
            "神经网络是如何工作的？",
            "什么是强化学习？"
        ]
        
        # 测试不同长度的生成
        test_configs = [
            {"max_new_tokens": 50, "name": "短文本生成"},
            {"max_new_tokens": 100, "name": "中等文本生成"},
            {"max_new_tokens": 200, "name": "长文本生成"}
        ]
        
        for config in test_configs:
            print(f"\n{config['name']} (max_new_tokens={config['max_new_tokens']}):")
            
            total_time = 0
            total_tokens = 0
            
            for i, prompt in enumerate(test_prompts):
                start_time = time.time()
                response = sgl.generate(
                    prompt, 
                    max_new_tokens=config['max_new_tokens'],
                    temperature=0.7
                )
                generation_time = time.time() - start_time
                
                total_time += generation_time
                total_tokens += len(response.text.split())
                
                print(f"  测试 {i+1}: {generation_time:.2f}s, {len(response.text.split())} tokens")
            
            avg_time = total_time / len(test_prompts)
            avg_tokens = total_tokens / len(test_prompts)
            tokens_per_second = avg_tokens / avg_time
            
            print(f"  平均时间: {avg_time:.2f}s")
            print(f"  平均token数: {avg_tokens:.1f}")
            print(f"  生成速度: {tokens_per_second:.1f} tokens/s")
        
        return True
        
    except Exception as e:
        print(f"性能测试失败: {e}")
        return False


def main():
    """主函数"""
    print("SGLang混合精度功能测试")
    print("=" * 60)
    
    # 服务器地址
    server_url = "http://127.0.0.1:8080"
    
    # 检查服务器是否运行
    print(f"测试服务器: {server_url}")
    
    # 运行测试
    tests = [
        ("健康检查", test_health_check, server_url),
        ("SGLang API", test_sglang_api, server_url),
        ("批量生成", test_batch_generation, server_url),
        ("混合精度信息", test_mixed_precision_info, server_url),
        ("性能对比", test_performance_comparison, server_url)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func, *args in tests:
        try:
            if test_func(*args):
                print(f"✅ {test_name} 通过")
                passed_tests += 1
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed_tests}/{total_tests} 通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！混合精度功能正常工作。")
    else:
        print("⚠️  部分测试失败，请检查服务器状态和配置。")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
