#!/usr/bin/env python3
"""
混合精度模型部署系统使用示例
包含专家激活跟踪功能
"""

import requests
import json
import time
from typing import Dict, Any


class MixedPrecisionModelClient:
    """混合精度模型客户端"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        """
        初始化客户端
        
        Args:
            base_url: API服务器地址
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def generate_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """生成文本"""
        data = {
            "prompt": prompt,
            **kwargs
        }
        
        response = self.session.post(
            f"{self.base_url}/generate",
            json=data
        )
        return response.json()
    
    def batch_generate(self, prompts: list, **kwargs) -> Dict[str, Any]:
        """批量生成"""
        data = {
            "prompts": prompts,
            **kwargs
        }
        
        response = self.session.post(
            f"{self.base_url}/batch_generate",
            json=data
        )
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        response = self.session.get(f"{self.base_url}/model_info")
        return response.json()
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """获取专家激活统计"""
        response = self.session.get(f"{self.base_url}/expert_stats")
        return response.json()
    
    def reset_expert_stats(self) -> Dict[str, Any]:
        """重置专家激活统计"""
        response = self.session.post(f"{self.base_url}/reset_expert_stats")
        return response.json()
    
    def export_expert_stats(self) -> Dict[str, Any]:
        """导出专家激活统计"""
        response = self.session.post(f"{self.base_url}/export_expert_stats")
        return response.json()


def example_1_basic_usage():
    """示例1：基本使用"""
    print("=" * 60)
    print("示例1：基本使用")
    print("=" * 60)
    
    client = MixedPrecisionModelClient()
    
    # 健康检查
    print("1. 健康检查")
    health = client.health_check()
    print(f"   状态: {health['status']}")
    print(f"   消息: {health['message']}")
    
    # 获取模型信息
    print("\n2. 获取模型信息")
    model_info = client.get_model_info()
    print(f"   模型名称: {model_info['model_name']}")
    print(f"   模型路径: {model_info['model_path']}")
    print(f"   设备: {model_info['device']}")
    
    # 生成文本
    print("\n3. 生成文本")
    result = client.generate_text(
        prompt="请介绍一下人工智能：",
        max_new_tokens=100,
        temperature=0.7
    )
    print(f"   输入: {result['prompt']}")
    print(f"   输出: {result['generated_text']}")
    print(f"   生成时间: {result['generation_time']:.2f}秒")


def example_2_expert_tracking():
    """示例2：专家激活跟踪"""
    print("\n" + "=" * 60)
    print("示例2：专家激活跟踪")
    print("=" * 60)
    
    client = MixedPrecisionModelClient()
    
    # 重置统计
    print("1. 重置专家激活统计")
    result = client.reset_expert_stats()
    print(f"   结果: {result['message']}")
    
    # 生成多个文本以触发专家激活
    print("\n2. 生成多个文本以触发专家激活")
    prompts = [
        "什么是机器学习？",
        "深度学习与传统机器学习有什么区别？",
        "请解释神经网络的工作原理。",
        "什么是强化学习？",
        "自然语言处理有哪些应用？"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"   生成文本 {i+1}: {prompt}")
        result = client.generate_text(
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.8
        )
        print(f"   生成完成，长度: {len(result['generated_text'])} 字符")
        time.sleep(0.5)  # 短暂延迟
    
    # 获取专家统计
    print("\n3. 获取专家激活统计")
    stats = client.get_expert_stats()
    
    summary = stats['summary']
    print(f"   总激活次数: {summary['total_activations']}")
    print(f"   总token数: {summary['total_tokens']}")
    print(f"   总请求数: {summary['total_requests']}")
    print(f"   总层数: {summary['total_layers']}")
    print(f"   总专家数: {summary['total_experts']}")
    
    # 显示前5个激活最多的专家
    print("\n4. 前5个激活最多的专家:")
    top_experts = stats['top_experts'][:5]
    for i, expert in enumerate(top_experts):
        print(f"   {i+1}. Layer {expert['layer_id']}, Expert {expert['expert_id']}: {expert['activation_count']} 次激活")


def example_3_batch_generation():
    """示例3：批量生成"""
    print("\n" + "=" * 60)
    print("示例3：批量生成")
    print("=" * 60)
    
    client = MixedPrecisionModelClient()
    
    # 批量生成
    batch_prompts = [
        "人工智能是什么？",
        "机器学习的基本原理是什么？",
        "深度学习与传统机器学习有什么区别？",
        "神经网络是如何工作的？",
        "什么是强化学习？"
    ]
    
    print("1. 批量生成文本")
    start_time = time.time()
    
    result = client.batch_generate(
        prompts=batch_prompts,
        max_new_tokens=80,
        temperature=0.7
    )
    
    end_time = time.time()
    print(f"   批量生成完成，耗时: {end_time - start_time:.2f}秒")
    
    # 显示结果
    print("\n2. 生成结果:")
    for i, (prompt, text) in enumerate(zip(result['prompts'], result['generated_texts'])):
        print(f"   输入 {i+1}: {prompt}")
        print(f"   输出 {i+1}: {text}")
        print()


def example_4_advanced_analysis():
    """示例4：高级分析"""
    print("\n" + "=" * 60)
    print("示例4：高级分析")
    print("=" * 60)
    
    client = MixedPrecisionModelClient()
    
    # 获取详细统计
    print("1. 获取详细专家统计")
    stats = client.get_expert_stats()
    
    # 层统计
    print("\n2. 层统计:")
    layer_stats = stats['layer_stats']
    for layer_name, layer_info in layer_stats.items():
        print(f"   {layer_name}:")
        print(f"     总激活: {layer_info['total_activations']}")
        print(f"     专家数: {layer_info['expert_count']}")
        print(f"     平均激活/专家: {layer_info['avg_activations_per_expert']:.2f}")
        print(f"     激活率: {layer_info['activation_rate']:.4f}")
    
    # 专家利用率分析
    print("\n3. 专家利用率分析:")
    all_experts = stats['all_experts']
    total_experts = 0
    active_experts = 0
    
    for layer_name, layer_data in all_experts.items():
        for expert_name, expert_data in layer_data.items():
            total_experts += 1
            if expert_data['activation_count'] > 0:
                active_experts += 1
    
    utilization_rate = (active_experts / total_experts) * 100 if total_experts > 0 else 0
    print(f"   总专家数: {total_experts}")
    print(f"   激活专家数: {active_experts}")
    print(f"   专家利用率: {utilization_rate:.2f}%")
    
    # 导出统计
    print("\n4. 导出专家激活统计")
    result = client.export_expert_stats()
    print(f"   结果: {result['message']}")


def example_5_performance_monitoring():
    """示例5：性能监控"""
    print("\n" + "=" * 60)
    print("示例5：性能监控")
    print("=" * 60)
    
    client = MixedPrecisionModelClient()
    
    # 重置统计
    client.reset_expert_stats()
    
    # 性能测试
    print("1. 性能测试")
    test_prompts = [
        "请介绍一下人工智能的发展历史。",
        "什么是机器学习？请详细解释其基本原理。",
        "深度学习与传统机器学习有什么区别？请举例说明。",
        "请解释神经网络的工作原理和训练过程。",
        "什么是强化学习？它有哪些实际应用？"
    ]
    
    total_tokens = 0
    total_time = 0
    
    for i, prompt in enumerate(test_prompts):
        print(f"   测试 {i+1}: {prompt}")
        
        start_time = time.time()
        result = client.generate_text(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7
        )
        end_time = time.time()
        
        generation_time = end_time - start_time
        total_time += generation_time
        total_tokens += len(result['generated_text'])
        
        print(f"   生成时间: {generation_time:.2f}秒")
        print(f"   生成长度: {len(result['generated_text'])} 字符")
    
    # 性能统计
    print("\n2. 性能统计:")
    stats = client.get_expert_stats()
    summary = stats['summary']
    
    print(f"   总生成时间: {total_time:.2f}秒")
    print(f"   总生成字符: {total_tokens}")
    print(f"   平均生成时间: {total_time / len(test_prompts):.2f}秒/请求")
    print(f"   平均生成速度: {total_tokens / total_time:.2f}字符/秒")
    print(f"   总激活次数: {summary['total_activations']}")
    print(f"   激活/秒: {summary['activations_per_second']:.2f}")
    print(f"   token/秒: {summary['tokens_per_second']:.2f}")


def main():
    """主函数"""
    print("混合精度模型部署系统使用示例")
    print("包含专家激活跟踪功能")
    print("=" * 80)
    
    try:
        # 运行所有示例
        example_1_basic_usage()
        example_2_expert_tracking()
        example_3_batch_generation()
        example_4_advanced_analysis()
        example_5_performance_monitoring()
        
        print("\n" + "=" * 80)
        print("所有示例运行完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n运行示例时出错: {e}")
        print("请确保服务器正在运行并且配置正确")


if __name__ == "__main__":
    main()
