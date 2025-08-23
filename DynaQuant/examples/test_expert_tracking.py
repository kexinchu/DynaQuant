#!/usr/bin/env python3
"""
测试专家激活跟踪功能
"""
import sys
import requests
import json
import time
from typing import Dict, Any


class ExpertTrackingClient:
    """专家激活跟踪客户端"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        """
        初始化客户端
        
        Args:
            base_url: API服务器地址
        """
        self.base_url = base_url
        self.session = requests.Session()
    
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
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """获取专家激活统计"""
        response = self.session.get(f"{self.base_url}/expert_stats", timeout=(2, 100))
        return response.json()
    
    def get_expert_stats_with_params(self, top_k: int = 10, minutes: int = 5) -> Dict[str, Any]:
        """获取专家激活统计（带参数）"""
        data = {
            "top_k": top_k,
            "minutes": minutes
        }
        
        response = self.session.post(
            f"{self.base_url}/expert_stats",
            json=data
        )
        return response.json()
    
    def reset_expert_stats(self) -> Dict[str, Any]:
        """重置专家激活统计"""
        response = self.session.post(f"{self.base_url}/reset_expert_stats")
        return response.json()
    
    def export_expert_stats(self) -> Dict[str, Any]:
        """导出专家激活统计"""
        response = self.session.post(f"{self.base_url}/export_expert_stats")
        return response.json()


def test_expert_tracking(port):
    """测试专家激活跟踪"""
    print("=" * 60)
    print("专家激活跟踪测试")
    print("=" * 60)
    
    client = ExpertTrackingClient("http://127.0.0.1:" + str(port))
    
    # 重置统计
    # print("1. 重置专家激活统计...")
    # result = client.reset_expert_stats()
    # print(f"结果: {result['message']}")
    
    # 生成一些文本
    print("\n2. 生成文本以触发专家激活...")
    prompts = [
        "请介绍一下人工智能的发展历史：",
        "什么是机器学习？请详细解释。",
        "深度学习与传统机器学习有什么区别？",
        "请解释一下神经网络的工作原理。",
        "什么是强化学习？它有什么应用？"
    ]
    
    # for i, prompt in enumerate(prompts):
    #     print(f"\n生成文本 {i+1}: {prompt}")
    #     try:
    #         result = client.generate_text(
    #             prompt=prompt,
    #             max_new_tokens=100,
    #             temperature=0.7
    #         )
    #         print(f"生成成功，长度: {len(result['generated_text'])} 字符")
    #     except Exception as e:
    #         print(f"生成失败: {e}")
        
    #     time.sleep(1)  # 短暂延迟
    
    # 获取专家统计
    print("\n3. 获取专家激活统计...")
    try:
        stats = client.get_expert_stats()
        
        print("\n摘要统计:")
        summary = stats['summary']
        print(f"  总激活次数: {summary['total_activations']}")
        print(f"  总token数: {summary['total_tokens']}")
        print(f"  总请求数: {summary['total_requests']}")
        print(f"  总层数: {summary['total_layers']}")
        print(f"  总专家数: {summary['total_experts']}")
        print(f"  运行时间: {summary['runtime_seconds']:.2f}秒")
        print(f"  激活/秒: {summary['activations_per_second']:.2f}")
        print(f"  token/秒: {summary['tokens_per_second']:.2f}")
        
        print("\n层统计:")
        layer_stats = stats['layer_stats']
        for layer_name, layer_info in layer_stats.items():
            print(f"  {layer_name}:")
            print(f"    总激活: {layer_info['total_activations']}")
            print(f"    专家数: {layer_info['expert_count']}")
            print(f"    平均激活/专家: {layer_info['avg_activations_per_expert']:.2f}")
            print(f"    激活率: {layer_info['activation_rate']:.4f}")
        
        print("\n前10个激活最多的专家:")
        top_experts = stats['top_experts']
        for i, expert in enumerate(top_experts[:10]):
            print(f"  {i+1}. Layer {expert['layer_id']}, Expert {expert['expert_id']}: {expert['activation_count']} 次激活")
        
    except Exception as e:
        print(f"获取统计失败: {e}")
    
    # 获取详细统计
    print("\n4. 获取详细专家统计...")
    try:
        detailed_stats = client.get_expert_stats_with_params(top_k=5, minutes=10)
        
        print("\n最近激活记录:")
        recent_activations = detailed_stats['recent_activations']
        for activation in recent_activations[:5]:  # 只显示前5个
            timestamp = time.strftime('%H:%M:%S', time.localtime(activation['timestamp']))
            print(f"  {timestamp}: Layer {activation['layer_id']}, Expert {activation['expert_id']}, Tokens: {activation['token_count']}")
        
    except Exception as e:
        print(f"获取详细统计失败: {e}")
    
    # 导出统计
    print("\n5. 导出专家激活统计...")
    try:
        result = client.export_expert_stats()
        print(f"结果: {result['message']}")
    except Exception as e:
        print(f"导出失败: {e}")
    
    print("\n" + "=" * 60)
    print("专家激活跟踪测试完成")
    print("=" * 60)


def test_batch_generation(port):
    """测试批量生成"""
    print("\n" + "=" * 60)
    print("批量生成测试")
    print("=" * 60)
    
    client = ExpertTrackingClient(port)
    
    # 重置统计
    client.reset_expert_stats()
    
    # 批量生成
    batch_prompts = [
        "人工智能是什么？",
        "机器学习的基本原理是什么？",
        "深度学习与传统机器学习有什么区别？",
        "神经网络是如何工作的？",
        "什么是强化学习？",
        "自然语言处理有哪些应用？",
        "计算机视觉的发展历程如何？",
        "推荐系统的工作原理是什么？",
        "知识图谱的构建方法有哪些？",
        "大语言模型的发展趋势如何？"
    ]
    
    print(f"开始批量生成 {len(batch_prompts)} 个请求...")
    start_time = time.time()
    
    for i, prompt in enumerate(batch_prompts):
        try:
            result = client.generate_text(
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.8
            )
            print(f"请求 {i+1} 完成")
        except Exception as e:
            print(f"请求 {i+1} 失败: {e}")
    
    end_time = time.time()
    print(f"批量生成完成，耗时: {end_time - start_time:.2f}秒")
    
    # 获取最终统计
    stats = client.get_expert_stats()
    summary = stats['summary']
    
    print(f"\n最终统计:")
    print(f"  总激活次数: {summary['total_activations']}")
    print(f"  总token数: {summary['total_tokens']}")
    print(f"  总请求数: {summary['total_requests']}")
    print(f"  平均激活/请求: {summary['total_activations'] / max(summary['total_requests'], 1):.2f}")
    print(f"  平均token/请求: {summary['total_tokens'] / max(summary['total_requests'], 1):.2f}")


def main(port):
    """主函数"""
    print("专家激活跟踪功能测试")
    
    # 测试基本功能
    test_expert_tracking(port)
    
    # 测试批量生成
    # test_batch_generation(port)


if __name__ == "__main__":
    port = sys.argv[1]
    main(port)
