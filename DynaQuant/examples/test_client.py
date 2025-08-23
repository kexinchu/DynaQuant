#!/usr/bin/env python3
"""
测试客户端示例
演示如何使用混合精度Transformer模型API
"""
import sys
import requests
import json
import time
from typing import Dict, Any


class MixedPrecisionAPIClient:
    """混合精度API客户端"""
    
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
        """
        生成文本
        
        Args:
            prompt: 输入提示
            **kwargs: 其他生成参数
            
        Returns:
            生成结果
        """
        data = {
            "prompt": prompt,
            **kwargs
        }
        
        response = self.session.post(
            f"{self.base_url}/generate",
            json=data,
            timeout=(2,100)
        )
        return response.json()
    
    def batch_generate(self, prompts: list, **kwargs) -> Dict[str, Any]:
        """
        批量生成文本
        
        Args:
            prompts: 输入提示列表
            **kwargs: 其他生成参数
            
        Returns:
            批量生成结果
        """
        data = {
            "prompts": prompts,
            **kwargs
        }
        
        response = self.session.post(
            f"{self.base_url}/batch_generate",
            json=data,
            timeout=(2,100)
        )
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        response = self.session.get(f"{self.base_url}/model_info")
        return response.json()
    
    def update_weight_mapping(self, weight_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        更新权重映射
        
        Args:
            weight_mapping: 权重映射配置
            
        Returns:
            更新结果
        """
        data = {"weight_mapping": weight_mapping}
        response = self.session.post(
            f"{self.base_url}/update_weight_mapping",
            json=data
        )
        return response.json()
    
    def reload_weights(self) -> Dict[str, Any]:
        """重新加载权重"""
        response = self.session.post(f"{self.base_url}/reload_weights")
        return response.json()


def test_health_check(client: MixedPrecisionAPIClient):
    """测试健康检查"""
    print("=" * 50)
    print("测试健康检查")
    print("=" * 50)
    
    try:
        result = client.health_check()
        print(f"健康状态: {result['status']}")
        print(f"模型已加载: {result['model_loaded']}")
        print(f"设备: {result['device']}")
        print(f"时间戳: {result['timestamp']}")
    except Exception as e:
        print(f"健康检查失败: {e}")


def test_single_generation(client: MixedPrecisionAPIClient):
    """测试单次生成"""
    print("\n" + "=" * 50)
    print("测试单次文本生成")
    print("=" * 50)
    
    prompt = "请介绍一下人工智能的发展历史："
    
    try:
        result = client.generate_text(
            prompt=prompt,
            max_new_tokens=20,
            temperature=0.9,
            top_p=0.9
        )
        
        print(f"输入提示: {result['prompt']}")
        print(f"生成文本: {result['generated_text']}")
        print(f"生成时间: {result['generation_time']:.2f}秒")
        print(f"模型信息: {result['model_info']['model_name']}")
        
    except Exception as e:
        print(f"单次生成失败: {e}")


def test_batch_generation(client: MixedPrecisionAPIClient):
    """测试批量生成"""
    print("\n" + "=" * 50)
    print("测试批量文本生成")
    print("=" * 50)
    
    prompts = [
        "什么是机器学习？",
        "深度学习与传统机器学习有什么区别？",
        "请解释一下神经网络的工作原理。"
    ]
    
    try:
        result = client.batch_generate(
            prompts=prompts,
            max_new_tokens=20,
            temperature=0.9,
            top_p=0.9
        )
        
        print(f"批量生成时间: {result['generation_time']:.2f}秒")
        print("\n生成结果:")
        for i, (prompt, generated) in enumerate(zip(result['prompts'], result['generated_texts'])):
            print(f"\n{i+1}. 输入: {prompt}")
            print(f"   输出: {generated}")
            
    except Exception as e:
        print(f"批量生成失败: {e}")


def test_model_info(client: MixedPrecisionAPIClient):
    """测试获取模型信息"""
    print("\n" + "=" * 50)
    print("测试获取模型信息")
    print("=" * 50)
    
    try:
        result = client.get_model_info()
        
        print("模型信息:")
        model_info = result['model_info']
        print(f"  模型名称: {model_info['model_name']}")
        print(f"  设备: {model_info['device']}")
        print(f"  数据类型: {model_info['dtype']}")
        print(f"  最大序列长度: {model_info['max_seq_length']}")
        print(f"  最大批处理大小: {model_info['max_batch_size']}")
        
        print("\n权重信息:")
        weight_info = result['weight_info']
        print(f"  精度路径: {weight_info['precision_paths']}")
        print(f"  权重映射数量: {len(weight_info['weight_mapping'])}")
        print(f"  缓存文件数量: {len(weight_info['cached_files'])}")
        
    except Exception as e:
        print(f"获取模型信息失败: {e}")


def test_weight_mapping_update(client: MixedPrecisionAPIClient):
    """测试权重映射更新"""
    print("\n" + "=" * 50)
    print("测试权重映射更新")
    print("=" * 50)
    
    # 示例权重映射更新
    new_mapping = {
        "model.layers.0.self_attn.q_proj.weight": "fp8",
        "model.layers.0.self_attn.k_proj.weight": "fp8"
    }
    
    try:
        result = client.update_weight_mapping(new_mapping)
        print(f"权重映射更新结果: {result['message']}")
        
        # 重新加载权重
        reload_result = client.reload_weights()
        print(f"权重重载结果: {reload_result['message']}")
        
    except Exception as e:
        print(f"权重映射更新失败: {e}")


def main(port):
    """主函数"""
    print("混合精度Transformer模型API测试客户端")
    print("=" * 60)
    
    # 创建客户端
    client = MixedPrecisionAPIClient("http://127.0.0.1:" + str(port))
    
    # 运行测试
    test_health_check(client)
    test_single_generation(client)
    test_batch_generation(client)
    test_model_info(client)
    test_weight_mapping_update(client)
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    port = sys.argv[1]
    main(port)
