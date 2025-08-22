#!/usr/bin/env python3
"""
专家激活统计可视化工具
"""

import requests
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
import seaborn as sns


class ExpertStatsVisualizer:
    """专家激活统计可视化器"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        """
        初始化可视化器
        
        Args:
            base_url: API服务器地址
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """获取专家激活统计"""
        response = self.session.get(f"{self.base_url}/expert_stats")
        return response.json()
    
    def plot_expert_activation_heatmap(self, stats: Dict[str, Any], save_path: str = None):
        """绘制专家激活热力图"""
        expert_info = stats['all_experts']
        
        # 提取层和专家信息
        layers = []
        experts = []
        activations = []
        
        for layer_name, layer_data in expert_info.items():
            layer_id = int(layer_name.split('_')[1])
            for expert_name, expert_data in layer_data.items():
                expert_id = int(expert_name.split('_')[1])
                activation_count = expert_data['activation_count']
                
                layers.append(layer_id)
                experts.append(expert_id)
                activations.append(activation_count)
        
        if not activations:
            print("没有找到专家激活数据")
            return
        
        # 创建热力图数据
        max_layer = max(layers)
        max_expert = max(experts)
        
        heatmap_data = np.zeros((max_layer + 1, max_expert + 1))
        
        for layer_id, expert_id, activation_count in zip(layers, experts, activations):
            heatmap_data[layer_id, expert_id] = activation_count
        
        # 绘制热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='d', 
                   cmap='YlOrRd',
                   xticklabels=range(max_expert + 1),
                   yticklabels=range(max_layer + 1))
        
        plt.title('专家激活热力图')
        plt.xlabel('专家ID')
        plt.ylabel('层ID')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"热力图已保存到: {save_path}")
        
        plt.show()
    
    def plot_top_experts_bar(self, stats: Dict[str, Any], top_k: int = 10, save_path: str = None):
        """绘制前K个激活最多的专家柱状图"""
        top_experts = stats['top_experts'][:top_k]
        
        if not top_experts:
            print("没有找到专家激活数据")
            return
        
        # 提取数据
        expert_labels = []
        activation_counts = []
        
        for expert in top_experts:
            label = f"L{expert['layer_id']}-E{expert['expert_id']}"
            expert_labels.append(label)
            activation_counts.append(expert['activation_count'])
        
        # 绘制柱状图
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(expert_labels)), activation_counts, color='skyblue')
        
        plt.title(f'前{top_k}个激活最多的专家')
        plt.xlabel('专家 (层-专家)')
        plt.ylabel('激活次数')
        plt.xticks(range(len(expert_labels)), expert_labels, rotation=45)
        
        # 添加数值标签
        for bar, count in zip(bars, activation_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"柱状图已保存到: {save_path}")
        
        plt.show()
    
    def plot_layer_stats(self, stats: Dict[str, Any], save_path: str = None):
        """绘制层统计图"""
        layer_stats = stats['layer_stats']
        
        if not layer_stats:
            print("没有找到层统计数据")
            return
        
        # 提取数据
        layer_ids = []
        total_activations = []
        expert_counts = []
        avg_activations = []
        
        for layer_name, layer_data in layer_stats.items():
            layer_id = int(layer_name.split('_')[1])
            layer_ids.append(layer_id)
            total_activations.append(layer_data['total_activations'])
            expert_counts.append(layer_data['expert_count'])
            avg_activations.append(layer_data['avg_activations_per_expert'])
        
        # 创建子图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 总激活次数
        ax1.bar(layer_ids, total_activations, color='lightcoral')
        ax1.set_title('每层总激活次数')
        ax1.set_xlabel('层ID')
        ax1.set_ylabel('激活次数')
        
        # 专家数量
        ax2.bar(layer_ids, expert_counts, color='lightgreen')
        ax2.set_title('每层专家数量')
        ax2.set_xlabel('层ID')
        ax2.set_ylabel('专家数量')
        
        # 平均激活次数
        ax3.bar(layer_ids, avg_activations, color='lightblue')
        ax3.set_title('每层平均激活次数/专家')
        ax3.set_xlabel('层ID')
        ax3.set_ylabel('平均激活次数')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"层统计图已保存到: {save_path}")
        
        plt.show()
    
    def plot_summary_stats(self, stats: Dict[str, Any], save_path: str = None):
        """绘制摘要统计图"""
        summary = stats['summary']
        
        # 创建饼图显示激活分布
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 总激活次数
        ax1.pie([summary['total_activations']], 
               labels=['总激活次数'], 
               autopct='%1.0f',
               colors=['lightcoral'])
        ax1.set_title('总激活次数')
        
        # 总token数
        ax2.pie([summary['total_tokens']], 
               labels=['总token数'], 
               autopct='%1.0f',
               colors=['lightgreen'])
        ax2.set_title('总token数')
        
        # 性能指标
        performance_metrics = ['激活/秒', 'token/秒', '请求/秒']
        performance_values = [
            summary['activations_per_second'],
            summary['tokens_per_second'],
            summary['requests_per_second']
        ]
        
        ax3.bar(performance_metrics, performance_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax3.set_title('性能指标')
        ax3.set_ylabel('次数/秒')
        
        # 模型信息
        model_info = ['总层数', '总专家数', '总请求数']
        model_values = [
            summary['total_layers'],
            summary['total_experts'],
            summary['total_requests']
        ]
        
        ax4.bar(model_info, model_values, color=['gold', 'orange', 'red'])
        ax4.set_title('模型信息')
        ax4.set_ylabel('数量')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"摘要统计图已保存到: {save_path}")
        
        plt.show()
    
    def create_dashboard(self, save_dir: str = "expert_stats_plots"):
        """创建完整的统计仪表板"""
        import os
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取统计数据
        stats = self.get_expert_stats()
        
        print("创建专家激活统计仪表板...")
        
        # 绘制各种图表
        self.plot_expert_activation_heatmap(stats, f"{save_dir}/expert_heatmap.png")
        self.plot_top_experts_bar(stats, 15, f"{save_dir}/top_experts.png")
        self.plot_layer_stats(stats, f"{save_dir}/layer_stats.png")
        self.plot_summary_stats(stats, f"{save_dir}/summary_stats.png")
        
        # 保存原始数据
        with open(f"{save_dir}/expert_stats_data.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"仪表板已保存到目录: {save_dir}")


def main():
    """主函数"""
    print("专家激活统计可视化工具")
    print("=" * 50)
    
    visualizer = ExpertStatsVisualizer()
    
    try:
        # 创建完整仪表板
        visualizer.create_dashboard()
        
    except Exception as e:
        print(f"可视化失败: {e}")
        print("请确保服务器正在运行并且有专家激活数据")


if __name__ == "__main__":
    main()
