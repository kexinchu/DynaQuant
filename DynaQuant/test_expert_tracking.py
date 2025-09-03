#!/usr/bin/env python3
"""
测试专家激活跟踪功能
验证hot-cold分数计算和导出功能
"""

import sys
import os
import time
import json
import logging

# 添加SGLang路径
sys.path.insert(0, 'sglang-0.4.7/python')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_expert_tracker():
    """测试专家激活跟踪器"""
    try:
        from sglang.srt.model_loader.enhanced_mixed_precision_loader import (
            init_global_expert_tracker, 
            get_global_expert_tracker,
            record_expert_activation,
            record_request
        )
        
        print("✓ 成功导入expert tracker模块")
        
        # 初始化全局expert tracker
        tracker = init_global_expert_tracker()
        print("✓ 全局expert tracker初始化成功")
        
        # 模拟一些expert激活
        print("\n模拟expert激活...")
        
        # 模拟请求1
        record_request("req_001", input_length=100, output_length=50)
        
        # 模拟layer 0的expert激活
        record_expert_activation(layer_id=0, expert_id=0, tokens_processed=50, activation_strength=0.9)
        record_expert_activation(layer_id=0, expert_id=2, tokens_processed=30, activation_strength=0.7)
        record_expert_activation(layer_id=0, expert_id=5, tokens_processed=20, activation_strength=0.5)
        
        # 模拟layer 1的expert激活
        record_expert_activation(layer_id=1, expert_id=1, tokens_processed=40, activation_strength=0.8)
        record_expert_activation(layer_id=1, expert_id=3, tokens_processed=35, activation_strength=0.6)
        
        # 等待一段时间让hot-cold分数更新
        time.sleep(0.1)
        
        # 模拟请求2
        record_request("req_002", input_length=80, output_length=40)
        
        # 再次激活一些expert
        record_expert_activation(layer_id=0, expert_id=0, tokens_processed=40, activation_strength=0.9)
        record_expert_activation(layer_id=0, expert_id=1, tokens_processed=25, activation_strength=0.8)
        record_expert_activation(layer_id=1, expert_id=1, tokens_processed=30, activation_strength=0.8)
        
        print("✓ 模拟expert激活完成")
        
        # 获取统计信息
        print("\n获取expert统计信息...")
        expert_stats = tracker.get_expert_stats()
        print(f"✓ 获取到 {len(expert_stats)} 个expert的统计信息")
        
        # 获取hot-cold分数
        print("\n获取hot-cold分数...")
        hot_cold_scores = tracker.get_hot_cold_scores()
        print(f"✓ 获取到 {len(hot_cold_scores)} 个expert的hot-cold分数")
        
        # 显示前几个expert的分数
        print("\n前5个expert的hot-cold分数:")
        for i, (key, info) in enumerate(list(hot_cold_scores.items())[:5]):
            print(f"  {key}: {info['hot_cold_score']:.4f} (激活次数: {info['activation_count']})")
        
        # 获取最hot的expert
        print("\n最hot的5个expert:")
        top_hot_experts = tracker.get_top_hot_experts(5)
        for i, expert in enumerate(top_hot_experts):
            print(f"  {i+1}. Layer {expert['layer_id']} Expert {expert['expert_id']}: {expert['hot_cold_score']:.4f}")
        
        # 导出报告
        print("\n导出expert统计报告...")
        tracker.export_stats("test_expert_stats.json")
        print("✓ 统计报告已导出到 test_expert_stats.json")
        
        # 导出hot-cold报告
        print("\n导出hot-cold报告...")
        tracker.export_hot_cold_report("test_hot_cold_report.json")
        print("✓ Hot-cold报告已导出到 test_hot_cold_report.json")
        
        # 显示报告内容
        print("\n报告文件内容预览:")
        if os.path.exists("test_hot_cold_report.json"):
            with open("test_hot_cold_report.json", 'r', encoding='utf-8') as f:
                report = json.load(f)
                print(f"  总expert数: {report['summary']['total_experts']}")
                print(f"  总激活数: {report['summary']['total_activations']}")
                print(f"  导出时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report['export_time']))}")
        
        print("\n✓ 所有测试完成！")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_moe_tracker_integration():
    """测试MoE跟踪器集成"""
    try:
        from sglang.srt.models.moe_tracker import MoETracker, MoEModuleWrapper
        
        print("\n测试MoE跟踪器集成...")
        
        # 创建一个模拟的MoE模块
        class MockMoEModule:
            def __init__(self):
                self.experts = [None] * 8  # 8个expert
                self.top_k = 2
                self._gate_outputs = None
            
            def forward(self, x):
                # 模拟gate输出
                import torch
                self._gate_outputs = torch.randn(x.shape[0], 8)  # 8个expert
                return x
        
        # 创建模拟模型
        mock_model = MockMoEModule()
        
        # 包装MoE模块
        wrapped_modules = MoETracker.wrap_moe_modules(mock_model)
        print(f"✓ 成功包装了 {len(wrapped_modules)} 个MoE模块")
        
        # 测试前向传播
        import torch
        test_input = torch.randn(10, 512)  # 10个token，512维
        
        for name, wrapper in wrapped_modules.items():
            print(f"测试模块: {name}")
            output = wrapper(test_input)
            print(f"  ✓ 前向传播成功，输出形状: {output.shape}")
        
        print("✓ MoE跟踪器集成测试完成！")
        return True
        
    except Exception as e:
        print(f"✗ MoE跟踪器集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("专家激活跟踪功能测试")
    print("=" * 60)
    
    # 测试1: 基本expert tracker功能
    success1 = test_expert_tracker()
    
    # 测试2: MoE跟踪器集成
    success2 = test_moe_tracker_integration()
    
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print(f"  基本功能测试: {'✓ 通过' if success1 else '✗ 失败'}")
    print(f"  集成测试: {'✓ 通过' if success2 else '✗ 失败'}")
    
    if success1 and success2:
        print("\n🎉 所有测试通过！expert tracking功能正常工作")
        print("\n下一步:")
        print("1. 启动SGLang服务")
        print("2. 发送一些请求")
        print("3. 查看生成的expert报告文件")
    else:
        print("\n❌ 部分测试失败，请检查错误信息")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
