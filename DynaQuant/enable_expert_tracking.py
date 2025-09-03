#!/usr/bin/env python3
"""
在SGLang启动时启用expert tracking功能
"""

import sys
import os
import logging

# 添加SGLang路径
sys.path.insert(0, 'sglang-0.4.7/python')

def enable_expert_tracking():
    """启用expert tracking功能"""
    try:
        from sglang.srt.model_loader.enhanced_mixed_precision_loader import (
            init_global_expert_tracker,
            get_global_expert_tracker
        )
        
        # 初始化全局expert tracker
        tracker = init_global_expert_tracker()
        print("✓ 全局expert tracker初始化成功")
        
        # 验证tracker是否可用
        current_tracker = get_global_expert_tracker()
        if current_tracker:
            print("✓ Expert tracking功能已启用")
            print(f"  - 当前expert数量: {len(current_tracker.expert_stats)}")
            print(f"  - 激活历史长度: {len(current_tracker.activation_history)}")
            return True
        else:
            print("✗ Expert tracking功能启用失败")
            return False
            
    except Exception as e:
        print(f"✗ 启用expert tracking失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def setup_moe_tracking(model):
    """为模型设置MoE跟踪"""
    try:
        from sglang.srt.models.moe_tracker import MoETracker
        
        print("设置MoE模块跟踪...")
        
        # 包装MoE模块
        wrapped_modules = MoETracker.track_expert_activations_in_model(
            model, 
            enable_tracking=True
        )
        
        if wrapped_modules:
            print(f"✓ 成功包装了 {len(wrapped_modules)} 个MoE模块")
            for name, wrapper in wrapped_modules.items():
                print(f"  - {name} -> Layer {wrapper.layer_id}")
            return True
        else:
            print("⚠ 未找到MoE模块或跟踪已禁用")
            return False
            
    except Exception as e:
        print(f"✗ 设置MoE跟踪失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_tracking_status():
    """获取tracking状态"""
    try:
        from sglang.srt.model_loader.enhanced_mixed_precision_loader import (
            get_global_expert_tracker
        )
        
        tracker = get_global_expert_tracker()
        if tracker:
            print("\n📊 Expert Tracking 状态:")
            print(f"  - 全局tracker: {'✓ 已初始化' if tracker else '✗ 未初始化'}")
            print(f"  - Expert数量: {len(tracker.expert_stats)}")
            print(f"  - 激活记录: {len(tracker.activation_history)}")
            print(f"  - 请求记录: {len(tracker.request_history)}")
            
            # 显示前几个expert的分数
            if tracker.expert_stats:
                print("\n🔥 前5个Expert的Hot-Cold分数:")
                hot_cold_scores = tracker.get_hot_cold_scores()
                for i, (key, info) in enumerate(list(hot_cold_scores.items())[:5]):
                    print(f"  {i+1}. {key}: {info['hot_cold_score']:.4f}")
            
            return True
        else:
            print("✗ Expert tracker未初始化")
            return False
            
    except Exception as e:
        print(f"✗ 获取状态失败: {e}")
        return False


def export_current_stats():
    """导出当前统计信息"""
    try:
        from sglang.srt.model_loader.enhanced_mixed_precision_loader import (
            get_global_expert_tracker
        )
        
        tracker = get_global_expert_tracker()
        if tracker and tracker.expert_stats:
            print("\n📤 导出统计信息...")
            
            # 导出完整统计
            tracker.export_stats("current_expert_stats.json")
            print("✓ 完整统计已导出到 current_expert_stats.json")
            
            # 导出hot-cold报告
            tracker.export_hot_cold_report("current_hot_cold_report.json")
            print("✓ Hot-cold报告已导出到 current_hot_cold_report.json")
            
            return True
        else:
            print("⚠ 没有可导出的统计信息")
            return False
            
    except Exception as e:
        print(f"✗ 导出失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("SGLang Expert Tracking 启用脚本")
    print("=" * 60)
    
    # 1. 启用expert tracking
    print("\n1. 启用Expert Tracking功能...")
    success1 = enable_expert_tracking()
    
    # 2. 获取状态
    print("\n2. 检查功能状态...")
    success2 = get_tracking_status()
    
    # 3. 导出统计（如果有的话）
    print("\n3. 导出统计信息...")
    success3 = export_current_stats()
    
    print("\n" + "=" * 60)
    print("启用结果总结:")
    print(f"  Expert Tracking启用: {'✓ 成功' if success1 else '✗ 失败'}")
    print(f"  状态检查: {'✓ 成功' if success2 else '✗ 失败'}")
    print(f"  统计导出: {'✓ 成功' if success3 else '⚠ 跳过'}")
    
    if success1 and success2:
        print("\n🎉 Expert Tracking功能已成功启用！")
        print("\n下一步:")
        print("1. 启动SGLang服务")
        print("2. 发送一些请求")
        print("3. 使用 get_tracking_status() 查看状态")
        print("4. 使用 export_current_stats() 导出报告")
    else:
        print("\n❌ 部分功能启用失败，请检查错误信息")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
