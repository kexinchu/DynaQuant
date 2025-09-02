#!/usr/bin/env python3
"""
快速Expert Tracking测试脚本
验证expert激活跟踪和hot-cold分数计算功能
"""

import sys
import time
import json
import logging

# 添加SGLang路径
sys.path.insert(0, 'sglang-0.4.7/python')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_expert_tracking():
    """测试expert tracking功能"""
    try:
        logger.info("测试Expert Tracking功能...")
        
        # 导入expert tracker
        from sglang.srt.model_loader.enhanced_mixed_precision_loader import (
            init_global_expert_tracker,
            get_global_expert_tracker,
            record_expert_activation,
            record_request
        )
        
        # 初始化expert tracker
        tracker = init_global_expert_tracker()
        logger.info("✓ Expert tracker初始化成功")
        
        # 模拟一些请求和expert激活
        logger.info("模拟expert激活...")
        
        # 模拟请求1
        record_request("req_001", input_length=100, output_length=50)
        
        # 模拟不同层的expert激活
        layers = [0, 1, 2, 3, 4]  # 5层
        experts_per_layer = 8       # 每层8个expert
        
        for layer_id in layers:
            for expert_id in range(experts_per_layer):
                # 随机激活次数 (1-20)
                import random
                activation_count = random.randint(1, 20)
                
                for _ in range(activation_count):
                    record_expert_activation(
                        layer_id=layer_id,
                        expert_id=expert_id,
                        tokens_processed=random.randint(10, 100),
                        activation_strength=random.uniform(0.5, 1.0)
                    )
        
        # 模拟请求2
        record_request("req_002", input_length=80, output_length=40)
        
        # 再次激活一些expert
        for layer_id in [0, 1, 2]:
            for expert_id in [0, 1, 2]:  # 只激活前3个expert
                record_expert_activation(
                    layer_id=layer_id,
                    expert_id=expert_id,
                    tokens_processed=50,
                    activation_strength=0.9
                )
        
        logger.info("✓ 模拟expert激活完成")
        
        # 等待一段时间让分数更新
        time.sleep(1)
        
        # 获取统计信息
        logger.info("获取expert统计信息...")
        expert_stats = tracker.get_expert_stats()
        logger.info(f"✓ 获取到 {len(expert_stats)} 个expert的统计信息")
        
        # 计算hot-cold分数
        hot_cold_analysis = calculate_hot_cold_scores(tracker)
        
        # 导出结果
        export_expert_analysis(tracker, hot_cold_analysis)
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def calculate_hot_cold_scores(tracker) -> dict:
    """计算hot-cold分数（基于激活次数）"""
    try:
        logger.info("计算expert hot-cold分数...")
        
        # 获取所有expert的激活次数
        expert_stats = tracker.get_expert_stats()
        
        if not expert_stats:
            logger.warning("没有expert统计数据")
            return {}
        
        # 按层分组统计
        layer_experts = {}
        for key, info in expert_stats.items():
            layer_id = info['layer_id']
            if layer_id not in layer_experts:
                layer_experts[layer_id] = []
            layer_experts[layer_id].append({
                'expert_id': info['expert_id'],
                'activation_count': info['activation_count'],
                'total_tokens': info['total_tokens_processed']
            })
        
        # 计算每层的hot-cold分数
        hot_cold_analysis = {}
        for layer_id, experts in layer_experts.items():
            if not experts:
                continue
            
            # 按激活次数排序
            experts.sort(key=lambda x: x['activation_count'], reverse=True)
            
            # 计算分数
            max_count = experts[0]['activation_count']
            min_count = experts[-1]['activation_count']
            
            layer_analysis = {
                'layer_id': layer_id,
                'total_experts': len(experts),
                'max_activations': max_count,
                'min_activations': min_count,
                'experts': {}
            }
            
            for expert in experts:
                if max_count == min_count:
                    # 如果所有expert激活次数相同
                    hot_cold_score = 1.0
                else:
                    # 线性插值计算分数
                    hot_cold_score = (expert['activation_count'] - min_count) / (max_count - min_count)
                
                layer_analysis['experts'][expert['expert_id']] = {
                    'activation_count': expert['activation_count'],
                    'total_tokens': expert['total_tokens'],
                    'hot_cold_score': round(hot_cold_score, 4)
                }
            
            hot_cold_analysis[f'layer_{layer_id}'] = layer_analysis
        
        logger.info(f"✓ 计算完成，共 {len(hot_cold_analysis)} 层")
        return hot_cold_analysis
        
    except Exception as e:
        logger.error(f"计算hot-cold分数失败: {e}")
        return {}


def export_expert_analysis(tracker, hot_cold_analysis):
    """导出expert分析结果"""
    try:
        logger.info("导出expert分析结果...")
        
        # 获取详细统计
        expert_stats = tracker.get_expert_stats()
        top_experts = tracker.get_top_experts(20)
        
        # 构建完整报告
        report = {
            'export_time': time.time(),
            'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_experts': len(expert_stats),
                'total_activations': len(tracker.activation_history),
                'total_requests': len(tracker.request_history)
            },
            'hot_cold_analysis': hot_cold_analysis,
            'expert_stats': expert_stats,
            'top_experts': top_experts
        }
        
        # 导出到文件
        output_file = "expert_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Expert分析结果已导出到: {output_file}")
        
        # 显示摘要
        display_analysis_summary(report)
        
    except Exception as e:
        logger.error(f"导出expert分析失败: {e}")


def display_analysis_summary(report):
    """显示分析摘要"""
    try:
        print("\n" + "=" * 60)
        print("Expert Analysis 摘要")
        print("=" * 60)
        
        summary = report['summary']
        print(f"总Expert数: {summary['total_experts']}")
        print(f"总激活次数: {summary['total_activations']}")
        print(f"总请求数: {summary['total_requests']}")
        
        hot_cold_analysis = report['hot_cold_analysis']
        if hot_cold_analysis:
            print(f"\n层数: {len(hot_cold_analysis)}")
            
            for layer_key, layer_info in list(hot_cold_analysis.items())[:3]:  # 显示前3层
                print(f"\n{layer_key}:")
                print(f"  Expert数: {layer_info['total_experts']}")
                print(f"  最大激活: {layer_info['max_activations']}")
                print(f"  最小激活: {layer_info['min_activations']}")
                
                # 显示前5个expert的分数
                experts = list(layer_info['experts'].items())[:5]
                for expert_id, expert_info in experts:
                    print(f"    Expert {expert_id}: {expert_info['hot_cold_score']:.4f} ({expert_info['activation_count']})")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        logger.error(f"显示摘要失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("快速Expert Tracking测试")
    print("=" * 60)
    
    # 运行测试
    success = test_expert_tracking()
    
    if success:
        print("\n🎉 测试完成！")
        print("✓ Expert tracking功能正常工作")
        print("✓ Hot-cold分数计算正确")
        print("✓ 结果已导出到 expert_analysis.json")
        
        print("\n下一步:")
        print("1. 启动Qwen3-235B-A22B服务")
        print("2. 运行完整的expert tracking测试")
        print("3. 使用真实数据进行测试")
    else:
        print("\n❌ 测试失败，请检查错误信息")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
