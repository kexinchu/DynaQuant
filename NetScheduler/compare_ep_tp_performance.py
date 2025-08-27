#!/usr/bin/env python3
"""
EP vs TP 性能对比分析脚本
比较Expert Parallel和Tensor Parallel两种部署方式的性能差异
"""

import json
import statistics
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class PerformanceComparison:
    """性能对比数据类"""
    ep_results: Dict[str, Any]
    tp_results: Dict[str, Any]
    
    def analyze_deployment_differences(self) -> Dict[str, Any]:
        """分析部署差异"""
        print("=== 部署配置对比分析 ===")
        
        ep_deployment = self.ep_results.get('deployment_info', {})
        tp_deployment = self.tp_results.get('deployment_info', {})
        
        comparison = {
            'gpu_memory_usage': {},
            'gpu_utilization': {},
            'deployment_type': {},
            'model_loading': {},
            'internal_state': {}
        }
        
        # GPU内存使用对比
        ep_memory = ep_deployment.get('gpu_memory_usage', {})
        tp_memory = tp_deployment.get('gpu_memory_usage', {})
        
        if ep_memory and tp_memory:
            ep_values = list(ep_memory.values())
            tp_values = list(tp_memory.values())
            
            comparison['gpu_memory_usage'] = {
                'ep_mean': statistics.mean(ep_values),
                'tp_mean': statistics.mean(tp_values),
                'ep_std': statistics.stdev(ep_values) if len(ep_values) > 1 else 0,
                'tp_std': statistics.stdev(tp_values) if len(tp_values) > 1 else 0,
                'memory_efficiency': {
                    'ep': 'uniform' if statistics.stdev(ep_values) < 10 else 'non_uniform',
                    'tp': 'uniform' if statistics.stdev(tp_values) < 15 else 'non_uniform'
                }
            }
            
            print(f"GPU内存使用对比:")
            print(f"  EP平均使用率: {comparison['gpu_memory_usage']['ep_mean']:.2f}%")
            print(f"  TP平均使用率: {comparison['gpu_memory_usage']['tp_mean']:.2f}%")
            print(f"  EP标准差: {comparison['gpu_memory_usage']['ep_std']:.2f}%")
            print(f"  TP标准差: {comparison['gpu_memory_usage']['tp_std']:.2f}%")
        
        # 部署类型对比
        ep_dist = ep_deployment.get('expert_distribution', {})
        tp_dist = tp_deployment.get('expert_distribution', {})
        
        comparison['deployment_type'] = {
            'ep_type': ep_dist.get('type', 'unknown'),
            'tp_type': tp_dist.get('type', 'unknown'),
            'ep_distribution': ep_dist.get('distribution', 'unknown'),
            'tp_distribution': tp_dist.get('distribution', 'unknown')
        }
        
        print(f"\n部署类型对比:")
        print(f"  EP类型: {comparison['deployment_type']['ep_type']}")
        print(f"  TP类型: {comparison['deployment_type']['tp_type']}")
        print(f"  EP分布: {comparison['deployment_type']['ep_distribution']}")
        print(f"  TP分布: {comparison['deployment_type']['tp_distribution']}")
        
        # 内部状态对比
        ep_internal = ep_deployment.get('internal_state', {})
        tp_internal = tp_deployment.get('internal_state', {})
        
        if ep_internal or tp_internal:
            comparison['internal_state'] = {
                'ep_verification': ep_internal.get('verification_result', {}),
                'tp_verification': tp_internal.get('verification_result', {}),
                'ep_parallel_state': ep_internal.get('parallel_state', {}),
                'tp_parallel_state': tp_internal.get('parallel_state', {}),
                'ep_environment': ep_internal.get('environment_info', {}),
                'tp_environment': tp_internal.get('environment_info', {})
            }
            
            print(f"\n内部状态验证对比:")
            ep_valid = ep_internal.get('verification_result', {}).get('is_valid', False)
            tp_valid = tp_internal.get('verification_result', {}).get('is_valid', False)
            print(f"  EP验证结果: {'✅ 通过' if ep_valid else '❌ 失败'}")
            print(f"  TP验证结果: {'✅ 通过' if tp_valid else '❌ 失败'}")
            
            if not ep_valid:
                ep_error = ep_internal.get('verification_result', {}).get('error_message', '未知错误')
                print(f"  EP错误信息: {ep_error}")
            
            if not tp_valid:
                tp_error = tp_internal.get('verification_result', {}).get('error_message', '未知错误')
                print(f"  TP错误信息: {tp_error}")
        
        return comparison
    
    def analyze_performance_differences(self) -> Dict[str, Any]:
        """分析性能差异"""
        print("\n=== 性能测试结果对比分析 ===")
        
        ep_tests = self.ep_results.get('test_results', {})
        tp_tests = self.tp_results.get('test_results', {})
        
        comparison = {
            'query_length_performance': {},
            'qps_performance': {},
            'overall_performance': {}
        }
        
        # Query长度性能对比
        ep_query = ep_tests.get('query_length_test', [])
        tp_query = tp_tests.get('query_length_test', [])
        
        if ep_query and tp_query:
            comparison['query_length_performance'] = self._compare_query_length_performance(ep_query, tp_query)
        
        # QPS性能对比
        ep_qps = ep_tests.get('qps_test', [])
        tp_qps = tp_tests.get('qps_test', [])
        
        if ep_qps and tp_qps:
            comparison['qps_performance'] = self._compare_qps_performance(ep_qps, tp_qps)
        
        return comparison
    
    def _compare_query_length_performance(self, ep_results: List[Dict], tp_results: List[Dict]) -> Dict[str, Any]:
        """对比不同query长度的性能"""
        print("\n--- Query长度性能对比 ---")
        
        # 按query长度分组
        ep_by_length = {}
        tp_by_length = {}
        
        for result in ep_results:
            if result['success']:
                length = result['query_length']
                if length not in ep_by_length:
                    ep_by_length[length] = []
                ep_by_length[length].append(result)
        
        for result in tp_results:
            if result['success']:
                length = result['query_length']
                if length not in tp_by_length:
                    tp_by_length[length] = []
                tp_by_length[length].append(result)
        
        comparison = {}
        
        for length in sorted(set(ep_by_length.keys()) | set(tp_by_length.keys())):
            if length in ep_by_length and length in tp_by_length:
                ep_stats = self._calculate_stats(ep_by_length[length])
                tp_stats = self._calculate_stats(tp_by_length[length])
                
                comparison[length] = {
                    'ep': ep_stats,
                    'tp': tp_stats,
                    'improvement': {
                        'ttft': ((tp_stats['ttft_mean'] - ep_stats['ttft_mean']) / ep_stats['ttft_mean']) * 100,
                        'tpot': ((tp_stats['tpot_mean'] - ep_stats['tpot_mean']) / ep_stats['tpot_mean']) * 100,
                        'overall': ((tp_stats['overall_mean'] - ep_stats['overall_mean']) / ep_stats['overall_mean']) * 100
                    }
                }
                
                print(f"Query长度 {length}:")
                print(f"  EP - TTFT: {ep_stats['ttft_mean']:.2f}ms, TPOT: {ep_stats['tpot_mean']:.2f}ms, Overall: {ep_stats['overall_mean']:.2f}ms")
                print(f"  TP - TTFT: {tp_stats['ttft_mean']:.2f}ms, TPOT: {tp_stats['tpot_mean']:.2f}ms, Overall: {tp_stats['overall_mean']:.2f}ms")
                print(f"  改进 - TTFT: {comparison[length]['improvement']['ttft']:+.2f}%, TPOT: {comparison[length]['improvement']['tpot']:+.2f}%, Overall: {comparison[length]['improvement']['overall']:+.2f}%")
        
        return comparison
    
    def _compare_qps_performance(self, ep_results: List[Dict], tp_results: List[Dict]) -> Dict[str, Any]:
        """对比不同QPS的性能"""
        print("\n--- QPS性能对比 ---")
        
        # 按QPS分组
        ep_by_qps = {}
        tp_by_qps = {}
        
        for result in ep_results:
            if result['success']:
                qps = result['qps']
                if qps not in ep_by_qps:
                    ep_by_qps[qps] = []
                ep_by_qps[qps].append(result)
        
        for result in tp_results:
            if result['success']:
                qps = result['qps']
                if qps not in tp_by_qps:
                    tp_by_qps[qps] = []
                tp_by_qps[qps].append(result)
        
        comparison = {}
        
        for qps in sorted(set(ep_by_qps.keys()) | set(tp_by_qps.keys())):
            if qps in ep_by_qps and qps in tp_by_qps:
                ep_stats = self._calculate_stats(ep_by_qps[qps])
                tp_stats = self._calculate_stats(tp_by_qps[qps])
                
                comparison[qps] = {
                    'ep': ep_stats,
                    'tp': tp_stats,
                    'improvement': {
                        'ttft': ((tp_stats['ttft_mean'] - ep_stats['ttft_mean']) / ep_stats['ttft_mean']) * 100,
                        'tpot': ((tp_stats['tpot_mean'] - ep_stats['tpot_mean']) / ep_stats['tpot_mean']) * 100,
                        'overall': ((tp_stats['overall_mean'] - ep_stats['overall_mean']) / ep_stats['overall_mean']) * 100
                    }
                }
                
                print(f"QPS {qps}:")
                print(f"  EP - TTFT: {ep_stats['ttft_mean']:.2f}ms, TPOT: {ep_stats['tpot_mean']:.2f}ms, Overall: {ep_stats['overall_mean']:.2f}ms")
                print(f"  TP - TTFT: {tp_stats['ttft_mean']:.2f}ms, TPOT: {tp_stats['tpot_mean']:.2f}ms, Overall: {tp_stats['overall_mean']:.2f}ms")
                print(f"  改进 - TTFT: {comparison[qps]['improvement']['ttft']:+.2f}%, TPOT: {comparison[qps]['improvement']['tpot']:+.2f}%, Overall: {comparison[qps]['improvement']['overall']:+.2f}%")
        
        return comparison
    
    def _calculate_stats(self, results: List[Dict]) -> Dict[str, float]:
        """计算统计信息"""
        ttft_values = [r['ttft_ms'] for r in results]
        tpot_values = [r['tpot_ms'] for r in results]
        overall_values = [r['overall_latency_ms'] for r in results]
        
        return {
            'ttft_mean': statistics.mean(ttft_values),
            'ttft_median': statistics.median(ttft_values),
            'ttft_std': statistics.stdev(ttft_values) if len(ttft_values) > 1 else 0,
            'tpot_mean': statistics.mean(tpot_values),
            'tpot_median': statistics.median(tpot_values),
            'tpot_std': statistics.stdev(tpot_values) if len(tpot_values) > 1 else 0,
            'overall_mean': statistics.mean(overall_values),
            'overall_median': statistics.median(overall_values),
            'overall_std': statistics.stdev(overall_values) if len(overall_values) > 1 else 0
        }
    
    def generate_plots(self, save_path: str = "performance_comparison.png"):
        """生成性能对比图表"""
        print(f"\n=== 生成性能对比图表 ===")
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('EP vs TP 性能对比分析', fontsize=16)
        
        # 1. Query长度 vs TTFT
        self._plot_query_length_vs_metric(ax1, 'ttft_mean', 'TTFT (ms)', 'Query长度 vs TTFT')
        
        # 2. Query长度 vs TPOT
        self._plot_query_length_vs_metric(ax2, 'tpot_mean', 'TPOT (ms)', 'Query长度 vs TPOT')
        
        # 3. QPS vs Overall Latency
        self._plot_qps_vs_metric(ax3, 'overall_mean', 'Overall Latency (ms)', 'QPS vs Overall Latency')
        
        # 4. 性能改进百分比
        self._plot_improvement_percentage(ax4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
        
        # 显示图表
        plt.show()
    
    def _plot_query_length_vs_metric(self, ax, metric: str, ylabel: str, title: str):
        """绘制query长度与性能指标的关系"""
        query_perf = self.analyze_performance_differences()['query_length_performance']
        
        lengths = []
        ep_values = []
        tp_values = []
        
        for length, data in query_perf.items():
            lengths.append(length)
            ep_values.append(data['ep'][metric])
            tp_values.append(data['tp'][metric])
        
        x = np.arange(len(lengths))
        width = 0.35
        
        ax.bar(x - width/2, ep_values, width, label='Expert Parallel (EP)', alpha=0.8)
        ax.bar(x + width/2, tp_values, width, label='Tensor Parallel (TP)', alpha=0.8)
        
        ax.set_xlabel('Query长度')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(lengths)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_qps_vs_metric(self, ax, metric: str, ylabel: str, title: str):
        """绘制QPS与性能指标的关系"""
        qps_perf = self.analyze_performance_differences()['qps_performance']
        
        qps_values = []
        ep_values = []
        tp_values = []
        
        for qps, data in qps_perf.items():
            qps_values.append(qps)
            ep_values.append(data['ep'][metric])
            tp_values.append(data['tp'][metric])
        
        ax.plot(qps_values, ep_values, 'o-', label='Expert Parallel (EP)', linewidth=2, markersize=6)
        ax.plot(qps_values, tp_values, 's-', label='Tensor Parallel (TP)', linewidth=2, markersize=6)
        
        ax.set_xlabel('QPS')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    def _plot_improvement_percentage(self, ax):
        """绘制性能改进百分比"""
        query_perf = self.analyze_performance_differences()['query_length_performance']
        
        lengths = []
        ttft_improvement = []
        tpot_improvement = []
        overall_improvement = []
        
        for length, data in query_perf.items():
            lengths.append(length)
            ttft_improvement.append(data['improvement']['ttft'])
            tpot_improvement.append(data['improvement']['tpot'])
            overall_improvement.append(data['improvement']['overall'])
        
        x = np.arange(len(lengths))
        width = 0.25
        
        ax.bar(x - width, ttft_improvement, width, label='TTFT改进', alpha=0.8)
        ax.bar(x, tpot_improvement, width, label='TPOT改进', alpha=0.8)
        ax.bar(x + width, overall_improvement, width, label='Overall改进', alpha=0.8)
        
        ax.set_xlabel('Query长度')
        ax.set_ylabel('改进百分比 (%)')
        ax.set_title('TP相对于EP的性能改进')
        ax.set_xticks(x)
        ax.set_xticklabels(lengths)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

def load_test_results(ep_file: str = 'ep_test_results.json', tp_file: str = 'tp_test_results.json') -> PerformanceComparison:
    """加载测试结果"""
    try:
        with open(ep_file, 'r', encoding='utf-8') as f:
            ep_results = json.load(f)
        
        with open(tp_file, 'r', encoding='utf-8') as f:
            tp_results = json.load(f)
        
        return PerformanceComparison(ep_results, tp_results)
    except FileNotFoundError as e:
        print(f"错误: 找不到测试结果文件 - {e}")
        print("请先运行 test_single_expert_ep.py 和 test_single_expert_tp.py")
        return None

def generate_summary_report(comparison: PerformanceComparison) -> str:
    """生成总结报告"""
    print("\n=== 总结报告 ===")
    
    # 部署验证总结
    deployment_comp = comparison.analyze_deployment_differences()
    
    print("1. 部署验证结果:")
    if deployment_comp['gpu_memory_usage']:
        ep_memory = deployment_comp['gpu_memory_usage']['ep_mean']
        tp_memory = deployment_comp['gpu_memory_usage']['tp_mean']
        print(f"   - EP平均GPU内存使用率: {ep_memory:.2f}%")
        print(f"   - TP平均GPU内存使用率: {tp_memory:.2f}%")
        print(f"   - 内存使用差异: {abs(ep_memory - tp_memory):.2f}%")
    
    # 性能测试总结
    perf_comp = comparison.analyze_performance_differences()
    
    print("\n2. 性能测试总结:")
    
    # 计算平均改进
    query_improvements = []
    qps_improvements = []
    
    for length_data in perf_comp['query_length_performance'].values():
        query_improvements.append(length_data['improvement']['overall'])
    
    for qps_data in perf_comp['qps_performance'].values():
        qps_improvements.append(qps_data['improvement']['overall'])
    
    if query_improvements:
        avg_query_improvement = statistics.mean(query_improvements)
        print(f"   - Query长度测试平均改进: {avg_query_improvement:+.2f}%")
    
    if qps_improvements:
        avg_qps_improvement = statistics.mean(qps_improvements)
        print(f"   - QPS测试平均改进: {avg_qps_improvement:+.2f}%")
    
    # 推荐建议
    print("\n3. 推荐建议:")
    if query_improvements and qps_improvements:
        overall_improvement = (statistics.mean(query_improvements) + statistics.mean(qps_improvements)) / 2
        
        if overall_improvement > 5:
            print("   - TP部署方式在性能上明显优于EP部署方式")
            print("   - 建议在生产环境中使用TP部署方式")
        elif overall_improvement < -5:
            print("   - EP部署方式在性能上明显优于TP部署方式")
            print("   - 建议在生产环境中使用EP部署方式")
        else:
            print("   - 两种部署方式性能相近")
            print("   - 可以根据具体需求选择部署方式")
    
    return "总结报告生成完成"

def main():
    """主函数"""
    print("EP vs TP 性能对比分析")
    print("=" * 50)
    
    # 加载测试结果
    comparison = load_test_results()
    if comparison is None:
        return
    
    # 分析部署差异
    comparison.analyze_deployment_differences()
    
    # 分析性能差异
    comparison.analyze_performance_differences()
    
    # 生成总结报告
    generate_summary_report(comparison)
    
    # 生成图表
    try:
        comparison.generate_plots()
    except ImportError:
        print("警告: matplotlib未安装，跳过图表生成")
        print("安装命令: pip install matplotlib")
    
    # 保存对比结果
    comparison_data = {
        'deployment_comparison': comparison.analyze_deployment_differences(),
        'performance_comparison': comparison.analyze_performance_differences()
    }
    
    with open('ep_tp_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    print("\n对比结果已保存到 ep_tp_comparison.json")

if __name__ == "__main__":
    main()
