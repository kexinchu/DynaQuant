#!/usr/bin/env python3
"""
分析Motivation Test结果
对比FP16、全Int4、混合精度对Router激活模式的影响
输入格式: JSON (与expert activation格式兼容)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_results(result_dir: str) -> Dict:
    """加载所有实验组的JSON结果"""
    results = {}

    # 查找所有实验组
    for group_dir in Path(result_dir).iterdir():
        if not group_dir.is_dir():
            continue

        group_name = group_dir.name

        # 查找JSON文件 (格式: gsm8k_<group_name>.json)
        json_files = list(group_dir.glob("gsm8k_*.json"))

        if json_files:
            json_file = json_files[0]
            print(f"加载: {group_name} <- {json_file.name}")
            with open(json_file, 'r', encoding='utf-8') as f:
                results[group_name] = json.load(f)
        else:
            print(f"警告: {group_name} 没有JSON结果文件")

    return results


def compute_expert_activation_stats(layer_counts_json: List[Dict]) -> Dict:
    """
    计算expert激活统计
    输入: JSON格式 [{"0": count, "1": count, ...}, {...}, ...]
    """
    stats = {}

    # 遍历每一层
    for layer_idx, layer_dict in enumerate(layer_counts_json):
        if not layer_dict:
            continue

        # layer_dict: {"expert_id": count, ...}
        expert_counts = {int(k): v for k, v in layer_dict.items()}

        if not expert_counts:
            continue

        expert_ids = np.array(list(expert_counts.keys()))
        counts = np.array(list(expert_counts.values()))

        total_activations = counts.sum()

        stats[layer_idx] = {
            'expert_counts': expert_counts,
            'total_activations': int(total_activations),
            'unique_experts': len(expert_ids),
            'entropy': compute_entropy(counts),
            'top_10_experts': expert_ids[np.argsort(-counts)[:10]].tolist(),
            'top_10_ratios': (counts[np.argsort(-counts)[:10]] / total_activations).tolist()
        }

    return stats


def compute_entropy(counts: np.ndarray) -> float:
    """计算Shannon熵"""
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    return float(entropy)


def compute_js_divergence(p: Dict, q: Dict) -> float:
    """计算JS散度（Jensen-Shannon divergence）"""
    # 获取所有expert ids
    all_experts = set(p.keys()) | set(q.keys())

    # 构建概率分布
    p_probs = np.array([p.get(exp, 0) for exp in all_experts])
    q_probs = np.array([q.get(exp, 0) for exp in all_experts])

    # 归一化
    p_probs = p_probs / (p_probs.sum() + 1e-10)
    q_probs = q_probs / (q_probs.sum() + 1e-10)

    # 计算JS散度
    m = 0.5 * (p_probs + q_probs)
    kl_pm = np.sum(p_probs * np.log2((p_probs + 1e-10) / (m + 1e-10)))
    kl_qm = np.sum(q_probs * np.log2((q_probs + 1e-10) / (m + 1e-10)))
    js = 0.5 * (kl_pm + kl_qm)

    return float(js)


def analyze_router_shift(results: Dict, output_dir: str):
    """分析量化对Router选择的影响"""
    print("\n分析Router激活模式变化...")

    os.makedirs(output_dir, exist_ok=True)

    analysis_report = {
        'overall_summary': {}
    }

    # 计算每组的统计
    print(f"\n对比各实验组:")
    group_stats = {}
    for group_name, layer_counts_json in results.items():
        stats = compute_expert_activation_stats(layer_counts_json)
        group_stats[group_name] = stats

        print(f"\n  {group_name}:")
        # 计算平均熵
        avg_entropy = np.mean([s['entropy'] for s in stats.values()])
        print(f"    平均熵: {avg_entropy:.4f}")
        avg_unique = np.mean([s['unique_experts'] for s in stats.values()])
        print(f"    平均激活experts数: {avg_unique:.2f}")

    # 对比实验组与对照组
    comparisons = {}
    if 'control_fp16' in group_stats:
        control_stats = group_stats['control_fp16']

        # 对比实验组1
        if 'exp1_all_int4' in group_stats:
            exp1_stats = group_stats['exp1_all_int4']
            js_divergences = []

            for layer_idx in control_stats.keys():
                if layer_idx in exp1_stats:
                    js = compute_js_divergence(
                        control_stats[layer_idx]['expert_counts'],
                        exp1_stats[layer_idx]['expert_counts']
                    )
                    js_divergences.append(js)

            avg_js = np.mean(js_divergences)
            print(f"\n  实验组1 vs 对照组:")
            print(f"    平均JS散度: {avg_js:.4f}")
            print(f"    最大JS散度: {np.max(js_divergences):.4f}")

            comparisons['exp1_vs_control'] = {
                'avg_js_divergence': float(avg_js),
                'max_js_divergence': float(np.max(js_divergences)),
                'layer_js_divergences': [float(x) for x in js_divergences]
            }

        # 对比实验组2
        if 'exp2_mixed_hot10' in group_stats:
            exp2_stats = group_stats['exp2_mixed_hot10']
            js_divergences = []

            for layer_idx in control_stats.keys():
                if layer_idx in exp2_stats:
                    js = compute_js_divergence(
                        control_stats[layer_idx]['expert_counts'],
                        exp2_stats[layer_idx]['expert_counts']
                    )
                    js_divergences.append(js)

            avg_js = np.mean(js_divergences)
            print(f"\n  实验组2 vs 对照组:")
            print(f"    平均JS散度: {avg_js:.4f}")
            print(f"    最大JS散度: {np.max(js_divergences):.4f}")

            comparisons['exp2_vs_control'] = {
                'avg_js_divergence': float(avg_js),
                'max_js_divergence': float(np.max(js_divergences)),
                'layer_js_divergences': [float(x) for x in js_divergences]
            }

    # 整理分析报告
    analysis_report['group_stats'] = {k: {
        'avg_entropy': float(np.mean([s['entropy'] for s in v.values()])),
        'avg_unique_experts': float(np.mean([s['unique_experts'] for s in v.values()]))
    } for k, v in group_stats.items()}
    analysis_report['comparisons'] = comparisons
    analysis_report['all_group_stats'] = group_stats

    # 保存分析报告
    report_file = os.path.join(output_dir, 'router_shift_analysis.json')
    with open(report_file, 'w') as f:
        json.dump(analysis_report, f, indent=2)
    print(f"\n分析报告保存至: {report_file}")

    return analysis_report


def plot_expert_activation_comparison(results: Dict, output_dir: str):
    """绘制expert激活对比图"""
    print("\n生成可视化图表...")

    # 收集每组的激活数据
    group_activations = {}

    for group_name, layer_counts_json in results.items():
        stats = compute_expert_activation_stats(layer_counts_json)
        group_activations[group_name] = stats

    if not group_activations:
        print("  没有数据可以绘制")
        return

    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Expert Activation Analysis - GSM8K', fontsize=16)

    # 1. 每层的熵对比
    ax = axes[0, 0]
    for group_name, stats in group_activations.items():
        layers = sorted(stats.keys())
        entropies = [stats[l]['entropy'] for l in layers]
        ax.plot(layers, entropies, marker='o', label=group_name, alpha=0.7)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Entropy')
    ax.set_title('Expert Selection Entropy per Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 每层的unique experts数量对比
    ax = axes[0, 1]
    for group_name, stats in group_activations.items():
        layers = sorted(stats.keys())
        unique_experts = [stats[l]['unique_experts'] for l in layers]
        ax.plot(layers, unique_experts, marker='o',
                label=group_name, alpha=0.7)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Number of Unique Experts')
    ax.set_title('Unique Experts Activated per Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Top-10 expert激活比例对比 (选择中间层)
    ax = axes[1, 0]
    if group_activations:
        mid_layer = sorted(list(group_activations.values())[0].keys())[
            len(list(group_activations.values())[0]) // 2]

        x_pos = np.arange(10)
        width = 0.25

        for i, (group_name, stats) in enumerate(group_activations.items()):
            if mid_layer in stats:
                top_ratios = stats[mid_layer]['top_10_ratios']
                ax.bar(x_pos + i * width, top_ratios,
                       width, label=group_name, alpha=0.7)

        ax.set_xlabel('Top-K Expert Rank')
        ax.set_ylabel('Activation Ratio')
        ax.set_title(f'Top-10 Expert Activation (Layer {mid_layer})')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(range(1, 11))
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # 4. JS散度热力图 (如果有对照组)
    ax = axes[1, 1]
    if 'control_fp16' in group_activations:
        control_stats = group_activations['control_fp16']
        exp_groups = [k for k in group_activations.keys() if k !=
                      'control_fp16']

        if exp_groups:
            layers = sorted(control_stats.keys())
            js_matrix = []

            for exp_group in exp_groups:
                exp_stats = group_activations[exp_group]
                js_values = []

                for layer in layers:
                    if layer in exp_stats:
                        js = compute_js_divergence(
                            control_stats[layer]['expert_counts'],
                            exp_stats[layer]['expert_counts']
                        )
                        js_values.append(js)
                    else:
                        js_values.append(0)

                js_matrix.append(js_values)

            if js_matrix:
                im = ax.imshow(js_matrix, aspect='auto', cmap='YlOrRd')
                ax.set_yticks(range(len(exp_groups)))
                ax.set_yticklabels(exp_groups)
                ax.set_xlabel('Layer Index')
                ax.set_title('JS Divergence from Control (FP16)')
                plt.colorbar(im, ax=ax)

    plt.tight_layout()

    # 保存图表
    output_file = os.path.join(output_dir, 'gsm8k_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: {output_file}")
    plt.close()


def generate_text_report(analysis_report: Dict, output_dir: str):
    """生成文本报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Motivation Test: 量化对MoE Router激活模式的影响")
    report_lines.append("=" * 80)
    report_lines.append("")

    report_lines.append("数据集: GSM8K (数学推理)")
    report_lines.append("-" * 80)

    # 各组统计
    report_lines.append("\n各组基本统计:")
    for group_name, group_stats in analysis_report['group_stats'].items():
        report_lines.append(f"\n  {group_name}:")
        report_lines.append(f"    平均熵: {group_stats['avg_entropy']:.4f}")
        report_lines.append(
            f"    平均激活expert数: {group_stats['avg_unique_experts']:.2f}")

    # 对比分析
    if 'comparisons' in analysis_report and analysis_report['comparisons']:
        report_lines.append("\n\n对比分析:")

        if 'exp1_vs_control' in analysis_report['comparisons']:
            exp1_data = analysis_report['comparisons']['exp1_vs_control']
            report_lines.append("\n  实验组1 (全Int4) vs 对照组 (全FP16):")
            report_lines.append(
                f"    平均JS散度: {exp1_data['avg_js_divergence']:.4f}")
            report_lines.append(
                f"    最大JS散度: {exp1_data['max_js_divergence']:.4f}")
            report_lines.append(f"    解释: JS散度越大，表示量化后router选择模式变化越大")

        if 'exp2_vs_control' in analysis_report['comparisons']:
            exp2_data = analysis_report['comparisons']['exp2_vs_control']
            report_lines.append("\n  实验组2 (混合精度) vs 对照组 (全FP16):")
            report_lines.append(
                f"    平均JS散度: {exp2_data['avg_js_divergence']:.4f}")
            report_lines.append(
                f"    最大JS散度: {exp2_data['max_js_divergence']:.4f}")
            report_lines.append(f"    解释: 混合精度策略对router选择的影响")

    report_lines.append("\n")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("总结")
    report_lines.append("=" * 80)
    report_lines.append("\n关键发现:")
    report_lines.append("1. 量化对expert选择熵的影响")
    report_lines.append("2. 量化对激活expert分布的影响")
    report_lines.append("3. 混合精度策略能否缓解量化带来的router偏移")
    report_lines.append("\n")

    # 保存报告
    report_file = os.path.join(output_dir, 'quantization_impact_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\n文本报告保存至: {report_file}")

    # 同时打印到控制台
    print("\n" + '\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(description="分析Motivation Test结果")
    parser.add_argument('--result_dir', type=str, required=True,
                        help='结果目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='分析输出目录')

    args = parser.parse_args()

    print("加载实验结果...")
    results = load_results(args.result_dir)

    if not results:
        print("错误: 没有找到任何实验结果")
        return

    print(f"找到 {len(results)} 组实验结果:")
    for group_name in results.keys():
        print(f"  - {group_name}")

    # 分析router激活模式变化
    analysis_report = analyze_router_shift(results, args.output_dir)

    # 生成可视化
    plot_expert_activation_comparison(results, args.output_dir)

    # 生成文本报告
    generate_text_report(analysis_report, args.output_dir)

    print("\n分析完成!")


if __name__ == "__main__":
    main()
