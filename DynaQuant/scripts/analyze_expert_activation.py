"""
分析专家激活统计结果的辅助脚本
"""
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_activation_data(exp_name, results_dir="expert_activation_results"):
    """加载指定实验的激活数据"""
    file_path = os.path.join(results_dir, f"{exp_name}.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def compute_activation_statistics(data):
    """
    计算激活统计信息
    返回：每层的统计信息列表
    """
    stats = []
    for layer_idx, layer_counts in enumerate(data):
        if not layer_counts:
            stats.append({
                'layer': layer_idx,
                'total_activations': 0,
                'num_active_experts': 0,
                'top_5_experts': [],
                'activation_concentration': 0.0
            })
            continue

        total = sum(layer_counts.values())
        num_active = len(layer_counts)

        # 计算top-5专家
        sorted_experts = sorted(
            layer_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        top_5 = [(int(eid), count, count/total)
                 for eid, count in sorted_experts]

        # 计算激活集中度（top-10专家占比）
        top_10_count = sum([count for _, count in sorted_experts[:10]])
        concentration = top_10_count / total if total > 0 else 0.0

        stats.append({
            'layer': layer_idx,
            'total_activations': total,
            'num_active_experts': num_active,
            'top_5_experts': top_5,
            'activation_concentration': concentration
        })

    return stats


def compare_experiments(exp1_name, exp2_name, results_dir="expert_activation_results"):
    """
    比较两个实验的专家激活差异
    返回：每层的KL散度
    """
    data1 = load_activation_data(exp1_name, results_dir)
    data2 = load_activation_data(exp2_name, results_dir)

    kl_divergences = []

    for layer_idx in range(len(data1)):
        counts1 = data1[layer_idx]
        counts2 = data2[layer_idx]

        # 获取所有出现的expert ID
        all_ids = set(counts1.keys()) | set(counts2.keys())

        # 转换为概率分布
        v1 = np.array([counts1.get(str(i), 0)
                      for i in range(128)], dtype=float)
        v2 = np.array([counts2.get(str(i), 0)
                      for i in range(128)], dtype=float)

        # 归一化
        sum1 = v1.sum()
        sum2 = v2.sum()
        if sum1 > 0:
            v1 = v1 / sum1
        if sum2 > 0:
            v2 = v2 / sum2

        # 计算KL散度
        eps = 1e-8
        kl = (v1 * np.log((v1 + eps) / (v2 + eps))).sum()
        kl_divergences.append(kl)

    return kl_divergences


def print_summary(exp_name, results_dir="expert_activation_results"):
    """打印指定实验的统计摘要"""
    print(f"\n{'='*80}")
    print(f"实验: {exp_name}")
    print(f"{'='*80}")

    data = load_activation_data(exp_name, results_dir)
    stats = compute_activation_statistics(data)

    print(f"\n总层数: {len(stats)}")

    # 打印每层的top-5专家
    for layer_stat in stats:
        layer_idx = layer_stat['layer']
        total = layer_stat['total_activations']
        num_active = layer_stat['num_active_experts']
        top_5 = layer_stat['top_5_experts']
        concentration = layer_stat['activation_concentration']

        print(f"\n第 {layer_idx+1} 层:")
        print(f"  总激活次数: {total}")
        print(f"  活跃专家数: {num_active}")
        print(f"  Top-10集中度: {concentration:.2%}")
        if top_5:
            print(f"  Top-5 专家:")
            for expert_id, count, ratio in top_5:
                print(f"    Expert {expert_id:3d}: {count:6d} ({ratio:6.2%})")


def print_comparison(exp1_name, exp2_name, results_dir="expert_activation_results"):
    """打印两个实验的对比结果"""
    print(f"\n{'='*80}")
    print(f"对比: {exp1_name} vs {exp2_name}")
    print(f"{'='*80}")

    kl_divs = compare_experiments(exp1_name, exp2_name, results_dir)

    print(f"\n每层KL散度:")
    for layer_idx, kl in enumerate(kl_divs):
        print(f"  第 {layer_idx+1:2d} 层: KL = {kl:8.4f}")

    print(f"\n平均KL散度: {np.mean(kl_divs):.4f}")
    print(f"最大KL散度: {np.max(kl_divs):.4f} (第 {np.argmax(kl_divs)+1} 层)")
    print(f"最小KL散度: {np.min(kl_divs):.4f} (第 {np.argmin(kl_divs)+1} 层)")


def create_heatmap(data, exp_name, output_dir="plots"):
    """
    创建专家激活热力图（类似图片中的效果）
    参数:
        data: 加载的激活数据
        exp_name: 实验名称
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 确定层数和专家数
    num_layers = len(data)
    max_expert_id = 0
    for layer_data in data:
        if layer_data:
            max_expert_id = max(max_expert_id, max(
                [int(k) for k in layer_data.keys()]))

    # 创建激活矩阵 [layers x experts]
    activation_matrix = np.zeros((num_layers, max_expert_id + 1))

    for layer_idx, layer_data in enumerate(data):
        for expert_id_str, count in layer_data.items():
            expert_id = int(expert_id_str)
            activation_matrix[layer_idx, expert_id] = count

    # 创建热力图
    plt.figure(figsize=(15, 8))

    # 使用YlOrRd颜色映射（黄色到红色）
    im = plt.imshow(activation_matrix, cmap='YlOrRd', aspect='auto')

    # 设置坐标轴
    plt.xlabel('Expert ID', fontsize=12)
    plt.ylabel('Layer', fontsize=12)
    plt.title(f'{exp_name} - Expert Usage Heatmap Across Layers',
              fontsize=14, fontweight='bold')

    # 设置Y轴标签（从layer_1开始）
    layer_labels = [f'layer_{i+1}' for i in range(num_layers)]
    plt.yticks(range(num_layers), layer_labels)

    # 设置X轴标签（每5个专家显示一个）
    expert_ticks = range(0, max_expert_id + 1, 5)
    plt.xticks(expert_ticks, expert_ticks)

    # 添加颜色条
    cbar = plt.colorbar(im, label='Number of calls')
    cbar.set_label('Number of calls', fontsize=12)

    # 添加网格线
    plt.grid(True, alpha=0.3, color='white', linewidth=0.5)

    # 调整布局
    plt.tight_layout()

    # 保存图片为PDF
    output_file = os.path.join(output_dir, f'{exp_name}_heatmap.pdf')
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"  热力图已保存: {output_file}")

    return activation_matrix


def create_layer_comparison(data, exp_name, layer_indices=[0, -1], output_dir="plots"):
    """
    创建单层对比图（第一层和最后一层）
    参数:
        data: 加载的激活数据
        exp_name: 实验名称
        layer_indices: 要对比的层索引列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取实际层索引
    actual_indices = []
    for idx in layer_indices:
        if idx < 0:
            actual_idx = len(data) + idx
        else:
            actual_idx = idx
        if 0 <= actual_idx < len(data):
            actual_indices.append((idx, actual_idx))

    if not actual_indices:
        print(f"  警告: 没有有效的层索引")
        return

    # 创建子图
    n_layers = len(actual_indices)
    fig, axes = plt.subplots(1, n_layers, figsize=(8 * n_layers, 6))
    if n_layers == 1:
        axes = [axes]

    for i, (orig_idx, actual_idx) in enumerate(actual_indices):
        layer_data = data[actual_idx]

        if not layer_data:
            axes[i].text(0.5, 0.5, f'Layer {actual_idx+1}\nNo data',
                         ha='center', va='center', transform=axes[i].transAxes)
            continue

        # 获取专家ID和激活次数
        expert_ids = [int(k) for k in layer_data.keys()]
        counts = list(layer_data.values())

        # 按专家ID排序
        sorted_pairs = sorted(zip(expert_ids, counts))
        expert_ids, counts = zip(*sorted_pairs)

        # 创建柱状图
        bars = axes[i].bar(expert_ids, counts, color='lightblue',
                           alpha=0.7, edgecolor='black', linewidth=0.5)

        # 突出显示top-5专家
        if len(counts) >= 5:
            sorted_by_count = sorted(
                zip(expert_ids, counts), key=lambda x: x[1], reverse=True)
            top5_experts = {expert_id for expert_id, _ in sorted_by_count[:5]}

            for j, (expert_id, count) in enumerate(zip(expert_ids, counts)):
                if expert_id in top5_experts:
                    if j == 0:  # 最高激活的专家用深红色
                        bars[j].set_color('darkred')
                    else:
                        bars[j].set_color('orange')

        # 设置标题和标签
        layer_name = f'layer_{actual_idx+1}'
        if orig_idx != actual_idx:
            layer_name += f' (index {orig_idx})'

        axes[i].set_title(f'{exp_name} - Expert Usage Distribution - {layer_name}',
                          fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Expert ID', fontsize=11)
        axes[i].set_ylabel('Number of calls', fontsize=11)

        # 设置网格
        axes[i].grid(True, alpha=0.3)

        # 设置X轴刻度
        if len(expert_ids) > 20:
            step = max(1, len(expert_ids) // 10)
            axes[i].set_xticks(expert_ids[::step])

    plt.tight_layout()

    # 保存图片为PDF
    layer_str = '_'.join(
        [f'layer{actual_idx+1}' for _, actual_idx in actual_indices])
    output_file = os.path.join(
        output_dir, f'{exp_name}_layer_comparison_{layer_str}.pdf')
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"  单层对比图已保存: {output_file}")


def create_merged_visualization(exp_names, results_dir, output_dir="plots"):
    """
    创建合并可视化（多个实验的对比）
    参数:
        exp_names: 实验名称列表
        results_dir: 结果目录
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载所有实验数据
    all_data = {}
    for exp_name in exp_names:
        try:
            data = load_activation_data(exp_name, results_dir)
            all_data[exp_name] = data
        except FileNotFoundError:
            print(f"  警告: 未找到实验 {exp_name} 的数据")
            continue

    if not all_data:
        print("  错误: 没有找到任何实验数据")
        return

    # 创建合并热力图
    create_merged_heatmap(all_data, output_dir)

    # 创建合并单层对比
    create_merged_layer_comparison(all_data, output_dir)


def create_merged_heatmap(all_data, output_dir):
    """创建合并热力图"""
    # 计算所有实验的最大激活值，用于统一颜色范围
    max_activation = 0
    for data in all_data.values():
        for layer_data in data:
            if layer_data:
                max_activation = max(max_activation, max(layer_data.values()))

    # 为每个实验创建热力图
    for exp_name, data in all_data.items():
        create_heatmap(data, exp_name, output_dir)

    # 创建对比热力图（如果有多个实验）
    if len(all_data) > 1:
        create_comparison_heatmap(all_data, max_activation, output_dir)


def create_comparison_heatmap(all_data, max_activation, output_dir):
    """创建实验对比热力图"""
    exp_names = list(all_data.keys())
    n_experiments = len(exp_names)

    fig, axes = plt.subplots(1, n_experiments, figsize=(6 * n_experiments, 8))
    if n_experiments == 1:
        axes = [axes]

    for i, (exp_name, data) in enumerate(all_data.items()):
        # 创建激活矩阵
        num_layers = len(data)
        max_expert_id = 0
        for layer_data in data:
            if layer_data:
                max_expert_id = max(max_expert_id, max(
                    [int(k) for k in layer_data.keys()]))

        activation_matrix = np.zeros((num_layers, max_expert_id + 1))
        for layer_idx, layer_data in enumerate(data):
            for expert_id_str, count in layer_data.items():
                expert_id = int(expert_id_str)
                activation_matrix[layer_idx, expert_id] = count

        # 绘制热力图
        im = axes[i].imshow(activation_matrix, cmap='YlOrRd', aspect='auto',
                            vmin=0, vmax=max_activation)

        axes[i].set_title(f'{exp_name}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Expert ID', fontsize=10)
        if i == 0:
            axes[i].set_ylabel('Layer', fontsize=10)
            layer_labels = [f'layer_{j+1}' for j in range(num_layers)]
            axes[i].set_yticks(range(num_layers), layer_labels)
        else:
            axes[i].set_yticks([])

        axes[i].grid(True, alpha=0.3, color='white', linewidth=0.5)

    # 添加颜色条
    cbar = fig.colorbar(im, ax=axes, label='Number of calls')
    cbar.set_label('Number of calls', fontsize=12)

    plt.suptitle('Expert Usage Heatmap Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'merged_heatmap_comparison.pdf')
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"  合并对比热力图已保存: {output_file}")


def create_merged_layer_comparison(all_data, output_dir):
    """创建合并单层对比"""
    exp_names = list(all_data.keys())

    # 选择第一层和最后一层
    for layer_idx in [0, -1]:
        layer_name = f'layer_{len(list(all_data.values())[0]) + layer_idx if layer_idx < 0 else layer_idx + 1}'

        fig, axes = plt.subplots(
            1, len(exp_names), figsize=(6 * len(exp_names), 6))
        if len(exp_names) == 1:
            axes = [axes]

        for i, exp_name in enumerate(exp_names):
            data = all_data[exp_name]
            actual_idx = len(data) + layer_idx if layer_idx < 0 else layer_idx
            layer_data = data[actual_idx]

            if not layer_data:
                axes[i].text(0.5, 0.5, f'{exp_name}\nLayer {actual_idx+1}\nNo data',
                             ha='center', va='center', transform=axes[i].transAxes)
                continue

            # 获取专家ID和激活次数
            expert_ids = [int(k) for k in layer_data.keys()]
            counts = list(layer_data.values())

            # 按专家ID排序
            sorted_pairs = sorted(zip(expert_ids, counts))
            expert_ids, counts = zip(*sorted_pairs)

            # 创建柱状图
            bars = axes[i].bar(expert_ids, counts, color='lightblue', alpha=0.7,
                               edgecolor='black', linewidth=0.5)

            # 突出显示top-5专家
            if len(counts) >= 5:
                sorted_by_count = sorted(
                    zip(expert_ids, counts), key=lambda x: x[1], reverse=True)
                top5_experts = {expert_id for expert_id,
                                _ in sorted_by_count[:5]}

                for j, (expert_id, count) in enumerate(zip(expert_ids, counts)):
                    if expert_id in top5_experts:
                        if j == 0:  # 最高激活的专家用深红色
                            bars[j].set_color('darkred')
                        else:
                            bars[j].set_color('orange')

            axes[i].set_title(f'{exp_name}\n{layer_name}',
                              fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Expert ID', fontsize=10)
            if i == 0:
                axes[i].set_ylabel('Number of calls', fontsize=10)
            axes[i].grid(True, alpha=0.3)

        plt.suptitle(
            f'Expert Usage Distribution - {layer_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_file = os.path.join(
            output_dir, f'merged_layer_comparison_{layer_name}.pdf')
        plt.savefig(output_file, format='pdf', bbox_inches='tight')
        plt.close()

        print(f"  合并单层对比图已保存: {output_file}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="分析专家激活统计结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析默认目录的所有结果
  python analyze_expert_activation.py

  # 指定结果目录
  python analyze_expert_activation.py --results-dir ./my_results

  # 只显示摘要，不进行对比
  python analyze_expert_activation.py --summary-only

  # 只对比特定实验
  python analyze_expert_activation.py --compare wikitext_thinking_off wikitext_thinking_on

  # 生成可视化图表（热力图和单层对比）
  python analyze_expert_activation.py --plot

  # 只生成图表，不显示文本统计
  python analyze_expert_activation.py --plot-only

  # 生成合并可视化（多个实验对比）
  python analyze_expert_activation.py --merged-plot

  # 指定图表输出目录
  python analyze_expert_activation.py --plot --plot-dir ./my_plots
        """
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='./benchmark_results/expert_activation_results',
        help='结果目录路径（默认: ./benchmark_results/expert_activation_results）'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='只显示统计摘要，不进行对比分析'
    )
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('EXP1', 'EXP2'),
        help='对比两个指定的实验'
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        help='指定要分析的实验名称（不指定则分析所有）'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='生成可视化图表（热力图和单层对比图）'
    )
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='只生成图表，不显示文本统计'
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default='plots',
        help='图表输出目录（默认: plots）'
    )
    parser.add_argument(
        '--merged-plot',
        action='store_true',
        help='生成合并可视化（多个实验对比）'
    )

    return parser.parse_args()


def main():
    """主函数：分析所有实验结果"""
    args = parse_args()
    results_dir = args.results_dir

    print(f"{'='*80}")
    print(f"Expert Activation Analysis Results")
    print(f"{'='*80}")
    print(f"结果目录: {results_dir}\n")

    # 如果指定了对比两个实验
    if args.compare:
        exp1, exp2 = args.compare
        print_comparison(exp1, exp2, results_dir)
        return

    # 确定要分析的实验
    if args.experiments:
        experiments = args.experiments
    else:
        # 默认的6组实验
        experiments = [
            "wikitext_thinking_off",
            "wikitext_thinking_on",
            "gsm8k_thinking_off",
            "gsm8k_thinking_on",
            "humaneval_thinking_off",
            "humaneval_thinking_on"
        ]

    # 打印每个实验的摘要
    print("【实验统计摘要】")
    found_experiments = []
    for exp_name in experiments:
        try:
            print_summary(exp_name, results_dir)
            found_experiments.append(exp_name)
        except FileNotFoundError:
            print(f"\n警告: 未找到实验结果文件 {exp_name}.json")

    if not found_experiments:
        print(f"\n错误: 在 {results_dir} 中未找到任何实验结果")
        return

    # 生成可视化图表
    if args.plot or args.plot_only or args.merged_plot:
        print(f"\n\n{'#'*80}")
        print("# 生成可视化图表")
        print(f"{'#'*80}")

        if args.merged_plot:
            # 生成合并可视化
            print("\n生成合并可视化图表...")
            create_merged_visualization(
                found_experiments, results_dir, args.plot_dir)
        else:
            # 为每个实验生成单独的可视化
            for exp_name in found_experiments:
                print(f"\n生成 {exp_name} 的可视化图表...")
                data = load_activation_data(exp_name, results_dir)

                # 生成热力图
                create_heatmap(data, exp_name, args.plot_dir)

                # 生成单层对比图（第一层和最后一层）
                create_layer_comparison(data, exp_name, [0, -1], args.plot_dir)

        print(f"\n所有图表已保存到: {args.plot_dir}/")

    # 如果只生成图表，则结束
    if args.plot_only:
        return

    # 如果只显示摘要，则结束
    if args.summary_only:
        return

    # 比较同一任务类型在不同thinking模式下的差异
    print(f"\n\n{'#'*80}")
    print("# 对比分析: 不同Thinking模式的影响")
    print(f"{'#'*80}")

    task_pairs = [
        ("wikitext_thinking_off", "wikitext_thinking_on"),
        ("gsm8k_thinking_off", "gsm8k_thinking_on"),
        ("humaneval_thinking_off", "humaneval_thinking_on")
    ]

    for exp1, exp2 in task_pairs:
        if exp1 in found_experiments and exp2 in found_experiments:
            try:
                print_comparison(exp1, exp2, results_dir)
            except FileNotFoundError as e:
                print(f"\n警告: {e}")

    # 比较不同任务类型（thinking off）
    print(f"\n\n{'#'*80}")
    print("# 对比分析: 不同任务类型的差异 (Thinking Off)")
    print(f"{'#'*80}")

    task_comparison = [
        ("wikitext_thinking_off", "gsm8k_thinking_off"),
        ("wikitext_thinking_off", "humaneval_thinking_off"),
        ("gsm8k_thinking_off", "humaneval_thinking_off")
    ]

    for exp1, exp2 in task_comparison:
        if exp1 in found_experiments and exp2 in found_experiments:
            try:
                print_comparison(exp1, exp2, results_dir)
            except FileNotFoundError as e:
                print(f"\n警告: {e}")


if __name__ == "__main__":
    main()
