import json
import argparse
from collections import Counter
from typing import List, Tuple, Dict, Set

def get_top_n_experts(layer_data: Dict, n: int = 10) -> Set[int]:
    """获取激活次数top-N的expert ID"""
    sorted_experts = sorted(layer_data.items(), key=lambda x: x[1], reverse=True)
    return set([int(expert_id) for expert_id, _ in sorted_experts[:n]])

def calculate_difference(set1: Set[int], set2: Set[int], set3: Set[int]) -> float:
    """计算三个集合之间的差异度
    使用Jaccard距离的平均值来衡量差异"""
    def jaccard_distance(s1: Set[int], s2: Set[int]) -> float:
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        if union == 0:
            return 1.0
        return 1.0 - (intersection / union)
    
    # 计算两两之间的Jaccard距离
    d12 = jaccard_distance(set1, set2)
    d13 = jaccard_distance(set1, set3)
    d23 = jaccard_distance(set2, set3)
    
    # 返回平均距离（差异越大，值越大）
    return (d12 + d13 + d23) / 3.0

def get_top_n_experts_with_counts(layer_data: Dict, n: int = 10) -> List[Tuple[int, int]]:
    """获取激活次数top-N的expert ID和对应的激活次数"""
    sorted_experts = sorted(layer_data.items(), key=lambda x: x[1], reverse=True)
    return [(int(expert_id), int(count)) for expert_id, count in sorted_experts[:n]]

def analyze_three_files(file1_path: str, file2_path: str, file3_path: str):
    """分析三个JSON文件，找出top-10差异最大的层"""
    # 读取三个JSON文件
    print(f"正在读取文件...")
    print(f"  文件1: {file1_path}")
    print(f"  文件2: {file2_path}")
    print(f"  文件3: {file3_path}")
    print()
    
    with open(file1_path, 'r') as f:
        data1 = json.load(f)
    
    with open(file2_path, 'r') as f:
        data2 = json.load(f)
    
    with open(file3_path, 'r') as f:
        data3 = json.load(f)
    
    # 检查层数是否一致
    if len(data1) != len(data2) or len(data1) != len(data3):
        print(f"警告: 三个文件的层数不一致 (文件1: {len(data1)}, 文件2: {len(data2)}, 文件3: {len(data3)})")
        min_layers = min(len(data1), len(data2), len(data3))
        print(f"将只分析前 {min_layers} 层")
    else:
        min_layers = len(data1)
    
    # 遍历每一层，计算差异
    layer_differences: List[Tuple[int, float]] = []
    
    print(f"正在分析 {min_layers} 层的expert激活差异...")
    for layer_idx in range(min_layers):
        # 获取三种情况下的top-10 experts
        top10_1 = get_top_n_experts(data1[layer_idx], 10)
        top10_2 = get_top_n_experts(data2[layer_idx], 10)
        top10_3 = get_top_n_experts(data3[layer_idx], 10)
        
        # 计算差异
        diff = calculate_difference(top10_1, top10_2, top10_3)
        layer_differences.append((layer_idx, diff))
    
    # 按差异度排序，找出差异最大的层
    layer_differences.sort(key=lambda x: x[1], reverse=True)
    max_diff_layer_idx, max_diff = layer_differences[0]
    
    # 输出结果
    print("\n" + "="*80)
    print(f"差异最大的层: 层 {max_diff_layer_idx} (差异度: {max_diff:.4f})")
    print("="*80)
    print()
    
    # 显示该层在三个文件中的top-10 experts
    top10_1 = get_top_n_experts_with_counts(data1[max_diff_layer_idx], 10)
    top10_2 = get_top_n_experts_with_counts(data2[max_diff_layer_idx], 10)
    top10_3 = get_top_n_experts_with_counts(data3[max_diff_layer_idx], 10)
    
    print(f"层 {max_diff_layer_idx} - 文件1 (wikitext) top-10 experts:")
    for i, (expert_id, count) in enumerate(top10_1, 1):
        print(f"  {i}. Expert {expert_id}: {count:,} 次激活")
    
    print()
    print(f"层 {max_diff_layer_idx} - 文件2 (humaneval) top-10 experts:")
    for i, (expert_id, count) in enumerate(top10_2, 1):
        print(f"  {i}. Expert {expert_id}: {count:,} 次激活")
    
    print()
    print(f"层 {max_diff_layer_idx} - 文件3 (gsm8k) top-10 experts:")
    for i, (expert_id, count) in enumerate(top10_3, 1):
        print(f"  {i}. Expert {expert_id}: {count:,} 次激活")
    
    print()
    
    return max_diff_layer_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析三个JSON文件中expert激活的差异，找出top-10差异最大的层")
    parser.add_argument("file1", help="第一个JSON文件路径")
    parser.add_argument("file2", help="第二个JSON文件路径")
    parser.add_argument("file3", help="第三个JSON文件路径")
    
    args = parser.parse_args()
    
    analyze_three_files(args.file1, args.file2, args.file3)

