#!/usr/bin/env python3
"""
分析 coze_results.jsonl 文件中的评分数据
计算每个请求的平均分和整体平均分
"""

import json
import statistics
from typing import List, Dict, Any
from collections import defaultdict


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析失败: {e}")
                continue
    return data


def extract_scores_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """从单条记录中提取评分信息"""
    try:
        coze_result = record.get('coze_api_result', {})
        dimensions = coze_result.get('dimensions', [])
        overall_score = coze_result.get('overall_score', {})
        
        # 提取各个维度的评分
        dimension_scores = {}
        for dim in dimensions:
            dim_name = dim.get('dimension_name', '')
            score = dim.get('score', 0)
            dimension_scores[dim_name] = score
        
        # 提取整体评分信息
        overall_info = {
            'total_dimensions': overall_score.get('total_dimensions', 0),
            'total_score': overall_score.get('total_score', 0),
            'average_score': overall_score.get('average_score', 0),
            'max_score': overall_score.get('max_score', 0),
            'min_score': overall_score.get('min_score', 0)
        }
        
        return {
            'record_id': record.get('record_id', ''),
            'dimension_scores': dimension_scores,
            'overall_info': overall_info
        }
    except Exception as e:
        print(f"警告: 处理记录时出错: {e}")
        return None


def calculate_statistics(scores_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算统计信息"""
    if not scores_data:
        return {}
    
    # 收集所有维度名称
    all_dimensions = set()
    for record in scores_data:
        all_dimensions.update(record['dimension_scores'].keys())
    
    # 计算每个维度的统计信息
    dimension_stats = {}
    for dim_name in all_dimensions:
        scores = []
        for record in scores_data:
            if dim_name in record['dimension_scores']:
                scores.append(record['dimension_scores'][dim_name])
        
        if scores:
            dimension_stats[dim_name] = {
                'count': len(scores),
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'min': min(scores),
                'max': max(scores),
                'std': statistics.stdev(scores) if len(scores) > 1 else 0
            }
    
    # 计算整体评分的统计信息
    overall_scores = [record['overall_info']['average_score'] for record in scores_data]
    overall_stats = {
        'count': len(overall_scores),
        'mean': statistics.mean(overall_scores),
        'median': statistics.median(overall_scores),
        'min': min(overall_scores),
        'max': max(overall_scores),
        'std': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0
    }
    
    return {
        'dimension_stats': dimension_stats,
        'overall_stats': overall_stats,
        'total_records': len(scores_data)
    }


def print_results(stats: Dict[str, Any]):
    """打印分析结果"""
    print("=" * 80)
    print("Coze 评分数据分析结果")
    print("=" * 80)
    
    print(f"\n总记录数: {stats['total_records']}")
    
    # 打印整体评分统计
    print(f"\n整体评分统计:")
    print(f"  平均分: {stats['overall_stats']['mean']:.4f}")
    print(f"  中位数: {stats['overall_stats']['median']:.4f}")
    print(f"  最小值: {stats['overall_stats']['min']:.4f}")
    print(f"  最大值: {stats['overall_stats']['max']:.4f}")
    print(f"  标准差: {stats['overall_stats']['std']:.4f}")
    
    # 打印各维度评分统计
    print(f"\n各维度评分统计:")
    print("-" * 60)
    for dim_name, dim_stats in stats['dimension_stats'].items():
        print(f"{dim_name}:")
        print(f"  平均分: {dim_stats['mean']:.4f}")
        print(f"  中位数: {dim_stats['median']:.4f}")
        print(f"  最小值: {dim_stats['min']:.4f}")
        print(f"  最大值: {dim_stats['max']:.4f}")
        print(f"  标准差: {dim_stats['std']:.4f}")
        print(f"  样本数: {dim_stats['count']}")
        print()


def print_individual_scores(scores_data: List[Dict[str, Any]]):
    """打印每个请求的详细评分"""
    print("\n" + "=" * 80)
    print("各请求详细评分")
    print("=" * 80)
    
    for i, record in enumerate(scores_data, 1):
        print(f"\n请求 {i} (ID: {record['record_id']}):")
        print(f"  整体平均分: {record['overall_info']['average_score']:.4f}")
        print("  各维度评分:")
        for dim_name, score in record['dimension_scores'].items():
            print(f"    {dim_name}: {score}")


def main():
    """主函数"""
    file_path = "coze_results-fp16.jsonl"
    
    print("正在加载数据...")
    data = load_jsonl_data(file_path)
    print(f"成功加载 {len(data)} 条记录")
    
    print("正在提取评分数据...")
    scores_data = []
    for record in data:
        score_info = extract_scores_from_record(record)
        if score_info:
            scores_data.append(score_info)
    
    print(f"成功提取 {len(scores_data)} 条评分记录")
    
    if not scores_data:
        print("没有找到有效的评分数据")
        return
    
    print("正在计算统计信息...")
    stats = calculate_statistics(scores_data)
    
    # 打印结果
    print_results(stats)
    print_individual_scores(scores_data)
    
    # 保存结果到文件
    output_file = "score_analysis_results-fp16.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'statistics': stats,
            'individual_scores': scores_data
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
