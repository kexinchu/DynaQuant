#!/usr/bin/env python3
"""
Coze API 分析程序运行脚本
从配置文件读取配置并运行分析
"""

import json
import os
import sys
from coze_api_processor import main as coze_main

def load_config(config_file: str = "coze_config.json") -> dict:
    """加载配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python run_coze_analysis.py <输入文件> [输出文件] [摘要文件]")
        print("示例: python run_coze_analysis.py test_results.jsonl")
        print("示例: python run_coze_analysis.py test_results.jsonl coze_results.jsonl summary.json")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "coze_results.jsonl"
    summary_file = sys.argv[3] if len(sys.argv) > 3 else "summary_report.json"
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 加载配置
    config = load_config()
    if not config:
        print("错误: 无法加载配置文件")
        return
    
    # 构建命令行参数
    sys.argv = [
        'coze_api_processor.py',
        '--input', input_file,
        '--output', output_file,
        '--summary', summary_file,
        '--api-key', config['api_key'],
        '--workflow-id', config['workflow_id'],
        '--delay', str(config.get('default_delay', 2.0))
    ]
    
    # 运行主程序
    try:
        coze_main()
    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
