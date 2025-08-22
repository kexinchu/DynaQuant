#!/usr/bin/env python3
"""
SGLang混合精度服务器启动脚本
支持从不同精度文件中选择性加载权重参数
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# 添加sglang路径
sys.path.insert(0, str(Path(__file__).parent / "python"))

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def load_mixed_precision_config(config_path: str) -> dict:
    """加载混合精度配置"""
    if not os.path.exists(config_path):
        print(f"警告: 混合精度配置文件 {config_path} 不存在")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def modify_server_args_for_mixed_precision(server_args, mixed_precision_config: dict):
    """修改服务器参数以支持混合精度"""
    if not mixed_precision_config:
        return server_args
    
    # 设置混合精度配置路径
    config_path = os.path.abspath(mixed_precision_config.get('config_path', 'mixed_precision_config.yaml'))
    
    # 修改模型配置以包含混合精度配置
    if hasattr(server_args, 'model_config'):
        server_args.model_config.mixed_precision_config = config_path
    
    print(f"混合精度配置已加载: {config_path}")
    return server_args


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SGLang混合精度服务器")
    parser.add_argument(
        "--mixed-precision-config",
        type=str,
        default="mixed_precision_config.yaml",
        help="混合精度配置文件路径"
    )
    parser.add_argument(
        "--enable-mixed-precision",
        action="store_true",
        help="启用混合精度加载"
    )
    
    # 解析混合精度相关参数
    mixed_precision_args, remaining_args = parser.parse_known_args()
    
    # 加载混合精度配置
    mixed_precision_config = {}
    if mixed_precision_args.enable_mixed_precision:
        mixed_precision_config = load_mixed_precision_config(mixed_precision_args.mixed_precision_config)
        if mixed_precision_config:
            print("=" * 60)
            print("混合精度配置已启用")
            print("=" * 60)
            print(f"FP16路径: {mixed_precision_config.get('mixed_precision', {}).get('fp16_path', 'N/A')}")
            print(f"FP8路径: {mixed_precision_config.get('mixed_precision', {}).get('fp8_path', 'N/A')}")
            print(f"Int4路径: {mixed_precision_config.get('mixed_precision', {}).get('int4_path', 'N/A')}")
            print(f"权重映射数量: {len(mixed_precision_config.get('mixed_precision', {}).get('weight_mapping', {}))}")
            print("=" * 60)
    
    # 准备服务器参数
    server_args = prepare_server_args(remaining_args)
    
    # 如果启用了混合精度，修改服务器参数
    if mixed_precision_args.enable_mixed_precision and mixed_precision_config:
        server_args = modify_server_args_for_mixed_precision(server_args, mixed_precision_config)
    
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
