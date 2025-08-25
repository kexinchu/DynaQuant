#!/usr/bin/env python3
"""
兼容SGLang原生启动方式的混合精度启动脚本
支持通过sglang.launch_server命令启动混合精度推理
"""

import os
import sys
import argparse
from pathlib import Path

# 添加SGLang路径
sys.path.insert(0, str(Path(__file__).parent / "python"))

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def main():
    """主函数 - 兼容SGLang原生启动方式"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="SGLang Mixed Precision Server")
    
    # 添加混合精度相关参数
    parser.add_argument(
        "--enable-mixed-precision",
        action="store_true",
        help="Enable mixed precision loading for selective weight quantization."
    )
    parser.add_argument(
        "--mixed-precision-config",
        type=str,
        help="Path to the mixed precision configuration file (YAML format). "
        "This file specifies which model layers should use which precision "
        "(FP16, FP8, Int4, etc.) for optimal memory usage and performance."
    )
    
    # 解析混合精度参数
    mixed_precision_args, remaining_args = parser.parse_known_args()
    
    # 如果启用了混合精度，显示配置信息
    if mixed_precision_args.enable_mixed_precision:
        print("=" * 60)
        print("混合精度配置已启用")
        print("=" * 60)
        print(f"配置文件: {mixed_precision_args.mixed_precision_config}")
        print("=" * 60)
        
        # 验证配置文件
        if not mixed_precision_args.mixed_precision_config:
            print("错误: 启用混合精度时必须指定配置文件路径 (--mixed-precision-config)")
            sys.exit(1)
        
        if not os.path.exists(mixed_precision_args.mixed_precision_config):
            print(f"错误: 混合精度配置文件不存在: {mixed_precision_args.mixed_precision_config}")
            sys.exit(1)
    
    # 准备服务器参数
    server_args = prepare_server_args(remaining_args)
    
    # 如果启用了混合精度，设置相关参数
    if mixed_precision_args.enable_mixed_precision:
        server_args.enable_mixed_precision = True
        server_args.mixed_precision_config = mixed_precision_args.mixed_precision_config
        
        print(f"混合精度配置已设置:")
        print(f"  启用状态: {server_args.enable_mixed_precision}")
        print(f"  配置文件: {server_args.mixed_precision_config}")
    
    try:
        # 启动服务器
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
