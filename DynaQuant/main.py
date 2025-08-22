#!/usr/bin/env python3
"""
混合精度Transformer模型部署主程序
支持从不同精度文件中选择性加载权重参数，并进行混合精度推理
"""

import argparse
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.api_server import MixedPrecisionAPIServer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="混合精度Transformer模型部署")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/model_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="服务器主机地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="服务器端口"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="工作线程数"
    )
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 {args.config} 不存在")
        sys.exit(1)
    
    print("=" * 60)
    print("混合精度Transformer模型部署服务器")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"服务器地址: {args.host}:{args.port}")
    print(f"工作线程数: {args.workers}")
    print("=" * 60)
    
    try:
        # 创建并运行服务器
        server = MixedPrecisionAPIServer(args.config)
        
        # 更新服务器配置
        server.server_config['host'] = args.host
        server.server_config['port'] = args.port
        server.server_config['max_workers'] = args.workers
        
        print("启动服务器...")
        server.run()
        
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
