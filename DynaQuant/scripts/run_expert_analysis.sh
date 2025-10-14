#!/bin/bash

# Expert Activation Analysis 快速启动脚本

echo "=========================================="
echo "Expert Activation Analysis"
echo "=========================================="
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python环境"
    exit 1
fi

echo "Python版本:"
python --version
echo ""

# 检查必要的包
echo "检查依赖包..."
python -c "import torch; import transformers; import datasets; import tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: 部分依赖包未安装"
    echo "正在安装依赖..."
    pip install torch transformers datasets tqdm
fi

echo ""
echo "开始运行专家激活分析..."
echo "注意: 这可能需要较长时间（取决于GPU性能和网络速度）"
echo ""

# 运行分析脚本
cd "$(dirname "$0")"
python collect_expert_activation.py

# 检查是否成功生成结果
if [ -d "expert_activation_results" ]; then
    echo ""
    echo "=========================================="
    echo "分析完成！"
    echo "=========================================="
    echo ""
    echo "生成的文件:"
    ls -lh expert_activation_results/*.json
    echo ""
    echo "运行以下命令查看分析结果:"
    echo "  python analyze_expert_activation.py"
else
    echo ""
    echo "警告: 未找到结果目录，分析可能未成功完成"
fi

