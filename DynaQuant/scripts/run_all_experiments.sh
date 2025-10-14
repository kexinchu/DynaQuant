#!/bin/bash

# 运行所有6组实验的完整脚本

echo "=========================================="
echo "Expert Activation Analysis - Full Suite"
echo "=========================================="
echo ""
echo "这将运行所有6组实验:"
echo "  - WikiText (thinking off/on)"
echo "  - GSM8K (thinking off/on)"
echo "  - HumanEval (thinking off/on)"
echo ""
echo "每个数据集 256 个样本"
echo "预计运行时间: 1.5-4小时（取决于硬件）"
echo ""

# 询问用户确认
read -p "继续运行? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消运行"
    exit 0
fi

# 记录开始时间
START_TIME=$(date +%s)
echo ""
echo "开始时间: $(date)"
echo ""

# 运行所有实验
python3 scripts/collect_expert_activation.py --all

# 检查是否成功
if [ $? -ne 0 ]; then
    echo ""
    echo "错误: 实验运行失败"
    exit 1
fi

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "=========================================="
echo "结束时间: $(date)"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo ""
echo "开始分析结果..."
echo ""

# 分析结果
python3 scripts/analyze_expert_activation.py

echo ""
echo "=========================================="
echo "分析完成！"
echo "=========================================="
echo ""
echo "结果文件位于: ./benchmark_results/expert_activation_results/"
echo ""
echo "查看具体实验结果:"
echo "  ls -lh ./benchmark_results/expert_activation_results/"

