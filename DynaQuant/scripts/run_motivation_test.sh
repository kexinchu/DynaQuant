#!/bin/bash

# Motivation Test: 量化MoE对Router激活模式的影响
# 测试FP16 vs Int4量化如何影响expert selection

echo "=========================================="
echo "Motivation Test - Quantization Impact on Router"
echo "=========================================="
echo ""
echo "本测试将运行3组对比实验:"
echo "  对照组:  所有层使用FP16（包括experts）"
echo "  实验组1: Experts统一使用Int4，其余FP16"
echo "  实验组2: Hot 10% experts用FP16，90% cold experts用Int4"
echo ""
echo "模型: Qwen3-30B-A3B"
echo "数据集: GSM8K (数学推理)"
echo "样本数: 256"
echo "输出格式: JSON (兼容expert activation格式)"
echo "预计运行时间: 1-3小时（取决于硬件）"
echo ""

# 检查必要的模型路径
FP16_MODEL="/dev/shm/Qwen3-30B-A3B"
INT4_MODEL="/dev/shm/Qwen3-30B-A3B-AWQ"

echo ""
echo "检查模型路径..."
echo "FP16 模型: ${FP16_MODEL}"
echo "INT4 模型: ${INT4_MODEL}"
echo ""

# 检查expert激活统计文件是否存在（用于实验组2）
EXPERT_STATS_DIR="./benchmark_results/expert_activation_results"
if [ ! -d "$EXPERT_STATS_DIR" ]; then
    echo "警告: Expert激活统计目录不存在: $EXPERT_STATS_DIR"
    echo "实验组2需要先运行 expert activation 分析"
    echo ""
    read -p "是否先运行 expert activation 分析? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "运行 expert activation 分析..."
        bash scripts/run_collect_experts.sh
        if [ $? -ne 0 ]; then
            echo "Expert activation 分析失败，退出"
            exit 1
        fi
    else
        echo "跳过实验组2（需要expert统计数据）"
        SKIP_GROUP2=1
    fi
fi

# 记录开始时间
START_TIME=$(date +%s)
echo ""
echo "开始时间: $(date)"
echo ""

# 创建结果目录
RESULT_DIR="./benchmark_results/motivation_test"
mkdir -p ${RESULT_DIR}

# echo "=========================================="
# echo "对照组: 全FP16模型"
# echo "=========================================="
# python3 scripts/motivation_test.py \
#     --model_path ${FP16_MODEL} \
#     --test_group control \
#     --output_dir ${RESULT_DIR}/control_fp16 \
#     --num_samples 256

# if [ $? -ne 0 ]; then
#     echo "错误: 对照组运行失败"
#     exit 1
# fi

echo ""
echo "=========================================="
echo "实验组1: Experts全部Int4"
echo "=========================================="
python3 scripts/motivation_test.py \
    --model_path ${FP16_MODEL} \
    --int4_model_path ${INT4_MODEL} \
    --test_group exp1_all_int4 \
    --output_dir ${RESULT_DIR}/exp1_all_int4 \
    --num_samples 256 \
    --quantize_experts all

if [ $? -ne 0 ]; then
    echo "错误: 实验组1运行失败"
    exit 1
fi

if [ -z "$SKIP_GROUP2" ]; then
    echo ""
    echo "=========================================="
    echo "实验组2: Hot 10% FP16 + Cold 90% Int4"
    echo "=========================================="
    python3 scripts/motivation_test.py \
        --model_path ${FP16_MODEL} \
        --int4_model_path ${INT4_MODEL} \
        --test_group exp2_mixed \
        --output_dir ${RESULT_DIR}/exp2_mixed_hot10 \
        --num_samples 256 \
        --quantize_experts mixed \
        --hot_expert_ratio 0.1 \
        --expert_stats_dir ${EXPERT_STATS_DIR}
    
    if [ $? -ne 0 ]; then
        echo "错误: 实验组2运行失败"
        exit 1
    fi
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
echo "结果文件位于: ${RESULT_DIR}/"
echo ""
echo "JSON输出文件 (与expert activation格式兼容):"
echo "  - ${RESULT_DIR}/control_fp16/gsm8k_control.json"
echo "  - ${RESULT_DIR}/exp1_all_int4/gsm8k_exp1_all_int4.json"
if [ -z "$SKIP_GROUP2" ]; then
echo "  - ${RESULT_DIR}/exp2_mixed_hot10/gsm8k_exp2_mixed.json"
fi
echo ""
echo "这些JSON文件可以直接用之前的分析工具进行可视化:"
echo "  python3 scripts/analyze_expert_activation.py --results-dir ${RESULT_DIR}/control_fp16"
echo "  python3 scripts/analyze_expert_activation.py --results-dir ${RESULT_DIR}/exp1_all_int4"
if [ -z "$SKIP_GROUP2" ]; then
echo "  python3 scripts/analyze_expert_activation.py --results-dir ${RESULT_DIR}/exp2_mixed_hot10"
fi
echo ""
echo "对比不同实验组:"
echo "  python3 scripts/analyze_expert_activation.py \\"
echo "    --compare gsm8k_control gsm8k_exp1_all_int4 \\"
echo "    --results-dir ${RESULT_DIR}"
echo ""

