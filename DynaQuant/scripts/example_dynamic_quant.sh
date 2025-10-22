#!/bin/bash
#
# 动态量化MoE示例脚本
# 演示如何使用动态量化系统评估MoE模型
#

set -e

# ==================== 配置参数 ====================
FP16_MODEL="/dev/shm/Qwen3-30B-A3B"
INT4_MODEL="/dev/shm/Qwen3-30B-A3B-INT4"
DATA_DIR="./data"
OUTPUT_DIR="./eval_results_dynamic_quant"

# 时间窗口和hot比例
TIME_WINDOW=20.0
HOT_RATIO=0.1

# 数据集样本数
NUM_SAMPLES_WIKITEXT=1000
NUM_SAMPLES_MMLU=100
NUM_SAMPLES_GSM8K=100
NUM_SAMPLES_HELLASWAG=100

# GPU数量（设置为0则使用所有可用GPU）
NUM_GPUS=8

# ==================== 创建输出目录 ====================
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "动态量化MoE评估示例"
echo "=========================================="
echo "FP16模型: $FP16_MODEL"
echo "INT4模型: $INT4_MODEL"
echo "时间窗口: ${TIME_WINDOW}s"
echo "Hot比例: ${HOT_RATIO} (10%)"
echo "GPU数量: $NUM_GPUS"
echo "=========================================="
echo ""

# ==================== 方案1：多GPU并行评估（推荐） ====================
echo ">>> 方案1：多GPU并行评估（最快，推荐8×H20环境）"
echo ""

python scripts/evaluate_parallel_dynamic_quant.py \
    --fp16-model $FP16_MODEL \
    --int4-model $INT4_MODEL \
    --datasets wikitext mmlu gsm8k hellaswag \
    --num-gpus $NUM_GPUS \
    --time-window $TIME_WINDOW \
    --hot-ratio $HOT_RATIO \
    --num-samples-wikitext $NUM_SAMPLES_WIKITEXT \
    --num-samples-mmlu $NUM_SAMPLES_MMLU \
    --num-samples-gsm8k $NUM_SAMPLES_GSM8K \
    --num-samples-hellaswag $NUM_SAMPLES_HELLASWAG \
    --data-dir $DATA_DIR \
    --output $OUTPUT_DIR/results_parallel_dynamic.json

echo ""
echo "✓ 多GPU并行评估完成！"
echo ""

# ==================== 方案2：单GPU评估 ====================
echo ">>> 方案2：单GPU评估（适合单卡环境）"
echo ""

python scripts/evaluate_dynamic_quant.py \
    --fp16-model $FP16_MODEL \
    --int4-model $INT4_MODEL \
    --datasets wikitext mmlu \
    --time-window $TIME_WINDOW \
    --hot-ratio $HOT_RATIO \
    --num-samples-wikitext 500 \
    --num-samples-mmlu 50 \
    --data-dir $DATA_DIR \
    --output $OUTPUT_DIR/results_single_dynamic.json

echo ""
echo "✓ 单GPU评估完成！"
echo ""

# ==================== 方案3：基准测试（禁用动态路由） ====================
echo ">>> 方案3：基准测试（全FP16，用于对比）"
echo ""

python scripts/evaluate_dynamic_quant.py \
    --fp16-model $FP16_MODEL \
    --int4-model $INT4_MODEL \
    --datasets wikitext mmlu \
    --disable-dynamic-routing \
    --num-samples-wikitext 500 \
    --num-samples-mmlu 50 \
    --data-dir $DATA_DIR \
    --output $OUTPUT_DIR/results_baseline_fp16.json

echo ""
echo "✓ 基准测试完成！"
echo ""

# ==================== 结果对比 ====================
echo "=========================================="
echo "评估完成！结果保存在: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "查看结果："
echo "  - 多GPU并行: $OUTPUT_DIR/results_parallel_dynamic.json"
echo "  - 单GPU动态: $OUTPUT_DIR/results_single_dynamic.json"
echo "  - FP16基准: $OUTPUT_DIR/results_baseline_fp16.json"
echo ""

# 打印关键指标对比
echo "关键指标对比："
echo ""

if [ -f "$OUTPUT_DIR/results_parallel_dynamic.json" ]; then
    echo "=== 多GPU并行动态量化 ==="
    python3 -c "
import json
with open('$OUTPUT_DIR/results_parallel_dynamic.json') as f:
    data = json.load(f)
    print(f\"  评估时间: {data.get('total_evaluation_time', 0):.2f}s\")
    if 'wikitext' in data.get('evaluations', {}):
        ppl = data['evaluations']['wikitext'].get('perplexity', 0)
        print(f\"  WikiText PPL: {ppl:.2f}\")
    if 'mmlu' in data.get('evaluations', {}):
        acc = data['evaluations']['mmlu'].get('overall_accuracy', 0)
        print(f\"  MMLU Accuracy: {acc:.4f}\")
" 2>/dev/null || echo "  (结果文件解析失败)"
    echo ""
fi

if [ -f "$OUTPUT_DIR/results_baseline_fp16.json" ]; then
    echo "=== FP16基准 ==="
    python3 -c "
import json
with open('$OUTPUT_DIR/results_baseline_fp16.json') as f:
    data = json.load(f)
    if 'wikitext' in data.get('evaluations', {}):
        ppl = data['evaluations']['wikitext'].get('perplexity', 0)
        print(f\"  WikiText PPL: {ppl:.2f}\")
    if 'mmlu' in data.get('evaluations', {}):
        acc = data['evaluations']['mmlu'].get('overall_accuracy', 0)
        print(f\"  MMLU Accuracy: {acc:.4f}\")
" 2>/dev/null || echo "  (结果文件解析失败)"
    echo ""
fi

echo "=========================================="
echo "使用提示："
echo "  1. 对于8×H20环境，推荐使用多GPU并行评估（方案1）"
echo "  2. 调整--hot-ratio可以改变hot/cold专家比例"
echo "  3. 调整--time-window可以改变统计时间窗口"
echo "  4. 使用--disable-dynamic-routing可以禁用动态路由进行基准测试"
echo "=========================================="

