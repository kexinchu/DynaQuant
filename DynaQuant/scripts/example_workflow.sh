#!/bin/bash
#
# DynaQuant 完整工作流程示例
# 从量化到评估的完整流程
#

set -e

# 配置参数
MODEL_PATH="/dev/shm/Qwen3-30B-A3B"
CALIB_DATA="calibration_datasets/Qwen3-30B-A3B/calibration_Qwen3-30B-A3B.json"
DATA_DIR="data"

echo "=========================================="
echo "DynaQuant 量化与评估工作流程"
echo "=========================================="
echo ""

# Step 1: 检查环境
echo "Step 1: 检查环境..."
if ! python -c "import llmcompressor" 2>/dev/null; then
    echo "❌ llm-compressor 未安装，正在安装..."
    pip install llm-compressor
else
    echo "✓ llm-compressor 已安装"
fi

if ! python -c "import datasets" 2>/dev/null; then
    echo "❌ datasets 未安装，正在安装..."
    pip install datasets pandas
else
    echo "✓ datasets 已安装"
fi

echo ""

# Step 2: 评估原始模型（可选）
if [ -d "$MODEL_PATH" ]; then
    echo "Step 2: 评估原始模型（baseline）..."
    echo "跳过原始模型评估（耗时较长），如需评估请取消注释下面的命令"
    # python evaluate_model.py \
    #     --model "$MODEL_PATH" \
    #     --datasets wikitext mmlu \
    #     --output "eval_results_original.json"
    echo ""
else
    echo "⚠ 原始模型不存在: $MODEL_PATH"
    echo "请修改脚本中的 MODEL_PATH 参数"
    exit 1
fi

# Step 3: W4A16 量化
echo "Step 3: 执行 W4A16 量化..."
OUTPUT_DIR_W4A16="/dev/shm/Qwen3-30B-A3B-W4A16"

if [ ! -d "$OUTPUT_DIR_W4A16" ]; then
    echo "开始量化: $MODEL_PATH -> $OUTPUT_DIR_W4A16"
    python quantize_w4a16.py \
        --model "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR_W4A16" \
        --calib-data "$CALIB_DATA" \
        --num-samples 512 \
        --max-seq-length 8192
    echo "✓ W4A16 量化完成"
else
    echo "✓ W4A16 模型已存在，跳过量化: $OUTPUT_DIR_W4A16"
fi

echo ""

# Step 4: W2A16 量化（可选）
echo "Step 4: 执行 W2A16 量化（可选）..."
OUTPUT_DIR_W2A16="/dev/shm/Qwen3-30B-A3B-W2A16"

if [ ! -d "$OUTPUT_DIR_W2A16" ]; then
    read -p "是否执行 W2A16 量化？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "开始量化: $MODEL_PATH -> $OUTPUT_DIR_W2A16"
        python quantize_w2a16.py \
            --model "$MODEL_PATH" \
            --output-dir "$OUTPUT_DIR_W2A16" \
            --calib-data "$CALIB_DATA" \
            --num-samples 512 \
            --max-seq-length 8192
        echo "✓ W2A16 量化完成"
    else
        echo "跳过 W2A16 量化"
    fi
else
    echo "✓ W2A16 模型已存在，跳过量化: $OUTPUT_DIR_W2A16"
fi

echo ""

# Step 5: 评估 W4A16 模型
echo "Step 5: 评估 W4A16 量化模型..."
if [ -d "$OUTPUT_DIR_W4A16" ]; then
    python evaluate_model.py \
        --model "$OUTPUT_DIR_W4A16" \
        --datasets wikitext mmlu gsm8k \
        --output "eval_results_w4a16.json" \
        --data-dir "$DATA_DIR"
    echo "✓ W4A16 评估完成: eval_results_w4a16.json"
else
    echo "⚠ W4A16 模型不存在，跳过评估"
fi

echo ""

# Step 6: 评估 W2A16 模型（如果存在）
if [ -d "$OUTPUT_DIR_W2A16" ]; then
    echo "Step 6: 评估 W2A16 量化模型..."
    python evaluate_model.py \
        --model "$OUTPUT_DIR_W2A16" \
        --datasets wikitext mmlu gsm8k \
        --output "eval_results_w2a16.json" \
        --data-dir "$DATA_DIR"
    echo "✓ W2A16 评估完成: eval_results_w2a16.json"
fi

echo ""

# Step 7: 显示结果摘要
echo "=========================================="
echo "完成！结果摘要："
echo "=========================================="
echo ""

if [ -f "eval_results_w4a16.json" ]; then
    echo "W4A16 量化结果："
    python -c "
import json
with open('eval_results_w4a16.json') as f:
    data = json.load(f)
    evals = data.get('evaluations', {})
    if 'wikitext' in evals:
        print(f\"  WikiText PPL: {evals['wikitext'].get('perplexity', 'N/A'):.2f}\")
    if 'mmlu' in evals:
        print(f\"  MMLU Accuracy: {evals['mmlu'].get('overall_accuracy', 'N/A'):.4f}\")
    if 'gsm8k' in evals:
        print(f\"  GSM8K Accuracy: {evals['gsm8k'].get('accuracy', 'N/A'):.4f}\")
" || echo "  (结果文件解析失败)"
    echo ""
fi

if [ -f "eval_results_w2a16.json" ]; then
    echo "W2A16 量化结果："
    python -c "
import json
with open('eval_results_w2a16.json') as f:
    data = json.load(f)
    evals = data.get('evaluations', {})
    if 'wikitext' in evals:
        print(f\"  WikiText PPL: {evals['wikitext'].get('perplexity', 'N/A'):.2f}\")
    if 'mmlu' in evals:
        print(f\"  MMLU Accuracy: {evals['mmlu'].get('overall_accuracy', 'N/A'):.4f}\")
    if 'gsm8k' in evals:
        print(f\"  GSM8K Accuracy: {evals['gsm8k'].get('accuracy', 'N/A'):.4f}\")
" || echo "  (结果文件解析失败)"
    echo ""
fi

echo "详细结果保存在："
[ -f "eval_results_w4a16.json" ] && echo "  - eval_results_w4a16.json"
[ -f "eval_results_w2a16.json" ] && echo "  - eval_results_w2a16.json"

echo ""
echo "量化模型保存在："
[ -d "$OUTPUT_DIR_W4A16" ] && echo "  - $OUTPUT_DIR_W4A16"
[ -d "$OUTPUT_DIR_W2A16" ] && echo "  - $OUTPUT_DIR_W2A16"

echo ""
echo "=========================================="
echo "工作流程完成！"
echo "=========================================="

