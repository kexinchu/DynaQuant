#!/bin/bash
#
# MoEQuant W8A8 量化脚本
# 基于 MoEQuant 论文的 EBSS + AGQ 实现
# 
# W8A8: 8位权重 + 8位激活
# 压缩比: ~2x, 精度损失: <1%
#

set -e

# 默认参数
MODEL="/dev/shm/Qwen3-30B-A3B"
OUTPUT_DIR="/dev/shm/Qwen3-30B-A3B-W8A8"
CALIB_SIZE=1024
EBSS_BEAM_WIDTH=4
EBSS_TAU=1.2
EBSS_MAX_TOKENS=1024
AGQ_GROUP_SIZE=128
SEED_TEXT="calibration_datasets/Qwen3-30B-A3B/calibration_Qwen3-30B-A3B.txt"
CALIB_BATCH_SIZE=1
USE_AGQ_ERROR_COMP=1
USE_MULTI_GPU=0
NUM_GPUS=8
GPU_IDS=""  # 逗号分隔的GPU IDs，如 "0,1,2,3"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --calib-size)
            CALIB_SIZE="$2"
            shift 2
            ;;
        --ebss-beam)
            EBSS_BEAM_WIDTH="$2"
            shift 2
            ;;
        --ebss-tau)
            EBSS_TAU="$2"
            shift 2
            ;;
        --ebss-max-tokens)
            EBSS_MAX_TOKENS="$2"
            shift 2
            ;;
        --agq-group-size)
            AGQ_GROUP_SIZE="$2"
            shift 2
            ;;
        --seed-text)
            SEED_TEXT="$2"
            shift 2
            ;;
        --calib-batch-size)
            CALIB_BATCH_SIZE="$2"
            shift 2
            ;;
        --no-agq-error-comp)
            USE_AGQ_ERROR_COMP=0
            shift
            ;;
        --multi-gpu)
            USE_MULTI_GPU=1
            shift
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            USE_MULTI_GPU=1
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            USE_MULTI_GPU=1
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "MoEQuant W8A8 量化"
echo "========================================"
echo "模型: $MODEL"
echo "输出目录: $OUTPUT_DIR"
echo "校准样本数: $CALIB_SIZE"
echo "EBSS Beam Width: $EBSS_BEAM_WIDTH"
echo "EBSS Tau: $EBSS_TAU"
echo "EBSS Max Tokens: $EBSS_MAX_TOKENS"
echo "AGQ Group Size: $AGQ_GROUP_SIZE"
echo "AGQ Error Compensation: $USE_AGQ_ERROR_COMP"
echo "多GPU模式: $USE_MULTI_GPU"
if [ "$USE_MULTI_GPU" -eq 1 ]; then
    echo "GPU数量: $NUM_GPUS"
    if [ -n "$GPU_IDS" ]; then
        echo "指定GPU: $GPU_IDS"
    fi
fi
echo "种子文本: $SEED_TEXT"
echo "========================================"

# 检查模型是否存在
if [ ! -d "$MODEL" ]; then
    echo "错误: 模型目录不存在: $MODEL"
    exit 1
fi

# 检查种子文本
if [ ! -f "$SEED_TEXT" ] && [ "$SEED_TEXT" != "data/seed_text.txt" ]; then
    echo "警告: 种子文本文件不存在: $SEED_TEXT, 将使用默认文本"
    SEED_TEXT=""
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查GPU
if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "可用GPU数量: $NUM_GPUS"
    
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "GPU内存: ${GPU_MEMORY}MB"
else
    echo "警告: 未检测到NVIDIA GPU"
fi

# 选择运行模式
if [ "$USE_MULTI_GPU" -eq 1 ]; then
    echo ""
    echo "使用多GPU模式（$NUM_GPUS 个GPU）"
    echo ""
    
    # 设置CUDA可见设备
    if [ -n "$GPU_IDS" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        echo "设置 CUDA_VISIBLE_DEVICES=$GPU_IDS"
    else
        # 自动选择前N个GPU
        GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        echo "使用GPU: $GPU_IDS"
    fi
    
    # 使用多GPU模式（在moequant_ptq_runner中支持）
    CMD="python3 -m moe_quant.runners.moequant_ptq_runner \
        --model \"$MODEL\" \
        --output-dir \"$OUTPUT_DIR\" \
        --precision w8a8 \
        --seed-text \"$SEED_TEXT\" \
        --calib-size $CALIB_SIZE \
        --ebss-beam-width $EBSS_BEAM_WIDTH \
        --ebss-tau $EBSS_TAU \
        --ebss-max-tokens $EBSS_MAX_TOKENS \
        --calibration-batch-size $CALIB_BATCH_SIZE \
        --multi-gpu"
    
    if [ -n "$GPU_IDS" ]; then
        CMD="$CMD --gpu-ids \"$GPU_IDS\""
    fi
    
    if [ "$USE_AGQ_ERROR_COMP" -eq 0 ]; then
        CMD="$CMD --no-agq-error-compensation"
    fi
    
    if [ -n "$AGQ_GROUP_SIZE" ]; then
        CMD="$CMD --agq-group-size $AGQ_GROUP_SIZE"
    fi
else
    echo ""
    echo "使用单GPU模式"
    echo ""
    
    # 构建命令
    CMD="python3 -m moe_quant.runners.moequant_ptq_runner \
        --model \"$MODEL\" \
        --output-dir \"$OUTPUT_DIR\" \
        --precision w8a8 \
        --calib-size $CALIB_SIZE \
        --ebss-beam-width $EBSS_BEAM_WIDTH \
        --ebss-tau $EBSS_TAU \
        --ebss-max-tokens $EBSS_MAX_TOKENS \
        --calibration-batch-size $CALIB_BATCH_SIZE"

    # 添加可选参数
    if [ -n "$SEED_TEXT" ]; then
        CMD="$CMD --seed-text \"$SEED_TEXT\""
    fi

    if [ "$USE_AGQ_ERROR_COMP" -eq 0 ]; then
        CMD="$CMD --no-agq-error-compensation"
    fi

    if [ -n "$AGQ_GROUP_SIZE" ]; then
        CMD="$CMD --agq-group-size $AGQ_GROUP_SIZE"
    fi
fi

# 运行量化
echo ""
echo "开始W8A8量化..."
echo "命令: $CMD"
echo ""

eval $CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ W8A8量化完成!"
    echo "========================================"
    echo "输出目录: $OUTPUT_DIR"
    echo ""
    echo "模型文件:"
    ls -lh "$OUTPUT_DIR"/*.safetensors 2>/dev/null | head -5
    if [ $(ls "$OUTPUT_DIR"/*.safetensors 2>/dev/null | wc -l) -gt 5 ]; then
        echo "... 以及 $(ls "$OUTPUT_DIR"/*.safetensors 2>/dev/null | wc -l | awk '{print $1-5}') 个其他文件"
    fi
    echo ""
    
    # 显示量化统计
    if [ -f "$OUTPUT_DIR/quantization_summary.json" ]; then
        echo "量化摘要:"
        python3 -c "
import json
with open('$OUTPUT_DIR/quantization_summary.json', 'r') as f:
    summary = json.load(f)
    print(f\"  精度: {summary['precision'].upper()}\")
    stats = summary.get('quantization_stats_summary', {})
    print(f\"  成功量化: {stats.get('successful_layers', 0)} 层\")
    print(f\"  失败: {stats.get('failed_layers', 0)} 层\")
" 2>/dev/null || true
    fi
    
    echo ""
    echo "下一步:"
    echo "1. 测试量化模型:"
    echo "   python3 -m transformers.models.auto --model \"$OUTPUT_DIR\""
    echo ""
    echo "2. 评估模型质量:"
    echo "   python3 -m moe_quant.eval --model \"$OUTPUT_DIR\" --reference \"$MODEL\""
else
    echo ""
    echo "========================================"
    echo "❌ W8A8量化失败 (退出码: $EXIT_CODE)"
    echo "========================================"
    echo "请检查日志获取详细信息"
    exit $EXIT_CODE
fi

