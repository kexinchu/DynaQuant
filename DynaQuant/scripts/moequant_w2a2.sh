#!/bin/bash
#
# MoEQuant W2A2 量化脚本
# 基于 MoEQuant 论文的 EBSS + AGQ 实现
# 
# W2A2: 2位权重 + 2位激活
# 压缩比: ~8x, 精度损失: 3-5%
#

set -e

# 默认参数
MODEL="/dev/shm/Qwen3-30B-A3B"
OUTPUT_DIR="/dev/shm/Qwen3-30B-A3B-W2A2"
CALIB_SIZE=1024
EBSS_BEAM_WIDTH=8  # W2A2需要更大的beam width
EBSS_TAU=1.5  # W2A2需要更高的温度
EBSS_MAX_TOKENS=1024
AGQ_GROUP_SIZE=64  # W2A2使用较小的group size
SEED_TEXT="data/seed_text.txt"
CALIB_BATCH_SIZE=1
USE_AGQ_ERROR_COMP=1  # W2A2强烈推荐使用误差补偿
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
echo "MoEQuant W2A2 量化 (极限压缩)"
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
echo ""
echo "⚠️  W2A2是极限压缩配置，可能导致较大精度损失"
echo "    建议先尝试W4A4配置"
echo ""

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

# W2A2建议使用更多校准样本
if [ $CALIB_SIZE -lt 1024 ]; then
    echo "警告: W2A2建议使用至少1024个校准样本"
    echo "      当前设置: $CALIB_SIZE"
    read -p "是否继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 1
    fi
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查GPU
if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "可用GPU数量: $NUM_GPUS"
    
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "GPU内存: ${GPU_MEMORY}MB"
    
    # W2A2建议至少40GB内存
    if [ "$GPU_MEMORY" -lt 40000 ]; then
        echo "警告: GPU内存 (${GPU_MEMORY}MB) 可能不足以量化30B模型"
        echo "建议使用至少40GB内存的GPU"
    fi
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
        --precision w2a2 \
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
        echo "警告: 禁用误差补偿可能导致W2A2精度显著下降"
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
        --precision w2a2 \
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
        echo "警告: 禁用误差补偿可能导致W2A2精度显著下降"
    fi

    if [ -n "$AGQ_GROUP_SIZE" ]; then
        CMD="$CMD --agq-group-size $AGQ_GROUP_SIZE"
    fi
fi

# 运行量化
echo ""
echo "开始W2A2量化..."
echo "命令: $CMD"
echo ""
echo "注意: W2A2量化可能需要较长时间 (30-60分钟)"
echo ""

eval $CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ W2A2量化完成!"
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
    print(f\"  权重位宽: {summary['quantization_config']['weight_bits']}\")
    print(f\"  激活位宽: {summary['quantization_config']['activation_bits']}\")
    print(f\"  分组大小: {summary['quantization_config']['group_size']}\")
    stats = summary.get('quantization_stats_summary', {})
    print(f\"  成功量化: {stats.get('successful_layers', 0)} 层\")
    print(f\"  失败: {stats.get('failed_layers', 0)} 层\")
" 2>/dev/null || true
    fi
    
    echo ""
    echo "性能指标 (预期):"
    echo "  模型大小: ~7-8GB (原始 ~60GB, 压缩比 ~8x)"
    echo "  精度损失: 3-5%"
    echo "  推理速度: 提升 3-4x"
    echo ""
    echo "⚠️  重要: 请务必评估量化后的模型质量"
    echo ""
    echo "下一步:"
    echo "1. 测试量化模型:"
    echo "   python3 -m transformers.models.auto --model \"$OUTPUT_DIR\""
    echo ""
    echo "2. 评估模型质量 (必须!):"
    echo "   python3 -m moe_quant.eval --model \"$OUTPUT_DIR\" --reference \"$MODEL\""
    echo ""
    echo "3. 如果精度不满意, 尝试:"
    echo "   - 增加校准样本数 (--calib-size 1024)"
    echo "   - 增大EBSS beam width (--ebss-beam 16)"
    echo "   - 或使用W4A4配置"
else
    echo ""
    echo "========================================"
    echo "❌ W2A2量化失败 (退出码: $EXIT_CODE)"
    echo "========================================"
    echo "请检查日志获取详细信息"
    echo ""
    echo "常见问题:"
    echo "- 内存不足: 尝试减少校准样本数或使用更大的GPU"
    echo "- 数值不稳定: 确保启用AGQ误差补偿"
    echo "- 如果问题persist, 建议使用W4A4配置"
    exit $EXIT_CODE
fi

