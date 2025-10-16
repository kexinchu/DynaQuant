#!/bin/bash
#
# 批量生成EBSS数据集脚本
# 为 Qwen3-30B-A3B, Qwen3-Next-80B-A3B, Qwen3-235B-A22B 生成EBSS校准数据
#

set -e

# 默认参数
BASE_MODEL_DIR="/dev/shm"
OUTPUT_BASE_DIR="./ebss_datasets"
DATA_DIR="data"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-dir)
            BASE_MODEL_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--model-dir DIR] [--output-dir DIR] [--data-dir DIR]"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "批量生成EBSS数据集"
echo "========================================"
echo "模型目录: $BASE_MODEL_DIR"
echo "输出目录: $OUTPUT_BASE_DIR"
echo "数据目录: $DATA_DIR"
echo "========================================"
echo ""

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据目录不存在: $DATA_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_BASE_DIR"

# 定义模型配置
declare -A MODELS
MODELS["Qwen3-30B-A3B"]="$BASE_MODEL_DIR/Qwen3-30B-A3B"
MODELS["Qwen3-Next-80B-A3B"]="$BASE_MODEL_DIR/Qwen3-Next-80B-A3B"
MODELS["Qwen3-235B-A22B"]="$BASE_MODEL_DIR/Qwen3-235B-A22B"

# 定义推荐配置
declare -A BEAM_WIDTH
BEAM_WIDTH["Qwen3-30B-A3B"]=4
BEAM_WIDTH["Qwen3-Next-80B-A3B"]=6
BEAM_WIDTH["Qwen3-235B-A22B"]=8

declare -A TAU
TAU["Qwen3-30B-A3B"]=1.2
TAU["Qwen3-Next-80B-A3B"]=1.3
TAU["Qwen3-235B-A22B"]=1.5

declare -A NUM_SAMPLES
NUM_SAMPLES["Qwen3-30B-A3B"]=512
NUM_SAMPLES["Qwen3-Next-80B-A3B"]=768
NUM_SAMPLES["Qwen3-235B-A22B"]=1024

# 生成函数
generate_ebss_for_model() {
    local model_name=$1
    local model_path=$2
    
    echo ""
    echo "========================================"
    echo "处理模型: $model_name"
    echo "========================================"
    
    # 检查模型是否存在
    if [ ! -d "$model_path" ]; then
        echo "⚠️  模型不存在: $model_path"
        echo "   跳过此模型"
        return 1
    fi
    
    local beam_width=${BEAM_WIDTH[$model_name]}
    local tau=${TAU[$model_name]}
    local num_samples=${NUM_SAMPLES[$model_name]}
    
    echo "配置:"
    echo "  Beam Width: $beam_width"
    echo "  Tau: $tau"
    echo "  样本数: $num_samples"
    echo ""
    
    # 创建模型专属输出目录
    local output_dir="$OUTPUT_BASE_DIR/$model_name"
    mkdir -p "$output_dir"
    
    # 生成EBSS数据集
    echo "开始生成EBSS数据集..."
    
    python3 scripts/generate_ebss_datasets.py \
        --model "$model_path" \
        --model-name "$model_name" \
        --output-dir "$output_dir" \
        --data-dir "$DATA_DIR" \
        --beam-width "$beam_width" \
        --tau "$tau" \
        --num-samples "$num_samples" \
        --max-tokens 512 \
        --seed-ratio 0.05 \
        --wikitext-ratio 0.30 \
        --chinese-ratio 0.30 \
        --mmlu-ratio 0.20 \
        --gsm8k-ratio 0.15
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ $model_name EBSS数据集生成成功"
        echo "  输出目录: $output_dir"
        
        # 显示生成的文件
        echo "  生成的文件:"
        ls -lh "$output_dir"/*.json 2>/dev/null | awk '{print "    " $9 " (" $5 ")"}'
        ls -lh "$output_dir"/*.txt 2>/dev/null | awk '{print "    " $9 " (" $5 ")"}'
        
        return 0
    else
        echo "❌ $model_name EBSS数据集生成失败"
        return 1
    fi
}

# 记录结果
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# 按顺序处理每个模型
for model_name in "Qwen3-30B-A3B" "Qwen3-Next-80B-A3B" "Qwen3-235B-A22B"; do
    model_path="${MODELS[$model_name]}"
    
    if generate_ebss_for_model "$model_name" "$model_path"; then
        ((SUCCESS_COUNT++))
    else
        if [ -d "$model_path" ]; then
            ((FAIL_COUNT++))
        else
            ((SKIP_COUNT++))
        fi
    fi
    
    echo ""
    echo "----------------------------------------"
    echo ""
    
    # 在模型之间添加延迟，让GPU冷却
    if [ "$model_name" != "Qwen3-235B-A22B" ]; then
        echo "等待10秒让GPU冷却..."
        sleep 10
    fi
done

# 最终总结
echo ""
echo "========================================"
echo "EBSS数据集生成完成"
echo "========================================"
echo "成功: $SUCCESS_COUNT"
echo "失败: $FAIL_COUNT"
echo "跳过: $SKIP_COUNT"
echo ""
echo "输出目录: $OUTPUT_BASE_DIR"
echo ""

# 显示所有生成的数据集
if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "生成的EBSS数据集:"
    for model_name in "Qwen3-30B-A3B" "Qwen3-Next-80B-A3B" "Qwen3-235B-A22B"; do
        output_dir="$OUTPUT_BASE_DIR/$model_name"
        if [ -d "$output_dir" ] && [ -f "$output_dir/ebss_${model_name}_calibration.txt" ]; then
            txt_file="$output_dir/ebss_${model_name}_calibration.txt"
            json_file="$output_dir/ebss_${model_name}_calibration.json"
            
            txt_size=$(du -h "$txt_file" 2>/dev/null | cut -f1)
            json_size=$(du -h "$json_file" 2>/dev/null | cut -f1)
            num_lines=$(wc -l < "$txt_file" 2>/dev/null || echo "0")
            
            echo ""
            echo "  $model_name:"
            echo "    文本文件: $txt_file ($txt_size, $num_lines 行)"
            echo "    JSON文件: $json_file ($json_size)"
            
            # 显示统计信息
            if [ -f "$output_dir/ebss_${model_name}_calibration_stats.json" ]; then
                stats_file="$output_dir/ebss_${model_name}_calibration_stats.json"
                echo "    统计文件: $stats_file"
                
                # 提取关键统计
                python3 -c "
import json
try:
    with open('$stats_file', 'r') as f:
        stats = json.load(f)
        print(f\"      专家激活数: {stats.get('num_experts_activated', 'N/A')}\")
        print(f\"      总激活次数: {stats.get('total_activations', 'N/A')}\")
        print(f\"      激活标准差: {stats.get('activation_std', 'N/A'):.2f}\")
except Exception as e:
    pass
" 2>/dev/null || true
            fi
        fi
    done
    
    echo ""
    echo "后续使用EBSS数据集进行量化:"
    echo ""
    for model_name in "Qwen3-30B-A3B" "Qwen3-Next-80B-A3B" "Qwen3-235B-A22B"; do
        output_dir="$OUTPUT_BASE_DIR/$model_name"
        txt_file="$output_dir/ebss_${model_name}_calibration.txt"
        
        if [ -f "$txt_file" ]; then
            model_path="${MODELS[$model_name]}"
            
            echo "# $model_name (W4A4)"
            echo "bash scripts/moequant_w4a4.sh \\"
            echo "    --model \"$model_path\" \\"
            echo "    --output-dir \"./output/${model_name}_W4A4\" \\"
            echo "    --seed-text \"$txt_file\""
            echo ""
        fi
    done
fi

if [ $SUCCESS_COUNT -eq 0 ]; then
    echo "⚠️  没有成功生成任何EBSS数据集"
    echo ""
    echo "可能的原因:"
    echo "1. 模型文件不存在于 $BASE_MODEL_DIR"
    echo "2. GPU内存不足"
    echo "3. Python依赖缺失"
    echo ""
    echo "请检查错误日志并重试"
    exit 1
fi

echo "全部完成! 🎉"

