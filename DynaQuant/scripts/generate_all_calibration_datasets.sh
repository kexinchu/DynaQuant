#!/bin/bash
#
# 批量生成校准数据集脚本（简化版本）
# 为三个MoE模型生成混合校准数据集，无需EBSS
#

set -e

# 默认参数
OUTPUT_BASE_DIR="./calibration_datasets"
DATA_DIR="data"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
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
            echo "用法: $0 [--output-dir DIR] [--data-dir DIR]"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "批量生成校准数据集"
echo "========================================"
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
declare -A NUM_SAMPLES
NUM_SAMPLES["Qwen3-30B-A3B"]=512
NUM_SAMPLES["Qwen3-Next-80B-A3B"]=768
NUM_SAMPLES["Qwen3-235B-A22B"]=1024

# 生成函数
generate_calibration_for_model() {
    local model_name=$1
    local num_samples=$2
    
    echo ""
    echo "========================================"
    echo "处理模型: $model_name"
    echo "========================================"
    echo "样本数: $num_samples"
    echo ""
    
    # 创建模型专属输出目录
    local output_dir="$OUTPUT_BASE_DIR/$model_name"
    mkdir -p "$output_dir"
    
    # 生成校准数据集
    echo "开始生成校准数据集..."
    
    python3 scripts/generate_ebss_datasets_standalone.py \
        --model-name "$model_name" \
        --output-dir "$output_dir" \
        --data-dir "$DATA_DIR" \
        --num-samples "$num_samples" \
        --seed-ratio 0.05 \
        --wikitext-ratio 0.30 \
        --chinese-ratio 0.30 \
        --mmlu-ratio 0.20 \
        --gsm8k-ratio 0.15
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ $model_name 校准数据集生成成功"
        echo "  输出目录: $output_dir"
        
        # 显示生成的文件
        echo "  生成的文件:"
        ls -lh "$output_dir"/*.json 2>/dev/null | awk '{print "    " $9 " (" $5 ")"}'
        ls -lh "$output_dir"/*.txt 2>/dev/null | awk '{print "    " $9 " (" $5 ")"}'
        
        return 0
    else
        echo "❌ $model_name 校准数据集生成失败"
        return 1
    fi
}

# 记录结果
SUCCESS_COUNT=0
FAIL_COUNT=0

# 按顺序处理每个模型
for model_name in "Qwen3-30B-A3B" "Qwen3-Next-80B-A3B" "Qwen3-235B-A22B"; do
    num_samples="${NUM_SAMPLES[$model_name]}"
    
    if generate_calibration_for_model "$model_name" "$num_samples"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
    
    echo ""
    echo "----------------------------------------"
    echo ""
done

# 最终总结
echo ""
echo "========================================"
echo "校准数据集生成完成"
echo "========================================"
echo "成功: $SUCCESS_COUNT"
echo "失败: $FAIL_COUNT"
echo ""
echo "输出目录: $OUTPUT_BASE_DIR"
echo ""

# 显示所有生成的数据集
if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "生成的校准数据集:"
    for model_name in "Qwen3-30B-A3B" "Qwen3-Next-80B-A3B" "Qwen3-235B-A22B"; do
        output_dir="$OUTPUT_BASE_DIR/$model_name"
        if [ -d "$output_dir" ] && [ -f "$output_dir/calibration_${model_name}.txt" ]; then
            txt_file="$output_dir/calibration_${model_name}.txt"
            json_file="$output_dir/calibration_${model_name}.json"
            
            txt_size=$(du -h "$txt_file" 2>/dev/null | cut -f1)
            json_size=$(du -h "$json_file" 2>/dev/null | cut -f1)
            num_lines=$(wc -l < "$txt_file" 2>/dev/null || echo "0")
            
            echo ""
            echo "  $model_name:"
            echo "    文本文件: $txt_file ($txt_size, $num_lines 行)"
            echo "    JSON文件: $json_file ($json_size)"
        fi
    done
    
    echo ""
    echo "后续使用校准数据集进行量化:"
    echo ""
    for model_name in "Qwen3-30B-A3B" "Qwen3-Next-80B-A3B" "Qwen3-235B-A22B"; do
        output_dir="$OUTPUT_BASE_DIR/$model_name"
        txt_file="$output_dir/calibration_${model_name}.txt"
        
        if [ -f "$txt_file" ]; then
            model_path="/dev/shm/$model_name"
            
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
    echo "⚠️  没有成功生成任何校准数据集"
    echo ""
    echo "可能的原因:"
    echo "1. 数据目录为空或数据格式不正确"
    echo "2. Python依赖缺失 (需要pandas)"
    echo ""
    echo "请检查错误日志并重试"
    exit 1
fi

echo "全部完成! 🎉"

