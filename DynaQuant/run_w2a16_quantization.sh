#!/bin/bash
#
# W2A16 AWQ 量化完整工作流程
# ===========================
# 从量化到测试的一站式脚本
#

set -e

# ============================================================================
# 配置参数
# ============================================================================

# 模型路径
MODEL_PATH="${MODEL_PATH:-/dev/shm/Qwen3-30B-A3B}"
MODEL_NAME=$(basename "$MODEL_PATH")

# 输出路径
OUTPUT_DIR="${OUTPUT_DIR:-./output/${MODEL_NAME}-W2A16}"

# 校准数据
CALIB_DATA="${CALIB_DATA:-calibration_datasets/${MODEL_NAME}/calibration_${MODEL_NAME}.json}"

# 量化参数
NUM_SAMPLES="${NUM_SAMPLES:-512}"
GROUP_SIZE="${GROUP_SIZE:-128}"
SEARCH_MODE="${SEARCH_MODE:-global}"

# 测试参数
TEST_PROMPT="${TEST_PROMPT:-Once upon a time, there was a}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"

# ============================================================================
# 颜色输出
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# 主流程
# ============================================================================

echo "============================================================================"
echo "  W2A16 AWQ Quantization Workflow"
echo "============================================================================"
echo ""
info "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Calibration data: $CALIB_DATA"
echo "  Samples: $NUM_SAMPLES"
echo "  Group size: $GROUP_SIZE"
echo ""

# Step 1: 检查环境
info "Step 1/4: Checking environment..."

if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH/config.json" ]; then
    error "Model not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$CALIB_DATA" ]; then
    warning "Calibration data not found: $CALIB_DATA"
    info "Trying to find alternative calibration data..."
    
    # 尝试其他路径
    ALT_CALIB="calibration_datasets/calibration.json"
    if [ -f "$ALT_CALIB" ]; then
        CALIB_DATA="$ALT_CALIB"
        success "Found calibration data: $CALIB_DATA"
    else
        error "No calibration data found. Please specify CALIB_DATA environment variable."
        exit 1
    fi
fi

if ! python3 -c "import torch; import transformers; from quant.awq_w2 import W2AWQLinear" 2>/dev/null; then
    error "Required packages not found. Please install dependencies:"
    echo "  pip install torch transformers datasets safetensors"
    exit 1
fi

success "Environment check passed"
echo ""

# Step 2: 运行量化
info "Step 2/4: Running W2A16 quantization..."
echo "  This may take 60-120 minutes depending on model size"
echo "  Press Ctrl+C to cancel"
echo ""

START_TIME=$(date +%s)

python3 scripts/quantize_w2a16.py \
    --model "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --calib-data "$CALIB_DATA" \
    --num-samples "$NUM_SAMPLES" \
    --group-size "$GROUP_SIZE" \
    --search-mode "$SEARCH_MODE" \
    --ignore lm_head \
    --moe \
    --device cuda

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

success "Quantization completed in ${DURATION_MIN}m ${DURATION_SEC}s"
echo ""

# Step 3: 验证输出
info "Step 3/4: Verifying output..."

if [ ! -d "$OUTPUT_DIR" ]; then
    error "Output directory not created: $OUTPUT_DIR"
    exit 1
fi

REQUIRED_FILES=(
    "config.json"
    "quantization_config.json"
    "quantization_metadata.json"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$OUTPUT_DIR/$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    warning "Missing files in output directory:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
else
    success "All required files present"
fi

# 显示模型大小
if [ -f "$OUTPUT_DIR/model.safetensors" ]; then
    MODEL_SIZE=$(du -h "$OUTPUT_DIR/model.safetensors" | cut -f1)
    info "Quantized model size: $MODEL_SIZE"
elif [ -f "$OUTPUT_DIR/pytorch_model.bin" ]; then
    MODEL_SIZE=$(du -h "$OUTPUT_DIR/pytorch_model.bin" | cut -f1)
    info "Quantized model size: $MODEL_SIZE"
fi

# 显示元数据
if [ -f "$OUTPUT_DIR/quantization_metadata.json" ]; then
    info "Quantization metadata:"
    python3 -c "
import json
with open('$OUTPUT_DIR/quantization_metadata.json') as f:
    meta = json.load(f)
    print(f\"  Layers quantized: {meta.get('num_layers_quantized', 'N/A')}\")
    print(f\"  Average error: {meta.get('avg_error', 'N/A'):.6f}\")
    print(f\"  Group size: {meta.get('group_size', 'N/A')}\")
"
fi

echo ""

# Step 4: 测试推理
info "Step 4/4: Testing inference..."
echo "  Prompt: $TEST_PROMPT"
echo ""

python3 scripts/load_w2a16_model.py \
    --model-path "$OUTPUT_DIR" \
    --test-prompt "$TEST_PROMPT" \
    --max-new-tokens "$MAX_NEW_TOKENS"

echo ""

# 完成
echo "============================================================================"
success "W2A16 Quantization Workflow Completed!"
echo "============================================================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Total time: ${DURATION_MIN}m ${DURATION_SEC}s"
echo ""
echo "Next steps:"
echo "  1. Evaluate on benchmarks:"
echo "     python scripts/evaluate_model.py --model $OUTPUT_DIR --datasets wikitext mmlu"
echo ""
echo "  2. Compare with original:"
echo "     python scripts/load_w2a16_model.py \\"
echo "       --model-path $OUTPUT_DIR \\"
echo "       --compare-with-original $MODEL_PATH"
echo ""
echo "  3. Deploy for inference:"
echo "     python scripts/load_w2a16_model.py --model-path $OUTPUT_DIR"
echo ""
