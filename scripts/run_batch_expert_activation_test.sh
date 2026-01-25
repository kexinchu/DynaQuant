#!/bin/bash
# Run expert activation tests for different batch sizes on two models

set -e

# Set environment variable to optimize GPU memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
DATASET="ShareGPT_V3_unfiltered_cleaned_split.json"
NUM_QUERIES=128
# Start with smaller batch sizes to avoid OOM, can increase later
BATCH_SIZES="1 2 4 8 16 32"
MAX_LENGTH=512
MAX_NEW_TOKENS=10
OUTPUT_DIR="results/expert_activation_batch"
# Use device_map="auto" to automatically use all available GPUs
# Set a default device for input tensors (will be automatically distributed)
# Check which GPU has more free memory for input placement
FREE_MEM_0=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$1==0 {print $2}')
FREE_MEM_1=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$1==1 {print $2}')

if [ -n "$FREE_MEM_1" ] && [ "$FREE_MEM_1" -gt 20000 ]; then
    DEVICE="${DEVICE:-cuda:1}"
elif [ -n "$FREE_MEM_0" ] && [ "$FREE_MEM_0" -gt 20000 ]; then
    DEVICE="${DEVICE:-cuda:0}"
else
    # Default to cuda:0 if both are busy
    DEVICE="${DEVICE:-cuda:0}"
    echo "Warning: Both GPUs seem busy, using $DEVICE for input placement"
fi

echo "Note: Model will be automatically distributed across all available GPUs using device_map='auto'"
echo "Input tensors will be placed on: $DEVICE"
MODELS_BASE="/workspace/Models"

# Models to test - all from /workspace/Models/
# Use int4 quantized version for 30B model to reduce memory usage
MODEL_30B="${MODELS_BASE}/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"
MODEL_80B="${MODELS_BASE}/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound"
MODEL_GPT_OSS="${MODELS_BASE}/gpt-oss-20b"
MODEL_DEEPSEEK="${MODELS_BASE}/DeepSeek-V2-Lite"

echo "=========================================="
echo "Expert Activation Batch Size Test"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Number of queries: $NUM_QUERIES"
echo "Batch sizes: $BATCH_SIZES"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Models base: $MODELS_BASE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to test a model
test_model() {
    local model_path=$1
    local model_name=$2
    local quantization=${3:-none}
    
    if [ ! -d "$model_path" ]; then
        echo "Warning: Model path $model_path does not exist, skipping..."
        return 1
    fi
    
    echo "=========================================="
    echo "Testing Model: $model_name"
    echo "Path: $model_path"
    echo "Quantization: $quantization"
    echo "=========================================="
    
    # Set PyTorch memory allocation config to reduce fragmentation
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    python3 scripts/test_expert_activation_batch.py \
        --dataset "$DATASET" \
        --model-id "$model_path" \
        --num-queries "$NUM_QUERIES" \
        --batch-sizes $BATCH_SIZES \
        --max-length "$MAX_LENGTH" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --device "$DEVICE" \
        --quantization "$quantization" \
        --output-dir "$OUTPUT_DIR" \
        --log-level INFO
    
    echo ""
}

# Test all models
# Use autoround-int4 for the quantized 30B model
test_model "$MODEL_30B" "Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound" "autoround-int4"
test_model "$MODEL_80B" "Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound" "autoround-int2"
test_model "$MODEL_DEEPSEEK" "DeepSeek-V2-Lite" "none"
test_model "$MODEL_GPT_OSS" "gpt-oss-20b" "none"

echo ""
echo "=========================================="
echo "All tests completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

