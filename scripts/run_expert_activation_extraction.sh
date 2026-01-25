#!/bin/bash
# Extract expert activations for multiple models using mmlu_pro_200 dataset

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
DATASET="calibration_datasets/requests/mmlu_pro_200.jsonl"
OUTPUT_DIR="activations"
# Use GPU 1 if available and has more free memory, otherwise use GPU 0
if nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$1==1 && $2 > 30000 {exit 0} {exit 1}'; then
    DEVICE="${DEVICE:-cuda:1}"
else
    DEVICE="${DEVICE:-cuda:0}"
fi
MAX_LENGTH=512
TOP_K=8

# Models to process
declare -A MODELS=(
    ["gpt-oss-20b"]="../Models/gpt-oss-20b"
    ["deepseek-v2-lite"]="../Models/DeepSeek-V2-Lite"
)

echo "=========================================="
echo "Expert Activation Extraction"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Max length: $MAX_LENGTH"
echo "Top-K: $TOP_K"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to extract activations for a model
extract_activations() {
    local model_name=$1
    local model_path=$2
    local output_file="${OUTPUT_DIR}/activation_${model_name}_mmlu_pro.json"
    
    echo "----------------------------------------"
    echo "Processing: $model_name"
    echo "Model path: $model_path"
    echo "Output: $output_file"
    echo "----------------------------------------"
    
    python3 scripts/summarize_expert_activation.py \
        --dataset "$DATASET" \
        --model-id "$model_path" \
        --device "$DEVICE" \
        --max-length "$MAX_LENGTH" \
        --top-k "$TOP_K" \
        --output "$output_file" \
        --log-level INFO
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully extracted activations for $model_name"
        echo "  Output saved to: $output_file"
        # Show file size
        if [ -f "$output_file" ]; then
            echo "  File size: $(du -h "$output_file" | cut -f1)"
        fi
    else
        echo "✗ Failed to extract activations for $model_name"
        return 1
    fi
    echo ""
}

# Process each model
for model_name in "${!MODELS[@]}"; do
    model_path="${MODELS[$model_name]}"
    
    # Check if model path exists
    if [ ! -d "$model_path" ]; then
        echo "Warning: Model path does not exist: $model_path"
        echo "Skipping $model_name"
        echo ""
        continue
    fi
    
    extract_activations "$model_name" "$model_path"
done

echo "=========================================="
echo "Extraction Complete"
echo "=========================================="
echo "Results saved in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/activation_*_mmlu_pro.json 2>/dev/null || echo "No output files found"

