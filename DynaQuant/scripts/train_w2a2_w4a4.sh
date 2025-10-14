#!/bin/bash
#
# Train W2A2 and W4A4 models in parallel using 8 A100 GPUs
# Router layers use W8A8
#

set -e

# Configuration
BASE_MODEL="/dev/shm/Qwen3-30B-A3B"
OUTPUT_BASE="/dev/shm/quantized_models"
NUM_GPUS=8
CALIB_SIZE=256

echo "========================================"
echo "MoE Parallel Quantization Pipeline"
echo "========================================"
echo "Base Model: $BASE_MODEL"
echo "Output Base: $OUTPUT_BASE"
echo "GPUs: $NUM_GPUS"
echo "Calibration Size: $CALIB_SIZE"
echo "========================================"

# Create output directories
mkdir -p "$OUTPUT_BASE"

# Function to run PTQ
run_ptq() {
    local W_BIT=$1
    local A_BIT=$2
    local MODEL_NAME=$3
    
    echo ""
    echo "========================================"
    echo "Training ${MODEL_NAME}"
    echo "Expert: W${W_BIT}A${A_BIT}, Router: W8A8"
    echo "========================================"
    
    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
    
    python3 -m moe_quant.runners.run_parallel_ptq \
        --model "$BASE_MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --w-bit "$W_BIT" \
        --a-bit "$A_BIT" \
        --router-w-bit 8 \
        --router-a-bit 8 \
        --num-gpus "$NUM_GPUS" \
        --calib-size "$CALIB_SIZE"
    
    echo "✓ ${MODEL_NAME} completed!"
    echo "Results saved to: $OUTPUT_DIR"
}

# Train W2A2 model
run_ptq 2 2 "Qwen3-30B-A3B-W2A2"

# Train W4A4 model
run_ptq 4 4 "Qwen3-30B-A3B-W4A4"

echo ""
echo "========================================"
echo "All models quantized successfully!"
echo "========================================"
echo "W2A2 Model: ${OUTPUT_BASE}/Qwen3-30B-A3B-W2A2"
echo "W4A4 Model: ${OUTPUT_BASE}/Qwen3-30B-A3B-W4A4"
echo "========================================"

# Generate summary
python3 - << 'PYTHON_SCRIPT'
import json
from pathlib import Path

output_base = Path("/dev/shm/quantized_models")

print("\n" + "="*60)
print("Quantization Summary")
print("="*60)

for model_name in ["Qwen3-30B-A3B-W2A2", "Qwen3-30B-A3B-W4A4"]:
    model_dir = output_base / model_name
    stats_file = model_dir / "quantization_stats.json"
    
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print(f"\n{model_name}:")
        print(f"  Total components quantized: {len(stats)}")
        
        # Calculate average errors
        mses = [s.get('mse', 0) for s in stats.values() if 'mse' in s]
        if mses:
            print(f"  Average MSE: {sum(mses)/len(mses):.6f}")
        
        # Count routers and experts
        routers = [k for k in stats.keys() if 'router' in k]
        experts = [k for k in stats.keys() if 'expert' in k]
        print(f"  Routers quantized: {len(routers)}")
        print(f"  Expert layers quantized: {len(experts)}")

print("\n" + "="*60)
PYTHON_SCRIPT

echo ""
echo "Next steps:"
echo "1. Run tests: bash scripts/test_all_models.sh"
echo "2. Or test individual model:"
echo "   python3 scripts/test_quantized_models.py --model ${OUTPUT_BASE}/Qwen3-30B-A3B-W2A2"
