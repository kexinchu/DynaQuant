#!/bin/bash
# Run expert weight swap experiments for conditions A, B, and C

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
MODELS_BASE="/home/kec23008/Models"
MODEL_30B_FP16="${MODELS_BASE}/Qwen3-30B-A3B-Instruct-2507"
MODEL_30B_INT4="${MODELS_BASE}/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"

# Default parameters
DEVICE="${DEVICE:-cuda:0}"
EXPERIMENT_OUTPUT_DIR="${EXPERIMENT_OUTPUT_DIR:-results/expert_weight_swap}"
PROMPT_LENGTHS="${PROMPT_LENGTHS:-1 32 128 256 512}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 8 32 64}"
EXPERT_LAYER="${EXPERT_LAYER:-0}"
EXPERT_IDX="${EXPERT_IDX:-0}"
SWAP_TRIGGER_STEP="${SWAP_TRIGGER_STEP:-50}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"

echo "=========================================="
echo "Expert Weight Swap Experiment"
echo "=========================================="
echo "Model (FP16): $MODEL_30B_FP16"
echo "Model (Int4): $MODEL_30B_INT4"
echo "Device: $DEVICE"
echo "Output directory: $EXPERIMENT_OUTPUT_DIR"
echo "Prompt lengths: $PROMPT_LENGTHS"
echo "Concurrency levels: $CONCURRENCY_LEVELS"
echo "Expert: Layer $EXPERT_LAYER, Index $EXPERT_IDX"
echo "Swap trigger step: $SWAP_TRIGGER_STEP"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$EXPERIMENT_OUTPUT_DIR"

# Function to run a single experiment condition
run_condition() {
    local condition=$1
    local prompt_len=$2
    local concurrency=$3
    local pinned=$4

    local output_file="${EXPERIMENT_OUTPUT_DIR}/condition_${condition}_prompt${prompt_len}_concurrency${concurrency}_expertL${EXPERT_LAYER}E${EXPERT_IDX}.json"

    echo "----------------------------------------"
    echo "Running Condition $condition"
    echo "  Prompt length: $prompt_len"
    echo "  Concurrency: $concurrency"
    echo "  Pinned memory: $pinned"
    echo "  Output: $output_file"
    echo "----------------------------------------"

    local pinned_flag=""
    if [ "$pinned" = "true" ]; then
        pinned_flag="--pinned-memory"
    fi

    python3 scripts/experiment_expert_weight_swap.py \
        --model-path "$MODEL_30B_FP16" \
        --int4-model-path "$MODEL_30B_INT4" \
        --condition "$condition" \
        --prompt-length "$prompt_len" \
        --concurrency "$concurrency" \
        --expert-layer "$EXPERT_LAYER" \
        --expert-idx "$EXPERT_IDX" \
        --swap-trigger-step "$SWAP_TRIGGER_STEP" \
        $pinned_flag \
        --num-requests "$NUM_REQUESTS" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --device "$DEVICE" \
        --output "$output_file" \
        --log-level INFO

    echo "Completed: $output_file"
    echo ""
}

# Run experiments
# Condition A: Baseline (no swap)
echo "=========================================="
echo "Condition A: Baseline (No Swap)"
echo "=========================================="
for prompt_len in $PROMPT_LENGTHS; do
    for concurrency in $CONCURRENCY_LEVELS; do
        run_condition "A" "$prompt_len" "$concurrency" "false"
    done
done

# Condition B: No pre-allocated memory
echo "=========================================="
echo "Condition B: No Pre-allocated Memory"
echo "=========================================="
for prompt_len in $PROMPT_LENGTHS; do
    for concurrency in $CONCURRENCY_LEVELS; do
        run_condition "B" "$prompt_len" "$concurrency" "false"
    done
done

# Condition C: Pre-allocated pinned memory
echo "=========================================="
echo "Condition C: Pre-allocated Pinned Memory"
echo "=========================================="
for prompt_len in $PROMPT_LENGTHS; do
    for concurrency in $CONCURRENCY_LEVELS; do
        run_condition "C" "$prompt_len" "$concurrency" "true"
    done
done

echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $EXPERIMENT_OUTPUT_DIR"
echo "=========================================="

# Generate summary
echo ""
echo "Generating summary..."
python3 << EOF
import json
import glob
from pathlib import Path

output_dir = Path("$EXPERIMENT_OUTPUT_DIR")
results = {}

for json_file in glob.glob(str(output_dir / "*.json")):
    with open(json_file, 'r') as f:
        data = json.load(f)
        condition = data['config']['condition']
        prompt_len = data['config']['prompt_length']
        concurrency = data['config']['concurrency']

        key = f"{condition}_prompt{prompt_len}_concurrency{concurrency}"
        if key not in results:
            results[key] = {}

        results[key][condition] = {
            'decode_p50': data['metrics']['decode_latency_percentiles_ms'].get(50, 0),
            'decode_p95': data['metrics']['decode_latency_percentiles_ms'].get(95, 0),
            'decode_p99': data['metrics']['decode_latency_percentiles_ms'].get(99, 0),
            'decode_p99_9': data['metrics']['decode_latency_percentiles_ms'].get(99.9, 0),
            'total_p50': data['metrics']['total_latency_percentiles_ms'].get(50, 0),
            'swap_metrics': data['metrics'].get('swap_metrics'),
        }

# Save summary
summary_file = output_dir / "summary.json"
with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Summary saved to: {summary_file}")
EOF

echo "Done!"
