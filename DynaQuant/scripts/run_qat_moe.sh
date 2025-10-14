#!/bin/bash
#
# MoE W2A2 QAT Script
#
# Usage:
#   bash scripts/run_qat_moe.sh --model MODEL_NAME --load-ptq PTQ_CHECKPOINT [OPTIONS]
#

set -e

# Default parameters
MODEL="Qwen/Qwen-MoE-14B"
LOAD_PTQ=""
EPOCHS=2
LR=5e-6
BATCH_SIZE=1
GRAD_ACCUM=8
LAMBDA_TOPK=1.0
MU_MARGIN=0.2
TRAIN_DATA=""
OUTPUT_DIR="./output/qat_moe"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --load-ptq)
            LOAD_PTQ="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --lambda-topk)
            LAMBDA_TOPK="$2"
            shift 2
            ;;
        --mu-margin)
            MU_MARGIN="$2"
            shift 2
            ;;
        --train-data)
            TRAIN_DATA="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$LOAD_PTQ" ]; then
    echo "Error: --load-ptq is required"
    exit 1
fi

echo "================================"
echo "MoE W2A2 QAT Training"
echo "================================"
echo "Model: $MODEL"
echo "Load PTQ from: $LOAD_PTQ"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRAD_ACCUM"
echo "Lambda Top-k: $LAMBDA_TOPK"
echo "Mu Margin: $MU_MARGIN"
echo "Training Data: $TRAIN_DATA"
echo "Output Directory: $OUTPUT_DIR"
echo "================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run QAT
python3 -m moe_quant.qat.run_qat \
    --model "$MODEL" \
    --load-ptq "$LOAD_PTQ" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lambda-topk "$LAMBDA_TOPK" \
    --mu-margin "$MU_MARGIN" \
    --train-data "$TRAIN_DATA" \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo "================================"
echo "QAT complete! Results in $OUTPUT_DIR"
echo "================================"

