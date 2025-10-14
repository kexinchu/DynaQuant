#!/bin/bash
#
# MoE W2A2 PTQ Script
# 
# Usage:
#   bash scripts/run_ptq_moe.sh --model MODEL_NAME [OPTIONS]
#

set -e

# Default parameters
MODEL="Qwen/Qwen-MoE-14B"
CALIB_SIZE=128
EBSS_BEAM_WIDTH=4
EBSS_TAU=1.2
BIT_W=2
BIT_A=2
GROUP_SIZE=64
USE_ROTATION=1
ENABLE_FALLBACK=1
ROUTER_MODE="fp16"
STRICT_TOPK=1
OUTPUT_DIR="./output/ptq_moe"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
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
        --bit-w)
            BIT_W="$2"
            shift 2
            ;;
        --bit-a)
            BIT_A="$2"
            shift 2
            ;;
        --group-size)
            GROUP_SIZE="$2"
            shift 2
            ;;
        --no-rotation)
            USE_ROTATION=0
            shift
            ;;
        --no-fallback)
            ENABLE_FALLBACK=0
            shift
            ;;
        --router-mode)
            ROUTER_MODE="$2"
            shift 2
            ;;
        --no-strict-topk)
            STRICT_TOPK=0
            shift
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

echo "================================"
echo "MoE W2A2 PTQ with EBSS + AGQ"
echo "================================"
echo "Model: $MODEL"
echo "Calibration Size: $CALIB_SIZE"
echo "EBSS Beam Width: $EBSS_BEAM_WIDTH"
echo "EBSS Tau: $EBSS_TAU"
echo "Weight Bits: $BIT_W"
echo "Activation Bits: $BIT_A"
echo "Group Size: $GROUP_SIZE"
echo "Use Rotation: $USE_ROTATION"
echo "Enable Fallback: $ENABLE_FALLBACK"
echo "Router Mode: $ROUTER_MODE"
echo "Strict Top-k: $STRICT_TOPK"
echo "Output Directory: $OUTPUT_DIR"
echo "================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run PTQ
python3 -m moe_quant.runners.run_ptq \
    --model "$MODEL" \
    --calib-size "$CALIB_SIZE" \
    --ebss-beam-width "$EBSS_BEAM_WIDTH" \
    --ebss-tau "$EBSS_TAU" \
    --bit-w "$BIT_W" \
    --bit-a "$BIT_A" \
    --group-size "$GROUP_SIZE" \
    --use-rotation "$USE_ROTATION" \
    --enable-fallback "$ENABLE_FALLBACK" \
    --router-mode "$ROUTER_MODE" \
    --strict-topk "$STRICT_TOPK" \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo "================================"
echo "PTQ complete! Results in $OUTPUT_DIR"
echo "================================"

