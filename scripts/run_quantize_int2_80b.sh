#!/bin/bash
# Quantize Qwen3-Next-80B from Int4 -> Int2 using AutoRound (W2A16 mixed precision).
#
# GPU usage: device_map="auto" distributes across both A6000s.
#   GPU 0: ~4 GB free  |  GPU 1: ~48 GB free
#
# Estimated time: 4–8 hours for 80B model (128 samples, 200 iters).
#
# Run in background:
#   nohup bash scripts/run_quantize_int2_80b.sh > logs/quantize_int2_80b.log 2>&1 &
#   tail -f logs/quantize_int2_80b.log

set -euo pipefail

source /home/kec23008/auto-paper-reading/venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0   # run entirely on GPU 0 (~49 GB free)

echo "[$(date)] Starting Int4→Int2 quantization of Qwen3-Next-80B (GPU 0)"
python "${SCRIPT_DIR}/quantize_int2_qwen3_80b.py"
echo "[$(date)] Quantization finished"
