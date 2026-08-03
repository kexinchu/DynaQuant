#!/usr/bin/env bash
# run_80b_eval.sh — Evaluate Qwen3-Next-80B INT4 on GPU 0 after Phi-3.5 FP16 finishes
set -euo pipefail
source /home/kec23008/auto-paper-reading/venv/bin/activate
if [[ -z "${HF_TOKEN:-}" ]]; then echo "HF_TOKEN must be set in the environment" >&2; exit 2; fi
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p results/logs

CUDA_VISIBLE_DEVICES=0,1 python -u scripts/eval_qwen3next_80b.py \
    --n-samples 200 --n-aime 30 \
    --max-new-tokens 2048 \
    --batch-size 16 \
    --device-map auto \
    --output results/qwen3next_80b_int4.json \
    2>&1 | tee results/logs/qwen3next_80b.log
