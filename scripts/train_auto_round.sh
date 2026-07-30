#!/usr/bin/env bash
set -euo pipefail

# Rebuild the formal Phi-3.5-MoE W4A16 baseline from an immutable BF16 source.
# CUDA_VISIBLE_DEVICES keeps the offline quantizer on one physical A6000.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

quant_python="${PYTHON:-python}"
"${quant_python}" scripts/quantize_with_autoround.py \
  --model-path /dev/shm/dynaexq-models/phi35-moe-bf16 \
  --output-path /dev/shm/dynaexq-models/phi35-moe-w4a16-autoround-formal \
  --source-manifest results/model_manifests/phi35_moe_bf16.json \
  --calibration-jsonl calibration_datasets/formal/wikitext103_train_256x2048.jsonl \
  --scheme W4A16 \
  --iters 200 \
  --nsamples 256 \
  --seqlen 2048 \
  --seed 42 \
  --output-format auto_round \
  --no-trust-remote-code \
  --device-map 0 \
  --batch-size 1 \
  --low-gpu-mem-usage
