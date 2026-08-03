#!/usr/bin/env bash
set -euo pipefail

# Rebuild the formal Phi-3.5-MoE W4A16 baseline from an immutable BF16 source.
# CUDA_VISIBLE_DEVICES keeps the offline quantizer on one physical A6000.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

quant_python="${PYTHON:-python}"
: "${PHI35_BF16_MODEL_PATH:?Set PHI35_BF16_MODEL_PATH to the source checkpoint}"
: "${PHI35_W4_OUTPUT_PATH:?Set PHI35_W4_OUTPUT_PATH to a new output directory}"
calibration_jsonl="${CALIBRATION_JSONL:-calibration_datasets/formal/wikitext103_train_256x2048.jsonl}"
"${quant_python}" scripts/quantize_with_autoround.py \
  --model-path "${PHI35_BF16_MODEL_PATH}" \
  --output-path "${PHI35_W4_OUTPUT_PATH}" \
  --source-manifest results/model_manifests/phi35_moe_bf16.json \
  --calibration-jsonl "${calibration_jsonl}" \
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
