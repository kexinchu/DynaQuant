#!/bin/bash
# Run two MoE offload experiments in parallel, one per GPU.
set -e

cd /home/kec23008/DynaQuant
RESULTS_DIR=results/moe_offload_experiment
PROMPT_LENGTHS="1 2 4 8 16 32 48 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512"
mkdir -p "$RESULTS_DIR"

# GPU 0: Qwen3-30B
# python -u scripts/experiment_moe_offload_latency.py \
#   --model /home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound \
#   --dataset /home/kec23008/DynaQuant/ShareGPT_V3_unfiltered_cleaned_split.json \
#   --output "$RESULTS_DIR/Qwen3-30B_full_test.json" \
#   --prompt-lengths $PROMPT_LENGTHS --num-prompts 10 --device cuda:0 --log-level INFO \
#   > "$RESULTS_DIR/Qwen3-30B_full_test.log" 2>&1 &
# PID1=$!

# python -u scripts/experiment_moe_offload_latency.py \
#   --model /home/kec23008/Models/Phi-3.5-MoE-instruct-W4A16-AutoRound-formal \
#   --dataset /home/kec23008/DynaQuant/ShareGPT_V3_unfiltered_cleaned_split.json \
#   --output "$RESULTS_DIR/Phi-3.5-MoE_full_test.json" \
#   --prompt-lengths $PROMPT_LENGTHS --num-prompts 10 --device cuda:0 --log-level INFO \
#   > "$RESULTS_DIR/Phi-3.5-MoE_full_test.log" 2>&1 &
# PID1=$!

python -u scripts/experiment_moe_offload_latency.py \
  --model /home/kec23008/Models/gpt-oss-20b \
  --dataset /home/kec23008/DynaQuant/ShareGPT_V3_unfiltered_cleaned_split.json \
  --output "$RESULTS_DIR/GPT-OSS-20B_full_test.json" \
  --prompt-lengths $PROMPT_LENGTHS --num-prompts 10 --device cuda:0 --log-level INFO \
  > "$RESULTS_DIR/GPT-OSS-20B_full_test.log" 2>&1 &
PID1=$!

# GPU 1: Qwen3-80B
# python -u scripts/experiment_moe_offload_latency.py \
#   --model /home/kec23008/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
#   --dataset /home/kec23008/DynaQuant/ShareGPT_V3_unfiltered_cleaned_split.json \
#   --output "$RESULTS_DIR/Qwen3-80B_full_test.json" \
#   --prompt-lengths $PROMPT_LENGTHS --num-prompts 10 --device cuda:1 --log-level INFO \
#   > "$RESULTS_DIR/Qwen3-80B_full_test.log" 2>&1 &
# PID2=$!

# echo "Started: Phi-3.5-MoE on cuda:0 (PID $PID1), Qwen3-80B on cuda:1 (PID $PID2)"
echo "Logs: $RESULTS_DIR/GPT-OSS-20B_full_test.log, $RESULTS_DIR/Qwen3-80B_full_test.log"
wait $PID1 && echo "GPT-OSS-20B finished." || exit 1
# wait $PID2 && echo "Qwen3-80B finished." || exit 1
echo "Both experiments done."
