#/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m sglang.launch_server \
  --model-path /dev/shm/Qwen3-235B-A22B-FP8 \
  --tp-size 4 --dp-size 2 \
  --enable-ep-moe \
  --max-running-requests 32 \
  --host 127.0.0.1 --port 8080 \
  --max-total-tokens 163840 \
  --dtype bfloat16 \
  --trust-remote-code \
  --attention-backend torch_native \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  --disable-cuda-graph-padding \
  --kv-cache-dtype auto \
  --allow-auto-truncate \
  --chunked-prefill-size 16384 \
  --enable-mixed-precision \
  --mixed-precision-config ./sglang-0.4.7/mixed_precision_config.yaml \
  --expert-distribution-recorder-mode per_token \
  --enable-expert-distribution-metrics \
  --ep-dispatch-algorithm static \
  --enable-dynamic-quantization \
  --fp16-model-path /dev/shm/Qwen3-235B-A22B \
  --fp8-model-path /dev/shm/Qwen3-235B-A22B-FP8 \
  --gptq-int4-model-path /dev/shm/Qwen3-235B-A22B-GPTQ-Int4 \
  --quantization-high-threshold 0.5 \
  --quantization-medium-threshold 0.0 \
  --max-concurrent-swaps 1 \

# 测试
# curl -s http://127.0.0.1:8080/v1/chat/completions \
#   -H 'Content-Type: application/json' \
#   -H 'Authorization: Bearer sk-local' \
#   -d '{
#     "model": "qwen3-235b-a22b",
#     "messages": [
#       {"role":"system","content":"你是一个擅长混合精度/MoE 的助手"},
#       {"role":"user","content":"用一段话解释混合精度推理的优势"}
#     ],
#     "max_tokens": 128,
#     "temperature": 0.7,
#     "top_p": 0.9
#   }'
