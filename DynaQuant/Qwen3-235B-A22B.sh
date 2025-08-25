#/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m sglang.launch_server \
  --model-path /dcar-vepfs-trans-models/Qwen3-235B-A22B-FP8 \
  --tp-size 4 --dp-size 1 --ep-size 1\
  --enable-ep-moe \
  --max-running-requests 32 \
  --host 127.0.0.1 --port 8080 \
  --max-total-tokens 40960 \
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
  --mixed-precision-config ./sglang-0.4.7/mixed_precision_config.yaml

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
