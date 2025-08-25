#/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m sglang.launch_server \
  --model-path /dcar-vepfs-trans-models/Qwen3-30B-A3B \
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
