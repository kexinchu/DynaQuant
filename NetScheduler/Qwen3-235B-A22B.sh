#/bin/sh
export SGLANG_DISABLE_MARLIN=1
export SGL_DISABLE_AWQ_MARLIN=1
export SGLANG_DISABLE_SGL_KERNEL=1

# 拆分Qwen3-MoE: multi-experts to single-expert
# python3 prune_qwen3_a22b_to_single_expert.py \
#  --src "/dcar-vepfs-trans-models/Qwen3-30B-A3B" \
#  --dst "/dev/shm/Qwen3-30B-A3B" \
#  --expert-id 0 \
#  --dry-run  # 测试，注释之后开始真正的写入weights

# 场景A Expert使用DP，复制8份
export SINGLE_EXPERT_MODE=dp
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m sglang.launch_server \
  --model-path /dev/shm/Qwen3-30B-A3B \
  --tp-size 1 \
  --dp-size 8 \
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

# expert size 的影响
# network的影响： attention层的TP (all2all的通信量) <确认一下>；模拟一下带宽
# workload ：相同长度prompt, QPS增加； 相同QPS，长度增加,超长prompt

# 场景B Expert使用TP，切8片
export SINGLE_EXPERT_MODE=tp
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python3 -m sglang.launch_server \
#   --model-path /dev/shm/Qwen3-30B-A3B \
#   --tp 8 \
#   --pp 1 \
#   --max-running-requests 32 \
#   --host 0.0.0.0 --port 8080
# #   --args-json '{"single_expert_mode":"tp"}' \