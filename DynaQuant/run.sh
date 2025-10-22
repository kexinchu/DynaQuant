#/bin/sh

python3 scripts/quantize_w2a16.py --model /dev/shm/Qwen3-30B-A3B --output-dir /dev/shm/Qwen3-30B-A3B-W2A16 --num-samples 1024 --group-size 128 --moe
# python3 scripts/quantize_w2a16.py --model /dev/shm/Qwen3-Next-80B-A3B --output-dir /dev/shm/Qwen3-Next-80B-A3B-W2A16 --num-samples 1024 --group-size 128 --moe

# python scripts/quantize_w4a16.py --model /dev/shm/Qwen3-Next-80B-A3B --output-dir /dev/shm/Qwen3-Next-80B-A3B-W4A16 --num-samples 512 --max-seq-length 8192

# python scripts/evaluate_model.py \
#     --model /dev/shm/Qwen3-30B-A3B-W4A16 \
#     --datasets wikitext mmlu gsm8k

# python scripts/evaluate_model.py \
#     --model /dev/shm/Qwen3-30B-A3B-W2A16 \
#     --datasets wikitext mmlu gsm8k

# python scripts/evaluate_model.py \
#     --model /dev/shm/Qwen3-Next-80B-A3B-W4A16 \
#     --datasets wikitext mmlu gsm8k

# python scripts/evaluate_model.py \
#     --model /dev/shm/Qwen3-Next-80B-A3B-W2A16 \
#     --datasets wikitext mmlu gsm8k