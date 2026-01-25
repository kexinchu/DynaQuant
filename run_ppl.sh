python scripts/evaluate_mmlu_perplexity_mixed.py --fp16 /workspace/Models/gpt-oss-20b   --int4 /workspace/Models/gpt-oss-20b-mixed-AutoRound   --activation-file ./activations/activation_gpt-oss-20b_mmlu_pro_sorted.json   --tail-count 0   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > gpt-oss-20b-0.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py --fp16 /workspace/Models/gpt-oss-20b   --int4 /workspace/Models/gpt-oss-20b-mixed-AutoRound   --activation-file ./activations/activation_gpt-oss-20b_mmlu_pro_sorted.json   --tail-count 5   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > gpt-oss-20b-15.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py --fp16 /workspace/Models/gpt-oss-20b   --int4 /workspace/Models/gpt-oss-20b-mixed-AutoRound   --activation-file ./activations/activation_gpt-oss-20b_mmlu_pro_sorted.json   --tail-count 9   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > gpt-oss-20b-30.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py --fp16 /workspace/Models/gpt-oss-20b   --int4 /workspace/Models/gpt-oss-20b-mixed-AutoRound   --activation-file ./activations/activation_gpt-oss-20b_mmlu_pro_sorted.json   --tail-count 14   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > gpt-oss-20b-45.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py --fp16 /workspace/Models/gpt-oss-20b   --int4 /workspace/Models/gpt-oss-20b-mixed-AutoRound   --activation-file ./activations/activation_gpt-oss-20b_mmlu_pro_sorted.json   --tail-count 19   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > gpt-oss-20b-60.txt   
# python scripts/evaluate_mmlu_perplexity_mixed.py --fp16 /workspace/Models/gpt-oss-20b   --int4 /workspace/Models/gpt-oss-20b-mixed-AutoRound   --activation-file ./activations/activation_gpt-oss-20b_mmlu_pro_sorted.json   --tail-count 24   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > gpt-oss-20b-75.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py --fp16 /workspace/Models/gpt-oss-20b   --int4 /workspace/Models/gpt-oss-20b-mixed-AutoRound   --activation-file ./activations/activation_gpt-oss-20b_mmlu_pro_sorted.json   --tail-count 29   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > gpt-oss-20b-90.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py --fp16 /workspace/Models/gpt-oss-20b   --int4 /workspace/Models/gpt-oss-20b-mixed-AutoRound   --activation-file ./activations/activation_gpt-oss-20b_mmlu_pro_sorted.json   --tail-count 32   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > gpt-oss-20b-100.txt


# 获取激活
# python scripts/summarize_expert_activation.py \
#     --dataset calibration_datasets/requests/mmlu_pro_200.jsonl \
#     --model-id /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
#     --device cuda:0 \
#     --max-prompts 50 \
#     --max-length 1 \
#     --top-k 10 \
#     --output activations/activation_qwen80b_mmlu_pro.json \
#     --quantization autoround-int2
# sort
# python scripts/sort_expert_activations.py \
#     --input /workspace/DynaQuant/DynaQuant_New/activations/activation_qwen80b_mmlu_pro.json \
#     --output /workspace/DynaQuant/DynaQuant_New/activations/activation_qwen80b_mmlu_pro_sorted.json
# ppl
# python scripts/evaluate_mmlu_perplexity_mixed.py \
#     --fp16 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
#     --int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
#     --activation-file ./activations/activation_qwen80b_mmlu_pro_sorted.json \
#     --tail-count 0 \
#     --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 80b-0.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py \
#     --fp16 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
#     --int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
#     --activation-file ./activations/activation_qwen80b_mmlu_pro_sorted.json \
#     --tail-count 77 \
#     --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 80b-15.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py \
#     --fp16 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
#     --int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
#     --activation-file ./activations/activation_qwen80b_mmlu_pro_sorted.json \
#     --tail-count 153 \
#     --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 80b-30.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py \
#     --fp16 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
#     --int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
#     --activation-file ./activations/activation_qwen80b_mmlu_pro_sorted.json \
#     --tail-count 230 \
#     --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 80b-45.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py \
#     --fp16 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
#     --int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
#     --activation-file ./activations/activation_qwen80b_mmlu_pro_sorted.json \
#     --tail-count 307 \
#     --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 80b-60.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py \
#     --fp16 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
#     --int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
#     --activation-file ./activations/activation_qwen80b_mmlu_pro_sorted.json \
#     --tail-count 384 \
#     --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 80b-75.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py \
#     --fp16 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
#     --int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
#     --activation-file ./activations/activation_qwen80b_mmlu_pro_sorted.json \
#     --tail-count 460 \
#     --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 80b-90.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py \
#     --fp16 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
#     --int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
#     --activation-file ./activations/activation_qwen80b_mmlu_pro_sorted.json \
#     --tail-count 512 \
#     --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 80b-100.txt


# python3 scripts/benchmark_latency.py \
#   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl \
#   --qwen30-fp16 /workspace/Models/Qwen3-30B-A3B-Instruct-2507 \
#   --max-prompts 120 \
#   --max-new-tokens 256 \
#   --output scripts/results/latency_summary_30b_fp16.json

# python3 scripts/benchmark_latency_transformers.py \
#   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl \
#   --qwen80-int2 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
#   --max-prompts 120 \
#   --max-new-tokens 256 \
#   --output scripts/results/latency_summary_80b_int4.json

