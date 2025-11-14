# python scripts/evaluate_mmlu_perplexity_mixed.py --fp16 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound   --int4 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int2-mixed-AutoRound   --activation-file ./activations/activation_qwen30b_mmlu_pro_sorted.json   --tail-count 0   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 30b-0.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py --fp16 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound   --int4 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int2-mixed-AutoRound   --activation-file ./activations/activation_qwen30b_mmlu_pro_sorted.json   --tail-count 19   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 30b-15.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py --fp16 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound   --int4 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int2-mixed-AutoRound   --activation-file ./activations/activation_qwen30b_mmlu_pro_sorted.json   --tail-count 38   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 30b-30.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py --fp16 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound   --int4 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int2-mixed-AutoRound   --activation-file ./activations/activation_qwen30b_mmlu_pro_sorted.json   --tail-count 58   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 30b-45.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py --fp16 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound   --int4 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int2-mixed-AutoRound   --activation-file ./activations/activation_qwen30b_mmlu_pro_sorted.json   --tail-count 77   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 30b-60.txt   
# python scripts/evaluate_mmlu_perplexity_mixed.py --fp16 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound   --int4 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int2-mixed-AutoRound   --activation-file ./activations/activation_qwen30b_mmlu_pro_sorted.json   --tail-count 96   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 30b-75.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py --fp16 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound   --int4 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int2-mixed-AutoRound   --activation-file ./activations/activation_qwen30b_mmlu_pro_sorted.json   --tail-count 115   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 30b-90.txt
# python scripts/evaluate_mmlu_perplexity_mixed.py --fp16 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound   --int4 /workspace/Models/Qwen3-30B-A3B-Instruct-2507-int2-mixed-AutoRound   --activation-file ./activations/activation_qwen30b_mmlu_pro_sorted.json   --tail-count 128   --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 30b-100.txt


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
# # sort
# python scripts/sort_expert_activations.py \
#     --input /workspace/DynaQuant/DynaQuant_New/activations/activation_qwen80b_mmlu_pro.json \
#     --output /workspace/DynaQuant/DynaQuant_New/activations/activation_qwen80b_mmlu_pro_sorted.json
# ppl
python scripts/evaluate_mmlu_perplexity_mixed.py \
    --fp16 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
    --int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
    --activation-file ./activations/activation_qwen80b_mmlu_pro_sorted.json \
    --tail-count 0 \
    --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 80b-0.txt
python scripts/evaluate_mmlu_perplexity_mixed.py \
    --fp16 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
    --int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
    --activation-file ./activations/activation_qwen80b_mmlu_pro_sorted.json \
    --tail-count 77 \
    --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 80b-15.txt
python scripts/evaluate_mmlu_perplexity_mixed.py \
    --fp16 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
    --int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
    --activation-file ./activations/activation_qwen80b_mmlu_pro_sorted.json \
    --tail-count 153 \
    --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 80b-30.txt
python scripts/evaluate_mmlu_perplexity_mixed.py \
    --fp16 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
    --int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
    --activation-file ./activations/activation_qwen80b_mmlu_pro_sorted.json \
    --tail-count 230 \
    --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 80b-45.txt
python scripts/evaluate_mmlu_perplexity_mixed.py \
    --fp16 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
    --int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
    --activation-file ./activations/activation_qwen80b_mmlu_pro_sorted.json \
    --tail-count 307 \
    --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 80b-60.txt
python scripts/evaluate_mmlu_perplexity_mixed.py \
    --fp16 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
    --int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
    --activation-file ./activations/activation_qwen80b_mmlu_pro_sorted.json \
    --tail-count 384 \
    --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 80b-75.txt
python scripts/evaluate_mmlu_perplexity_mixed.py \
    --fp16 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
    --int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
    --activation-file ./activations/activation_qwen80b_mmlu_pro_sorted.json \
    --tail-count 460 \
    --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 80b-90.txt
python scripts/evaluate_mmlu_perplexity_mixed.py \
    --fp16 /workspace/Models/Qwen3-80B-A3B-Instruct-int4-mixed-AutoRound \
    --int4 /workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound \
    --activation-file ./activations/activation_qwen80b_mmlu_pro_sorted.json \
    --tail-count 512 \
    --dataset calibration_datasets/requests/wikitext2_128x2048.jsonl > 80b-100.txt