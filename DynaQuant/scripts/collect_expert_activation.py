import torch
import json
import os
import argparse
import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def load_dataset_prompts(dataset_path, dataset_name, num_samples=256):
    """
    从本地parquet文件加载数据集并抽取样本作为prompts。
    参数:
        dataset_path (str): parquet文件路径
        dataset_name (str): 数据集名称，用于确定处理逻辑 "wikitext", "gsm8k", "humaneval"
        num_samples (int): 抽取样本数量
    返回:
        prompts (list[str]): prompt列表
    """
    import random
    random.seed(42)

    # 读取parquet文件
    print(f"  正在读取文件: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    print(f"  文件包含 {len(df)} 条记录")

    prompts = []

    if dataset_name == "wikitext":
        # WikiText数据集处理
        # 假设parquet文件有 'text' 列
        if 'text' in df.columns:
            texts = [str(text).strip()
                     for text in df['text'] if len(str(text).strip()) > 50]
        else:
            # 尝试其他可能的列名
            text_col = df.columns[0]
            print(f"  使用列: {text_col}")
            texts = [str(text).strip()
                     for text in df[text_col] if len(str(text).strip()) > 50]

        # 采样指定数量
        sampled = random.sample(texts, min(num_samples, len(texts)))
        # 构造简单的提问格式
        prompts = [f"请解释以下内容：{text[:200]}" for text in sampled]

    elif dataset_name == "gsm8k":
        # GSM8K数据集处理
        # 假设parquet文件有 'question' 列
        if 'question' in df.columns:
            questions = [str(q).strip() for q in df['question']]
        else:
            # 尝试其他可能的列名
            text_col = df.columns[0]
            print(f"  使用列: {text_col}")
            questions = [str(q).strip() for q in df[text_col]]

        sampled = random.sample(questions, min(num_samples, len(questions)))
        prompts = sampled

    elif dataset_name == "humaneval":
        # HumanEval数据集处理
        # 假设parquet文件有 'prompt' 列
        if 'prompt' in df.columns:
            problems = [str(p).strip() for p in df['prompt']]
        else:
            # 尝试其他可能的列名
            text_col = df.columns[0]
            print(f"  使用列: {text_col}")
            problems = [str(p).strip() for p in df[text_col]]

        # HumanEval样本较少，可能需要重复采样
        if len(problems) >= num_samples:
            sampled = random.sample(problems, num_samples)
        else:
            # 重复采样以达到目标数量
            sampled = random.choices(problems, k=num_samples)
        prompts = [f"请完成以下Python函数：\n{p}" for p in sampled]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return prompts


def collect_expert_distribution(model, tokenizer, prompts, enable_thinking=True, debug=False):
    """
    统计给定 prompts 列表在 Qwen3 MoE 模型各层的专家激活频率。
    参数:
        model, tokenizer: Qwen3-30B-A3B 模型及其 tokenizer。
        prompts (list[str]): 待分析的用户输入列表，每个元素是一条完整的用户问题。
        enable_thinking (bool): 是否启用"思考模式"；思考模式会插入 <think> 标签。
        debug (bool): 是否打印调试信息
    返回:
        expert_counts (list[dict]): 长度为 MoE 层数的列表，每个元素是 {expert_id: count} 计数字典。
    """
    # 初始化计数器: 每层有 num_experts 个专家
    num_layers = getattr(model.config, "num_hidden_layers", 48)
    num_experts = getattr(model.config, "num_experts", 128)
    k = getattr(model.config, "num_experts_per_tok", 8)  # top‑k
    expert_counts = [defaultdict(int) for _ in range(num_layers)]

    if debug:
        print(f"\n[DEBUG] 模型配置:")
        print(f"  num_layers: {num_layers}")
        print(f"  num_experts: {num_experts}")
        print(f"  num_experts_per_tok (k): {k}")

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        # 构造聊天模板，enable_thinking 控制 <think> 部分
        messages = [{"role": "user", "content": prompt.strip()}]
        # Qwen3 提供 apply_chat_template 帮助包裹对话格式
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking
        )
        # 编码
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        if debug and prompt_idx == 0:
            print(
                f"\n[DEBUG] 第一个 prompt 的 token 数量: {inputs['input_ids'].shape[1]}")

        # 前向推理：设置 output_router_logits=True 以获得路由层输出
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_router_logits=True,
                return_dict=True,
                use_cache=False  # 统计 router logits 时不允许缓存
            )

        # router_logits 是一个包含每个 MoE 层路由 logits 的元组，
        # 某些层可能没有 MoE（返回 None），其形状为 (batch, seq_len, num_experts)
        router_logits = outputs.router_logits

        if debug and prompt_idx == 0:
            print(f"\n[DEBUG] router_logits 信息:")
            print(f"  类型: {type(router_logits)}")
            print(f"  长度: {len(router_logits) if router_logits else 'None'}")
            if router_logits:
                for i, logits in enumerate(router_logits[:3]):  # 只看前3层
                    if logits is not None:
                        print(
                            f"  Layer {i}: shape={logits.shape}, dtype={logits.dtype}")
        for layer_idx, logits in enumerate(router_logits):
            if logits is None:
                continue

            # 处理不同维度的 logits
            # logits 可能是 [seq_len, num_experts] (batch=1时) 或 [batch, seq_len, num_experts]
            if logits.dim() == 3:
                # [batch=1, seq_len, num_experts]
                seq_logits = logits[0]  # shape [seq_len, num_experts]
                # 对每个 token 计算 top‑k expert id
                topk = torch.topk(seq_logits, k, dim=-
                                  1).indices  # [seq_len, k]

                if debug and prompt_idx == 0 and layer_idx == 0:
                    print(f"\n[DEBUG] Layer {layer_idx} (3D) 专家选择:")
                    print(f"  seq_len: {seq_logits.shape[0]}")
                    print(f"  前3个token的top-{k}专家: {topk[:3].tolist()}")

                for token_topk in topk:
                    for expert_id in token_topk.tolist():
                        expert_counts[layer_idx][expert_id] += 1
            elif logits.dim() == 2:
                # [seq_len, num_experts] - batch=1时transformers会自动squeeze掉batch维度
                seq_logits = logits  # shape [seq_len, num_experts]

                # 检查是否确实是 [seq_len, num_experts] 格式
                if seq_logits.shape[1] == num_experts:
                    # 对每个 token 计算 top‑k expert id
                    topk = torch.topk(seq_logits, k, dim=-
                                      1).indices  # [seq_len, k]

                    if debug and prompt_idx == 0 and layer_idx == 0:
                        print(f"\n[DEBUG] Layer {layer_idx} (2D) 专家选择:")
                        print(
                            f"  seq_len: {seq_logits.shape[0]}, num_experts: {seq_logits.shape[1]}")
                        print(f"  前3个token的top-{k}专家: {topk[:3].tolist()}")

                    # 遍历所有token的专家选择
                    for token_topk in topk:
                        for expert_id in token_topk.tolist():
                            expert_counts[layer_idx][expert_id] += 1
                else:
                    # 如果第二维不是num_experts，可能是单个token的情况
                    print(
                        f"Warning: Unexpected 2D shape {seq_logits.shape} at layer {layer_idx}")
            else:
                print(
                    f"Warning: Unexpected logits dimension {logits.dim()} at layer {layer_idx}")
                continue

    return expert_counts


def summarize_distribution(expert_counts, num_experts):
    """
    根据每层专家计数统计分布信息，返回每层 top‑5 激活专家及其比例。
    """
    summary = []
    for layer_idx, counts in enumerate(expert_counts):
        total = sum(counts.values())
        if total == 0:
            summary.append((layer_idx, []))
            continue
        # 按频次降序获取 top‑5
        top5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top5 = [(eid, c / total) for eid, c in top5]
        summary.append((layer_idx, top5))
    return summary


def save_results_to_json(results, output_dir="./benchmark_results/expert_activation_results"):
    """
    将专家激活统计结果保存为JSON文件。
    参数:
        results (dict): 包含所有实验结果的字典
        output_dir (str): 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    for exp_name, expert_counts in results.items():
        # 将 defaultdict 转换为普通 dict 以便序列化
        serializable_counts = []
        for layer_counts in expert_counts:
            serializable_counts.append(dict(layer_counts))

        output_file = os.path.join(output_dir, f"{exp_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_counts, f, indent=2, ensure_ascii=False)
        print(f"已保存结果到: {output_file}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Expert Activation Analysis for MoE Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行所有实验（6组）
  python collect_expert_activation.py --all

  # 只测试WikiText数据集
  python collect_expert_activation.py --datasets wikitext

  # 测试多个数据集，不开启thinking
  python collect_expert_activation.py --datasets wikitext gsm8k --no-thinking

  # 快速测试（少量样本）
  python collect_expert_activation.py --datasets wikitext --num-samples 32 --thinking-only

  # 指定模型路径和输出目录
  python collect_expert_activation.py --all --model-path /path/to/model --output-dir ./my_results
        """
    )

    # 数据集选择
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['wikitext', 'gsm8k', 'humaneval'],
        help='选择要测试的数据集（可多选）'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='运行所有数据集的所有实验（6组）'
    )

    # Thinking模式选择
    parser.add_argument(
        '--thinking-only',
        action='store_true',
        help='只测试thinking开启的情况'
    )
    parser.add_argument(
        '--no-thinking',
        action='store_true',
        help='只测试thinking关闭的情况'
    )

    # 模型和数据配置
    parser.add_argument(
        '--model-path',
        type=str,
        default='/dev/shm/Qwen3-30B-A3B',
        help='模型路径（默认: /dev/shm/Qwen3-30B-A3B）'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=256,
        help='每个数据集的样本数量（默认: 256）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./benchmark_results/expert_activation_results',
        help='结果输出目录（默认: ./benchmark_results/expert_activation_results）'
    )

    # 数据集路径配置
    parser.add_argument(
        '--wikitext-path',
        type=str,
        default='./data/Wikitext/arz_train-00000-of-00003.parquent',
        help='WikiText数据集路径'
    )
    parser.add_argument(
        '--gsm8k-path',
        type=str,
        default='./data/GSM8K/test-00000-of-00001.parquet',
        help='GSM8K数据集路径'
    )
    parser.add_argument(
        '--humaneval-path',
        type=str,
        default='./data/HumanEval/test-00000-of-00001.parquet',
        help='HumanEval数据集路径'
    )

    args = parser.parse_args()

    # 参数验证
    if not args.all and not args.datasets:
        parser.error("必须指定 --all 或 --datasets")

    if args.thinking_only and args.no_thinking:
        parser.error("--thinking-only 和 --no-thinking 不能同时使用")

    return args


def run_analysis(args):
    """
    运行专家激活分析实验
    参数:
        args: 命令行参数
    """
    # 加载模型
    model_name = args.model_path
    print(f"{'='*80}")
    print(f"Expert Activation Analysis")
    print(f"{'='*80}")
    print(f"\n加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    print(f"模型加载完成，设备: {model.device}")

    # 数据集配置
    dataset_paths = {
        "wikitext": args.wikitext_path,
        "gsm8k": args.gsm8k_path,
        "humaneval": args.humaneval_path
    }

    dataset_names_cn = {
        "wikitext": "通用闲聊 (WikiText)",
        "gsm8k": "数学推理 (GSM8K)",
        "humaneval": "代码生成 (HumanEval)"
    }

    # 确定要测试的数据集
    if args.all:
        datasets = ["wikitext", "gsm8k", "humaneval"]
    else:
        datasets = args.datasets

    # 确定要测试的thinking模式
    if args.thinking_only:
        thinking_modes = [True]
    elif args.no_thinking:
        thinking_modes = [False]
    else:
        thinking_modes = [False, True]

    # 打印实验配置
    print(f"\n实验配置:")
    print(f"  数据集: {', '.join(datasets)}")
    print(
        f"  Thinking模式: {', '.join(['On' if t else 'Off' for t in thinking_modes])}")
    print(f"  每个数据集样本数: {args.num_samples}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  总实验数: {len(datasets) * len(thinking_modes)}")

    # 存储所有实验结果
    all_results = {}

    # 运行实验
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"数据集: {dataset_names_cn[dataset_name]}")
        print(f"{'='*60}")

        # 加载数据集prompts
        dataset_path = dataset_paths[dataset_name]
        prompts = load_dataset_prompts(
            dataset_path, dataset_name, num_samples=args.num_samples
        )
        print(f"  成功加载 {len(prompts)} 个样本")

        for enable_thinking in thinking_modes:
            thinking_str = "thinking_on" if enable_thinking else "thinking_off"
            exp_name = f"{dataset_name}_{thinking_str}"

            print(f"\n▶ 实验: {exp_name}")
            print(f"  Thinking模式: {'开启' if enable_thinking else '关闭'}")

            # 统计专家激活分布 (第一个实验开启调试)
            is_first_exp = (
                dataset_name == datasets[0] and enable_thinking == thinking_modes[0])
            expert_counts = collect_expert_distribution(
                model, tokenizer, prompts, enable_thinking=enable_thinking, debug=is_first_exp
            )

            # 保存结果
            all_results[exp_name] = expert_counts

            # 打印简要统计
            summary = summarize_distribution(
                expert_counts, model.config.num_experts)
            print(f"  ✓ 完成统计，共 {len(summary)} 层")

            # 显示第一层和最后一层的top-5专家
            if len(summary) > 0 and len(summary[0][1]) > 0:
                print(f"  第1层 top-3 专家: {summary[0][1][:3]}")
                if len(summary) > 1 and len(summary[-1][1]) > 0:
                    print(f"  第{len(summary)}层 top-3 专家: {summary[-1][1][:3]}")

    # 保存所有结果到JSON文件
    print(f"\n{'='*60}")
    print("保存结果到JSON文件")
    print(f"{'='*60}")
    save_results_to_json(all_results, output_dir=args.output_dir)

    print(f"\n{'='*80}")
    print(f"✓ 所有实验完成！")
    print(f"{'='*80}")
    print(f"  完成实验数: {len(all_results)}")
    print(f"  结果目录: {args.output_dir}")
    print(f"\n查看分析结果:")
    print(
        f"  python analyze_expert_activation.py --results-dir {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)
