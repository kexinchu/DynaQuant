#!/usr/bin/env python3
"""
EBSS数据集生成脚本 - 独立版本

为MoE模型生成专家均衡的校准数据集
不依赖moe_quant模块，直接使用transformers

支持的模型:
- Qwen3-30B-A3B
- Qwen3-Next-80B-A3B
- Qwen3-235B-A22B
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """加载和处理不同类型的数据集"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def load_seed_texts(self, num_samples: int = 20) -> List[str]:
        """加载seed_text.txt文件"""
        seed_file = self.data_dir / "seed_text.txt"

        if not seed_file.exists():
            logger.warning(f"seed_text.txt 不存在，返回空列表")
            return []

        with open(seed_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        # 如果样本不够，循环复制
        if len(texts) < num_samples:
            texts = texts * (num_samples // len(texts) + 1)

        return texts[:num_samples]

    def load_wikitext(self, num_samples: int = 100, min_length: int = 50) -> List[str]:
        """加载Wikitext数据集"""
        wikitext_dir = self.data_dir / "Wikitext"

        if not wikitext_dir.exists():
            logger.warning(f"Wikitext目录不存在: {wikitext_dir}")
            return []

        texts = []

        # 尝试加载parquet文件（注意：文件扩展名可能是 .parquent）
        try:
            import pandas as pd

            # 同时查找 .parquet 和 .parquent 文件
            parquet_files = list(wikitext_dir.glob(
                "*.parquet")) + list(wikitext_dir.glob("*.parquent"))

            for parquet_file in parquet_files:
                try:
                    df = pd.read_parquet(parquet_file)
                    logger.info(f"读取 {parquet_file.name}: {len(df)} 行")

                    if 'text' in df.columns:
                        for text in df['text']:
                            if isinstance(text, str) and len(text) >= min_length:
                                texts.append(text)
                                if len(texts) >= num_samples:
                                    logger.info(
                                        f"Wikitext已收集足够样本: {len(texts)}")
                                    return texts[:num_samples]
                except Exception as e:
                    logger.warning(f"读取 {parquet_file} 失败: {e}")

            logger.info(f"Wikitext总共收集到 {len(texts)} 个样本")
        except ImportError:
            logger.warning("pandas未安装，跳过Wikitext")

        return texts[:num_samples]

    def load_chinese_c4(self, num_samples: int = 100, min_length: int = 50) -> List[str]:
        """加载Chinese-C4数据集 (.jsonl.zst格式)"""
        c4_dir = self.data_dir / "Chinese-C4"

        if not c4_dir.exists():
            logger.warning(f"Chinese-C4目录不存在: {c4_dir}")
            return []

        texts = []

        # 查找 .jsonl.zst 文件
        zst_files = list(c4_dir.glob("*.jsonl.zst"))

        if not zst_files:
            logger.warning(f"在 {c4_dir} 中未找到 .jsonl.zst 文件")
            return []

        try:
            import zstandard as zstd

            logger.info(f"找到 {len(zst_files)} 个 Chinese-C4 文件")

            # 只处理第一个文件（避免加载太多数据）
            for zst_file in zst_files[:1]:
                try:
                    logger.info(f"正在读取 {zst_file.name}...")

                    # 使用 zstandard 解压
                    dctx = zstd.ZstdDecompressor()

                    with open(zst_file, 'rb') as f_in:
                        with dctx.stream_reader(f_in) as reader:
                            # 流式读取并解析
                            buffer = b''
                            chunk_size = 1024 * 1024  # 每次读取1MB

                            while len(texts) < num_samples:
                                # 读取数据块
                                chunk = reader.read(chunk_size)
                                if not chunk:
                                    break

                                buffer += chunk

                                # 按行分割
                                lines = buffer.split(b'\n')
                                buffer = lines[-1]  # 保留不完整的行

                                # 处理完整的行
                                for line_bytes in lines[:-1]:
                                    try:
                                        line_str = line_bytes.decode(
                                            'utf-8').strip()
                                        if not line_str:
                                            continue

                                        item = json.loads(line_str)
                                        text = item.get(
                                            'text') or item.get('content')

                                        if text and isinstance(text, str) and len(text) >= min_length:
                                            texts.append(text)

                                            if len(texts) >= num_samples:
                                                logger.info(
                                                    f"Chinese-C4已收集足够样本: {len(texts)}")
                                                return texts[:num_samples]

                                    except (json.JSONDecodeError, UnicodeDecodeError):
                                        continue

                    logger.info(f"从 {zst_file.name} 收集到 {len(texts)} 个样本")

                except Exception as e:
                    logger.warning(f"读取 {zst_file} 失败: {e}")

            logger.info(f"Chinese-C4总共收集到 {len(texts)} 个样本")

        except ImportError:
            logger.warning("zstandard未安装，跳过Chinese-C4")
            logger.info("安装方法: pip install zstandard")
            return []

        return texts[:num_samples]

    def load_mmlu(self, num_samples: int = 50) -> List[str]:
        """加载MMLU数据集"""
        mmlu_dir = self.data_dir / "MMLU/data/dev"

        if not mmlu_dir.exists():
            logger.warning(f"MMLU目录不存在: {mmlu_dir}")
            return []

        texts = []

        try:
            import pandas as pd

            for csv_file in list(mmlu_dir.glob("*.csv"))[:10]:  # 前10个任务
                try:
                    df = pd.read_csv(csv_file, header=None)
                    for _, row in df.iterrows():
                        # 构建问题文本
                        question = str(row[0])
                        options = [
                            f"{chr(65+i)}: {str(row[i+1])}" for i in range(4) if i+1 < len(row)]
                        text = f"{question}\n" + "\n".join(options)
                        texts.append(text)

                        if len(texts) >= num_samples:
                            return texts[:num_samples]
                except Exception as e:
                    logger.warning(f"读取 {csv_file} 失败: {e}")
        except ImportError:
            logger.warning("pandas未安装，跳过MMLU")

        return texts[:num_samples]

    def load_gsm8k(self, num_samples: int = 50) -> List[str]:
        """加载GSM8K数据集（支持parquet和jsonl格式）"""
        gsm8k_dir = self.data_dir / "GSM8K"

        if not gsm8k_dir.exists():
            logger.warning(f"GSM8K目录不存在: {gsm8k_dir}")
            return []

        texts = []

        # 方法1: 尝试加载parquet文件
        try:
            import pandas as pd

            parquet_files = list(gsm8k_dir.glob("*.parquet")) + \
                list(gsm8k_dir.glob("*.parquent"))

            for parquet_file in parquet_files:
                try:
                    df = pd.read_parquet(parquet_file)
                    logger.info(f"读取 GSM8K: {parquet_file.name} ({len(df)} 行)")

                    if 'question' in df.columns:
                        for question in df['question']:
                            if isinstance(question, str) and len(question) > 10:
                                texts.append(question)
                                if len(texts) >= num_samples:
                                    logger.info(f"GSM8K已收集足够样本: {len(texts)}")
                                    return texts[:num_samples]
                    elif 'problem' in df.columns:
                        for problem in df['problem']:
                            if isinstance(problem, str) and len(problem) > 10:
                                texts.append(problem)
                                if len(texts) >= num_samples:
                                    logger.info(f"GSM8K已收集足够样本: {len(texts)}")
                                    return texts[:num_samples]
                except Exception as e:
                    logger.warning(f"读取 {parquet_file} 失败: {e}")
        except ImportError:
            logger.warning("pandas未安装，跳过GSM8K parquet格式")

        # 方法2: 尝试加载jsonl文件
        for jsonl_file in gsm8k_dir.glob("**/*.jsonl"):
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            if 'question' in item:
                                texts.append(item['question'])
                            elif 'problem' in item:
                                texts.append(item['problem'])

                            if len(texts) >= num_samples:
                                return texts[:num_samples]
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"读取 {jsonl_file} 失败: {e}")

        logger.info(f"GSM8K总共收集到 {len(texts)} 个样本")
        return texts[:num_samples]

    def load_humaneval(self, num_samples: int = 50) -> List[str]:
        """加载HumanEval数据集（代码生成任务）"""
        humaneval_dir = self.data_dir / "HumanEval"

        if not humaneval_dir.exists():
            logger.warning(f"HumanEval目录不存在: {humaneval_dir}")
            return []

        texts = []

        try:
            import pandas as pd

            # 查找parquet文件
            parquet_files = list(humaneval_dir.glob("*.parquet")) + \
                list(humaneval_dir.glob("*.parquent")) + \
                list(humaneval_dir.glob("**/*.parquet")) + \
                list(humaneval_dir.glob("**/*.parquent"))

            for parquet_file in parquet_files:
                try:
                    df = pd.read_parquet(parquet_file)
                    logger.info(
                        f"读取 HumanEval: {parquet_file.name} ({len(df)} 行)")

                    # HumanEval通常包含: prompt, canonical_solution, test等字段
                    if 'prompt' in df.columns:
                        for prompt in df['prompt']:
                            if isinstance(prompt, str) and len(prompt) > 20:
                                texts.append(prompt)
                                if len(texts) >= num_samples:
                                    logger.info(
                                        f"HumanEval已收集足够样本: {len(texts)}")
                                    return texts[:num_samples]
                    elif 'task_id' in df.columns and 'prompt' in df.columns:
                        # 完整格式：task_id + prompt
                        for _, row in df.iterrows():
                            if pd.notna(row.get('prompt')):
                                prompt = str(row['prompt'])
                                if len(prompt) > 20:
                                    texts.append(prompt)
                                    if len(texts) >= num_samples:
                                        logger.info(
                                            f"HumanEval已收集足够样本: {len(texts)}")
                                        return texts[:num_samples]
                except Exception as e:
                    logger.debug(f"读取 {parquet_file} 失败: {e}")
        except ImportError:
            logger.warning("pandas未安装，跳过HumanEval")

        logger.info(f"HumanEval总共收集到 {len(texts)} 个样本")
        return texts[:num_samples]

    def load_hellaswag(self, num_samples: int = 50) -> List[str]:
        """加载HELLASWAG数据集（常识推理任务）"""
        hellaswag_dir = self.data_dir / "HELLASWAG"

        if not hellaswag_dir.exists():
            logger.warning(f"HELLASWAG目录不存在: {hellaswag_dir}")
            return []

        texts = []

        try:
            import pandas as pd

            # 查找parquet文件
            parquet_files = list(hellaswag_dir.glob("*.parquet")) + \
                list(hellaswag_dir.glob("*.parquent")) + \
                list(hellaswag_dir.glob("**/*.parquet")) + \
                list(hellaswag_dir.glob("**/*.parquent"))

            for parquet_file in parquet_files:
                try:
                    df = pd.read_parquet(parquet_file)
                    logger.info(
                        f"读取 HELLASWAG: {parquet_file.name} ({len(df)} 行)")

                    # HELLASWAG通常包含: ctx, endings, activity_label等字段
                    if 'ctx' in df.columns:
                        # ctx是上下文，endings是多个可能的结局
                        for idx, row in df.iterrows():
                            ctx = str(row['ctx']) if pd.notna(
                                row.get('ctx')) else ""

                            if not ctx or len(ctx) < 20:
                                continue

                            # 如果有endings字段，添加选项
                            if 'endings' in df.columns:
                                try:
                                    endings = row['endings']
                                    # endings 可能是 numpy.ndarray, list, tuple, 或 str
                                    if hasattr(endings, '__iter__') and not isinstance(endings, str):
                                        # 转换为列表
                                        endings_list = list(endings) if not isinstance(
                                            endings, list) else endings

                                        if len(endings_list) > 0:
                                            # 构建完整问题
                                            text = f"{ctx}\nChoose the best continuation:\n"
                                            # 最多4个选项
                                            for i, ending in enumerate(endings_list[:4]):
                                                text += f"{chr(65+i)}: {ending}\n"
                                            texts.append(text.strip())
                                        else:
                                            texts.append(ctx)
                                    else:
                                        texts.append(ctx)
                                except Exception as e:
                                    logger.debug(f"处理endings失败: {e}")
                                    texts.append(ctx)
                            else:
                                texts.append(ctx)

                            if len(texts) >= num_samples:
                                logger.info(f"HELLASWAG已收集足够样本: {len(texts)}")
                                return texts[:num_samples]
                    elif 'context' in df.columns:
                        # 另一种格式
                        for context in df['context']:
                            if isinstance(context, str) and len(context) > 20:
                                texts.append(context)
                                if len(texts) >= num_samples:
                                    logger.info(
                                        f"HELLASWAG已收集足够样本: {len(texts)}")
                                    return texts[:num_samples]
                except Exception as e:
                    logger.debug(f"读取 {parquet_file} 失败: {e}")
        except ImportError:
            logger.warning("pandas未安装，跳过HELLASWAG")

        logger.info(f"HELLASWAG总共收集到 {len(texts)} 个样本")
        return texts[:num_samples]

    def load_mixed_dataset(
        self,
        total_samples: int = 1024,
        seed_ratio: float = 0.05,       # 5% 种子文本
        wikitext_ratio: float = 0.2,   # 20% 英文通用文本
        chinese_ratio: float = 0.2,    # 20% 中文数据
        mmlu_ratio: float = 0.15,       # 15% 多任务理解
        gsm8k_ratio: float = 0.15,      # 15% 数学推理
        humaneval_ratio: float = 0.125,  # 12.5% 代码生成
        hellaswag_ratio: float = 0.125  # 12.5% 常识推理
    ) -> List[str]:
        """
        加载混合数据集，确保多样性

        Args:
            total_samples: 总样本数
            seed_ratio: seed_text占比
            wikitext_ratio: Wikitext占比
            chinese_ratio: Chinese-C4占比
            mmlu_ratio: MMLU占比
            gsm8k_ratio: GSM8K占比
            humaneval_ratio: HumanEval占比
            hellaswag_ratio: HELLASWAG占比
        """
        logger.info(f"加载混合数据集 (总计 {total_samples} 样本)")

        all_texts = []

        # 计算各数据集的样本数
        n_seed = int(total_samples * seed_ratio)
        n_wikitext = int(total_samples * wikitext_ratio)
        n_chinese = int(total_samples * chinese_ratio)
        n_mmlu = int(total_samples * mmlu_ratio)
        n_gsm8k = int(total_samples * gsm8k_ratio)
        n_humaneval = int(total_samples * humaneval_ratio)
        n_hellaswag = int(total_samples * hellaswag_ratio)

        # 加载各数据集
        seed_texts = self.load_seed_texts(n_seed)
        logger.info(f"  ✓ Seed texts: {len(seed_texts)} 样本")
        all_texts.extend(seed_texts)

        wikitext = self.load_wikitext(n_wikitext)
        logger.info(f"  ✓ Wikitext: {len(wikitext)} 样本")
        all_texts.extend(wikitext)

        chinese = self.load_chinese_c4(n_chinese)
        logger.info(f"  ✓ Chinese-C4: {len(chinese)} 样本")
        all_texts.extend(chinese)

        mmlu = self.load_mmlu(n_mmlu)
        logger.info(f"  ✓ MMLU: {len(mmlu)} 样本")
        all_texts.extend(mmlu)

        gsm8k = self.load_gsm8k(n_gsm8k)
        logger.info(f"  ✓ GSM8K: {len(gsm8k)} 样本")
        all_texts.extend(gsm8k)

        humaneval = self.load_humaneval(n_humaneval)
        logger.info(f"  ✓ HumanEval: {len(humaneval)} 样本")
        all_texts.extend(humaneval)

        hellaswag = self.load_hellaswag(n_hellaswag)
        logger.info(f"  ✓ HELLASWAG: {len(hellaswag)} 样本")
        all_texts.extend(hellaswag)

        # 打乱顺序
        random.shuffle(all_texts)

        # 确保达到目标数量
        if len(all_texts) < total_samples:
            logger.warning(
                f"样本不足 ({len(all_texts)}/{total_samples}), 将复制现有样本")
            all_texts = all_texts * (total_samples // len(all_texts) + 1)

        return all_texts[:total_samples]


def get_model_config(model_name: str) -> Dict:
    """获取模型的推荐配置"""
    configs = {
        "Qwen3-30B-A3B": {
            "num_samples": 1024
        },
        "Qwen3-Next-80B-A3B": {
            "num_samples": 768
        },
        "Qwen3-235B-A22B": {
            "num_samples": 1024
        }
    }

    for key in configs:
        if key in model_name:
            return configs[key]

    # 默认配置
    return configs["Qwen3-30B-A3B"]


def main():
    parser = argparse.ArgumentParser(
        description="为MoE模型生成校准数据集（不需要EBSS，使用混合采样）"
    )

    parser.add_argument("--model-name", type=str,
                        choices=["Qwen3-30B-A3B",
                                 "Qwen3-Next-80B-A3B", "Qwen3-235B-A22B"],
                        help="模型名称（用于自动配置）")
    parser.add_argument("--output-dir", type=str, default="./ebss_datasets",
                        help="输出目录")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="数据集目录")

    # 数据集参数
    parser.add_argument("--num-samples", type=int, default=None,
                        help="生成样本数")
    parser.add_argument("--seed-ratio", type=float, default=0.05,
                        help="seed_text占比")
    parser.add_argument("--wikitext-ratio", type=float, default=0.30,
                        help="Wikitext占比")
    parser.add_argument("--chinese-ratio", type=float, default=0.30,
                        help="Chinese-C4占比")
    parser.add_argument("--mmlu-ratio", type=float, default=0.20,
                        help="MMLU占比")
    parser.add_argument("--gsm8k-ratio", type=float, default=0.15,
                        help="GSM8K占比")
    parser.add_argument("--humaneval-ratio", type=float, default=0.10,
                        help="HumanEval占比")
    parser.add_argument("--hellaswag-ratio", type=float, default=0.10,
                        help="HELLASWAG占比")

    args = parser.parse_args()

    # 自动配置
    if args.model_name:
        config_dict = get_model_config(args.model_name)
        logger.info(f"使用 {args.model_name} 的推荐配置")
    else:
        config_dict = get_model_config("Qwen3-30B-A3B")
        logger.info("使用默认配置")
        args.model_name = "default"

    # 覆盖配置
    num_samples = args.num_samples or config_dict["num_samples"]

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    logger.info("=" * 60)
    logger.info("加载混合校准数据集")
    logger.info("=" * 60)

    dataset_loader = DatasetLoader(args.data_dir)
    calibration_texts = dataset_loader.load_mixed_dataset(
        total_samples=num_samples,
        seed_ratio=args.seed_ratio,
        wikitext_ratio=args.wikitext_ratio,
        chinese_ratio=args.chinese_ratio,
        mmlu_ratio=args.mmlu_ratio,
        gsm8k_ratio=args.gsm8k_ratio,
        humaneval_ratio=args.humaneval_ratio,
        hellaswag_ratio=args.hellaswag_ratio
    )

    logger.info(f"✓ 加载完成，共 {len(calibration_texts)} 个校准文本")

    # 生成输出文件名
    output_name = f"calibration_{args.model_name}"

    # 保存JSON格式
    json_file = output_dir / f"{output_name}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": args.model_name,
            "num_samples": len(calibration_texts),
            "data_sources": {
                "seed_ratio": args.seed_ratio,
                "wikitext_ratio": args.wikitext_ratio,
                "chinese_ratio": args.chinese_ratio,
                "mmlu_ratio": args.mmlu_ratio,
                "gsm8k_ratio": args.gsm8k_ratio,
                "humaneval_ratio": args.humaneval_ratio,
                "hellaswag_ratio": args.hellaswag_ratio
            },
            "samples": calibration_texts
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"  JSON文件: {json_file}")

    # 保存TXT格式
    txt_file = output_dir / f"{output_name}.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        for sample in calibration_texts:
            f.write(sample.strip() + "\n")

    logger.info(f"  TXT文件: {txt_file}")

    logger.info("=" * 60)
    logger.info("✓ 校准数据集生成完成!")
    logger.info("=" * 60)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"样本数: {len(calibration_texts)}")
    logger.info("\n后续使用:")
    logger.info(f"  bash scripts/moequant_w4a4.sh \\")
    logger.info(f"      --seed-text {txt_file}")


if __name__ == "__main__":
    main()
