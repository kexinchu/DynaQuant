"""
W4A16 Quantization Script using llm-compressor
================================================
量化配置：4位权重 + 16位激活
- 目标：所有Linear层（包括MoE专家层）
- 保留：lm_head为FP16以保持输出质量
"""
import os
import json
import argparse
from pathlib import Path
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def load_calibration_data(calib_path):
    """加载校准数据集"""
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with open(calib_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 支持多种格式：列表、包含samples字段的字典、包含data字段的字典
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "samples" in data:
            return data["samples"]
        elif "data" in data:
            return data["data"]
        else:
            raise ValueError(
                f"Calibration data must be a list or dict with 'samples'/'data' field. "
                f"Found keys: {list(data.keys())}"
            )
    else:
        raise ValueError(f"Unexpected calibration data type: {type(data)}")


def main():
    parser = argparse.ArgumentParser(
        description="W4A16 Quantization with llm-compressor")
    parser.add_argument("--model", type=str, default="/dev/shm/Qwen3-30B-A3B",
                        help="Path to the model")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for quantized model")
    parser.add_argument("--calib-data", type=str, default=None,
                        help="Path to calibration data (JSON file)")
    parser.add_argument("--num-samples", type=int, default=1024,
                        help="Number of calibration samples to use")
    parser.add_argument("--max-seq-length", type=int, default=8192,
                        help="Maximum sequence length")
    args = parser.parse_args()

    # 设置输出目录
    model_name = Path(args.model).name
    output_dir = args.output_dir or f"/dev/shm/{model_name}-W4A16"

    # 确定校准数据路径
    if args.calib_data:
        calib_path = args.calib_data
    else:
        # 尝试常见的校准数据位置
        possible_paths = [
            f"calibration_datasets/{model_name}/calibration_{model_name}.json",
            f"calibration_datasets/Qwen3-30B-A3B/calibration_Qwen3-30B-A3B.json",
            "calibration_datasets/calibration.json",
        ]
        calib_path = None
        for path in possible_paths:
            if os.path.exists(path):
                calib_path = path
                break

        if calib_path is None:
            raise FileNotFoundError(
                f"No calibration data found. Tried: {possible_paths}\n"
                f"Please specify --calib-data explicitly."
            )

    print(f"Loading calibration data from: {calib_path}")
    calib_data = load_calibration_data(calib_path)
    print(f"Loaded {len(calib_data)} calibration samples")

    # 限制样本数量
    calib_samples = calib_data[:args.num_samples]

    # 转换为 HuggingFace Dataset 格式
    calib_dataset = Dataset.from_dict({"text": calib_samples})
    print(f"Using {len(calib_dataset)} samples for calibration")

    # 定义量化配方：W4A16
    recipe = [
        QuantizationModifier(
            scheme="W4A16",
            targets="Linear",           # 量化所有Linear层
            ignore=["lm_head"],         # 保留输出头为FP16
        )
    ]

    print(f"\n{'='*60}")
    print(f"Starting W4A16 Quantization")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    print(f"Calibration samples: {len(calib_dataset)}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"{'='*60}\n")

    # 执行量化
    oneshot(
        model=args.model,
        dataset=calib_dataset,
        recipe=recipe,
        output_dir=output_dir,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(calib_dataset),
    )

    print(f"\n{'='*60}")
    print(f"Quantization completed successfully!")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
