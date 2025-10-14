#!/usr/bin/env python3
"""
CLI for Parallel PTQ Runner
"""

from moe_quant.runners.ptq_parallel import run_parallel_ptq
import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel PTQ for MoE models across multiple GPUs"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to base model"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for quantized model"
    )

    parser.add_argument(
        "--w-bit",
        type=int,
        default=2,
        help="Expert weight bits (default: 2)"
    )

    parser.add_argument(
        "--a-bit",
        type=int,
        default=2,
        help="Expert activation bits (default: 2)"
    )

    parser.add_argument(
        "--router-w-bit",
        type=int,
        default=8,
        help="Router weight bits (default: 8)"
    )

    parser.add_argument(
        "--router-a-bit",
        type=int,
        default=8,
        help="Router activation bits (default: 8)"
    )

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs to use (default: 8)"
    )

    parser.add_argument(
        "--calib-size",
        type=int,
        default=128,
        help="Calibration set size (default: 128)"
    )

    args = parser.parse_args()

    # Run parallel PTQ
    run_parallel_ptq(
        model_path=args.model,
        output_dir=args.output_dir,
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        router_w_bit=args.router_w_bit,
        router_a_bit=args.router_a_bit,
        num_gpus=args.num_gpus,
        calib_size=args.calib_size
    )


if __name__ == "__main__":
    main()
