"""
CLI entry point for PTQ
"""

import argparse
import logging
from pathlib import Path

from ..models.load_moe import load_moe_model
from ..quant.ebss import EBSSConfig
from ..quant.agq import AGQConfig
from ..quant.quantizers import W2A2Config
from ..quant.router_guard_enhanced import EnhancedRouterConfig
from .ptq_runner import PTQRunner


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="MoE W2A2 PTQ with EBSS and AGQ")

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path")

    # Calibration
    parser.add_argument("--calib-size", type=int,
                        default=128, help="Calibration set size")
    parser.add_argument("--calib-data", type=str, default=None,
                        help="Path to calibration data file")
    parser.add_argument("--seed-text", type=str, default=None,
                        help="Path to seed text file for EBSS")

    # EBSS
    parser.add_argument("--ebss-beam-width", type=int,
                        default=4, help="EBSS beam width")
    parser.add_argument("--ebss-tau", type=float,
                        default=1.2, help="EBSS temperature")
    parser.add_argument("--ebss-max-tokens", type=int,
                        default=512, help="EBSS max tokens")
    parser.add_argument("--no-ebss", action="store_true",
                        help="Disable EBSS sampling")

    # Quantization
    parser.add_argument("--bit-w", type=int, default=2,
                        help="Weight bit width")
    parser.add_argument("--bit-a", type=int, default=2,
                        help="Activation bit width")
    parser.add_argument("--group-size", type=int, default=64,
                        help="Group size for quantization")

    # W2A2
    parser.add_argument("--use-rotation", type=int,
                        default=1, help="Use activation rotation")
    parser.add_argument("--use-whitening", type=int,
                        default=1, help="Use activation whitening")
    parser.add_argument("--enable-fallback", type=int,
                        default=1, help="Enable progressive fallback")

    # Router
    parser.add_argument("--router-mode", type=str, default="fp16",
                        choices=["fp16", "int8_acc32"], help="Router mode")
    parser.add_argument("--strict-topk", type=int, default=1,
                        help="Strict top-k consistency")

    # Output
    parser.add_argument("--output-dir", type=str,
                        default="./output/ptq_moe", help="Output directory")

    args = parser.parse_args()

    # Create configs
    ebss_config = EBSSConfig(
        beam_width=args.ebss_beam_width,
        tau=args.ebss_tau,
        max_tokens=args.ebss_max_tokens,
        num_samples=args.calib_size
    )

    w2a2_config = W2A2Config(
        w_bit=args.bit_w,
        a_bit=args.bit_a,
        w_group_size=args.group_size,
        a_group_size=args.group_size,
        use_rotation=bool(args.use_rotation),
        use_whitening=bool(args.use_whitening),
        enable_fallback=bool(args.enable_fallback)
    )

    router_config = EnhancedRouterConfig(
        mode=args.router_mode,
        strict_topk=bool(args.strict_topk)
    )

    # Load model
    logger.info(f"Loading model: {args.model}")
    model_loader = load_moe_model(args.model)

    # Create runner
    runner = PTQRunner(
        model_loader,
        ebss_config=ebss_config,
        w2a2_config=w2a2_config,
        router_config=router_config,
        output_dir=args.output_dir
    )

    # Prepare seed texts
    if args.seed_text:
        with open(args.seed_text, 'r', encoding='utf-8') as f:
            seed_texts = [line.strip() for line in f if line.strip()]
    else:
        # Default seed texts
        seed_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "In a distant galaxy, a new civilization emerges.",
            "The scientific method involves observation and experimentation.",
        ] * 8  # Repeat to get more samples

    # Prepare calibration data
    if args.calib_data:
        with open(args.calib_data, 'r', encoding='utf-8') as f:
            calib_data = [line.strip()
                          for line in f if line.strip()][:args.calib_size]
    else:
        calib_data = None

    # Run PTQ
    logger.info("Starting PTQ pipeline...")
    results = runner.run_full_pipeline(
        seed_texts=seed_texts,
        use_ebss=not args.no_ebss,
        external_calib_data=calib_data
    )

    # Save quantized model
    model_path = Path(args.output_dir) / "quantized_model.pt"
    runner.save_quantized_model(str(model_path))

    logger.info("PTQ complete!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
