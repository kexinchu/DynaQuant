#!/usr/bin/env python3
"""
Quantize models using Intel AutoRound to generate Int4 versions.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def quantize_model(
    model_path: str,
    output_path: str,
    scheme: str = "W4A16",
    iters: int = 200,
    device: str = "auto",
    device_map: str = None,
    trust_remote_code: bool = True,
    batch_size: int = None,
    enable_torch_compile: bool = False,
):
    """Quantize a model using AutoRound."""
    try:
        from auto_round import AutoRound
    except ImportError as e:
        LOGGER.error(
            "Failed to import auto_round. Please install it: pip install auto-round"
        )
        raise

    LOGGER.info(f"Starting quantization of {model_path}")
    LOGGER.info(f"Output path: {output_path}")
    LOGGER.info(f"Scheme: {scheme}, Iterations: {iters}")
    if device_map:
        LOGGER.info(f"Using device_map: {device_map}")
    if batch_size:
        LOGGER.info(f"Using batch_size: {batch_size}")

    # Create output directory
    output_dir = Path(output_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Prepare AutoRound parameters
    ar_kwargs = {
        "model": model_path,
        "scheme": scheme,
        "iters": iters,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        "enable_torch_compile": enable_torch_compile,
    }
    
    # Use device_map if specified, otherwise use device
    if device_map:
        ar_kwargs["device_map"] = device_map
    else:
        ar_kwargs["device"] = device
    
    # Add batch_size if specified (for memory optimization)
    if batch_size:
        ar_kwargs["batch_size"] = batch_size

    # Initialize AutoRound
    ar = AutoRound(**ar_kwargs)

    # Quantize and save
    LOGGER.info("Starting quantization process...")
    ar.quantize_and_save(output_path)

    LOGGER.info(f"Quantization completed! Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize models using Intel AutoRound"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the input model",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the quantized model",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="W4A16",
        help="Quantization scheme (default: W4A16 for Int4)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Number of iterations for quantization (default: 200)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for quantization (default: auto)",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Device map for model loading (e.g., '1' for GPU 1, 'auto' for auto distribution)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for quantization (smaller values use less memory)",
    )
    parser.add_argument(
        "--enable-torch-compile",
        action="store_true",
        help="Enable torch.compile for faster quantization (may reduce memory)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code when loading model",
    )

    args = parser.parse_args()

    try:
        quantize_model(
            model_path=args.model_path,
            output_path=args.output_path,
            scheme=args.scheme,
            iters=args.iters,
            device=args.device,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            batch_size=args.batch_size,
            enable_torch_compile=args.enable_torch_compile,
        )
    except Exception as e:
        LOGGER.error(f"Quantization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

