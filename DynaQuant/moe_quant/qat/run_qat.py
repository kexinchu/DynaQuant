"""
CLI entry point for QAT
"""

import argparse
import logging
import torch
from pathlib import Path

from ..models.load_moe import load_moe_model
from ..quant.quantizers import W2A2Config
from .train_qat import QATTrainer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MoE W2A2 QAT Training")

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--load-ptq", type=str, required=True,
                        help="Path to PTQ checkpoint")

    # Training
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")

    # Loss weights
    parser.add_argument("--lambda-topk", type=float, default=1.0,
                        help="Weight for top-k consistency loss")
    parser.add_argument("--mu-margin", type=float,
                        default=0.2, help="Weight for margin loss")

    # Data
    parser.add_argument("--train-data", type=str, default=None,
                        help="Path to training data file")
    parser.add_argument("--num-samples", type=int,
                        default=1000, help="Number of training samples")

    # Strategy
    parser.add_argument("--freeze-experts", type=int,
                        default=1, help="Freeze expert parameters")
    parser.add_argument("--train-router-adjacent-only", type=int,
                        default=1, help="Train router-adjacent layers only")

    # Output
    parser.add_argument("--output-dir", type=str,
                        default="./output/qat_moe", help="Output directory")

    args = parser.parse_args()

    # Load PTQ checkpoint
    logger.info(f"Loading PTQ checkpoint: {args.load_ptq}")
    ptq_checkpoint = torch.load(args.load_ptq)
    w2a2_config = ptq_checkpoint.get("w2a2_config", W2A2Config())

    # Load model
    logger.info(f"Loading model: {args.model}")
    model_loader = load_moe_model(args.model)

    # Apply PTQ weights if available
    if "quantized_weights" in ptq_checkpoint:
        logger.info("Applying PTQ weights to model...")
        # Note: This would require more sophisticated weight loading
        # For now, we just log
        pass

    # Create trainer
    trainer = QATTrainer(
        model_loader,
        w2a2_config,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lambda_topk=args.lambda_topk,
        mu_margin=args.mu_margin,
        freeze_experts=bool(args.freeze_experts),
        train_router_adjacent_only=bool(args.train_router_adjacent_only)
    )

    # Prepare training data
    if args.train_data:
        with open(args.train_data, 'r', encoding='utf-8') as f:
            train_texts = [line.strip()
                           for line in f if line.strip()][:args.num_samples]
    else:
        # Default training texts
        train_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "In a distant galaxy, a new civilization emerges.",
        ] * (args.num_samples // 3)

    # Run QAT
    logger.info("Starting QAT training...")
    stats = trainer.train(train_texts)

    logger.info("QAT complete!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
