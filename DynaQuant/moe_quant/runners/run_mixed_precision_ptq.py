"""
Main PTQ Runner for Mixed Precision MoE Quantization

Supports:
- W2A2 and W4A4 expert quantization
- W8A8 router and non-expert layers
- RouterRank optimization for router consistency
- Distributed processing across multiple GPUs
- Integration with existing quantization methods
"""

import argparse
import logging
import torch
import json
from pathlib import Path
from typing import List, Dict, Optional

from ..quant.mixed_precision_quantizer import (
    MixedPrecisionQuantizer,
    MixedPrecisionConfig
)
from ..runners.distributed_ptq_runner import (
    launch_distributed_ptq,
    run_distributed_ptq_worker,
    create_w2a2_config,
    create_w4a4_config
)
from ..models.load_moe import load_moe_model
from ..quant.ebss import EBSSSampler, EBSSConfig
from ..runners.collect_calib import CalibrationCollector
from ..runners.eval_metrics import MetricsEvaluator


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MixedPrecisionPTQRunner:
    """
    Main PTQ runner for mixed precision MoE quantization
    """

    def __init__(
        self,
        model_name: str,
        config: MixedPrecisionConfig,
        output_dir: str = "./output/mixed_precision_ptq"
    ):
        self.model_name = model_name
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.model_loader = None
        self.quantizer = None
        self.calibration_data = None

    def load_model(self):
        """Load MoE model"""
        logger.info(f"Loading model: {self.model_name}")

        self.model_loader = load_moe_model(
            model_name=self.model_name,
            device="cuda",
            torch_dtype=torch.float16
        )

        logger.info(
            f"Model loaded with {self.model_loader.get_num_moe_layers()} MoE layers")

    def generate_calibration_data(
        self,
        seed_texts: List[str],
        use_ebss: bool = True,
        num_samples: int = 512
    ) -> Dict[str, torch.Tensor]:
        """
        Generate calibration data

        Args:
            seed_texts: Seed texts for calibration
            use_ebss: Whether to use EBSS sampling
            num_samples: Number of calibration samples

        Returns:
            calibration_data: Processed calibration data
        """
        logger.info("Generating calibration data...")

        # Generate EBSS samples if requested
        if use_ebss:
            logger.info("Using EBSS for expert-balanced sampling...")

            ebss_config = EBSSConfig(
                beam_width=4,
                tau=1.2,
                max_tokens=512,
                num_samples=num_samples
            )

            ebss_sampler = EBSSSampler(
                model=self.model_loader.model,
                tokenizer=self.model_loader.tokenizer,
                config=ebss_config,
                device="cuda"
            )

            # Repeat seed texts to reach desired number of samples
            extended_seeds = (seed_texts * (num_samples //
                              len(seed_texts) + 1))[:num_samples]
            ebss_samples = ebss_sampler.generate(extended_seeds)

            # Save EBSS samples
            ebss_file = self.output_dir / "ebss_samples.txt"
            with open(ebss_file, 'w', encoding='utf-8') as f:
                for sample in ebss_samples:
                    f.write(sample + "\n\n")

            logger.info(
                f"Generated {len(ebss_samples)} EBSS samples -> {ebss_file}")
            calibration_texts = ebss_samples
        else:
            calibration_texts = seed_texts

        # Collect calibration data
        logger.info("Collecting activation and affinity data...")

        calib_collector = CalibrationCollector(
            model_loader=self.model_loader,
            num_samples=len(calibration_texts),
            max_seq_len=512
        )

        self.calibration_data = calib_collector.collect_from_dataset(
            calibration_texts, batch_size=1
        )

        # Save calibration data
        calib_file = self.output_dir / "calibration_data.pkl"
        self.calibration_data.save(str(calib_file))

        logger.info(f"Calibration data saved to {calib_file}")

        # Convert to format expected by quantizer
        processed_data = self._process_calibration_data(self.calibration_data)

        return processed_data

    def _process_calibration_data(self, calib_data) -> Dict[str, torch.Tensor]:
        """Process calibration data for quantizer"""
        processed = {}

        # Process per-layer data
        for layer_idx in calib_data.layer_activations:
            activations = calib_data.layer_activations[layer_idx]
            affinities = calib_data.layer_affinities.get(layer_idx)

            processed[f"layer_{layer_idx}_activations"] = activations
            if affinities is not None:
                processed[f"layer_{layer_idx}_affinities"] = affinities

        # Process per-expert data
        for (layer_idx, expert_id) in calib_data.expert_activations:
            activations = calib_data.expert_activations[(layer_idx, expert_id)]
            affinities = calib_data.expert_affinities.get(
                (layer_idx, expert_id))

            processed[f"expert_{layer_idx}_{expert_id}_activations"] = activations
            if affinities is not None:
                processed[f"expert_{layer_idx}_{expert_id}_affinities"] = affinities

        return processed

    def quantize_model(
        self,
        calibration_data: Dict[str, torch.Tensor]
    ) -> Dict:
        """
        Quantize model with mixed precision

        Args:
            calibration_data: Calibration data

        Returns:
            quantization_stats: Statistics for all layers
        """
        logger.info("Starting mixed precision quantization...")

        self.quantizer = MixedPrecisionQuantizer(self.config)

        # Create layer mapping
        layer_mapping = self._create_layer_mapping()

        # Quantize model
        quantized_model, stats = self.quantizer.quantize_model(
            model=self.model_loader.model,
            calibration_data=calibration_data,
            layer_mapping=layer_mapping
        )

        # Save quantization statistics
        stats_file = self.output_dir / "quantization_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Quantization statistics saved to {stats_file}")

        return stats

    def _create_layer_mapping(self) -> Dict[str, str]:
        """Create layer type mapping"""
        layer_mapping = {}

        for name, module in self.model_loader.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Determine layer type
                name_lower = name.lower()

                if any(keyword in name_lower for keyword in ["gate", "router", "gating"]):
                    layer_mapping[name] = "router"
                elif any(keyword in name_lower for keyword in ["expert", "ffn", "mlp"]):
                    layer_mapping[name] = "expert"
                else:
                    layer_mapping[name] = "non_expert"

        logger.info(f"Created layer mapping for {len(layer_mapping)} layers")
        return layer_mapping

    def save_quantized_model(self, model_path: str):
        """Save quantized model"""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        torch.save({
            "model_state_dict": self.model_loader.model.state_dict(),
            "config": self.config.__dict__,
            "model_name": self.model_name,
        }, model_path)

        logger.info(f"Quantized model saved to {model_path}")

    def run_full_ptq(
        self,
        seed_texts: List[str],
        use_ebss: bool = True,
        num_samples: int = 512
    ) -> Dict:
        """
        Run complete PTQ pipeline

        Args:
            seed_texts: Seed texts for calibration
            use_ebss: Whether to use EBSS sampling
            num_samples: Number of calibration samples

        Returns:
            results: Complete PTQ results
        """
        logger.info("Starting mixed precision PTQ pipeline...")

        # Load model
        self.load_model()

        # Generate calibration data
        calibration_data = self.generate_calibration_data(
            seed_texts, use_ebss, num_samples
        )

        # Quantize model
        quantization_stats = self.quantize_model(calibration_data)

        # Save quantized model
        model_path = self.output_dir / "quantized_model.pt"
        self.save_quantized_model(str(model_path))

        # Prepare results
        results = {
            "model_name": self.model_name,
            "config": self.config.__dict__,
            "quantization_stats": quantization_stats,
            "model_path": str(model_path),
            "output_dir": str(self.output_dir)
        }

        # Save results
        results_file = self.output_dir / "ptq_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(
            f"PTQ pipeline complete! Results saved to {self.output_dir}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Mixed Precision MoE PTQ")

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path")

    # Quantization configuration
    parser.add_argument("--expert-precision", type=str, choices=["w2a2", "w4a4"],
                        default="w2a2", help="Expert quantization precision")
    parser.add_argument("--router-bits", type=int, default=8,
                        help="Router quantization bits")
    parser.add_argument("--non-expert-bits", type=int,
                        default=8, help="Non-expert quantization bits")

    # RouterRank configuration
    parser.add_argument("--router-rank-gamma", type=float, default=0.1,
                        help="RouterRank guard band margin")
    parser.add_argument("--router-rank-lambda", type=float, default=1.0,
                        help="RouterRank loss weight")

    # Calibration
    parser.add_argument("--calib-size", type=int,
                        default=512, help="Calibration set size")
    parser.add_argument("--seed-text", type=str,
                        default=None, help="Path to seed text file")
    parser.add_argument("--no-ebss", action="store_true",
                        help="Disable EBSS sampling")

    # Distributed processing
    parser.add_argument("--distributed", action="store_true",
                        help="Use distributed processing")
    parser.add_argument("--world-size", type=int, default=8,
                        help="Number of GPUs for distributed")

    # Output
    parser.add_argument("--output-dir", type=str, default="./output/mixed_precision_ptq",
                        help="Output directory")

    args = parser.parse_args()

    # Create configuration
    if args.expert_precision == "w2a2":
        config = create_w2a2_config()
    else:  # w4a4
        config = create_w4a4_config()

    # Override configuration
    config.router_bits_w = args.router_bits
    config.router_bits_a = args.router_bits
    config.non_expert_bits_w = args.non_expert_bits
    config.non_expert_bits_a = args.non_expert_bits
    config.router_rank_gamma = args.router_rank_gamma
    config.router_rank_lambda = args.router_rank_lambda

    # Prepare seed texts
    if args.seed_text:
        with open(args.seed_text, 'r', encoding='utf-8') as f:
            seed_texts = [line.strip() for line in f if line.strip()]
    else:
        # Default seed texts
        seed_texts = [
            "The quick brown fox jumps over the lazy dog, demonstrating agility and speed.",
            "Artificial intelligence is transforming industries across the globe through machine learning.",
            "In a distant galaxy far beyond our reach, a new civilization emerges with advanced technology.",
            "The scientific method involves careful observation, hypothesis formation, and experimentation.",
            "Climate change poses significant challenges to ecosystems worldwide, requiring urgent action.",
        ] * (args.calib_size // 5 + 1)

    seed_texts = seed_texts[:args.calib_size]

    # Run PTQ
    if args.distributed and torch.cuda.device_count() > 1:
        logger.info(
            f"Using distributed processing with {args.world_size} GPUs")

        launch_distributed_ptq(
            model_name=args.model,
            config=config,
            seed_texts=seed_texts,
            output_dir=args.output_dir,
            world_size=min(args.world_size, torch.cuda.device_count()),
            use_ebss=not args.no_ebss
        )
    else:
        logger.info("Using single GPU processing")

        runner = MixedPrecisionPTQRunner(
            model_name=args.model,
            config=config,
            output_dir=args.output_dir
        )

        results = runner.run_full_ptq(
            seed_texts=seed_texts,
            use_ebss=not args.no_ebss,
            num_samples=args.calib_size
        )

        logger.info("PTQ completed successfully!")


if __name__ == "__main__":
    main()
