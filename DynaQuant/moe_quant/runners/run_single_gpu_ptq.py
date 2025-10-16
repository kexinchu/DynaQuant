"""
Single GPU PTQ Runner for MoE Models

Optimized for large models that don't fit in distributed setup.
Uses memory-efficient strategies for 30B+ models.
"""

import argparse
import logging
import torch
import json
from pathlib import Path
from typing import List, Dict, Optional
import gc

from ..quant.mixed_precision_quantizer import (
    MixedPrecisionQuantizer,
    MixedPrecisionConfig
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


class SingleGPUMixedPrecisionRunner:
    """
    Single GPU mixed precision PTQ runner for large MoE models
    """

    def __init__(
        self,
        model_name: str,
        config: MixedPrecisionConfig,
        output_dir: str = "./output/single_gpu_ptq"
    ):
        self.model_name = model_name
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.model_loader = None
        self.quantizer = None
        self.calibration_data = None

    def setup_memory_optimization(self):
        """Setup memory optimization settings"""
        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)

        # Set memory fraction to avoid OOM
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)

        # Enable memory efficient mode
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        logger.info("Memory optimization settings applied")

    def load_model_with_optimization(self):
        """Load model with memory optimization"""
        logger.info(f"Loading model: {self.model_name}")

        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()

        self.model_loader = load_moe_model(
            model_name=self.model_name,
            device="cuda",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        logger.info(
            f"Model loaded with {self.model_loader.get_num_moe_layers()} MoE layers")

        # Log memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"GPU memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

    def generate_calibration_data_optimized(
        self,
        seed_texts: List[str],
        use_ebss: bool = True,
        num_samples: int = 128  # Reduced for memory efficiency
    ) -> Dict[str, torch.Tensor]:
        """
        Generate calibration data with memory optimization
        """
        logger.info("Generating calibration data with memory optimization...")

        # Limit number of samples for large models
        num_samples = min(num_samples, 128)
        seed_texts = seed_texts[:num_samples]

        # Generate EBSS samples if requested
        if use_ebss:
            logger.info("Using EBSS for expert-balanced sampling...")

            ebss_config = EBSSConfig(
                beam_width=2,  # Reduced beam width
                tau=1.2,
                max_tokens=256,  # Reduced max tokens
                num_samples=num_samples
            )

            ebss_sampler = EBSSSampler(
                model=self.model_loader.model,
                tokenizer=self.model_loader.tokenizer,
                config=ebss_config,
                device="cuda"
            )

            try:
                ebss_samples = ebss_sampler.generate(seed_texts)

                # Save EBSS samples
                ebss_file = self.output_dir / "ebss_samples.txt"
                with open(ebss_file, 'w', encoding='utf-8') as f:
                    for sample in ebss_samples:
                        f.write(sample + "\n\n")

                logger.info(
                    f"Generated {len(ebss_samples)} EBSS samples -> {ebss_file}")
                calibration_texts = ebss_samples

                # Clear memory
                del ebss_sampler
                torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(
                    f"EBSS generation failed: {e}, using original texts")
                calibration_texts = seed_texts
        else:
            calibration_texts = seed_texts

        # Collect calibration data with memory management
        logger.info("Collecting activation and affinity data...")

        calib_collector = CalibrationCollector(
            model_loader=self.model_loader,
            num_samples=len(calibration_texts),
            max_seq_len=256  # Reduced sequence length
        )

        try:
            self.calibration_data = calib_collector.collect_from_dataset(
                calibration_texts, batch_size=1  # Force batch size 1
            )

            # Save calibration data
            calib_file = self.output_dir / "calibration_data.pkl"
            self.calibration_data.save(str(calib_file))

            logger.info(f"Calibration data saved to {calib_file}")

        except Exception as e:
            logger.error(f"Calibration data collection failed: {e}")
            raise

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

    def quantize_model_with_memory_management(
        self,
        calibration_data: Dict[str, torch.Tensor]
    ) -> Dict:
        """
        Quantize model with memory management
        """
        logger.info(
            "Starting mixed precision quantization with memory management...")

        self.quantizer = MixedPrecisionQuantizer(self.config)

        # Create layer mapping
        layer_mapping = self._create_layer_mapping()

        # Get all linear layers
        linear_layers = {}
        for name, module in self.model_loader.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_layers[name] = module

        logger.info(f"Found {len(linear_layers)} linear layers to quantize")

        # Quantize layers in batches to manage memory
        batch_size = 10  # Process 10 layers at a time
        all_stats = {}

        layer_names = list(linear_layers.keys())
        for i in range(0, len(layer_names), batch_size):
            batch_layers = layer_names[i:i+batch_size]

            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(layer_names) + batch_size - 1)//batch_size}")

            for layer_name in batch_layers:
                if layer_name not in calibration_data:
                    logger.warning(
                        f"No calibration data for layer {layer_name}")
                    continue

                # Determine layer type
                if layer_mapping and layer_name in layer_mapping:
                    layer_type = layer_mapping[layer_name]
                else:
                    layer_type = self.quantizer.identify_layer_type(layer_name)

                logger.info(f"Quantizing {layer_name} as {layer_type}")

                try:
                    layer = linear_layers[layer_name]
                    calib_data = {
                        "inputs": calibration_data[layer_name].to("cuda")
                    }

                    if layer_type == "router":
                        quantized_layer, stats = self.quantizer.quantize_router_layer(
                            layer, calib_data
                        )
                    elif layer_type == "expert":
                        expert_id = self._extract_expert_id(layer_name)
                        quantized_layer, stats = self.quantizer.quantize_expert_layer(
                            layer, calib_data, expert_id
                        )
                    else:  # non_expert
                        quantized_layer, stats = self.quantizer.quantize_non_expert_layer(
                            layer, calib_data
                        )

                    # Replace layer in model
                    self._replace_layer_in_model(layer_name, quantized_layer)
                    all_stats[layer_name] = stats

                    logger.info(
                        f"✓ Quantized {layer_name}: {stats.get('quantization_bits', 'unknown')}")

                    # Clear memory after each layer
                    del calib_data, quantized_layer
                    torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"Failed to quantize {layer_name}: {e}")
                    all_stats[layer_name] = {"error": str(e)}

            # Clear memory after each batch
            gc.collect()
            torch.cuda.empty_cache()

        # Save quantization statistics
        stats_file = self.output_dir / "quantization_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)

        logger.info(f"Quantization statistics saved to {stats_file}")

        return all_stats

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

    def _extract_expert_id(self, layer_name: str) -> Optional[int]:
        """Extract expert ID from layer name"""
        import re
        match = re.search(r'expert[_\s]*(\d+)', layer_name.lower())
        if match:
            return int(match.group(1))
        return None

    def _replace_layer_in_model(self, layer_name: str, new_layer: torch.nn.Module):
        """Replace layer in model by name"""
        parts = layer_name.split('.')
        current = self.model_loader.model

        # Navigate to parent module
        for part in parts[:-1]:
            current = getattr(current, part)

        # Replace the layer
        setattr(current, parts[-1], new_layer)

    def save_quantized_model(self, model_path: str):
        """Save quantized model with memory optimization"""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving quantized model...")

        # Clear memory before saving
        torch.cuda.empty_cache()
        gc.collect()

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
        num_samples: int = 128
    ) -> Dict:
        """
        Run complete single GPU PTQ pipeline
        """
        logger.info("Starting single GPU mixed precision PTQ pipeline...")

        try:
            # Setup memory optimization
            self.setup_memory_optimization()

            # Load model
            self.load_model_with_optimization()

            # Generate calibration data
            calibration_data = self.generate_calibration_data_optimized(
                seed_texts, use_ebss, num_samples
            )

            # Quantize model
            quantization_stats = self.quantize_model_with_memory_management(
                calibration_data
            )

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
                f"Single GPU PTQ pipeline complete! Results saved to {self.output_dir}")

            return results

        except Exception as e:
            logger.error(f"PTQ pipeline failed: {e}")
            raise


def create_w2a2_config() -> MixedPrecisionConfig:
    """Create W2A2 configuration"""
    return MixedPrecisionConfig(
        expert_bits_w=2,
        expert_bits_a=2,
        router_bits_w=8,
        router_bits_a=8,
        non_expert_bits_w=8,
        non_expert_bits_a=8,
        router_rank_gamma=0.1,
        router_rank_lambda=1.0,
        use_agq_for_experts=True,
        use_activation_shaping=True
    )


def create_w4a4_config() -> MixedPrecisionConfig:
    """Create W4A4 configuration"""
    return MixedPrecisionConfig(
        expert_bits_w=4,
        expert_bits_a=4,
        router_bits_w=8,
        router_bits_a=8,
        non_expert_bits_w=8,
        non_expert_bits_a=8,
        router_rank_gamma=0.1,
        router_rank_lambda=1.0,
        use_agq_for_experts=True,
        use_activation_shaping=False
    )


def main():
    parser = argparse.ArgumentParser(
        description="Single GPU Mixed Precision MoE PTQ")

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
                        default=128, help="Calibration set size")
    parser.add_argument("--seed-text", type=str,
                        default=None, help="Path to seed text file")
    parser.add_argument("--no-ebss", action="store_true",
                        help="Disable EBSS sampling")

    # Output
    parser.add_argument("--output-dir", type=str, default="./output/single_gpu_ptq",
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

    # Run single GPU PTQ
    logger.info("Using single GPU processing for memory efficiency")

    runner = SingleGPUMixedPrecisionRunner(
        model_name=args.model,
        config=config,
        output_dir=args.output_dir
    )

    results = runner.run_full_ptq(
        seed_texts=seed_texts,
        use_ebss=not args.no_ebss,
        num_samples=args.calib_size
    )

    logger.info("Single GPU PTQ completed successfully!")


if __name__ == "__main__":
    main()
