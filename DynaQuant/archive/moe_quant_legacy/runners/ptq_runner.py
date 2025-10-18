"""
PTQ Runner for MoE W2A2 Quantization

Orchestrates the complete PTQ pipeline:
1. EBSS sampling (optional)
2. Calibration data collection
3. AGQ + W2A2 quantization
4. Router guard optimization
5. Evaluation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import json
import logging
from pathlib import Path
from tqdm import tqdm

from ..quant.ebss import EBSSSampler, EBSSConfig
from ..quant.agq import AGQuantizer, AGQConfig
from ..quant.quantizers import W2A2Quantizer, W2A2Config, QuantizedLinearW2A2
from ..quant.router_guard_enhanced import EnhancedRouterGuard, EnhancedRouterConfig
from ..models.load_moe import MoEModelLoader
from ..runners.collect_calib import CalibrationCollector, CalibrationData


logger = logging.getLogger(__name__)


class PTQRunner:
    """
    Post-Training Quantization Runner

    Complete pipeline for W2A2 MoE quantization with EBSS and AGQ
    """

    def __init__(
        self,
        model_loader: MoEModelLoader,
        ebss_config: Optional[EBSSConfig] = None,
        agq_config: Optional[AGQConfig] = None,
        w2a2_config: Optional[W2A2Config] = None,
        router_config: Optional[EnhancedRouterConfig] = None,
        output_dir: str = "./ptq_output"
    ):
        self.model_loader = model_loader
        self.ebss_config = ebss_config or EBSSConfig()
        self.agq_config = agq_config or AGQConfig()
        self.w2a2_config = w2a2_config or W2A2Config()
        self.router_config = router_config or EnhancedRouterConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.ebss_sampler = None
        self.calib_collector = None
        self.agq_quantizer = AGQuantizer(agq_config)
        self.w2a2_quantizer = W2A2Quantizer(w2a2_config)
        self.router_guard = EnhancedRouterGuard(router_config)

        # Results
        self.calibration_data = None
        self.quantized_weights = {}
        self.quantization_stats = {}

    def step1_generate_ebss_samples(
        self,
        seed_texts: List[str],
        num_samples: Optional[int] = None
    ) -> List[str]:
        """
        Step 1: Generate calibration samples using EBSS

        Args:
            seed_texts: Initial seed texts
            num_samples: Number of samples to generate (default: use config)

        Returns:
            List of generated samples
        """
        logger.info("Step 1: Generating EBSS samples...")

        num_samples = num_samples or self.ebss_config.num_samples

        # Create EBSS sampler
        self.ebss_sampler = EBSSSampler(
            self.model_loader.model,
            self.model_loader.tokenizer,
            self.ebss_config,
            self.model_loader.device
        )

        # Generate samples
        # Repeat seed texts to reach desired num_samples
        num_repeats = (num_samples + len(seed_texts) - 1) // len(seed_texts)
        extended_seeds = (seed_texts * num_repeats)[:num_samples]

        ebss_samples = self.ebss_sampler.generate(extended_seeds)

        # Save samples
        samples_file = self.output_dir / "ebss_samples.txt"
        with open(samples_file, 'w', encoding='utf-8') as f:
            for sample in ebss_samples:
                f.write(sample + "\n\n")

        logger.info(
            f"Generated {len(ebss_samples)} EBSS samples -> {samples_file}")

        return ebss_samples

    def step2_collect_calibration_data(
        self,
        dataset: List[str],
        batch_size: int = 1
    ) -> CalibrationData:
        """
        Step 2: Collect calibration data (activations + affinities)

        Args:
            dataset: Text samples for calibration
            batch_size: Batch size

        Returns:
            CalibrationData object
        """
        logger.info("Step 2: Collecting calibration data...")

        self.calib_collector = CalibrationCollector(
            self.model_loader,
            num_samples=len(dataset)
        )

        self.calibration_data = self.calib_collector.collect_from_dataset(
            dataset, batch_size
        )

        # Save calibration data
        calib_file = self.output_dir / "calibration_data.pkl"
        self.calibration_data.save(str(calib_file))

        # Log statistics
        logger.info(
            f"Collected calibration data for {len(self.calibration_data.layer_activations)} layers")
        logger.info(
            f"Expert call counts: {self.calibration_data.expert_call_counts}")

        return self.calibration_data

    def step3_quantize_experts(self) -> Dict:
        """
        Step 3: Quantize expert layers using AGQ + W2A2

        Returns:
            Dictionary of quantization statistics
        """
        logger.info("Step 3: Quantizing experts with AGQ + W2A2...")

        all_stats = {}

        for layer_idx in tqdm(range(self.model_loader.get_num_moe_layers()), desc="Quantizing layers"):
            # Get experts
            experts = self.model_loader.get_experts(layer_idx)
            if experts is None:
                continue

            layer_stats = {}

            for expert_id, expert_module in enumerate(experts):
                # Find linear layers in expert
                for name, module in expert_module.named_modules():
                    if not isinstance(module, nn.Linear):
                        continue

                    # Get calibration data for this expert
                    expert_key = (layer_idx, expert_id)
                    if expert_key not in self.calibration_data.expert_activations:
                        logger.warning(
                            f"No calibration data for layer {layer_idx} expert {expert_id}")
                        continue

                    X = self.calibration_data.expert_activations[expert_key]
                    c = self.calibration_data.expert_affinities[expert_key]

                    # Ensure data is on correct device
                    X = X.to(self.model_loader.device)
                    c = c.to(self.model_loader.device)

                    # AGQ quantization (weight)
                    W_agq, scales_agq, stats_agq = self.agq_quantizer.quantize_linear(
                        module, X, c,
                        bit_width=self.w2a2_config.w_bit,
                        group_size=self.w2a2_config.w_group_size
                    )

                    # W2A2 quantization (weight + activation shaping)
                    W_w2a2, W_absorbed, stats_w2a2 = self.w2a2_quantizer.quantize_linear_layer(
                        module, X, layer_id=layer_idx
                    )

                    # Replace module with quantized version
                    q_module = QuantizedLinearW2A2.from_float(
                        module, X, self.w2a2_config)

                    # Store quantized weights
                    key = f"layer{layer_idx}_expert{expert_id}_{name}"
                    self.quantized_weights[key] = {
                        "W_agq": W_agq.cpu(),
                        "W_w2a2": W_w2a2.cpu(),
                        "W_absorbed": W_absorbed.cpu(),
                        "scales_agq": scales_agq.cpu() if scales_agq is not None else None,
                    }

                    # Merge statistics
                    stats = {**stats_agq, **stats_w2a2}
                    layer_stats[key] = stats

            all_stats[f"layer_{layer_idx}"] = layer_stats

        self.quantization_stats = all_stats

        # Save stats
        stats_file = self.output_dir / "quantization_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)

        logger.info(f"Quantization complete. Stats saved to {stats_file}")

        return all_stats

    def step4_optimize_router_guard(self) -> Dict:
        """
        Step 4: Optimize router guard parameters

        Returns:
            Router guard optimization results
        """
        logger.info("Step 4: Optimizing router guard...")

        router_stats = {}

        for layer_idx in range(self.model_loader.get_num_moe_layers()):
            if layer_idx not in self.calibration_data.layer_activations:
                continue

            X = self.calibration_data.layer_activations[layer_idx].to(
                self.model_loader.device)

            # Get router
            router = self.model_loader.get_router(layer_idx)
            if router is None:
                continue

            # Test different router modes
            logits_fp, ids_fp = self.router_guard.forward_router_fp16(
                X, router.weight, router.bias
            )

            logits_int8, ids_int8 = self.router_guard.forward_router_int8(
                X, router.weight, router.bias
            )

            # Check consistency
            consistency = self.router_guard.check_consistency(
                ids_fp, ids_int8, layer_idx
            )

            router_stats[f"layer_{layer_idx}"] = consistency

        # Save router stats
        router_file = self.output_dir / "router_stats.json"
        with open(router_file, 'w') as f:
            json.dump(router_stats, f, indent=2)

        logger.info(
            f"Router optimization complete. Stats saved to {router_file}")

        return router_stats

    def run_full_pipeline(
        self,
        seed_texts: List[str],
        use_ebss: bool = True,
        external_calib_data: Optional[List[str]] = None
    ) -> Dict:
        """
        Run complete PTQ pipeline

        Args:
            seed_texts: Seed texts for EBSS
            use_ebss: Whether to use EBSS sampling
            external_calib_data: External calibration data (if not using EBSS)

        Returns:
            Dictionary with all results
        """
        results = {}

        # Step 1: Generate/prepare calibration data
        if use_ebss:
            ebss_samples = self.step1_generate_ebss_samples(seed_texts)
            calib_dataset = ebss_samples
        else:
            calib_dataset = external_calib_data or seed_texts

        # Step 2: Collect calibration data
        calib_data = self.step2_collect_calibration_data(calib_dataset)
        results["calibration"] = {
            "num_samples": len(calib_dataset),
            "num_layers": len(calib_data.layer_activations),
            "expert_counts": calib_data.expert_call_counts,
        }

        # Step 3: Quantize experts
        quant_stats = self.step3_quantize_experts()
        results["quantization"] = quant_stats

        # Step 4: Optimize router guard
        router_stats = self.step4_optimize_router_guard()
        results["router"] = router_stats

        # Save overall results
        results_file = self.output_dir / "ptq_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(
            f"PTQ pipeline complete! Results saved to {self.output_dir}")

        return results

    def save_quantized_model(self, path: str):
        """Save quantized model weights"""
        save_path = Path(path)
        torch.save({
            "quantized_weights": self.quantized_weights,
            "quantization_stats": self.quantization_stats,
            "w2a2_config": self.w2a2_config,
            "agq_config": self.agq_config,
        }, save_path)
        logger.info(f"Saved quantized model to {save_path}")


def create_ptq_runner(
    model_name: str,
    output_dir: str = "./ptq_output",
    w_bit: int = 2,
    a_bit: int = 2,
    use_rotation: bool = True,
    enable_fallback: bool = True
) -> PTQRunner:
    """
    Convenience function to create PTQ runner

    Args:
        model_name: MoE model name
        output_dir: Output directory
        w_bit: Weight bit width
        a_bit: Activation bit width
        use_rotation: Use activation rotation
        enable_fallback: Enable progressive fallback

    Returns:
        PTQRunner instance
    """
    from ..models.load_moe import load_moe_model

    # Load model
    model_loader = load_moe_model(model_name)

    # Create configs
    w2a2_config = W2A2Config(
        w_bit=w_bit,
        a_bit=a_bit,
        use_rotation=use_rotation,
        enable_fallback=enable_fallback
    )

    # Create runner
    runner = PTQRunner(
        model_loader,
        w2a2_config=w2a2_config,
        output_dir=output_dir
    )

    return runner
