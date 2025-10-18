"""
Distributed PTQ Runner for MoE Models

Leverages multiple GPUs to accelerate quantization process:
- Data parallel calibration
- Model parallel quantization
- Distributed expert processing
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time

from ..models.load_moe import MoEModelLoader
from ..quant.mixed_precision_quantizer import MixedPrecisionQuantizer, MixedPrecisionConfig
from ..runners.collect_calib import CalibrationCollector
from ..quant.ebss import EBSSSampler
from ..runners.eval_metrics import MetricsEvaluator


logger = logging.getLogger(__name__)


class DistributedPTQRunner:
    """
    Distributed PTQ runner for MoE models

    Supports:
    - Multi-GPU calibration data collection
    - Distributed expert quantization
    - Parallel router optimization
    - Efficient memory management
    """

    def __init__(
        self,
        model_name: str,
        config: MixedPrecisionConfig,
        output_dir: str = "./output/distributed_ptq",
        world_size: int = 8,
        rank: int = 0
    ):
        self.model_name = model_name
        self.config = config
        self.output_dir = Path(output_dir)
        self.world_size = world_size
        self.rank = rank

        # Setup distributed environment
        self.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(self.device)

        # Components
        self.model_loader = None
        self.quantizer = None
        self.calib_collector = None
        self.ebss_sampler = None

        # Results
        self.calibration_data = None
        self.quantization_stats = {}

    def setup_distributed(self):
        """Setup distributed training environment"""
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(
                backend='nccl',
                rank=self.rank,
                world_size=self.world_size
            )

        logger.info(f"Rank {self.rank} initialized on device {self.device}")

    def cleanup_distributed(self):
        """Cleanup distributed environment"""
        if dist.is_initialized():
            dist.destroy_process_group()

    def load_model(self):
        """Load model on current rank with memory optimization"""
        logger.info(f"Rank {self.rank}: Loading model {self.model_name}")

        # Clear GPU cache before loading
        torch.cuda.empty_cache()

        self.model_loader = MoEModelLoader(
            model_name=self.model_name,
            device=str(self.device),
            torch_dtype=torch.float16
        )
        self.model_loader.load()

        # For large MoE models, avoid DDP wrapping to save memory
        # Instead, we'll handle distributed processing at the data level
        logger.info(f"Rank {self.rank}: Model loaded successfully")

        # Log memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(
                self.device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            logger.info(
                f"Rank {self.rank}: GPU memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

    def generate_calibration_data_distributed(
        self,
        seed_texts: List[str],
        use_ebss: bool = True,
        batch_size: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Generate calibration data using distributed processing

        Args:
            seed_texts: Seed texts for calibration
            use_ebss: Whether to use EBSS sampling
            batch_size: Batch size per GPU

        Returns:
            calibration_data: Distributed calibration data
        """
        logger.info(f"Rank {self.rank}: Generating calibration data")

        # Use smaller batch size for large models
        if batch_size > 1:
            batch_size = 1  # Force batch_size=1 for memory efficiency
            logger.info(
                f"Rank {self.rank}: Reduced batch size to 1 for memory efficiency")

        # Distribute texts across ranks
        texts_per_rank = len(seed_texts) // self.world_size
        start_idx = self.rank * texts_per_rank
        end_idx = start_idx + texts_per_rank if self.rank < self.world_size - \
            1 else len(seed_texts)
        local_texts = seed_texts[start_idx:end_idx]

        # Limit the number of texts per rank for large models
        # Limit to 32 texts per rank
        max_texts_per_rank = min(len(local_texts), 32)
        local_texts = local_texts[:max_texts_per_rank]

        logger.info(
            f"Rank {self.rank}: Processing {len(local_texts)} texts (limited for memory)")

        # Generate EBSS samples if requested (with memory optimization)
        if use_ebss:
            try:
                self.ebss_sampler = EBSSSampler(
                    model=self.model_loader.model,
                    tokenizer=self.model_loader.tokenizer,
                    device=str(self.device)
                )
                # Limit EBSS generation to avoid memory issues
                ebss_config = getattr(self.ebss_sampler, 'config', None)
                if ebss_config:
                    ebss_config.max_tokens = min(
                        getattr(ebss_config, 'max_tokens', 512), 256)
                    ebss_config.beam_width = min(
                        getattr(ebss_config, 'beam_width', 4), 2)

                local_texts = self.ebss_sampler.generate(local_texts)
            except Exception as e:
                logger.warning(
                    f"Rank {self.rank}: EBSS generation failed: {e}, using original texts")

        # Collect calibration data on this rank with memory management
        self.calib_collector = CalibrationCollector(
            model_loader=self.model_loader,
            num_samples=len(local_texts),
            max_seq_len=256  # Reduced sequence length for memory efficiency
        )

        try:
            local_calib_data = self.calib_collector.collect_from_dataset(
                local_texts, batch_size
            )
        except Exception as e:
            logger.error(
                f"Rank {self.rank}: Calibration data collection failed: {e}")
            # Return empty calibration data
            local_calib_data = type('CalibrationData', (), {
                'layer_activations': {},
                'layer_affinities': {},
                'expert_activations': {},
                'expert_affinities': {},
                'routing_logits': {},
                'expert_ids': {},
                'expert_call_counts': {}
            })()

        # Clear memory after calibration
        torch.cuda.empty_cache()

        # Gather calibration data from all ranks
        gathered_calib_data = self._gather_calibration_data(local_calib_data)

        logger.info(f"Rank {self.rank}: Calibration data collection complete")
        return gathered_calib_data

    def _gather_calibration_data(
        self,
        local_calib_data
    ) -> Dict[str, torch.Tensor]:
        """
        Gather calibration data from all ranks

        Args:
            local_calib_data: Local calibration data

        Returns:
            gathered_data: Combined calibration data
        """
        if self.world_size == 1:
            return local_calib_data

        gathered_data = {}

        # Gather per-layer activations
        for layer_idx in local_calib_data.layer_activations:
            local_acts = local_calib_data.layer_activations[layer_idx]

            # Gather tensors from all ranks
            gathered_acts = [None] * self.world_size
            dist.all_gather_object(gathered_acts, local_acts.cpu())

            # Concatenate
            all_acts = torch.cat(gathered_acts, dim=0)
            gathered_data[f"layer_{layer_idx}_activations"] = all_acts

        # Gather per-layer affinities
        for layer_idx in local_calib_data.layer_affinities:
            local_affs = local_calib_data.layer_affinities[layer_idx]

            gathered_affs = [None] * self.world_size
            dist.all_gather_object(gathered_affs, local_affs.cpu())

            all_affs = torch.cat(gathered_affs, dim=0)
            gathered_data[f"layer_{layer_idx}_affinities"] = all_affs

        return gathered_data

    def quantize_experts_distributed(
        self,
        calibration_data: Dict[str, torch.Tensor]
    ) -> Dict:
        """
        Quantize expert layers using distributed processing

        Args:
            calibration_data: Calibration data

        Returns:
            quantization_stats: Statistics from all ranks
        """
        logger.info(
            f"Rank {self.rank}: Starting distributed expert quantization")

        self.quantizer = MixedPrecisionQuantizer(self.config)

        # Get expert layers
        expert_layers = self._get_expert_layers()

        # Distribute experts across ranks
        experts_per_rank = len(expert_layers) // self.world_size
        start_idx = self.rank * experts_per_rank
        end_idx = start_idx + experts_per_rank if self.rank < self.world_size - \
            1 else len(expert_layers)
        local_experts = expert_layers[start_idx:end_idx]

        logger.info(
            f"Rank {self.rank}: Processing {len(local_experts)} experts")

        local_stats = {}

        # Quantize local experts with memory management
        for i, (layer_name, layer) in enumerate(local_experts):
            if f"{layer_name}_activations" in calibration_data:
                try:
                    # Move calibration data to device
                    inputs = calibration_data[f"{layer_name}_activations"]
                    affinities = calibration_data.get(
                        f"{layer_name}_affinities")

                    # Handle case where affinities might not exist
                    if affinities is None:
                        # Create dummy affinities if not available
                        affinities = torch.ones(inputs.shape[0])

                    calib_data = {
                        "inputs": inputs.to(self.device),
                        "affinities": affinities.to(self.device)
                    }

                    # Quantize expert layer
                    quantized_layer, stats = self.quantizer.quantize_expert_layer(
                        layer, calib_data
                    )
                    local_stats[layer_name] = stats

                    # Replace layer in model
                    self._replace_layer_in_model(layer_name, quantized_layer)

                    logger.info(
                        f"Rank {self.rank}: ✓ Quantized {layer_name} ({i+1}/{len(local_experts)})")

                    # Clear memory after each quantization
                    del calib_data
                    torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(
                        f"Rank {self.rank}: Failed to quantize {layer_name}: {e}")
                    local_stats[layer_name] = {"error": str(e)}
                    # Continue with next expert even if one fails

        # Gather statistics from all ranks
        all_stats = self._gather_quantization_stats(local_stats)

        logger.info(f"Rank {self.rank}: Expert quantization complete")
        return all_stats

    def quantize_router_layers_distributed(
        self,
        calibration_data: Dict[str, torch.Tensor]
    ) -> Dict:
        """
        Quantize router layers with RouterRank optimization

        Args:
            calibration_data: Calibration data

        Returns:
            router_stats: Router quantization statistics
        """
        logger.info(f"Rank {self.rank}: Starting router quantization")

        # Get router layers
        router_layers = self._get_router_layers()

        router_stats = {}

        for layer_name, layer in router_layers:
            if f"{layer_name}_activations" in calibration_data:
                calib_data = {
                    "inputs": calibration_data[f"{layer_name}_activations"].to(self.device)
                }

                try:
                    quantized_layer, stats = self.quantizer.quantize_router_layer(
                        layer, calib_data
                    )
                    router_stats[layer_name] = stats

                    # Replace layer in model
                    self._replace_layer_in_model(layer_name, quantized_layer)

                    logger.info(
                        f"Rank {self.rank}: ✓ Quantized router {layer_name}")

                except Exception as e:
                    logger.error(
                        f"Rank {self.rank}: Failed to quantize router {layer_name}: {e}")
                    router_stats[layer_name] = {"error": str(e)}

        return router_stats

    def _get_expert_layers(self) -> List[Tuple[str, nn.Module]]:
        """Get all expert layers"""
        expert_layers = []

        for name, module in self.model_loader.model.named_modules():
            if isinstance(module, nn.Linear) and "expert" in name.lower():
                expert_layers.append((name, module))

        return expert_layers

    def _get_router_layers(self) -> List[Tuple[str, nn.Module]]:
        """Get all router layers"""
        router_layers = []

        for name, module in self.model_loader.model.named_modules():
            if isinstance(module, nn.Linear) and any(keyword in name.lower() for keyword in ["gate", "router", "gating"]):
                router_layers.append((name, module))

        return router_layers

    def _replace_layer_in_model(self, layer_name: str, new_layer: nn.Module):
        """Replace layer in model by name"""
        parts = layer_name.split('.')
        current = self.model_loader.model

        # Navigate to parent module
        for part in parts[:-1]:
            current = getattr(current, part)

        # Replace the layer
        setattr(current, parts[-1], new_layer)

    def _gather_quantization_stats(self, local_stats: Dict) -> Dict:
        """Gather quantization statistics from all ranks"""
        if self.world_size == 1:
            return local_stats

        all_stats = [None] * self.world_size
        dist.all_gather_object(all_stats, local_stats)

        # Combine stats from all ranks
        combined_stats = {}
        for stats in all_stats:
            combined_stats.update(stats)

        return combined_stats

    def save_model(self, save_path: str):
        """Save quantized model"""
        if self.rank == 0:  # Only save on rank 0
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save model state dict
            model_state = self.model_loader.model.state_dict()
            torch.save(model_state, save_path)

            logger.info(f"Model saved to {save_path}")

    def run_distributed_ptq(
        self,
        seed_texts: List[str],
        use_ebss: bool = True
    ) -> Dict:
        """
        Run complete distributed PTQ pipeline

        Args:
            seed_texts: Seed texts for calibration
            use_ebss: Whether to use EBSS sampling

        Returns:
            results: Complete PTQ results
        """
        logger.info(f"Rank {self.rank}: Starting distributed PTQ")

        try:
            # Setup
            self.setup_distributed()

            # Load model
            self.load_model()

            # Generate calibration data
            calibration_data = self.generate_calibration_data_distributed(
                seed_texts, use_ebss
            )

            # Quantize experts
            expert_stats = self.quantize_experts_distributed(calibration_data)

            # Quantize routers
            router_stats = self.quantize_router_layers_distributed(
                calibration_data)

            # Combine results
            results = {
                "expert_stats": expert_stats,
                "router_stats": router_stats,
                "config": self.config.__dict__,
                "world_size": self.world_size,
                "rank": self.rank
            }

            # Save results
            if self.rank == 0:
                results_file = self.output_dir / "distributed_ptq_results.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)

                logger.info(f"Results saved to {results_file}")

            return results

        finally:
            self.cleanup_distributed()


def run_distributed_ptq_worker(
    rank: int,
    world_size: int,
    model_name: str,
    config: MixedPrecisionConfig,
    seed_texts: List[str],
    output_dir: str,
    use_ebss: bool = True
):
    """
    Worker function for distributed PTQ

    Args:
        rank: Process rank
        world_size: Total number of processes
        model_name: Model name to quantize
        config: Quantization configuration
        seed_texts: Seed texts for calibration
        output_dir: Output directory
        use_ebss: Whether to use EBSS sampling
    """
    runner = DistributedPTQRunner(
        model_name=model_name,
        config=config,
        output_dir=output_dir,
        world_size=world_size,
        rank=rank
    )

    results = runner.run_distributed_ptq(seed_texts, use_ebss)
    return results


def launch_distributed_ptq(
    model_name: str,
    config: MixedPrecisionConfig,
    seed_texts: List[str],
    output_dir: str = "./output/distributed_ptq",
    world_size: int = 8,
    use_ebss: bool = True
):
    """
    Launch distributed PTQ across multiple GPUs

    Args:
        model_name: Model name to quantize
        config: Quantization configuration
        seed_texts: Seed texts for calibration
        output_dir: Output directory
        world_size: Number of GPUs to use
        use_ebss: Whether to use EBSS sampling
    """
    logger.info(f"Launching distributed PTQ with {world_size} GPUs")

    mp.spawn(
        run_distributed_ptq_worker,
        args=(world_size, model_name, config,
              seed_texts, output_dir, use_ebss),
        nprocs=world_size,
        join=True
    )

    logger.info("Distributed PTQ completed")


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
        use_activation_shaping=False  # Not needed for W4A4
    )
