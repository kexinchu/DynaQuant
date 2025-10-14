"""
Parallel PTQ Runner for Multi-GPU MoE Quantization

Supports distributed quantization across multiple GPUs for faster processing.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json

from ..quant.agq import AGQuantizer, AGQConfig
from ..quant.quantizers import W2A2Quantizer, W2A2Config
from ..models.load_moe import MoEModelLoader
from ..runners.collect_calib import CalibrationCollector


logger = logging.getLogger(__name__)


class ParallelPTQRunner:
    """
    Parallel PTQ Runner for multi-GPU quantization

    Distributes expert quantization across GPUs for faster processing.
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        w_bit: int = 2,
        a_bit: int = 2,
        router_w_bit: int = 8,
        router_a_bit: int = 8,
        group_size: int = 64,
        calib_size: int = 128,
        use_rotation: bool = True,
        num_gpus: int = 8
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.w_bit = w_bit
        self.a_bit = a_bit
        self.router_w_bit = router_w_bit
        self.router_a_bit = router_a_bit
        self.group_size = group_size
        self.calib_size = calib_size
        self.use_rotation = use_rotation
        self.num_gpus = num_gpus

        # Configs
        self.w2a2_config = W2A2Config(
            w_bit=w_bit,
            a_bit=a_bit,
            w_group_size=group_size,
            a_group_size=group_size,
            use_rotation=use_rotation,
            use_whitening=True,
            enable_fallback=True
        )

        self.router_config = W2A2Config(
            w_bit=router_w_bit,
            a_bit=router_a_bit,
            w_group_size=group_size,
            a_group_size=group_size,
            use_rotation=False,
            use_whitening=False,
            enable_fallback=False
        )

        self.agq_config = AGQConfig(
            bit_width=w_bit,
            group_size=group_size,
            use_error_compensation=True
        )

    def setup_distributed(self, rank: int, world_size: int):
        """Setup distributed training"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

        torch.cuda.set_device(rank)

    def cleanup_distributed(self):
        """Cleanup distributed training"""
        dist.destroy_process_group()

    def collect_calibration_data(self, rank: int):
        """Collect calibration data on rank 0"""
        if rank != 0:
            return None

        logger.info("Collecting calibration data on GPU 0...")

        # Load model on GPU 0
        model_loader = MoEModelLoader(
            self.model_path,
            device=f"cuda:{rank}",
            torch_dtype=torch.float16
        )
        model_loader.load()

        # Collect calibration data
        collector = CalibrationCollector(
            model_loader,
            num_samples=self.calib_size,
            max_seq_len=512
        )

        # Use seed texts for calibration
        seed_texts = self._get_seed_texts()[:self.calib_size]

        calib_data = collector.collect_from_dataset(seed_texts, batch_size=1)

        # Save calibration data
        calib_file = self.output_dir / "calibration_data.pkl"
        calib_data.save(str(calib_file))

        logger.info(f"Calibration data saved to {calib_file}")

        return calib_data

    def _get_seed_texts(self) -> List[str]:
        """Get seed texts for calibration"""
        seed_file = Path("data/seed_text.txt")
        if seed_file.exists():
            with open(seed_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            # Default seed texts
            texts = [
                "Artificial intelligence is transforming the world through advanced machine learning.",
                "The scientific method involves careful observation and experimentation.",
                "Climate change poses significant challenges to global ecosystems.",
                "Quantum computing represents a paradigm shift in computational power.",
                "Natural language processing enables machines to understand human language.",
            ] * 50

        return texts

    def quantize_experts_parallel(
        self,
        rank: int,
        world_size: int,
        calib_data_path: str
    ):
        """Quantize experts in parallel across GPUs"""

        # Setup distributed
        self.setup_distributed(rank, world_size)

        logger.info(f"GPU {rank}: Starting expert quantization...")

        # Load model
        model_loader = MoEModelLoader(
            self.model_path,
            device=f"cuda:{rank}",
            torch_dtype=torch.float16
        )
        model_loader.load()

        # Load calibration data
        from ..runners.collect_calib import CalibrationData
        calib_data = CalibrationData.load(calib_data_path)

        # Create quantizers
        expert_quantizer = W2A2Quantizer(self.w2a2_config)
        router_quantizer = W2A2Quantizer(self.router_config)
        agq_quantizer = AGQuantizer(self.agq_config)

        # Get layers to process
        num_layers = model_loader.get_num_moe_layers()
        layers_per_gpu = (num_layers + world_size - 1) // world_size
        start_layer = rank * layers_per_gpu
        end_layer = min(start_layer + layers_per_gpu, num_layers)

        logger.info(
            f"GPU {rank}: Processing layers {start_layer} to {end_layer-1}")

        quantized_results = {}

        for layer_idx in range(start_layer, end_layer):
            logger.info(f"GPU {rank}: Quantizing layer {layer_idx}")

            # Quantize router (W8A8)
            router = model_loader.get_router(layer_idx)
            if router is not None and layer_idx in calib_data.layer_activations:
                X_router = calib_data.layer_activations[layer_idx].to(
                    f"cuda:{rank}")

                # Simple quantization for router (no complex shaping)
                W_router_quant, _, router_stats = router_quantizer.quantize_linear_layer(
                    router, X_router, layer_id=layer_idx
                )

                quantized_results[f"router_{layer_idx}"] = {
                    "weight": W_router_quant.cpu(),
                    "stats": router_stats
                }

            # Quantize experts (W2A2 or W4A4)
            experts = model_loader.get_experts(layer_idx)
            if experts is None:
                continue

            for expert_id, expert in enumerate(experts):
                expert_key = (layer_idx, expert_id)

                if expert_key not in calib_data.expert_activations:
                    logger.warning(
                        f"GPU {rank}: No calib data for expert {expert_key}")
                    continue

                X_expert = calib_data.expert_activations[expert_key].to(
                    f"cuda:{rank}")
                c_expert = calib_data.expert_affinities[expert_key].to(
                    f"cuda:{rank}")

                # Process each linear layer in expert
                for name, module in expert.named_modules():
                    if not isinstance(module, torch.nn.Linear):
                        continue

                    # AGQ quantization
                    W_agq, scales_agq, stats_agq = agq_quantizer.quantize_linear(
                        module, X_expert, c_expert,
                        bit_width=self.w_bit,
                        group_size=self.group_size
                    )

                    # W2A2 quantization
                    W_w2a2, W_absorbed, stats_w2a2 = expert_quantizer.quantize_linear_layer(
                        module, X_expert, layer_id=layer_idx
                    )

                    # Store results
                    result_key = f"expert_{layer_idx}_{expert_id}_{name}"
                    quantized_results[result_key] = {
                        "W_agq": W_agq.cpu(),
                        "W_w2a2": W_w2a2.cpu(),
                        "W_absorbed": W_absorbed.cpu(),
                        "scales_agq": scales_agq.cpu() if scales_agq is not None else None,
                        "stats": {**stats_agq, **stats_w2a2}
                    }

                    logger.info(f"GPU {rank}: Quantized {result_key}")

        # Save results for this GPU
        result_file = self.output_dir / f"quantized_gpu{rank}.pt"
        torch.save(quantized_results, result_file)

        logger.info(f"GPU {rank}: Results saved to {result_file}")

        # Wait for all GPUs
        dist.barrier()

        # Cleanup
        self.cleanup_distributed()

    def merge_results(self):
        """Merge results from all GPUs"""
        logger.info("Merging results from all GPUs...")

        all_results = {}
        all_stats = {}

        for rank in range(self.num_gpus):
            result_file = self.output_dir / f"quantized_gpu{rank}.pt"
            if result_file.exists():
                results = torch.load(result_file)
                all_results.update(results)

                # Extract stats
                for key, value in results.items():
                    if "stats" in value:
                        all_stats[key] = value["stats"]

        # Save merged results
        merged_file = self.output_dir / "quantized_model_full.pt"
        torch.save(all_results, merged_file)

        # Save stats
        stats_file = self.output_dir / "quantization_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)

        logger.info(f"Merged results saved to {merged_file}")
        logger.info(f"Statistics saved to {stats_file}")

        # Clean up individual GPU files
        for rank in range(self.num_gpus):
            result_file = self.output_dir / f"quantized_gpu{rank}.pt"
            if result_file.exists():
                result_file.unlink()

    def run(self):
        """Run complete parallel PTQ pipeline"""
        logger.info("=" * 60)
        logger.info(f"Starting Parallel PTQ for {self.model_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(
            f"Config: W{self.w_bit}A{self.a_bit} (experts), W{self.router_w_bit}A{self.router_a_bit} (router)")
        logger.info(f"Using {self.num_gpus} GPUs")
        logger.info("=" * 60)

        # Step 1: Collect calibration data on GPU 0
        calib_data = self.collect_calibration_data(rank=0)
        calib_data_path = str(self.output_dir / "calibration_data.pkl")

        # Step 2: Launch parallel quantization
        logger.info("Launching parallel expert quantization...")

        mp.spawn(
            self.quantize_experts_parallel,
            args=(self.num_gpus, calib_data_path),
            nprocs=self.num_gpus,
            join=True
        )

        # Step 3: Merge results
        self.merge_results()

        logger.info("=" * 60)
        logger.info("Parallel PTQ Complete!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 60)


def run_parallel_ptq(
    model_path: str,
    output_dir: str,
    w_bit: int = 2,
    a_bit: int = 2,
    router_w_bit: int = 8,
    router_a_bit: int = 8,
    num_gpus: int = 8,
    calib_size: int = 128
):
    """
    Convenience function to run parallel PTQ

    Args:
        model_path: Path to base model
        output_dir: Output directory for quantized model
        w_bit: Expert weight bits
        a_bit: Expert activation bits
        router_w_bit: Router weight bits (default 8)
        router_a_bit: Router activation bits (default 8)
        num_gpus: Number of GPUs to use
        calib_size: Calibration set size
    """
    runner = ParallelPTQRunner(
        model_path=model_path,
        output_dir=output_dir,
        w_bit=w_bit,
        a_bit=a_bit,
        router_w_bit=router_w_bit,
        router_a_bit=router_a_bit,
        num_gpus=num_gpus,
        calib_size=calib_size
    )

    runner.run()
