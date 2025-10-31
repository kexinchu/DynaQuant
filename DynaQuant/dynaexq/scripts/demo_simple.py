#!/usr/bin/env python3
"""
Simple end-to-end demo of DynaExQ runtime

This demonstrates the basic usage of DynaExQ with a simulated MoE workload.
"""

from dynaexq.runtime.types import ExpertID
from dynaexq.integration.hooks_base import DynaExQRuntime
from dynaexq.config import load_config
import sys
import logging
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simulate_moe_inference(runtime: DynaExQRuntime, num_batches: int = 100):
    """
    Simulate MoE inference with varying expert activation patterns.

    Args:
        runtime: DynaExQ runtime instance
        num_batches: Number of batches to process
    """
    num_layers = runtime.config["num_layers"]
    num_experts = runtime.config["num_experts_per_layer"]
    top_k = 2
    batch_size = 4

    logger.info(
        f"Starting simulation: {num_batches} batches, {num_layers} layers")

    # Define different workload phases
    # Phase 1 (batches 0-33): Focus on experts 0-7 (math domain)
    # Phase 2 (batches 34-66): Focus on experts 8-15 (code domain)
    # Phase 3 (batches 67-99): Mixed workload

    for batch_idx in range(num_batches):
        if batch_idx % 10 == 0:
            logger.info(f"Processing batch {batch_idx}/{num_batches}")

        # Determine workload phase
        if batch_idx < 33:
            # Math domain: experts 0-7 are hot
            hot_experts = list(range(0, 8))
        elif batch_idx < 67:
            # Code domain: experts 8-15 are hot
            hot_experts = list(range(8, 16))
        else:
            # Mixed: all experts with varying probability
            hot_experts = list(range(0, num_experts))

        # Process each layer
        for layer_id in range(num_layers):
            # Simulate router output
            # Sample top-k experts from hot set with noise
            topk_indices = []
            logits_list = []

            for _ in range(batch_size):
                # Sample from hot experts with bias
                if np.random.random() < 0.8:  # 80% from hot set
                    selected = np.random.choice(
                        hot_experts, size=top_k, replace=False)
                else:  # 20% random
                    selected = np.random.choice(
                        num_experts, size=top_k, replace=False)

                # Generate logits (higher for selected experts)
                layer_logits = np.random.exponential(0.3, size=top_k)
                layer_logits = layer_logits / layer_logits.sum()  # Normalize

                topk_indices.append(selected)
                logits_list.append(layer_logits)

            topk_indices = np.array(topk_indices)
            logits = np.array(logits_list)

            # DynaExQ hooks
            runtime.on_layer_start(layer_id)
            runtime.on_router_output(layer_id, topk_indices, logits)
            runtime.ensure_experts_ready(layer_id, topk_indices, timeout=1.0)
            runtime.on_layer_end(layer_id)

    logger.info("Simulation complete")


def main():
    """Main demo function"""
    logger.info("=" * 60)
    logger.info("DynaExQ Simple Demo")
    logger.info("=" * 60)

    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    config = load_config(str(config_path))

    logger.info(f"Loaded config: tau_h={config.get('thresholds.tau_h')}, "
                f"tau_c={config.get('thresholds.tau_c')}")

    # Override some settings for demo
    config.update({
        "model.num_layers": 8,
        "model.num_experts_per_layer": 32,
        "pool.hot_w4_slots": 8,
        "hotness.window": 30.0,  # 30 second epochs for demo
    })

    # Create runtime
    runtime = DynaExQRuntime(config.to_dict())

    # Start runtime
    runtime.start()

    try:
        # Run simulation
        simulate_moe_inference(runtime, num_batches=100)

        # Print statistics
        logger.info("\n" + "=" * 60)
        logger.info("Final Statistics")
        logger.info("=" * 60)

        stats = runtime.get_statistics()

        logger.info("\n--- Monitor ---")
        logger.info(f"  Current epoch: {stats['monitor']['current_epoch']}")
        logger.info(
            f"  Total experts tracked: {stats['monitor']['total_experts_tracked']}")
        logger.info(f"  Mean hotness: {stats['monitor']['mean_hotness']:.4f}")
        logger.info(f"  Max hotness: {stats['monitor']['max_hotness']:.4f}")

        logger.info("\n--- Controller ---")
        logger.info(f"  tau_h: {stats['controller']['tau_h']:.3f}")
        logger.info(f"  tau_c: {stats['controller']['tau_c']:.3f}")
        logger.info(
            f"  Current W4 experts: {stats['controller']['current_w4_experts']}")
        logger.info(
            f"  Current W2 experts: {stats['controller']['current_w2_experts']}")

        logger.info("\n--- Memory ---")
        logger.info(
            f"  Hot pool utilization: {stats['memory']['hot_pool']['utilization']:.2%}")
        logger.info(
            f"  Cold pool utilization: {stats['memory']['cold_pool']['utilization']:.2%}")
        logger.info(f"  HBM pressure: {stats['memory']['hbm_pressure']:.2%}")
        logger.info(f"  Eviction count: {stats['memory']['eviction_count']}")

        logger.info("\n--- Swap Engine ---")
        logger.info(
            f"  Total upgrades: {stats['swap_engine']['upgrade_count']}")
        logger.info(
            f"  Total downgrades: {stats['swap_engine']['downgrade_count']}")
        logger.info(
            f"  Ready before use: {stats['swap_engine']['ready_before_use']}")
        logger.info(f"  Misses: {stats['swap_engine']['miss_count']}")
        logger.info(
            f"  Ready ratio: {stats['swap_engine']['ready_ratio']:.2%}")

        logger.info("\n--- Prefetch ---")
        logger.info(f"  Prefetch count: {stats['prefetch']['prefetch_count']}")
        logger.info(f"  Hit rate: {stats['prefetch']['hit_rate']:.2%}")

        # Export telemetry
        telemetry_path = "demo_telemetry_summary.json"
        runtime.telemetry.export_summary(telemetry_path)
        logger.info(f"\nTelemetry exported to {telemetry_path}")

    finally:
        # Stop runtime
        runtime.stop()

    logger.info("\n" + "=" * 60)
    logger.info("Demo completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
