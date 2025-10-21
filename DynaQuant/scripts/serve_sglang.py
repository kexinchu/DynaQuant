"""
Serving script for DynaQuant with SGLang 0.4.7.
Serves quantized MoE models with dynamic precision scheduling.
"""

from dynaquant.hooks import inject_dynaquant_into_sglang
import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
import logging
import time
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_sglang_server(config: dict):
    """
    Setup SGLang server with DynaQuant.
    """
    try:
        # Try to import SGLang
        try:
            import sglang as sgl
            from sglang import LLM
        except ImportError:
            logger.error(
                "SGLang not found. Please install from sglang-0.4.7 directory")
            logger.info("Falling back to standard transformers serving")
            return setup_transformers_server(config)

        logger.info("Setting up SGLang server with DynaQuant")

        model_name = config['model']['name']
        serving_config = config['serving']

        # Create LLM instance
        llm = LLM(
            model=model_name,
            tokenizer=model_name,
            tensor_parallel_size=serving_config['tensor_parallel_size'],
            max_total_tokens=serving_config['max_total_tokens'],
            mem_fraction_static=serving_config['mem_fraction_static'],
            trust_remote_code=True,
        )

        # Inject DynaQuant
        if any([serving_config['enable_rcg'], serving_config['enable_ps'], serving_config['enable_ec']]):
            logger.info("Injecting DynaQuant hooks")
            hook_manager = inject_dynaquant_into_sglang(
                model=llm.model,
                config=config,
                enable_rcg=serving_config['enable_rcg'],
                enable_ps=serving_config['enable_ps'],
                enable_ec=serving_config['enable_ec'],
            )
        else:
            logger.warning("All DynaQuant features disabled")
            hook_manager = None

        return llm, hook_manager

    except Exception as e:
        logger.error(f"Failed to setup SGLang server: {e}")
        logger.info("Falling back to transformers serving")
        return setup_transformers_server(config)


def setup_transformers_server(config: dict):
    """
    Fallback to transformers-based serving.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    logger.info("Setting up transformers-based serving")

    model_name = config['model']['name']
    cache_dir = config['model'].get('cache_dir', None)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Inject DynaQuant
    serving_config = config['serving']
    if any([serving_config['enable_rcg'], serving_config['enable_ps'], serving_config['enable_ec']]):
        logger.info("Injecting DynaQuant hooks")
        hook_manager = inject_dynaquant_into_sglang(
            model=model,
            config=config,
            enable_rcg=serving_config['enable_rcg'],
            enable_ps=serving_config['enable_ps'],
            enable_ec=serving_config['enable_ec'],
        )
    else:
        hook_manager = None

    # Create pipeline
    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device_map='auto',
    )

    return pipe, hook_manager


def run_server(llm, hook_manager, config):
    """
    Run serving loop.
    """
    logger.info("Starting server")

    serving_config = config['serving']

    # Simple serving loop (in practice, would use FastAPI or similar)
    logger.info(
        f"Server ready on {serving_config['host']}:{serving_config['port']}")
    logger.info("Press Ctrl+C to stop")

    # Statistics tracking
    request_count = 0
    last_stats_time = time.time()
    stats_interval = serving_config.get('stats_interval_secs', 10)

    try:
        while True:
            # In practice, this would be a proper HTTP server
            # For now, just demonstrate statistics collection

            time.sleep(stats_interval)

            # Print statistics
            if hook_manager is not None:
                stats = hook_manager.get_statistics()

                logger.info("="*80)
                logger.info("DynaQuant Statistics")
                logger.info("="*80)

                for block_name, block_stats in stats.items():
                    logger.info(f"\n{block_name}:")
                    logger.info(
                        f"  Forward count: {block_stats['forward_count']}")

                    if 'rcg' in block_stats:
                        rcg_stats = block_stats['rcg']
                        logger.info(
                            f"  RCG - Promotion rate: {rcg_stats['promotion_rate']:.4f}")

                    if 'ps' in block_stats:
                        ps_stats = block_stats['ps']
                        logger.info(
                            f"  PS - Precision counts: {ps_stats['precision_counts']}")
                        logger.info(
                            f"  PS - VRAM usage: {ps_stats['vram_usage_gb']:.2f} GB")

                    if 'ec' in block_stats:
                        ec_stats = block_stats['ec']
                        logger.info(
                            f"  EC - Cache hit rate: {ec_stats['hit_rate']:.4f}")

                logger.info("="*80)

    except KeyboardInterrupt:
        logger.info("\nShutting down server")


def main():
    parser = argparse.ArgumentParser(
        description='Serve DynaQuant model with SGLang')
    parser.add_argument('--config', type=str, default='experiments/config_ptq_qat.yaml',
                        help='Path to configuration file')
    parser.add_argument('--host', type=str, default=None,
                        help='Server host (overrides config)')
    parser.add_argument('--port', type=int, default=None,
                        help='Server port (overrides config)')
    parser.add_argument('--enable-rcg', action='store_true',
                        help='Enable Router-Consistency Guard')
    parser.add_argument('--enable-ps', action='store_true',
                        help='Enable Precision Scheduler')
    parser.add_argument('--enable-ec', action='store_true',
                        help='Enable Expert Cache')

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Override config with command-line args
    if args.host:
        config['serving']['host'] = args.host
    if args.port:
        config['serving']['port'] = args.port

    # Override feature flags if specified
    if args.enable_rcg:
        config['serving']['enable_rcg'] = True
    if args.enable_ps:
        config['serving']['enable_ps'] = True
    if args.enable_ec:
        config['serving']['enable_ec'] = True

    # Setup server
    llm, hook_manager = setup_sglang_server(config)

    # Run server
    run_server(llm, hook_manager, config)


if __name__ == '__main__':
    main()
