"""
PTQ Calibration script for DynaQuant.
Collects activation statistics and calibrates quantization parameters.
"""

from dynaquant.hooks import inject_dynaquant_into_sglang
from dynaquant import fake_quant, router_guard, pack
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import logging
from tqdm import tqdm
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


def load_model(config: dict):
    """Load model for calibration."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = config['model']['name']
        cache_dir = config['model'].get('cache_dir', None)
        trust_remote_code = config['model'].get('trust_remote_code', True)

        logger.info(f"Loading model: {model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16,
            device_map='auto',
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )

        logger.info(f"Model loaded successfully")

        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_calibration_data(config: dict, tokenizer):
    """Load calibration dataset."""
    try:
        from datasets import load_dataset

        calib_config = config['calibration']
        num_samples = calib_config['num_samples']
        seq_length = calib_config['seq_length']

        # Load data from different sources
        data_sources = calib_config['data']['sources']

        all_texts = []

        for source_info in data_sources:
            source_name = source_info['name']
            source_weight = source_info['weight']
            num_source_samples = int(num_samples * source_weight)

            logger.info(
                f"Loading {num_source_samples} samples from {source_name}")

            if source_name == 'pile':
                # Load from Pile
                try:
                    dataset = load_dataset(
                        'EleutherAI/pile', split='train', streaming=True)
                    texts = []
                    for i, example in enumerate(dataset):
                        if i >= num_source_samples:
                            break
                        texts.append(example['text'])
                    all_texts.extend(texts)
                except:
                    logger.warning(
                        f"Could not load {source_name}, using placeholder")
                    all_texts.extend(
                        ["Sample text for calibration."] * num_source_samples)

            elif source_name == 'c4':
                # Load from C4
                try:
                    dataset = load_dataset(
                        'c4', 'en', split='train', streaming=True)
                    texts = []
                    for i, example in enumerate(dataset):
                        if i >= num_source_samples:
                            break
                        texts.append(example['text'])
                    all_texts.extend(texts)
                except:
                    logger.warning(
                        f"Could not load {source_name}, using placeholder")
                    all_texts.extend(
                        ["Sample text for calibration."] * num_source_samples)

            else:
                logger.warning(
                    f"Unknown data source: {source_name}, using placeholder")
                all_texts.extend(
                    ["Sample text for calibration."] * num_source_samples)

        # Tokenize
        logger.info(f"Tokenizing {len(all_texts)} samples")

        input_ids_list = []
        for text in tqdm(all_texts, desc="Tokenizing"):
            tokens = tokenizer(
                text,
                max_length=seq_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
            input_ids_list.append(tokens['input_ids'])

        # Stack
        input_ids = torch.cat(input_ids_list, dim=0)

        logger.info(f"Calibration data shape: {input_ids.shape}")

        return input_ids

    except Exception as e:
        logger.error(f"Failed to load calibration data: {e}")
        # Return dummy data
        logger.warning("Using dummy calibration data")
        num_samples = config['calibration']['num_samples']
        seq_length = config['calibration']['seq_length']
        return torch.randint(0, 1000, (num_samples, seq_length))


def calibrate_model(model, calibration_data, config):
    """Run calibration on model."""
    logger.info("Starting calibration")

    calib_config = config['calibration']
    batch_size = calib_config['batch_size']
    num_samples = calibration_data.shape[0]

    # Inject observers into model
    logger.info("Injecting activation observers")

    observers = {}

    def add_observer_hook(module, name):
        """Add observer hook to a module."""
        observer = fake_quant.ActivationObserver(
            bits=4,
            symmetric=True,
            per_token=True,
            percentile=calib_config['histograms']['percentile'],
        )

        def hook(module, input, output):
            observer(output[0] if isinstance(output, tuple) else output)

        handle = module.register_forward_hook(hook)
        observers[name] = (observer, handle)

    # Add observers to all linear layers (simplified)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Only observe MoE expert layers
            if 'expert' in name.lower() or 'mlp' in name.lower():
                add_observer_hook(module, name)

    logger.info(f"Added {len(observers)} observers")

    # Run forward passes
    model.eval()
    device = next(model.parameters()).device

    logger.info("Running calibration forward passes")

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Calibrating"):
            batch = calibration_data[i:i+batch_size].to(device)

            try:
                outputs = model(batch, labels=batch)
            except:
                # Some models don't support labels during calibration
                try:
                    outputs = model(batch)
                except Exception as e:
                    logger.error(f"Forward pass failed: {e}")
                    continue

    # Calibrate observers
    logger.info("Calibrating quantization parameters")

    calibration_params = {}

    for name, (observer, handle) in observers.items():
        observer.calibrate(method=calib_config['scale_method'])
        params = observer.get_calibration_params()
        calibration_params[name] = {
            'scale': params['scale'].cpu().item(),
            'zero_point': params['zero_point'].cpu().item(),
            'bits': params['bits'],
        }

        # Remove hook
        handle.remove()

    logger.info(f"Calibrated {len(calibration_params)} layers")

    # Save calibration results
    save_dir = calib_config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    params_path = os.path.join(save_dir, 'calibration_params.json')
    with open(params_path, 'w') as f:
        json.dump(calibration_params, f, indent=2)

    logger.info(f"Saved calibration parameters to {params_path}")

    return calibration_params


def calibrate_router(model, calibration_data, config):
    """
    Calibrate router temperature and clip range via grid search.
    """
    logger.info("Calibrating router parameters")

    rcg_config = config['router_guard']

    # Extract temperature and clip range to search
    temp_range = rcg_config['grid_search']['temperature_range']
    clip_range = rcg_config['grid_search']['clip_range']

    logger.info(
        f"Grid search over temperature: {temp_range}, clip: {clip_range}")

    # Collect router logits
    logger.info("Collecting router logits")

    router_logits_fp = []

    # Add hook to collect router logits
    def router_hook(module, input, output):
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        router_logits_fp.append(logits.detach().cpu())

    # Find router (gate) modules
    router_handles = []
    for name, module in model.named_modules():
        if 'gate' in name.lower() and isinstance(module, nn.Linear):
            if 'mlp' in name.lower() or 'moe' in name.lower():
                handle = module.register_forward_hook(router_hook)
                router_handles.append(handle)

    # Run forward passes
    model.eval()
    device = next(model.parameters()).device
    batch_size = config['calibration']['batch_size']

    with torch.no_grad():
        for i in tqdm(range(0, min(1000, calibration_data.shape[0]), batch_size), desc="Collecting router logits"):
            batch = calibration_data[i:i+batch_size].to(device)
            try:
                _ = model(batch)
            except:
                pass

    # Remove hooks
    for handle in router_handles:
        handle.remove()

    if not router_logits_fp:
        logger.warning(
            "No router logits collected, skipping router calibration")
        return

    # Concatenate logits
    router_logits_fp = torch.cat(router_logits_fp, dim=0)[
        :1000]  # Use first 1000

    logger.info(f"Collected router logits shape: {router_logits_fp.shape}")

    # Simulate quantized logits (add noise)
    router_logits_quant = router_logits_fp + \
        torch.randn_like(router_logits_fp) * 0.05

    # Grid search
    best_temp, best_clip, best_agreement = router_guard.grid_search_temperature_clip(
        router_logits_fp,
        router_logits_quant,
        top_k=2,
        temperature_range=temp_range,
        clip_range=clip_range,
    )

    logger.info(f"Best temperature: {best_temp}")
    logger.info(f"Best clip range: {best_clip}")
    logger.info(f"Best top-k agreement: {best_agreement:.4f}")

    # Save router calibration
    save_dir = config['calibration']['save_dir']
    router_params_path = os.path.join(save_dir, 'router_params.json')

    router_params = {
        'temperature': best_temp,
        'clip_range': best_clip,
        'top_k_agreement': best_agreement,
    }

    with open(router_params_path, 'w') as f:
        json.dump(router_params, f, indent=2)

    logger.info(f"Saved router parameters to {router_params_path}")


def main():
    parser = argparse.ArgumentParser(description='Calibrate DynaQuant model')
    parser.add_argument('--config', type=str, default='experiments/config_ptq_qat.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (overrides config)')

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    if args.output_dir:
        config['calibration']['save_dir'] = args.output_dir

    # Load model
    model, tokenizer = load_model(config)

    # Load calibration data
    calibration_data = load_calibration_data(config, tokenizer)

    # Calibrate activations
    calibration_params = calibrate_model(model, calibration_data, config)

    # Calibrate router
    calibrate_router(model, calibration_data, config)

    logger.info("Calibration complete!")


if __name__ == '__main__':
    main()
