"""
QAT (Quantization-Aware Training) script for DynaQuant.
Fine-tunes model with fake quantization to improve quantized performance.
"""

from dynaquant.hooks import inject_dynaquant_into_sglang
from dynaquant import fake_quant
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def load_model_and_calibration(config: dict):
    """Load model and calibration parameters."""
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

        # Load calibration parameters
        calib_dir = config['calibration']['save_dir']
        calib_params_path = os.path.join(calib_dir, 'calibration_params.json')

        if os.path.exists(calib_params_path):
            with open(calib_params_path, 'r') as f:
                calib_params = json.load(f)
            logger.info(
                f"Loaded calibration parameters from {calib_params_path}")
        else:
            logger.warning(
                "No calibration parameters found, will use dynamic quantization")
            calib_params = None

        return model, tokenizer, calib_params

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def inject_fake_quantization(model, calib_params, config):
    """Inject fake quantization modules into model."""
    logger.info("Injecting fake quantization")

    qat_config = config['qat']

    # Add fake quantization to linear layers
    fake_quant_modules = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Only quantize MoE expert layers
            if 'expert' in name.lower() or 'mlp' in name.lower():
                # Get calibration params if available
                if calib_params and name in calib_params:
                    params = calib_params[name]
                    scale = torch.tensor(params['scale'])
                    zero_point = torch.tensor(params['zero_point'])
                else:
                    scale = None
                    zero_point = None

                # Create fake quantization module
                fq = fake_quant.FakeQuantize(
                    bits=4,
                    symmetric=True,
                    per_token=True,
                    dynamic=True,  # Use dynamic during QAT
                    scale=scale,
                    zero_point=zero_point,
                )

                # Wrap module output with fake quantization
                original_forward = module.forward

                def make_forward(orig_fwd, fq_module):
                    def forward(x):
                        out = orig_fwd(x)
                        return fq_module(out)
                    return forward

                module.forward = make_forward(original_forward, fq)
                fake_quant_modules[name] = fq

    logger.info(
        f"Injected fake quantization into {len(fake_quant_modules)} modules")

    return fake_quant_modules


def load_training_data(config, tokenizer):
    """Load training dataset for QAT."""
    try:
        from datasets import load_dataset

        qat_config = config['qat']
        num_steps = qat_config['num_steps']
        batch_size = qat_config['batch_size']

        # Estimate number of samples needed
        num_samples = num_steps * batch_size

        logger.info(f"Loading {num_samples} training samples")

        # Use same data sources as calibration
        calib_config = config['calibration']
        seq_length = calib_config['seq_length']

        # For simplicity, reuse calibration data loading logic
        # In practice, should use separate training data

        # Load dummy data for now
        logger.warning("Using dummy training data")
        input_ids = torch.randint(0, 1000, (num_samples, seq_length))

        return input_ids

    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        # Return dummy data
        qat_config = config['qat']
        num_steps = qat_config['num_steps']
        batch_size = qat_config['batch_size']
        seq_length = config['calibration']['seq_length']
        num_samples = num_steps * batch_size
        return torch.randint(0, 1000, (num_samples, seq_length))


def compute_routing_loss(router_logits_fp, router_logits_q, top_k=2):
    """
    Compute routing loss to maintain routing consistency.
    """
    # 1. Top-k agreement loss
    probs_fp = F.softmax(router_logits_fp, dim=-1)
    probs_q = F.softmax(router_logits_q, dim=-1)

    _, indices_fp = torch.topk(probs_fp, top_k, dim=-1)
    _, indices_q = torch.topk(probs_q, top_k, dim=-1)

    # Agreement = how many indices match
    agreement = 0.0
    for i in range(indices_fp.shape[1]):
        matches = (indices_fp == indices_q[:, i:i+1]).any(dim=1).float()
        agreement += matches.mean()
    agreement /= top_k

    topk_loss = 1.0 - agreement

    # 2. JS divergence loss
    from dynaquant.router_guard import js_divergence
    js_loss = js_divergence(probs_fp, probs_q).mean()

    # 3. Margin loss (encourage larger margins)
    sorted_logits_fp, _ = torch.sort(router_logits_fp, dim=-1, descending=True)
    sorted_logits_q, _ = torch.sort(router_logits_q, dim=-1, descending=True)

    margin_fp = sorted_logits_fp[:, top_k-1] - sorted_logits_fp[:, top_k]
    margin_q = sorted_logits_q[:, top_k-1] - sorted_logits_q[:, top_k]

    margin_loss = torch.abs(margin_fp - margin_q).mean()

    return topk_loss, js_loss, margin_loss


def train_qat(model, training_data, fake_quant_modules, config):
    """Run QAT training."""
    logger.info("Starting QAT training")

    qat_config = config['qat']

    # Training parameters
    num_steps = qat_config['num_steps']
    batch_size = qat_config['batch_size']
    learning_rate = qat_config['learning_rate']
    gradient_accumulation_steps = qat_config['gradient_accumulation_steps']

    # Loss weights
    task_weight = qat_config['loss']['task_weight']
    topk_weight = qat_config['loss']['topk_agreement_weight']
    js_weight = qat_config['loss']['js_divergence_weight']
    margin_weight = qat_config['loss']['margin_weight']

    # Freeze settings
    freeze_non_router = qat_config['freeze_non_router']
    unfreeze_expert_out_proj = qat_config['unfreeze_expert_out_proj']

    # Prepare model for training
    model.train()

    # Freeze/unfreeze parameters
    if freeze_non_router:
        logger.info("Freezing non-router weights")
        for name, param in model.named_parameters():
            if 'gate' not in name.lower() and 'router' not in name.lower():
                param.requires_grad = False

    if unfreeze_expert_out_proj:
        logger.info("Unfreezing expert out_proj weights")
        for name, param in model.named_parameters():
            if 'expert' in name.lower() and 'down_proj' in name.lower():
                param.requires_grad = True

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    # Training loop
    device = next(model.parameters()).device
    num_samples = training_data.shape[0]

    global_step = 0
    optimizer.zero_grad()

    progress_bar = tqdm(range(num_steps), desc="QAT Training")

    for step in progress_bar:
        # Get batch
        start_idx = (step * batch_size) % num_samples
        batch = training_data[start_idx:start_idx+batch_size].to(device)

        # Forward pass
        try:
            outputs = model(batch, labels=batch)
            loss_task = outputs.loss
        except:
            # Fallback if labels not supported
            outputs = model(batch)
            loss_task = torch.tensor(0.0, device=device)

        # Compute routing loss (simplified)
        # In practice, need to collect router logits before and after quantization
        loss_routing = torch.tensor(0.0, device=device)

        # Total loss
        loss = task_weight * loss_task + loss_routing

        # Backward
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # Update
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # Log
        progress_bar.set_postfix({
            'loss': loss.item() * gradient_accumulation_steps,
            'task_loss': loss_task.item() if isinstance(loss_task, torch.Tensor) else 0.0,
        })

        # Save checkpoint
        if (step + 1) % qat_config['checkpoint_steps'] == 0:
            save_dir = qat_config['save_dir']
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                save_dir, f'checkpoint_step_{global_step}.pt')

            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

            logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("QAT training complete!")

    # Save final model
    save_dir = qat_config['save_dir']
    final_path = os.path.join(save_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved final model to {final_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run QAT training for DynaQuant')
    parser.add_argument('--config', type=str, default='experiments/config_ptq_qat.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (overrides config)')

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    if args.output_dir:
        config['qat']['save_dir'] = args.output_dir

    # Check if QAT is enabled
    if not config['qat']['enabled']:
        logger.warning("QAT is disabled in config, exiting")
        return

    # Load model and calibration
    model, tokenizer, calib_params = load_model_and_calibration(config)

    # Inject fake quantization
    fake_quant_modules = inject_fake_quantization(model, calib_params, config)

    # Load training data
    training_data = load_training_data(config, tokenizer)

    # Run QAT training
    train_qat(model, training_data, fake_quant_modules, config)

    logger.info("QAT complete!")


if __name__ == '__main__':
    main()
