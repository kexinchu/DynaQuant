#!/usr/bin/env python3
"""
AWQ W2A16 Quantization Tool
============================
Quantize LLM models to 2-bit weights with AWQ activation-aware calibration.

Usage:
    python scripts/quantize_w2a16.py \
        --model Qwen/Qwen3-30B-A3B \
        --output-dir ./output/Qwen3-30B-A3B-W2A16 \
        --group-size 128 \
        --calib-data ./calibration_datasets/calib.json \
        --num-samples 512 \
        --ignore lm_head \
        --moe
"""

# ============================================================================
# ⚠️  CRITICAL: Import Order MUST NOT Be Changed!
# ============================================================================
# Python executes code top-to-bottom. DO NOT let auto-formatters reorder!
#
# Required Order:
#   1. Import standard library (os, sys, Path, json, argparse, etc.)
#   2. Set sys.path to add project root
#   3. Import local quant modules (ONLY AFTER sys.path is set!)
#
# Auto-formatters (isort, black, etc.) may try to reorder - DON'T LET THEM!
# ============================================================================

# Step 1: Standard library imports ONLY (no local modules!)
import os
import sys
from pathlib import Path
import json
import argparse
from typing import List, Dict

# Step 1b: Third-party imports
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import Dataset
import safetensors.torch as safetensors

# Step 2: Set sys.path BEFORE importing local modules
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Step 3: NOW import local quant modules (after sys.path is set)
# fmt: off  # Prevent formatters from moving these
# isort: skip_file  # Prevent isort from reordering
from quant.awq_w2 import (
    pack_2bit,
    quantize_weight_w2,
    collect_activations,
    calibrate_layer,
    W2AWQLinear,
)
from quant.awq_w2.quantize import QuantizationConfig
# fmt: on


def load_calibration_data(calib_path: str, num_samples: int = 512) -> List[str]:
    """Load calibration dataset."""
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with open(calib_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Support multiple formats
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        if 'samples' in data:
            samples = data['samples']
        elif 'data' in data:
            samples = data['data']
        else:
            raise ValueError(
                f"Unknown calibration format: {list(data.keys())}")
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")

    return samples[:num_samples]


def find_linear_layers(
    model: nn.Module,
    ignore: List[str] = None,
    include_moe: bool = True
) -> Dict[str, nn.Linear]:
    """
    Find all linear layers in the model.

    Args:
        model: Model to search
        ignore: List of module names to ignore (e.g., ['lm_head'])
        include_moe: Whether to include MoE expert layers

    Returns:
        Dictionary mapping layer names to nn.Linear modules
    """
    if ignore is None:
        ignore = []

    linear_layers = {}

    for name, module in model.named_modules():
        # Skip ignored modules
        if any(ig in name for ig in ignore):
            continue

        # Include standard Linear layers
        if isinstance(module, nn.Linear):
            linear_layers[name] = module

    print(f"Found {len(linear_layers)} linear layers")

    # Print some examples
    if linear_layers:
        examples = list(linear_layers.keys())[:5]
        print(f"Examples: {examples}")

    return linear_layers


def create_calibration_dataloader(
    samples: List[str],
    tokenizer,
    batch_size: int = 1,
    max_length: int = 512
):
    """Create dataloader for calibration."""
    # Tokenize
    encodings = tokenizer(
        samples,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    )

    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask']
    })

    # Simple batching
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([torch.tensor(b['input_ids']) for b in batch]),
            'attention_mask': torch.stack([torch.tensor(b['attention_mask']) for b in batch])
        }

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )

    return dataloader


def quantize_model(
    model_path: str,
    output_dir: str,
    calib_data_path: str,
    group_size: int = 128,
    num_samples: int = 512,
    ignore_modules: List[str] = None,
    search_mode: str = 'global',
    device: str = 'cuda',
    include_moe: bool = True,
):
    """
    Main quantization function.

    Args:
        model_path: Path to the model or HuggingFace model ID
        output_dir: Output directory for quantized model
        calib_data_path: Path to calibration data (JSON)
        group_size: Group size for quantization
        num_samples: Number of calibration samples
        ignore_modules: List of module names to skip (e.g., ['lm_head'])
        search_mode: 'global' or 'per_group' for alpha search
        device: Device for computation
        include_moe: Whether to quantize MoE expert layers
    """
    print("="*80)
    print("AWQ W2A16 Quantization")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Group size: {group_size}")
    print(f"Calibration samples: {num_samples}")
    print(f"Search mode: {search_mode}")
    print(f"Ignore modules: {ignore_modules}")
    print("="*80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("\n[1/6] Loading model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
    )
    model.eval()

    print(f"Model loaded: {config.model_type}")
    print(
        f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Load calibration data
    print("\n[2/6] Loading calibration data...")
    calib_samples = load_calibration_data(calib_data_path, num_samples)
    print(f"Loaded {len(calib_samples)} samples")

    calib_dataloader = create_calibration_dataloader(
        calib_samples, tokenizer, batch_size=1, max_length=512
    )

    # Find layers to quantize
    print("\n[3/6] Finding layers to quantize...")
    linear_layers = find_linear_layers(
        model, ignore=ignore_modules, include_moe=include_moe)

    # Collect activations
    print("\n[4/6] Collecting activations...")
    layer_names = list(linear_layers.keys())

    print(f"  Total layers: {len(layer_names)}")
    print(f"  Calibration samples: {num_samples}")

    # Strategy: Balance between speed and memory
    # - Too small batch (50): 373 batches × 1024 samples = 382k forward passes (3 days!)
    # - Too large batch (18672): OOM killed
    # - Sweet spot: 500-1000 layers per batch = ~20-40 batches × 1024 samples = reasonable

    max_layers_per_batch = 1000  # Increased from 50 to 1000 (20x speedup)
    num_batches = (len(layer_names) + max_layers_per_batch -
                   1) // max_layers_per_batch

    print(
        f"  Processing in {num_batches} batches ({max_layers_per_batch} layers/batch)")
    print(f"  Estimated time: ~{num_batches * 5}-{num_batches * 10} minutes")

    activations_dict = {}

    for i in range(0, len(layer_names), max_layers_per_batch):
        batch_names = layer_names[i:i+max_layers_per_batch]
        batch_num = i // max_layers_per_batch + 1

        print(
            f"\n  Batch {batch_num}/{num_batches}: Collecting {len(batch_names)} layers...")

        batch_activations = collect_activations(
            model, calib_dataloader,
            layer_names=batch_names,
            max_samples=128  # Reduce samples per layer to save memory
        )
        activations_dict.update(batch_activations)

        # Clear cache after each batch
        torch.cuda.empty_cache()

        print(
            f"  ✓ Batch {batch_num} done ({len(batch_activations)} layers collected)")

    print(f"\n  ✓ Total collected: {len(activations_dict)} layers")
    torch.cuda.empty_cache()

    # Calibrate and quantize layers
    print("\n[5/6] Calibrating and quantizing layers...")
    quantized_weights = {}

    for name, layer in tqdm(linear_layers.items(), desc="Quantizing"):
        if name not in activations_dict:
            print(f"  Warning: No activations for {name}, skipping...")
            continue

        X = activations_dict[name]

        # Calibrate
        calib_result = calibrate_layer(
            layer=layer,
            X=X,
            group_size=group_size,
            search_mode=search_mode,
            device=device
        )

        quantized_weights[name] = {
            'weight_q': calib_result['weight_q'],
            'scale': calib_result['scale'],
            'error': calib_result['error'],
        }

    avg_error = sum(w['error']
                    for w in quantized_weights.values()) / len(quantized_weights)
    print(f"Average reconstruction error: {avg_error:.6f}")

    # Save quantized model
    print("\n[6/6] Saving quantized model...")

    # Replace Linear layers with W2AWQLinear in the model
    print("  Replacing Linear layers with W2AWQLinear...")
    replaced_count = 0
    for name, layer in list(model.named_modules()):
        if name in quantized_weights:
            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Create W2AWQLinear layer
            original_layer = getattr(parent, child_name)
            w2_layer = W2AWQLinear(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                bias=original_layer.bias is not None,
                group_size=group_size,
                device=original_layer.weight.device,
                dtype=original_layer.weight.dtype,
            )

            # Load quantized weights using the proper method
            w2_layer.load_weights(
                weight_q=quantized_weights[name]['weight_q'],
                scale=quantized_weights[name]['scale'],
                bias=original_layer.bias.data if original_layer.bias is not None else None,
                packed=False  # weight_q is not packed yet
            )

            # Replace the layer
            setattr(parent, child_name, w2_layer)
            replaced_count += 1

    print(f"  Replaced {replaced_count} layers with W2AWQLinear")

    # Save the model with replaced layers
    print("  Saving model with quantized layers...")
    model.save_pretrained(
        output_dir,
        max_shard_size="5GB",
        safe_serialization=True
    )
    print(f"  Saved model to {output_dir}")

    # Also save packed weights separately for backup
    state_dict = {}
    for name, weights in quantized_weights.items():
        weight_packed = pack_2bit(weights['weight_q'])
        state_dict[f"{name}.weight_packed"] = weight_packed
        state_dict[f"{name}.scale"] = weights['scale']

    packed_path = os.path.join(output_dir, "quantized_weights.safetensors")
    safetensors.save_file(state_dict, packed_path)
    print(f"  Saved packed weights to {packed_path}")

    # Save quantization config
    quant_config = QuantizationConfig(
        algorithm="awq",
        bits=2,
        group_size=group_size,
        symmetric=True,
        packed_layout="4x2bit_per_byte",
    )

    config_path = os.path.join(output_dir, "quantization_config.json")
    with open(config_path, 'w') as f:
        json.dump(quant_config.to_dict(), f, indent=2)
    print(f"Saved quantization config to {config_path}")

    # Save original model config
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model config and tokenizer")

    # Save metadata
    metadata = {
        'model_path': model_path,
        'group_size': group_size,
        'num_samples': num_samples,
        'avg_error': avg_error,
        'num_layers_quantized': len(quantized_weights),
        'ignore_modules': ignore_modules,
    }

    metadata_path = os.path.join(output_dir, "quantization_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("✅ Quantization completed successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Layers quantized: {len(quantized_weights)}")
    print(f"Average error: {avg_error:.6f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="AWQ W2A16 Quantization")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model or HuggingFace model ID')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for quantized model')
    parser.add_argument('--calib-data', type=str, default=None,
                        help='Path to calibration data (JSON)')
    parser.add_argument('--group-size', type=int, default=128,
                        choices=[64, 128], help='Group size for quantization')
    parser.add_argument('--num-samples', type=int, default=512,
                        help='Number of calibration samples')
    parser.add_argument('--ignore', type=str, nargs='+', default=['lm_head'],
                        help='Module names to ignore (e.g., lm_head)')
    parser.add_argument('--search-mode', type=str, default='global',
                        choices=['global', 'per_group'],
                        help='Alpha search mode: global or per_group')
    parser.add_argument('--moe', action='store_true',
                        help='Enable MoE expert quantization')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for computation')

    args = parser.parse_args()

    # Auto-find calibration data if not specified
    if args.calib_data is None:
        model_name = Path(args.model).name
        possible_paths = [
            f"calibration_datasets/{model_name}/calibration_{model_name}.json",
            "calibration_datasets/calibration.json",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                args.calib_data = path
                break

        if args.calib_data is None:
            print("ERROR: No calibration data found. Please specify --calib-data")
            sys.exit(1)

    print(f"Using calibration data: {args.calib_data}")

    # Run quantization
    quantize_model(
        model_path=args.model,
        output_dir=args.output_dir,
        calib_data_path=args.calib_data,
        group_size=args.group_size,
        num_samples=args.num_samples,
        ignore_modules=args.ignore,
        search_mode=args.search_mode,
        device=args.device,
        include_moe=args.moe,
    )


if __name__ == '__main__':
    main()
