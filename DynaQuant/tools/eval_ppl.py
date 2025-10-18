#!/usr/bin/env python3
"""
Perplexity Evaluation Tool for Quantized Models
================================================
Evaluate perplexity on WikiText2 and compare FP16 vs W4A16 vs W2A16.

Usage:
    python tools/eval_ppl.py \
        --model ./output/Qwen3-30B-A3B-W2A16 \
        --baseline /path/to/fp16/model \
        --dataset wikitext2 \
        --output results.json
"""

from quant.awq_w2 import W2AWQLinear, unpack_2bit
import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import safetensors.torch as safetensors

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_quantized_model(model_path: str, device: str = 'cuda'):
    """
    Load a W2A16 quantized model.

    Args:
        model_path: Path to quantized model directory
        device: Device to load model on

    Returns:
        Model with quantized layers
    """
    print(f"Loading quantized model from {model_path}...")

    # Load config and create model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    # Check if quantization metadata exists
    quant_config_path = os.path.join(model_path, "quantization_config.json")
    if not os.path.exists(quant_config_path):
        print("Warning: No quantization config found, assuming FP16 model")
        return model

    with open(quant_config_path, 'r') as f:
        quant_config = json.load(f)

    print(f"Quantization config: {quant_config}")

    # Load quantized weights
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        state_dict = safetensors.load_file(safetensors_path)

        # Replace Linear layers with W2AWQLinear
        group_size = quant_config.get('group_size', 128)
        replace_with_quantized_layers(model, state_dict, group_size)

    return model


def replace_with_quantized_layers(model: nn.Module, state_dict: dict, group_size: int):
    """Replace nn.Linear with W2AWQLinear based on state_dict."""
    quantized_layers = set()

    # Find which layers have quantized weights
    for key in state_dict.keys():
        if key.endswith('.weight_packed'):
            layer_name = key.replace('.weight_packed', '')
            quantized_layers.add(layer_name)

    print(f"Found {len(quantized_layers)} quantized layers")

    # Replace layers
    for name in quantized_layers:
        try:
            # Get the module
            module = dict(model.named_modules())[name]

            if isinstance(module, nn.Linear):
                # Create quantized layer
                quant_layer = W2AWQLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    group_size=group_size,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )

                # Load weights
                weight_packed = state_dict[f"{name}.weight_packed"]
                scale = state_dict[f"{name}.scale"]
                bias = module.bias.data if module.bias is not None else None

                quant_layer.load_weights(
                    weight_packed, scale, bias, packed=True)

                # Replace in parent module
                parent_name, child_name = name.rsplit('.', 1)
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, child_name, quant_layer)

        except Exception as e:
            print(f"Warning: Failed to replace {name}: {e}")


@torch.no_grad()
def evaluate_perplexity(
    model,
    tokenizer,
    dataset_name: str = 'wikitext2',
    max_length: int = 512,
    stride: int = 512,
    device: str = 'cuda'
) -> float:
    """
    Evaluate perplexity on a dataset.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset_name: Dataset name ('wikitext2' or 'ptb')
        max_length: Maximum sequence length
        stride: Stride for sliding window
        device: Device

    Returns:
        Perplexity score
    """
    model.eval()

    # Load dataset
    if dataset_name == 'wikitext2':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        text = '\n\n'.join(dataset['text'])
    elif dataset_name == 'ptb':
        dataset = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        text = '\n\n'.join(dataset['sentence'])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Tokenize
    encodings = tokenizer(text, return_tensors='pt')

    # Compute perplexity with sliding window
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0

    print(f"Evaluating perplexity on {dataset_name} (seq_len={seq_len})...")

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()


def compare_models(
    quantized_path: str,
    baseline_path: str = None,
    dataset: str = 'wikitext2',
    output_file: str = None,
    device: str = 'cuda'
):
    """
    Compare perplexity of quantized model against baseline.

    Args:
        quantized_path: Path to quantized model
        baseline_path: Path to FP16 baseline (optional)
        dataset: Dataset name
        output_file: Output JSON file
        device: Device
    """
    results = {
        'dataset': dataset,
        'models': {}
    }

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        quantized_path, trust_remote_code=True)

    # Evaluate quantized model
    print("\n" + "="*80)
    print("Evaluating Quantized Model (W2A16)")
    print("="*80)

    quant_model = load_quantized_model(quantized_path, device)
    quant_ppl = evaluate_perplexity(
        quant_model, tokenizer, dataset, device=device)

    results['models']['w2a16'] = {
        'path': quantized_path,
        'perplexity': quant_ppl,
    }

    print(f"\n✓ W2A16 Perplexity: {quant_ppl:.4f}")

    # Evaluate baseline if provided
    if baseline_path:
        print("\n" + "="*80)
        print("Evaluating Baseline Model (FP16)")
        print("="*80)

        baseline_model = AutoModelForCausalLM.from_pretrained(
            baseline_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )

        baseline_ppl = evaluate_perplexity(
            baseline_model, tokenizer, dataset, device=device)

        results['models']['fp16'] = {
            'path': baseline_path,
            'perplexity': baseline_ppl,
        }

        print(f"\n✓ FP16 Perplexity: {baseline_ppl:.4f}")

        # Compute degradation
        degradation = ((quant_ppl - baseline_ppl) / baseline_ppl) * 100
        results['degradation_percent'] = degradation

        print("\n" + "="*80)
        print("Comparison Results")
        print("="*80)
        print(f"FP16 PPL:         {baseline_ppl:.4f}")
        print(f"W2A16 PPL:        {quant_ppl:.4f}")
        print(f"Degradation:      {degradation:+.2f}%")
        print("="*80)

    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model perplexity")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to quantized model')
    parser.add_argument('--baseline', type=str, default=None,
                        help='Path to baseline FP16 model (optional)')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'ptb'],
                        help='Dataset for evaluation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for computation')

    args = parser.parse_args()

    compare_models(
        quantized_path=args.model,
        baseline_path=args.baseline,
        dataset=args.dataset,
        output_file=args.output,
        device=args.device
    )


if __name__ == '__main__':
    main()
