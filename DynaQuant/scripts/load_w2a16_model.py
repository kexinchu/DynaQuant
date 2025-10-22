#!/usr/bin/env python3
"""
Load and Test W2A16 Quantized Model
====================================
Load a W2A16 AWQ quantized model and test inference.

Usage:
    python scripts/load_w2a16_model.py \
        --model-path ./output/Qwen3-30B-A3B-W2A16 \
        --test-prompt "Hello, how are you?"
"""

import os
import sys
from pathlib import Path
import json
import argparse

import torch
from transformers import AutoTokenizer, AutoConfig

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from quant.awq_w2 import W2AWQLinear


def load_quantized_model(model_path: str, device: str = 'cuda'):
    """
    Load a W2A16 quantized model.
    
    Args:
        model_path: Path to the quantized model directory
        device: Device to load the model on
    
    Returns:
        model: Loaded quantized model
        tokenizer: Tokenizer
        config: Model config
    """
    print(f"Loading quantized model from: {model_path}")
    
    # Load config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"Model type: {config.model_type}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load quantization metadata
    metadata_path = os.path.join(model_path, "quantization_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Quantization info:")
        print(f"  Layers quantized: {metadata.get('num_layers_quantized', 'N/A')}")
        print(f"  Average error: {metadata.get('avg_error', 'N/A'):.6f}")
        print(f"  Group size: {metadata.get('group_size', 'N/A')}")
    
    # Load the model
    # Note: The model should have been saved with W2AWQLinear layers already
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
    )
    
    # Count W2AWQLinear layers
    w2_layers = sum(1 for m in model.modules() if isinstance(m, W2AWQLinear))
    print(f"Model loaded with {w2_layers} W2AWQLinear layers")
    
    model.eval()
    
    return model, tokenizer, config


def test_inference(model, tokenizer, prompt: str, max_new_tokens: int = 50):
    """Test model inference."""
    print(f"\nTesting inference with prompt: {prompt}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\nGenerated text:")
    print("-" * 80)
    print(generated_text)
    print("-" * 80)
    
    return generated_text


def compare_with_original(model_path: str, original_model_path: str, prompt: str):
    """Compare quantized model with original model."""
    print("\n" + "="*80)
    print("Comparing Quantized vs Original Model")
    print("="*80)
    
    # Load quantized
    print("\n[1/2] Loading quantized model...")
    quant_model, tokenizer, _ = load_quantized_model(model_path)
    
    # Test quantized
    print("\n[Quantized Model Output]")
    quant_output = test_inference(quant_model, tokenizer, prompt)
    
    # Load original
    print("\n[2/2] Loading original model...")
    from transformers import AutoModelForCausalLM
    orig_model = AutoModelForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    orig_model.eval()
    
    # Test original
    print("\n[Original Model Output]")
    orig_output = test_inference(orig_model, tokenizer, prompt)
    
    print("\n" + "="*80)
    print("Comparison Complete")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Load and test W2A16 quantized model")
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to quantized model directory')
    parser.add_argument('--test-prompt', type=str,
                       default="Once upon a time, there was a",
                       help='Test prompt for inference')
    parser.add_argument('--max-new-tokens', type=int, default=50,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--compare-with-original', type=str, default=None,
                       help='Path to original model for comparison')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    if args.compare_with_original:
        # Compare mode
        compare_with_original(
            args.model_path,
            args.compare_with_original,
            args.test_prompt
        )
    else:
        # Load and test mode
        model, tokenizer, config = load_quantized_model(args.model_path, args.device)
        test_inference(model, tokenizer, args.test_prompt, args.max_new_tokens)
    
    print("\n✅ Test completed successfully!")


if __name__ == '__main__':
    main()

