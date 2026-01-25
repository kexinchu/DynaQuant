#!/usr/bin/env python3
"""
Test expert activation patterns across different batch sizes for MoE models.

This script tests expert activations for:
- Different batch sizes: 1, 2, 4, 8, 16, 32, 64
- Separate statistics for prefill and decode phases
- Multiple models: Qwen3-30B-A3B-Instruct-2507 and Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound
- Dataset: ShareGPT_V3 (128 queries)
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.activations as _hf_activations


# ---------------------------------------------------------------------------
# Compatibility patch for deprecated activations expected by AutoAWQ.
# ---------------------------------------------------------------------------
if not hasattr(_hf_activations, "PytorchGELUTanh"):
    class PytorchGELUTanh(nn.Module):
        """Drop-in replacement for legacy Transformers activation."""

        # type: ignore[override]
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.gelu(input, approximate="tanh")

    _hf_activations.PytorchGELUTanh = PytorchGELUTanh


LOGGER = logging.getLogger("expert_activation_batch")


def load_sharegpt_queries(
    dataset_path: Path,
    num_queries: int = 128,
) -> List[str]:
    """Load queries from ShareGPT_V3 dataset."""
    queries: List[str] = []
    
    with dataset_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)}")
    
    for item in data:
        if len(queries) >= num_queries:
            break
        
        if "conversations" not in item:
            continue
        
        # Extract the first human message as the query
        for conv in item["conversations"]:
            if conv.get("from") == "human" and "value" in conv:
                query = str(conv["value"]).strip()
                if query:
                    queries.append(query)
                    break
    
    if len(queries) < num_queries:
        LOGGER.warning(
            f"Only found {len(queries)} queries, requested {num_queries}"
        )
    
    LOGGER.info(f"Loaded {len(queries)} queries from {dataset_path}")
    return queries[:num_queries]


def collect_expert_activations_from_router_logits(
    router_logits: List[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    top_k: int,
    device: torch.device,
) -> List[Counter]:
    """Extract expert activations from router logits."""
    num_layers = len(router_logits)
    counters: List[Counter] = [Counter() for _ in range(num_layers)]
    
    for layer_id, layer_logits in enumerate(router_logits):
        flat_logits = layer_logits.real if torch.is_complex(layer_logits) else layer_logits
        
        if flat_logits.dim() == 3:
            # Shape: (batch, seq_len, num_experts)
            top_indices = torch.topk(flat_logits, k=top_k, dim=-1).indices
        elif flat_logits.dim() == 2:
            # Shape: (batch, num_experts) - need to expand to (batch, seq_len, num_experts)
            seq_len = input_ids.shape[1]
            tile = flat_logits.unsqueeze(1).repeat(1, seq_len, 1)
            top_indices = torch.topk(tile, k=top_k, dim=-1).indices
        else:
            raise ValueError(
                f"Unexpected router_logits dimensions: {flat_logits.shape}"
            )
        
        mask_cpu = (
            attention_mask.bool().cpu()
            if attention_mask is not None
            else torch.ones(
                input_ids.shape[:2], dtype=torch.bool
            )
        )
        if mask_cpu.dim() == 1:
            mask_cpu = mask_cpu.unsqueeze(1)
        
        batch_size, seq_len, _ = top_indices.shape
        if mask_cpu.size(1) != seq_len:
            mask_cpu = mask_cpu.expand(-1, seq_len)
        
        top_indices = top_indices.cpu()
        
        for b in range(batch_size):
            row = mask_cpu[min(b, mask_cpu.size(0) - 1)]
            if row.dim() == 0:
                row = row.unsqueeze(0)
            for t in range(seq_len):
                if t >= row.size(0):
                    continue
                if not row[t]:
                    continue
                for expert_id in top_indices[b, t]:
                    counters[layer_id][int(expert_id)] += 1
    
    return counters


def get_top_k_from_config(model) -> int:
    """Get top-k value from model config."""
    # Try different possible config attribute names
    top_k = getattr(model.config, 'num_experts_per_tok', None)
    if top_k is None:
        top_k = getattr(model.config, 'num_experts_per_token', None)
    if top_k is None:
        top_k = getattr(model.config, 'top_k', None)
    if top_k is None:
        top_k = getattr(model.config, 'moe_top_k', None)
    
    if top_k is None or not isinstance(top_k, int) or top_k <= 0:
        LOGGER.warning(
            f"Could not find top_k in config, using default 8. "
            f"Config attributes: {[attr for attr in dir(model.config) if 'expert' in attr.lower() or 'top' in attr.lower()]}"
        )
        top_k = 8
    
    LOGGER.info(f"Using top_k={top_k} from model config")
    return top_k


def get_num_experts_from_config(model) -> int:
    """Get total number of experts per layer from model config."""
    # Try different possible config attribute names
    num_experts = getattr(model.config, 'num_experts', None)
    if num_experts is None:
        num_experts = getattr(model.config, 'num_local_experts', None)
    if num_experts is None:
        num_experts = getattr(model.config, 'num_experts_per_layer', None)
    if num_experts is None:
        num_experts = getattr(model.config, 'moe_num_experts', None)
    
    if num_experts is None or not isinstance(num_experts, int) or num_experts <= 0:
        # Try to infer from router logits shape if available
        # This is a fallback, will be called after model is loaded
        LOGGER.warning(
            f"Could not find num_experts in config. "
            f"Config attributes: {[attr for attr in dir(model.config) if 'expert' in attr.lower()]}"
        )
        return None
    
    LOGGER.info(f"Using num_experts={num_experts} from model config")
    return num_experts


def collect_expert_statistics_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Collect expert activations for prefill and decode phases separately.
    
    For each batch, track which experts are activated (as a set).
    This allows us to compute the average number of unique experts per batch.
    
    Returns:
        (prefill_batch_experts, decode_batch_experts)
        - Each is List[List[Set[int]]]: [layer][batch_idx] -> set of expert IDs
    """
    # Get top_k from model config
    top_k = get_top_k_from_config(model)
    
    num_layers = model.config.num_hidden_layers
    # For each layer, store average experts per step for each batch
    # prefill: average experts per token position (prefill step) per batch
    # decode: average experts per decode step per batch
    prefill_batch_experts: List[List[float]] = [[] for _ in range(num_layers)]
    decode_batch_experts: List[List[float]] = [[] for _ in range(num_layers)]
    
    model.eval()
    model.config.output_router_logits = True
    
    # Process prompts in batches
    batch_idx = 0
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        LOGGER.info(f"Processing batch {batch_idx + 1}/{(len(prompts)) // batch_size}")
        
        # Tokenize batch
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        # Move inputs to the same device as the model
        # For device_map="auto", we need to get the device from the model
        model_device = next(model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        attention_mask = encoded.get("attention_mask")
        input_ids = encoded["input_ids"]
        
        # Prefill phase: process all input tokens
        LOGGER.debug(f"Prefill: input_ids shape={input_ids.shape}, device={input_ids.device}")
        LOGGER.debug(f"Prefill: attention_mask shape={attention_mask.shape if attention_mask is not None else None}")
        
        # Clear cache before prefill to free up memory
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            LOGGER.debug("Calling model forward for prefill...")
            outputs = model(
                **encoded,
                use_cache=True,
                output_router_logits=True,
                return_dict=True,
            )
            LOGGER.debug("Model forward completed")
        
        # Clear cache after prefill to free up intermediate activations
        torch.cuda.empty_cache()
        
        router_logits = outputs.router_logits
        if router_logits is None:
            LOGGER.error("Model did not return router_logits!")
            LOGGER.error(f"Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'N/A'}")
            LOGGER.error(f"Model config output_router_logits: {getattr(model.config, 'output_router_logits', 'not set')}")
            raise RuntimeError(
                "Model did not return `router_logits`. Ensure the checkpoint supports MoE router outputs."
            )
        
        LOGGER.debug(f"Prefill: Got router_logits, num_layers={len(router_logits)}")
        
        # Collect prefill activations for this batch
        # We need to count experts per token position (prefill step)
        prefill_experts_per_token: List[List[int]] = [[] for _ in range(num_layers)]  # [layer][token_pos] -> num experts
        
        # Process router logits to get experts per token position
        for layer_id, layer_logits in enumerate(router_logits):
            flat_logits = layer_logits.real if torch.is_complex(layer_logits) else layer_logits
            
            if flat_logits.dim() == 3:
                # Shape: (batch, seq_len, num_experts)
                top_indices = torch.topk(flat_logits, k=top_k, dim=-1).indices
            elif flat_logits.dim() == 2:
                # Shape: (batch, num_experts) - need to expand
                seq_len = input_ids.shape[1]
                tile = flat_logits.unsqueeze(1).repeat(1, seq_len, 1)
                top_indices = torch.topk(tile, k=top_k, dim=-1).indices
            else:
                continue
            
            mask_cpu = (
                attention_mask.bool().cpu()
                if attention_mask is not None
                else torch.ones(input_ids.shape[:2], dtype=torch.bool)
            )
            if mask_cpu.dim() == 1:
                mask_cpu = mask_cpu.unsqueeze(1)
            
            batch_size, seq_len, _ = top_indices.shape
            if mask_cpu.size(1) != seq_len:
                mask_cpu = mask_cpu.expand(-1, seq_len)
            
            top_indices = top_indices.cpu()
            
            # For each token position, count unique experts activated
            for b in range(batch_size):
                row = mask_cpu[min(b, mask_cpu.size(0) - 1)]
                if row.dim() == 0:
                    row = row.unsqueeze(0)
                for t in range(seq_len):
                    if t >= row.size(0):
                        continue
                    if not row[t]:
                        continue
                    # Count unique experts for this token position
                    token_experts = set(int(expert_id) for expert_id in top_indices[b, t])
                    prefill_experts_per_token[layer_id].append(len(token_experts))
        
        # Store average experts per token position (prefill step) for this batch
        for layer_id in range(num_layers):
            if prefill_experts_per_token[layer_id]:
                avg_experts_per_token = sum(prefill_experts_per_token[layer_id]) / len(prefill_experts_per_token[layer_id])
                prefill_batch_experts[layer_id].append(avg_experts_per_token)
            else:
                prefill_batch_experts[layer_id].append(0.0)
        
        # Clear cache after prefill to free memory
        torch.cuda.empty_cache()
        
        # Decode phase: generate tokens one by one
        past_key_values = outputs.past_key_values
        # Get the last token logits to start decode
        next_input_ids = input_ids[:, -1:]  # Shape: (batch_size, 1)
        
        # Track experts activated per decode step for this batch
        decode_experts_per_step: List[List[int]] = [[] for _ in range(num_layers)]  # [layer][step] -> num experts
        
        for decode_step in range(max_new_tokens):
            # Clear cache before each decode step to free up memory
            if decode_step > 0:  # Don't clear on first step (right after prefill)
                torch.cuda.empty_cache()
            
            # Prepare inputs for decode step
            decode_inputs = {
                "input_ids": next_input_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
                "output_router_logits": True,
                "return_dict": True,
            }
            
            LOGGER.debug(f"Decode step {decode_step}: input_ids shape={next_input_ids.shape}, device={next_input_ids.device}")
            
            with torch.no_grad():
                decode_outputs = model(**decode_inputs)
            
            decode_router_logits = decode_outputs.router_logits
            past_key_values = decode_outputs.past_key_values
            
            # Clear cache after each decode step
            torch.cuda.empty_cache()
            
            if decode_router_logits is None:
                LOGGER.warning(f"Decode step {decode_step}: No router_logits returned!")
            
            if decode_router_logits is not None:
                # Create attention mask for single token
                # Get device from model
                model_device = next(model.parameters()).device
                decode_attention_mask = torch.ones(
                    next_input_ids.shape[:2],
                    dtype=torch.bool,
                    device=model_device
                )
                
                # Collect decode activations for this step
                decode_batch_counters = collect_expert_activations_from_router_logits(
                    router_logits=decode_router_logits,
                    attention_mask=decode_attention_mask,
                    input_ids=next_input_ids,
                    top_k=top_k,
                    device=device,
                )
                
                # Count unique experts activated in this decode step
                for layer_id, counter in enumerate(decode_batch_counters):
                    num_active_experts = len(counter)  # Number of unique experts in this step
                    decode_experts_per_step[layer_id].append(num_active_experts)
            
            # Get next token (greedy decoding)
            next_token_logits = decode_outputs.logits[:, -1, :]  # Shape: (batch_size, vocab_size)
            next_token_ids = next_token_logits.argmax(dim=-1, keepdim=True)  # Shape: (batch_size, 1)
            
            # Check if all sequences have reached EOS
            if tokenizer.eos_token_id is not None:
                eos_mask = (next_token_ids == tokenizer.eos_token_id).squeeze(1)
                if eos_mask.all():
                    break
            
            # Update input for next iteration
            next_input_ids = next_token_ids
        
        # Store average experts per decode step for this batch
        for layer_id in range(num_layers):
            if decode_experts_per_step[layer_id]:
                avg_experts_per_step = sum(decode_experts_per_step[layer_id]) / len(decode_experts_per_step[layer_id])
                decode_batch_experts[layer_id].append(avg_experts_per_step)
            else:
                decode_batch_experts[layer_id].append(0.0)
        
        # Clear cache after each batch to prevent OOM
        torch.cuda.empty_cache()
        
        batch_idx += 1
    
    return prefill_batch_experts, decode_batch_experts


def format_summary(
    prefill_batch_experts: List[List[float]],  # avg experts per token position (prefill step) per batch
    decode_batch_experts: List[List[float]],  # avg experts per decode step per batch
    num_experts_per_layer: int,
) -> Dict[str, any]:
    """
    Format summary with expert activation ratio.
    
    Computes the average number of experts activated per step (prefill step or decode step).
    
    Args:
        prefill_batch_experts: List[List[float]] - [layer][batch_idx] -> avg experts per token position
        decode_batch_experts: List[List[float]] - [layer][batch_idx] -> avg experts per decode step
        num_experts_per_layer: Total number of experts per layer
    
    Returns:
        Dictionary with activation ratios per layer and overall statistics.
    """
    summary = {
        "prefill": {
            "per_layer": {},
            "overall": {},
        },
        "decode": {
            "per_layer": {},
            "overall": {},
        },
    }
    
    num_layers = len(prefill_batch_experts)
    num_batches = len(prefill_batch_experts[0]) if num_layers > 0 and len(prefill_batch_experts[0]) > 0 else 0
    
    # Process prefill: compute average experts per token position (prefill step) for each layer
    prefill_avg_active_experts_per_layer = []
    
    for layer_id in range(num_layers):
        batch_experts = prefill_batch_experts[layer_id]
        if not batch_experts:
            avg_active_experts = 0
        else:
            # For prefill, batch_experts contains average experts per token position for each batch
            # We need to average across all batches
            avg_active_experts = sum(batch_experts) / len(batch_experts)
        
        activation_ratio = avg_active_experts / num_experts_per_layer if num_experts_per_layer > 0 else 0.0
        
        summary["prefill"]["per_layer"][f"layer_{layer_id}"] = {
            "avg_active_experts_per_prefill_step": avg_active_experts,
            "total_experts": num_experts_per_layer,
            "activation_ratio": activation_ratio,
            "num_batches": len(batch_experts),
        }
        
        prefill_avg_active_experts_per_layer.append(avg_active_experts)
    
    # Overall prefill statistics: average across all layers
    overall_avg_active_experts = sum(prefill_avg_active_experts_per_layer) / num_layers if num_layers > 0 else 0
    overall_activation_ratio = overall_avg_active_experts / num_experts_per_layer if num_experts_per_layer > 0 else 0.0
    
    summary["prefill"]["overall"] = {
        "num_layers": num_layers,
        "num_batches": num_batches,
        "num_experts_per_layer": num_experts_per_layer,
        "avg_active_experts_per_prefill_step": overall_avg_active_experts,
        "avg_activation_ratio": overall_activation_ratio,
        "min_avg_active_experts": min(prefill_avg_active_experts_per_layer) if prefill_avg_active_experts_per_layer else 0,
        "max_avg_active_experts": max(prefill_avg_active_experts_per_layer) if prefill_avg_active_experts_per_layer else 0,
    }
    
    # Process decode: compute average experts per decode step for each layer
    decode_avg_active_experts_per_layer = []
    
    for layer_id in range(num_layers):
        batch_experts = decode_batch_experts[layer_id]
        if not batch_experts:
            avg_active_experts = 0
        else:
            # For decode, batch_experts contains average experts per decode step for each batch
            # We need to average across all batches
            avg_active_experts = sum(batch_experts) / len(batch_experts)
        
        activation_ratio = avg_active_experts / num_experts_per_layer if num_experts_per_layer > 0 else 0.0
        
        summary["decode"]["per_layer"][f"layer_{layer_id}"] = {
            "avg_active_experts_per_decode_step": avg_active_experts,
            "total_experts": num_experts_per_layer,
            "activation_ratio": activation_ratio,
            "num_batches": len(batch_experts),
        }
        
        decode_avg_active_experts_per_layer.append(avg_active_experts)
    
    # Overall decode statistics: average across all layers
    overall_avg_active_experts = sum(decode_avg_active_experts_per_layer) / num_layers if num_layers > 0 else 0
    overall_activation_ratio = overall_avg_active_experts / num_experts_per_layer if num_experts_per_layer > 0 else 0.0
    
    summary["decode"]["overall"] = {
        "num_layers": num_layers,
        "num_batches": num_batches,
        "num_experts_per_layer": num_experts_per_layer,
        "avg_active_experts_per_decode_step": overall_avg_active_experts,
        "avg_activation_ratio": overall_activation_ratio,
        "min_avg_active_experts": min(decode_avg_active_experts_per_layer) if decode_avg_active_experts_per_layer else 0,
        "max_avg_active_experts": max(decode_avg_active_experts_per_layer) if decode_avg_active_experts_per_layer else 0,
    }
    
    return summary


def load_model(
    model_id: str,
    quantization: str,
    device: torch.device,
):
    """Load model with specified quantization."""
    LOGGER.info(f"Loading model from: {model_id}")
    LOGGER.info(f"Quantization: {quantization}, Target device: {device}")
    
    # Use device_map="auto" to automatically distribute model across available GPUs
    # For quantized models, we can be more aggressive with memory allocation
    num_gpus = torch.cuda.device_count()
    
    # Set max_memory for each GPU to ensure balanced distribution
    # Reserve significant memory for activations and KV cache during forward pass
    max_memory = {}
    if num_gpus >= 2:
        if quantization in {"autoround-int4", "autoround-int2", "awq-int4"}:
            # For quantized models, they use less memory, so we can allocate more
            # Allocate ~42GB each (leaving ~8GB for activations/KV cache per GPU)
            max_memory[0] = "42GiB"
            max_memory[1] = "42GiB"
            LOGGER.info(f"Using max_memory: {max_memory} for {num_gpus} GPUs (quantized model, reserving ~8GB per GPU for activations/KV cache)")
        else:
            # For FP16 models, be more conservative
            # Allocate ~35GB each (leaving ~14GB for activations/KV cache per GPU)
            max_memory[0] = "35GiB"
            max_memory[1] = "35GiB"
            LOGGER.info(f"Using max_memory: {max_memory} for {num_gpus} GPUs (FP16 model, reserving ~14GB per GPU for activations/KV cache)")
    elif num_gpus == 1:
        # Single GPU: use less to leave more headroom for activations
        max_memory[0] = "40GiB"
        LOGGER.info(f"Using max_memory: {max_memory} for single GPU")
    
    # Use device_map="auto" with max_memory to force distribution
    device_map = "auto"
    LOGGER.info(f"Using device_map: {device_map} (auto-distribute across {num_gpus} GPUs)")
    
    if quantization == "awq-int4":
        from autoawq.modeling import AutoAWQForCausalLM
        
        LOGGER.info("Loading AWQ quantized model...")
        model = AutoAWQForCausalLM.from_quantized(
            model_id,
            device_map=device_map,
            trust_remote_code=True,
        )
        LOGGER.info("AWQ model loaded, loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
    elif quantization in {"autoround-int4", "autoround-int2"}:
        LOGGER.info(f"Loading AutoRound quantized model ({quantization})...")
        LOGGER.info("Note: AutoRound models are already quantized, loading directly with device_map='auto'...")
        
        # AutoRound quantized models can be loaded directly with device_map="auto"
        # The quantization is already applied in the saved model files
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            max_memory=max_memory if max_memory else None,
        )
        LOGGER.info("AutoRound model loaded, loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
    else:
        LOGGER.info("Loading FP16 model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            max_memory=max_memory if max_memory else None,
        )
        LOGGER.info("FP16 model loaded, loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
    
    # Verify model is on the correct device(s)
    LOGGER.info("Verifying model device placement...")
    param_devices = set()
    for name, param in list(model.named_parameters())[:20]:  # Check first 20 params to see distribution
        param_devices.add(str(param.device))
        if len(param_devices) > 5:  # Check enough to see multi-GPU distribution
            break
    
    LOGGER.info(f"Model parameters found on devices: {sorted(param_devices)}")
    
    # Clear cache after loading
    torch.cuda.empty_cache()
    
    # Check GPU memory usage for all available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        LOGGER.info(f"Available GPUs: {num_gpus}")
        for gpu_id in range(num_gpus):
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
            gpu_allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
            gpu_reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
            LOGGER.info(f"GPU {gpu_id} memory - Total: {gpu_memory:.2f}GB, Allocated: {gpu_allocated:.2f}GB, Reserved: {gpu_reserved:.2f}GB")
    
    # Verify model can do a forward pass
    LOGGER.info("Testing model forward pass...")
    test_input = tokenizer("test", return_tensors="pt")
    # For device_map="auto", inputs should go to the first GPU or model's device
    model_device = next(model.parameters()).device
    test_input = {k: v.to(model_device) for k, v in test_input.items()}
    
    model.eval()
    with torch.no_grad():
        test_output = model(**test_input, output_router_logits=True)
    
    LOGGER.info(f"Forward pass successful. Output keys: {test_output.keys()}")
    if hasattr(test_output, 'router_logits') and test_output.router_logits is not None:
        LOGGER.info(f"Router logits shape: {[rl.shape if hasattr(rl, 'shape') else type(rl) for rl in test_output.router_logits[:3]]}")
    else:
        LOGGER.warning("No router_logits in output!")
    
    LOGGER.info("Model loading and verification complete")
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test expert activations across different batch sizes."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("ShareGPT_V3_unfiltered_cleaned_split.json"),
        help="Path to ShareGPT_V3 dataset JSON file.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID or path (e.g., Qwen3-30B-A3B-Instruct-2507 or Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound).",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=128,
        help="Number of queries to test (default: 128).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Batch sizes to test (default: 1 2 4 8 16 32 64).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10,
        help="Maximum number of tokens to generate in decode phase (default: 10).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda:0 if available else cpu).",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "awq-int4", "autoround-int4", "autoround-int2"],
        default="none",
        help="Quantization strategy for loading the model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/expert_activation_batch"),
        help="Output directory for results.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ...).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Load queries
    queries = load_sharegpt_queries(args.dataset, args.num_queries)
    
    # Setup device
    device = torch.device(args.device or "cuda:0")
    if device.type != "cuda":
        raise ValueError(
            "MoE models require a CUDA device. Please specify --device cuda:<id>."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine.")
    
    LOGGER.info(f"Using device {device}")
    
    # Load model
    LOGGER.info(f"Loading model: {args.model_id}")
    LOGGER.info(f"Target device: {device}")
    
    model, tokenizer = load_model(args.model_id, args.quantization, device)
    
    # Verify model is ready
    model_device = next(model.parameters()).device
    LOGGER.info(f"Model loaded. First parameter device: {model_device}")
    LOGGER.info(f"Model has {model.config.num_hidden_layers} layers")
    LOGGER.info(f"Model config output_router_logits: {getattr(model.config, 'output_router_logits', 'not set')}")
    
    # Set output_router_logits if not already set
    if not hasattr(model.config, 'output_router_logits') or not model.config.output_router_logits:
        model.config.output_router_logits = True
        LOGGER.info("Set model.config.output_router_logits = True")
    
    LOGGER.info(f"Loaded {args.model_id} with quantization mode {args.quantization}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract model name for output file
    model_name = args.model_id.split("/")[-1].replace("-", "_")
    
    # Test each batch size
    all_results = {}
    
    for batch_size in args.batch_sizes:
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Testing batch_size={batch_size}")
        LOGGER.info(f"{'='*60}")
        
        prefill_batch_experts, decode_batch_experts = collect_expert_statistics_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=queries,
            batch_size=batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        
        # Get num_experts from config or infer from router logits
        num_experts = get_num_experts_from_config(model)
        if num_experts is None:
            # Try to infer from router logits shape
            # Run a dummy forward pass to get router logits shape
            dummy_input = tokenizer("test", return_tensors="pt")
            model_device = next(model.parameters()).device
            dummy_input = {k: v.to(model_device) for k, v in dummy_input.items()}
            with torch.no_grad():
                dummy_output = model(**dummy_input, output_router_logits=True)
            if dummy_output.router_logits is not None and len(dummy_output.router_logits) > 0:
                # Get shape from first layer
                first_layer_logits = dummy_output.router_logits[0]
                if isinstance(first_layer_logits, torch.Tensor):
                    if first_layer_logits.dim() == 3:
                        num_experts = first_layer_logits.shape[-1]
                    elif first_layer_logits.dim() == 2:
                        num_experts = first_layer_logits.shape[-1]
                    else:
                        num_experts = 128  # Default fallback
                else:
                    num_experts = 128
            else:
                num_experts = 128  # Default fallback
            LOGGER.info(f"Inferred num_experts={num_experts} from router logits shape")
        
        summary = format_summary(prefill_batch_experts, decode_batch_experts, num_experts)
        
        # Get top_k from config for metadata
        top_k = get_top_k_from_config(model)
        
        # Add metadata
        summary["metadata"] = {
            "model_id": args.model_id,
            "batch_size": batch_size,
            "num_queries": len(queries),
            "top_k": top_k,
            "num_experts_per_layer": num_experts,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
        }
        
        all_results[f"batch_size_{batch_size}"] = summary
        
        # Save individual batch size result
        output_file = args.output_dir / f"{model_name}_batch{batch_size}.json"
        output_file.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8"
        )
        LOGGER.info(f"Saved results to {output_file}")
    
    # Save combined results
    combined_output = args.output_dir / f"{model_name}_all_batch_sizes.json"
    combined_output.write_text(
        json.dumps(all_results, indent=2),
        encoding="utf-8"
    )
    LOGGER.info(f"Saved combined results to {combined_output}")


if __name__ == "__main__":
    main()

