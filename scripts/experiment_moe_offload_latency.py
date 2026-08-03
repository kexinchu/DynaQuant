#!/usr/bin/env python3
"""
Experiment to measure MoE offloading+prefetch latency under workload shift.

This script:
1. Measures latency vs workload and prompt length (prefill only)
2. Real CPU-to-GPU fetch + offload: experts start on CPU, fetch to GPU when needed.
   When GPU memory is insufficient, offload previous layer's experts (layer 0 -> layer N-1).
3. GPU memory limit: half experts per layer + all other weights + KV cache for 2048 tokens
4. Uses ShareGPT prompts; when prompt is too short, concatenates with subsequent prompts
   until length >= target, then truncates to exact length
5. Records comprehensive metrics per request and aggregates
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup logging
LOGGER = logging.getLogger("moe_offload_experiment")


@dataclass
class ExpertID:
    """Unique identifier for an expert."""
    layer: int
    idx: int

    def __hash__(self):
        return hash((self.layer, self.idx))

    def __eq__(self, other):
        return isinstance(other, ExpertID) and self.layer == other.layer and self.idx == other.idx


@dataclass
class LayerMetrics:
    """Metrics for a single layer during forward pass."""
    layer_id: int
    activated_experts: List[int]
    num_activated_experts: int
    waiting_time_ms: float
    offload_count: int
    load_count: int


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    prompt_length: int
    total_latency_ms: float
    waiting_time_ms_due_to_loading: float
    thres_expert: float
    layers: List[LayerMetrics]
    offload_count_to_cpu: int
    load_count_to_gpu: int


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    model_path: str
    dataset_path: str
    prompt_lengths: List[int]
    num_prompts_per_length: int = 16
    gpu_memory_gb: float = 24.0
    seed: int = 42
    device: str = "cuda:0"
    max_new_tokens: int = 0  # Only prefill, no generation
    temperature: float = 0.0
    top_p: float = 1.0
    accumulation_mode: str = "per_layer"  # "per_layer" or "cumulative"
    offload_policy: str = "random"  # "random" or other policies


def _get_layer_experts_info(model: nn.Module) -> Tuple[Dict[int, List[ExpertID]], Dict[ExpertID, nn.Module], int, int, int]:
    """
    Extract expert structure from model.
    Returns: (layer_experts dict, expert_id_to_module, expert_size_bytes, experts_per_layer, num_layers)
    """
    layer_experts: Dict[int, List[ExpertID]] = {}  # layer_idx -> [ExpertID]
    expert_id_to_module: Dict[ExpertID, nn.Module] = {}
    expert_size_bytes = 0
    experts_per_layer = 0
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        experts = None
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            experts = layer.mlp.experts
        elif hasattr(layer, 'experts'):
            experts = layer.experts
        elif hasattr(layer, 'mlp'):
            mlp = layer.mlp
            if hasattr(mlp, 'experts'):
                experts = mlp.experts
            elif isinstance(mlp, nn.ModuleList):
                experts = mlp

        if experts is not None:
            layer_experts[layer_idx] = []
            if isinstance(experts, nn.ModuleList):
                if experts_per_layer == 0:
                    experts_per_layer = len(experts)
                for expert_idx in range(len(experts)):
                    expert_id = ExpertID(layer=layer_idx, idx=expert_idx)
                    expert_module = experts[expert_idx]
                    layer_experts[layer_idx].append(expert_id)
                    expert_id_to_module[expert_id] = expert_module
            elif hasattr(experts, 'num_experts'):
                num_experts = experts.num_experts
                if experts_per_layer == 0:
                    experts_per_layer = num_experts
                for expert_idx in range(num_experts):
                    expert_id = ExpertID(layer=layer_idx, idx=expert_idx)
                    layer_experts[layer_idx].append(expert_id)
                    expert_id_to_module[expert_id] = experts
            elif hasattr(experts, '__getitem__'):
                try:
                    for expert_idx in range(len(experts)):
                        expert_id = ExpertID(layer=layer_idx, idx=expert_idx)
                        expert_module = experts[expert_idx]
                        layer_experts[layer_idx].append(expert_id)
                        expert_id_to_module[expert_id] = expert_module
                    if experts_per_layer == 0:
                        experts_per_layer = len(experts)
                except (TypeError, AttributeError):
                    pass

    # Compute expert size - count each unique module once (shared-param models)
    seen_modules: Set[int] = set()
    total_expert_bytes = 0
    for mod in expert_id_to_module.values():
        mod_id = id(mod)
        if mod_id in seen_modules:
            continue
        seen_modules.add(mod_id)
        for param in mod.parameters():
            total_expert_bytes += param.numel() * param.element_size()
    num_unique_modules = len(seen_modules)
    # Normal MoE: per-expert = total/(layers*experts). Shared-param: loading one expert loads whole module
    if num_unique_modules and experts_per_layer:
        if num_unique_modules < len(layer_experts) * experts_per_layer:
            # Shared-param: one module per layer
            expert_size_bytes = total_expert_bytes // num_unique_modules
        else:
            expert_size_bytes = total_expert_bytes // (len(layer_experts) * experts_per_layer)
        if expert_size_bytes <= 0:
            expert_size_bytes = total_expert_bytes // max(1, len(expert_id_to_module))
    else:
        expert_size_bytes = 0

    return layer_experts, expert_id_to_module, expert_size_bytes, experts_per_layer, num_layers


def compute_gpu_memory_limit_bytes(model: nn.Module) -> Tuple[int, Dict]:
    """
    GPU memory limit = half experts per layer + all other weights + KV cache for 2048 tokens.

    Returns: (limit_bytes, metadata)
    """
    layer_experts, _, expert_size_bytes, experts_per_layer, num_layers = _get_layer_experts_info(model)

    # Non-expert weights
    total_model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_expert_bytes = len(layer_experts) * experts_per_layer * expert_size_bytes if experts_per_layer else 0
    non_expert_bytes = total_model_bytes - total_expert_bytes

    # Half experts per layer
    num_layers_with_experts = len(layer_experts)
    expert_budget_bytes = num_layers_with_experts * (experts_per_layer // 2) * expert_size_bytes

    # KV cache for 2048 tokens: num_layers * 2048 * num_kv_heads * head_dim * 2 * element_size
    config = getattr(model, 'config', None)
    kv_cache_bytes = 0
    if config is not None:
        num_hidden_layers = getattr(config, 'num_hidden_layers', num_layers)
        num_kv_heads = getattr(config, 'num_key_value_heads', getattr(config, 'num_attention_heads', 32))
        head_dim = getattr(config, 'head_dim', None) or (
            getattr(config, 'hidden_size', 2048) // getattr(config, 'num_attention_heads', 32)
        )
        # Use fp16/bf16 for KV cache (2 bytes)
        kv_cache_bytes = num_hidden_layers * 2048 * num_kv_heads * head_dim * 2 * 2

    gpu_limit_bytes = non_expert_bytes + expert_budget_bytes + kv_cache_bytes
    metadata = {
        "non_expert_bytes": non_expert_bytes,
        "expert_budget_bytes": expert_budget_bytes,
        "kv_cache_bytes": kv_cache_bytes,
        "gpu_limit_bytes": gpu_limit_bytes,
        "expert_size_bytes": expert_size_bytes,
        "experts_per_layer": experts_per_layer,
    }
    LOGGER.info(f"GPU memory limit: {gpu_limit_bytes / 1e9:.2f} GB "
                f"(non-expert: {non_expert_bytes/1e9:.2f}, expert_budget: {expert_budget_bytes/1e9:.2f}, "
                f"kv_cache: {kv_cache_bytes/1e9:.2f})")
    # Return expert_budget_bytes as the limit for ExpertMemoryManager (experts only)
    return expert_budget_bytes, metadata


class ExpertMemoryManager:
    """
    Manages expert residency with real CPU-to-GPU fetch and offload.
    Experts start on CPU. When needed, fetch to GPU. If GPU memory full, offload previous layer.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        gpu_memory_limit_bytes: int,
        offload_policy: str = "random",
    ):
        self.model = model
        self.device = device
        self.gpu_memory_limit_bytes = gpu_memory_limit_bytes
        self.offload_policy = offload_policy

        layer_experts, expert_id_to_module, _, self.experts_per_layer, self.num_layers = _get_layer_experts_info(model)
        self.layer_experts = layer_experts
        self.expert_id_to_module = expert_id_to_module

        # Track expert residency: ExpertID -> "gpu" or "cpu"
        self.expert_residency: Dict[ExpertID, str] = {}
        self._gpu_expert_bytes = 0  # Current expert bytes on GPU

        # Statistics
        self.total_offload_count = 0
        self.total_load_count = 0
        self.total_waiting_time_ms = 0.0

        # Precompute expert sizes
        self._expert_sizes: Dict[ExpertID, int] = {}
        for expert_id, mod in expert_id_to_module.items():
            sz = sum(p.numel() * p.element_size() for p in mod.parameters())
            sz += sum(b.numel() * b.element_size() for b in mod.buffers())
            self._expert_sizes[expert_id] = sz

        # Initialize all experts on CPU (model loading will have put them somewhere; we'll move to CPU)
        self._initialize_expert_residency()

        LOGGER.info(f"ExpertMemoryManager: {len(expert_id_to_module)} experts, GPU limit {gpu_memory_limit_bytes/1e9:.2f} GB")

    def _initialize_expert_residency(self):
        """Move all experts to CPU and track residency."""
        LOGGER.info("Moving experts to CPU, initializing residency...")
        for expert_id in self.expert_id_to_module:
            mod = self.expert_id_to_module[expert_id]
            # Move to CPU if on GPU
            for param in mod.parameters():
                if param.device.type == "cuda":
                    param.data = param.data.cpu()
            for buffer in mod.buffers():
                if buffer.device.type == "cuda":
                    buffer.data = buffer.data.cpu()
            self.expert_residency[expert_id] = "cpu"
        self._gpu_expert_bytes = 0
        LOGGER.info(f"Initialized {len(self.expert_residency)} experts on CPU")

    def get_expert_size_bytes(self, expert_id: ExpertID) -> int:
        """Calculate size of an expert in bytes."""
        return self._expert_sizes.get(expert_id, 0)

    def _get_layer_with_gpu_experts(self, exclude_layer: int) -> Optional[int]:
        """
        Get any layer that has experts currently on GPU (excluding exclude_layer).
        Start from layer before exclude_layer and wrap around.
        """
        num_layers = self.num_layers
        start = (exclude_layer - 1) if exclude_layer > 0 else (num_layers - 1)
        for offset in range(num_layers):
            layer_idx = (start - offset) % num_layers
            if layer_idx == exclude_layer:
                continue
            if layer_idx in self.layer_experts:
                # Check if any expert in this layer is on GPU
                for expert_id in self.layer_experts[layer_idx]:
                    if self.expert_residency.get(expert_id) == "gpu":
                        return layer_idx
        return None

    def _offload_layer_experts(self, layer_idx: int) -> float:
        """Offload all GPU experts of a layer to CPU. Returns total transfer time in ms."""
        if layer_idx not in self.layer_experts:
            return 0.0
        total_ms = 0.0
        for expert_id in self.layer_experts[layer_idx]:
            if self.expert_residency.get(expert_id) == "gpu":
                total_ms += self._offload_expert_to_cpu(expert_id)
        return total_ms

    def _offload_expert_to_cpu(self, expert_id: ExpertID) -> float:
        """Offload single expert from GPU to CPU, return transfer time in ms."""
        if expert_id not in self.expert_residency or self.expert_residency[expert_id] == "cpu":
            return 0.0

        expert_module = self.expert_id_to_module[expert_id]
        start_time = time.time()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        for param in expert_module.parameters():
            if param.device.type == "cuda":
                param.data = param.data.cpu()
        for buffer in expert_module.buffers():
            if buffer.device.type == "cuda":
                buffer.data = buffer.data.cpu()

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        transfer_ms = (time.time() - start_time) * 1000.0

        self.expert_residency[expert_id] = "cpu"
        self._gpu_expert_bytes -= self._expert_sizes.get(expert_id, 0)
        self.total_offload_count += 1
        return transfer_ms

    def _load_expert_to_gpu(self, expert_id: ExpertID) -> float:
        """Load expert from CPU to GPU, return transfer time in ms."""
        if expert_id not in self.expert_residency or self.expert_residency[expert_id] == "gpu":
            return 0.0

        expert_module = self.expert_id_to_module[expert_id]
        start_time = time.time()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        for param in expert_module.parameters():
            if param.device.type == "cpu":
                param.data = param.data.to(self.device)
        for buffer in expert_module.buffers():
            if buffer.device.type == "cpu":
                buffer.data = buffer.data.to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        transfer_ms = (time.time() - start_time) * 1000.0

        self.expert_residency[expert_id] = "gpu"
        self._gpu_expert_bytes += self._expert_sizes.get(expert_id, 0)
        self.total_load_count += 1
        return transfer_ms

    def ensure_experts_on_gpu(
        self,
        activated_experts: Set[ExpertID],
        current_layer: int,
    ) -> float:
        """
        Fetch activated experts to GPU. If GPU memory insufficient, offload other layers' experts.
        Returns total waiting time (fetch + offload) in ms.
        """
        total_ms = 0.0

        # For each activated expert on CPU, ensure we have space and fetch
        for expert_id in activated_experts:
            if expert_id not in self.expert_residency:
                continue
            if self.expert_residency[expert_id] == "gpu":
                continue

            sz = self._expert_sizes.get(expert_id, 0)
            # Ensure we have space: offload other layers' experts if needed
            attempts = 0
            max_attempts = self.num_layers + 1
            while self._gpu_expert_bytes + sz > self.gpu_memory_limit_bytes and attempts < max_attempts:
                offload_layer = self._get_layer_with_gpu_experts(current_layer)
                if offload_layer is None:
                    # No other layer has experts on GPU - we're at capacity
                    # Allow oversubscription for current layer (will load anyway)
                    break
                offload_ms = self._offload_layer_experts(offload_layer)
                total_ms += offload_ms
                attempts += 1

            total_ms += self._load_expert_to_gpu(expert_id)

        return total_ms


def load_sharegpt_dataset(dataset_path: Path) -> List[str]:
    """Load prompts from ShareGPT dataset."""
    LOGGER.info(f"Loading ShareGPT dataset from {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    for item in data:
        if "conversations" in item:
            # Extract user messages
            for conv in item["conversations"]:
                if conv.get("from") == "human" and "value" in conv:
                    prompts.append(conv["value"])
                    break  # Take first user message

    LOGGER.info(f"Loaded {len(prompts)} prompts from dataset")
    return prompts


def get_prompt_of_exact_length(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    prompt_start_idx: int,
    target_length: int,
) -> Tuple[List[int], int]:
    """
    Build a prompt with exactly target_length tokens.
    If single prompt is too short, concatenate with subsequent prompts until length >= target_length,
    then truncate to exact target_length.

    Returns:
        (token_ids, next_prompt_idx) - token_ids has exactly target_length tokens,
        next_prompt_idx is the index to use for next call
    """
    if target_length <= 0:
        return [], prompt_start_idx

    if prompt_start_idx >= len(prompts):
        raise ValueError(
            f"Not enough prompts: need more prompts after index {prompt_start_idx}, "
            f"only have {len(prompts)} total"
        )

    # Concatenate prompts until we have enough tokens
    combined_text = prompts[prompt_start_idx]
    combined_tokens = tokenizer.encode(combined_text, add_special_tokens=False)
    next_idx = prompt_start_idx + 1

    while len(combined_tokens) < target_length and next_idx < len(prompts):
        # Add separator (newline) and next prompt
        next_prompt = prompts[next_idx]
        combined_text = combined_text + "\n\n" + next_prompt
        combined_tokens = tokenizer.encode(combined_text, add_special_tokens=False)
        next_idx += 1

    if len(combined_tokens) < target_length:
        raise ValueError(
            f"Cannot reach target_length {target_length}: "
            f"concatenated {next_idx - prompt_start_idx} prompts "
            f"only gives {len(combined_tokens)} tokens. Need more/longer prompts in dataset."
        )

    # Truncate to exact target_length - return token IDs directly for guaranteed length
    truncated_tokens = combined_tokens[:target_length]
    return truncated_tokens, next_idx


# Global storage for captured router outputs and layer-wise metrics
_captured_router_outputs = []
_layer_metrics_data = []  # Store (layer_idx, activated_experts, waiting_time, offload_count, load_count)
_activated_experts_cumulative = None  # Global cumulative set for cumulative mode


def setup_router_hooks(
    model: nn.Module,
    mem_manager: ExpertMemoryManager,
    thres_expert: float,
    config: ExperimentConfig,
) -> Tuple[List, Dict]:
    """
    Setup hooks to capture router decisions and handle offloading during forward.

    Returns:
        (hooks_list, metrics_dict) where metrics_dict tracks per-layer data
    """
    global _captured_router_outputs, _layer_metrics_data, _activated_experts_cumulative

    # Reset global state
    _captured_router_outputs = []
    _layer_metrics_data = []
    _activated_experts_cumulative = set()  # Reset cumulative set

    # Track initial counts
    initial_offload = mem_manager.total_offload_count
    initial_load = mem_manager.total_load_count

    hooks = []

    def make_router_hook(layer_idx: int, num_experts: int = 0, top_k: int = 8, thres: float = 0.0):
        """Hook: capture activated experts, fetch to GPU, record only overflow waiting time."""
        def hook_fn(module, input, output):
            global _activated_experts_cumulative

            activated_indices = []
            if isinstance(output, tuple):
                if len(output) >= 1:
                    topk_idx = output[0]
                    if isinstance(topk_idx, torch.Tensor) and topk_idx.numel() > 0:
                        unique_experts = topk_idx.flatten().unique().cpu().tolist()
                        activated_indices = [int(x) for x in unique_experts]
            elif isinstance(output, torch.Tensor):
                out_num_experts = output.shape[-1] if output.dim() >= 2 else 0
                if output.dim() >= 2 and output.dtype in (torch.float16, torch.float32, torch.bfloat16) and out_num_experts > 1:
                    n_exp = num_experts if num_experts > 0 else out_num_experts
                    router_logits = output.float().view(-1, n_exp)
                    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
                    _, selected_experts = torch.topk(routing_weights, min(top_k, n_exp), dim=-1)
                    activated_indices = selected_experts.flatten().unique().cpu().tolist()
                    activated_indices = [int(x) for x in activated_indices]
                elif output.numel() > 0 and output.dtype in (torch.long, torch.int32, torch.int64):
                    activated_indices = [int(x) for x in output.flatten().unique().cpu().tolist()]

            num_activated = len(activated_indices)  # deduplicated count
            activated_experts = {ExpertID(layer=layer_idx, idx=idx) for idx in activated_indices}
            _captured_router_outputs.append((layer_idx, activated_indices))

            if config.accumulation_mode == "cumulative":
                _activated_experts_cumulative.update(activated_experts)
                active_set = _activated_experts_cumulative
            else:
                active_set = activated_experts

            actual_waiting_ms = mem_manager.ensure_experts_on_gpu(active_set, layer_idx)
            # Report waiting time only for "overflow" experts: when num_activated > thres_expert
            if num_activated <= thres:
                reported_waiting_ms = 0.0
            else:
                require_expert_num = num_activated - int(thres)
                reported_waiting_ms = actual_waiting_ms * (require_expert_num / num_activated)

            layer_offload_count = mem_manager.total_offload_count - initial_offload
            layer_load_count = mem_manager.total_load_count - initial_load
            _layer_metrics_data.append({
                "layer_idx": layer_idx,
                "num_activated_experts": num_activated,
                "waiting_time_ms": reported_waiting_ms,
                "offload_count": layer_offload_count,
                "load_count": layer_load_count,
            })
            return output
        return hook_fn

    import re
    registered_layers = set()
    config_obj = getattr(model, "config", None)
    num_experts_default = getattr(config_obj, "num_experts", 0) or getattr(config_obj, "num_local_experts", 0)
    top_k_default = getattr(config_obj, "num_experts_per_tok", None) or getattr(config_obj, "num_experts_per_token", None) or getattr(config_obj, "top_k", 8) or 8

    for name, module in model.named_modules():
        if hasattr(module, "gate") and hasattr(module, "experts") and "mlp" in name.lower():
            match = re.search(r"layers\.(\d+)", name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx not in registered_layers:
                    n_exp = getattr(module, "num_experts", num_experts_default)
                    tk = getattr(module, "top_k", top_k_default)
                    hook = module.gate.register_forward_hook(make_router_hook(layer_idx, n_exp, tk, thres_expert))
                    hooks.append(hook)
                    registered_layers.add(layer_idx)
                    LOGGER.debug(f"Registered hook on {name}.gate (layer {layer_idx})")

    # Also try standard structure (Qwen, GPT-OSS, etc.)
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx in registered_layers:
            continue

        if hasattr(layer, 'mlp'):
            mlp = layer.mlp

            # For GPT-OSS and similar models, hook the MLP forward to capture router output
            # GPT-OSS MLP returns (routed_out, router_scores) and router returns (router_scores, router_indices)
            if hasattr(mlp, "router"):
                def make_mlp_router_hook(layer_idx: int, thres: float):
                    def hook_fn(module, input, output):
                        global _activated_experts_cumulative
                        activated_indices = []
                        if isinstance(output, tuple):
                            if len(output) >= 2:
                                router_indices = output[1]
                                if isinstance(router_indices, torch.Tensor) and router_indices.numel() > 0:
                                    activated_indices = [int(x) for x in router_indices.flatten().unique().cpu().tolist()]
                            elif len(output) >= 1:
                                topk_idx = output[0]
                                if isinstance(topk_idx, torch.Tensor) and topk_idx.numel() > 0:
                                    activated_indices = [int(x) for x in topk_idx.flatten().unique().cpu().tolist()]
                        num_activated = len(activated_indices)
                        activated_experts = {ExpertID(layer=layer_idx, idx=idx) for idx in activated_indices}
                        _captured_router_outputs.append((layer_idx, activated_indices))
                        if config.accumulation_mode == "cumulative":
                            _activated_experts_cumulative.update(activated_experts)
                            active_set = _activated_experts_cumulative
                        else:
                            active_set = activated_experts
                        actual_waiting_ms = mem_manager.ensure_experts_on_gpu(active_set, layer_idx)
                        if num_activated <= thres:
                            reported_waiting_ms = 0.0
                        else:
                            require_expert_num = num_activated - int(thres)
                            reported_waiting_ms = actual_waiting_ms * (require_expert_num / num_activated)
                        layer_offload_count = mem_manager.total_offload_count - initial_offload
                        layer_load_count = mem_manager.total_load_count - initial_load
                        _layer_metrics_data.append({
                            "layer_idx": layer_idx,
                            "num_activated_experts": num_activated,
                            "waiting_time_ms": reported_waiting_ms,
                            "offload_count": layer_offload_count,
                            "load_count": layer_load_count,
                        })
                        return output
                    return hook_fn
                hook = mlp.router.register_forward_hook(make_mlp_router_hook(layer_idx, thres_expert))
                hooks.append(hook)
                registered_layers.add(layer_idx)
                LOGGER.debug(f"Registered hook on layer {layer_idx}.mlp.router")
            elif hasattr(mlp, "gate"):
                n_exp = getattr(mlp, "num_experts", num_experts_default)
                tk = getattr(mlp, "top_k", top_k_default)
                hook = mlp.gate.register_forward_hook(make_router_hook(layer_idx, n_exp, tk, thres_expert))
                hooks.append(hook)
                registered_layers.add(layer_idx)
                LOGGER.debug(f"Registered hook on layer {layer_idx}.mlp.gate")

    LOGGER.info(f"Registered {len(hooks)} router hooks with offloading support")

    metrics_dict = {
        "initial_offload": initial_offload,
        "initial_load": initial_load,
    }

    return hooks, metrics_dict


def clear_router_hooks(hooks: List):
    """Remove router hooks."""
    for hook in hooks:
        hook.remove()
    global _captured_router_outputs
    _captured_router_outputs = []


def run_experiment_for_prompt(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    input_token_ids: List[int],
    config: ExperimentConfig,
    mem_manager: ExpertMemoryManager,
    thres_expert: float,
    router_hooks_and_metrics: Tuple[List, Dict],
) -> RequestMetrics:
    """Run experiment for a single prompt. input_token_ids has exactly the desired prompt length."""
    global _captured_router_outputs, _layer_metrics_data, _activated_experts_cumulative

    # Clear global state for this request
    _captured_router_outputs = []
    _layer_metrics_data = []
    _activated_experts_cumulative = set()  # Reset cumulative set for new request

    router_hooks, metrics_dict = router_hooks_and_metrics
    initial_offload = metrics_dict["initial_offload"]
    initial_load = metrics_dict["initial_load"]

    device = torch.device(config.device)

    # Use pre-tokenized input for exact prompt length
    input_ids = torch.tensor([input_token_ids], dtype=torch.long, device=device)
    # DeepSeek and some models require attention_mask
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Forward pass with timing
    # Offloading is handled by hooks during forward pass
    torch.cuda.synchronize(device)
    start_time = time.time()

    # Try to get router logits if supported
    model_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "return_dict": True,
    }

    # Check if model supports output_router_logits
    # DeepSeek models don't support this parameter, so we need to check model type
    model_type = getattr(model.config, 'model_type', '').lower() if hasattr(model.config, 'model_type') else ''
    model_class_name = str(type(model)).lower()
    is_deepseek = 'deepseek' in model_type or 'deepseek' in model_class_name

    if is_deepseek:
        # DeepSeek models don't support output_router_logits
        LOGGER.debug("DeepSeek model detected, skipping output_router_logits parameter")
        # Remove output_router_logits from kwargs if it was added
        model_kwargs.pop("output_router_logits", None)
    elif hasattr(model.config, 'output_router_logits') or \
         hasattr(model, 'config') and getattr(model.config, 'output_router_logits', None) is not None:
        model_kwargs["output_router_logits"] = True
    else:
        # Try anyway - some models support it even if not in config
        try:
            # First check if the forward method accepts this parameter
            import inspect
            forward_sig = inspect.signature(model.forward)
            if 'output_router_logits' in forward_sig.parameters:
                model_kwargs["output_router_logits"] = True
        except (TypeError, AttributeError):
            # Model doesn't support this parameter
            pass

    with torch.no_grad():
        outputs = model(**model_kwargs)

    torch.cuda.synchronize(device)
    total_latency_ms = (time.time() - start_time) * 1000.0

    # Process captured data from hooks
    num_layers = len(model.model.layers)

    # Build layer metrics from hook data
    layer_metrics_dict = {data["layer_idx"]: data for data in _layer_metrics_data}
    layer_metrics_list = []
    total_waiting_time = 0.0

    # Also try to get from router_logits if available (Qwen models) as fallback
    if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
        router_logits = outputs.router_logits
        top_k = getattr(model.config, 'num_experts_per_tok', None) or \
                getattr(model.config, 'num_experts_per_token', None) or \
                getattr(model.config, 'top_k', None) or 8

        if isinstance(router_logits, (list, tuple)):
            for layer_idx, logits in enumerate(router_logits):
                if layer_idx not in layer_metrics_dict and logits is not None and isinstance(logits, torch.Tensor):
                    # Get top-k experts as fallback
                    if logits.dim() >= 2:
                        flat_logits = logits.view(-1, logits.shape[-1])
                        topk_values, topk_indices = torch.topk(
                            flat_logits,
                            k=min(top_k, logits.shape[-1]),
                            dim=-1
                        )
                        unique_experts = topk_indices.flatten().unique().cpu().tolist()
                        activated_experts = [int(x) for x in unique_experts]

                        # Add to metrics if not already captured by hooks
                        layer_metrics_dict[layer_idx] = {
                            "layer_idx": layer_idx,
                            "num_activated_experts": len(activated_experts),
                            "waiting_time_ms": 0.0,
                            "offload_count": 0,
                            "load_count": 0,
                        }

    # Build layer metrics list
    for layer_idx in range(num_layers):
        if layer_idx in layer_metrics_dict:
            data = layer_metrics_dict[layer_idx]
            layer_metrics = LayerMetrics(
                layer_id=data["layer_idx"],
                activated_experts=[],  # Not recording activated_experts in output
                num_activated_experts=data["num_activated_experts"],
                waiting_time_ms=data["waiting_time_ms"],
                offload_count=data["offload_count"],
                load_count=data["load_count"],
            )
            total_waiting_time += data["waiting_time_ms"]
        else:
            # No data for this layer (might not be MoE layer)
            layer_metrics = LayerMetrics(
                layer_id=layer_idx,
                activated_experts=[],
                num_activated_experts=0,
                waiting_time_ms=0.0,
                offload_count=0,
                load_count=0,
            )
        layer_metrics_list.append(layer_metrics)

    # Calculate offload/load counts for this request
    request_offload_count = mem_manager.total_offload_count - initial_offload
    request_load_count = mem_manager.total_load_count - initial_load

    return RequestMetrics(
        prompt_length=len(input_ids[0]),
        total_latency_ms=total_latency_ms,
        waiting_time_ms_due_to_loading=total_waiting_time,
        thres_expert=thres_expert,
        layers=layer_metrics_list,
        offload_count_to_cpu=request_offload_count,
        load_count_to_gpu=request_load_count,
    )


def compute_aggregates(metrics_list: List[RequestMetrics]) -> Dict:
    """Compute aggregate statistics."""
    if not metrics_list:
        return {}

    total_latencies = [m.total_latency_ms for m in metrics_list]
    waiting_times = [m.waiting_time_ms_due_to_loading for m in metrics_list]

    def percentile(data, p):
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100.0)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    return {
        "total_latency_ms": {
            "mean": sum(total_latencies) / len(total_latencies),
            "p50": percentile(total_latencies, 50),
            "p95": percentile(total_latencies, 95),
            "p99": percentile(total_latencies, 99),
        },
        "waiting_time_ms_due_to_loading": {
            "mean": sum(waiting_times) / len(waiting_times),
            "p50": percentile(waiting_times, 50),
            "p95": percentile(waiting_times, 95),
            "p99": percentile(waiting_times, 99),
        },
    }


def run_experiment(config: ExperimentConfig) -> Dict:
    """Run the complete experiment."""
    LOGGER.info(f"Starting experiment for model: {config.model_path}")

    # Set random seeds
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)

    # Load model
    LOGGER.info("Loading model...")
    # Try to detect model dtype from config, default to float16
    # For quantized models, let transformers handle dtype automatically
    model_dtype = None
    try:
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(config.model_path, trust_remote_code=True)
        # Check if model has a preferred dtype
        if hasattr(model_config, 'torch_dtype') and model_config.torch_dtype is not None:
            model_dtype = model_config.torch_dtype
        elif hasattr(model_config, 'dtype') and model_config.dtype is not None:
            model_dtype = model_config.dtype

        # Convert string dtype to torch dtype if needed
        if isinstance(model_dtype, str):
            if 'bfloat16' in model_dtype.lower() or 'bf16' in model_dtype.lower():
                model_dtype = torch.bfloat16
            elif 'float16' in model_dtype.lower() or 'fp16' in model_dtype.lower():
                model_dtype = torch.float16
            else:
                model_dtype = None
    except Exception as e:
        LOGGER.debug(f"Could not detect dtype from config: {e}")
        model_dtype = None

    # If still None, let transformers use default (usually based on model quantization)
    if model_dtype is None:
        LOGGER.info("Using default dtype (will be determined by model)")
    else:
        LOGGER.info(f"Using dtype: {model_dtype}")

    # Load on single device so expert offload/fetch (CPU<->GPU) stays consistent
    load_kwargs = {
        "device_map": config.device,  # e.g. "cuda:0" - entire model on one GPU
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if model_dtype is not None:
        load_kwargs["torch_dtype"] = model_dtype

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        **load_kwargs,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Compute GPU memory limit (half experts per layer + non-expert + KV cache 2048)
    expert_budget_bytes, gpu_limit_metadata = compute_gpu_memory_limit_bytes(model)

    # thres_expert for metadata: half experts per layer (our GPU budget)
    experts_per_layer = gpu_limit_metadata.get("experts_per_layer", 0)
    thres_expert = experts_per_layer // 2 if experts_per_layer else 1

    threshold_metadata = {**gpu_limit_metadata, "thres_expert": thres_expert}

    # Initialize memory manager with real CPU-to-GPU fetch + offload
    mem_manager = ExpertMemoryManager(
        model,
        device,
        expert_budget_bytes,
        config.offload_policy,
    )

    # Setup router hooks with offloading support
    router_hooks, hooks_metrics = setup_router_hooks(
        model,
        mem_manager,
        thres_expert,
        config,
    )

    # Load dataset
    prompts = load_sharegpt_dataset(Path(config.dataset_path))
    # Shuffle for variety, but iterate sequentially (concatenation uses subsequent prompts)
    prompts_shuffled = prompts.copy()
    random.shuffle(prompts_shuffled)

    # Run experiments for each prompt length
    all_results = []
    prompt_idx = 0

    for prompt_len in config.prompt_lengths:
        LOGGER.info(f"Testing prompt length: {prompt_len}")

        for _ in range(config.num_prompts_per_length):
            try:
                token_ids, prompt_idx = get_prompt_of_exact_length(
                    tokenizer, prompts_shuffled, prompt_idx, prompt_len
                )
            except ValueError as e:
                LOGGER.warning(
                    f"Skipping remaining samples for prompt_len={prompt_len}: {e}. "
                    f"Consider using a dataset with longer/more prompts."
                )
                break

            # Reset memory manager stats for this request
            hooks_metrics["initial_offload"] = mem_manager.total_offload_count
            hooks_metrics["initial_load"] = mem_manager.total_load_count

            # Run experiment
            metrics = run_experiment_for_prompt(
                model,
                tokenizer,
                token_ids,
                config,
                mem_manager,
                thres_expert,
                (router_hooks, hooks_metrics),
            )

            all_results.append(metrics)

    # Cleanup
    clear_router_hooks(router_hooks)

    # Compute aggregates
    aggregates = compute_aggregates(all_results)

    # Prepare output
    result = {
        "config": {
            "model_path": config.model_path,
            "dataset_path": config.dataset_path,
            "prompt_lengths": config.prompt_lengths,
            "num_prompts_per_length": config.num_prompts_per_length,
            "gpu_memory_gb": config.gpu_memory_gb,
            "seed": config.seed,
            "device": config.device,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "accumulation_mode": config.accumulation_mode,
            "offload_policy": config.offload_policy,
        },
        "threshold_metadata": threshold_metadata,
        "thres_expert": thres_expert,
        "aggregates": aggregates,
        "per_request_metrics": [
            {
                "prompt_length": m.prompt_length,
                "total_latency_ms": m.total_latency_ms,
                "waiting_time_ms_due_to_loading": m.waiting_time_ms_due_to_loading,
                "offload_count_to_cpu": m.offload_count_to_cpu,
                "load_count_to_gpu": m.load_count_to_gpu,
                "layers": [
                    {
                        "layer_id": lm.layer_id,
                        "num_activated_experts": lm.num_activated_experts,
                        "waiting_time_ms": lm.waiting_time_ms,
                        "offload_count": lm.offload_count,
                        "load_count": lm.load_count,
                    }
                    for lm in m.layers
                ],
            }
            for m in all_results
        ],
    }

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--dataset",
        default="/home/kec23008/DynaQuant/ShareGPT_V3_unfiltered_cleaned_split.json",
        help="Path to ShareGPT dataset",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--prompt-lengths",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512],
        help="Prompt lengths to test",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=16,
        help="Number of prompts per length",
    )
    parser.add_argument(
        "--gpu-memory-gb",
        type=float,
        default=24.0,
        help="GPU memory in GB",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use",
    )
    parser.add_argument(
        "--accumulation-mode",
        choices=["per_layer", "cumulative"],
        default="per_layer",
        help="How to accumulate activated experts",
    )
    parser.add_argument(
        "--offload-policy",
        default="random",
        help="Policy for selecting experts to offload",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = ExperimentConfig(
        model_path=args.model,
        dataset_path=args.dataset,
        prompt_lengths=args.prompt_lengths,
        num_prompts_per_length=args.num_prompts,
        gpu_memory_gb=args.gpu_memory_gb,
        seed=args.seed,
        device=args.device,
        accumulation_mode=args.accumulation_mode,
        offload_policy=args.offload_policy,
    )

    result = run_experiment(config)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)

    LOGGER.info(f"Results saved to {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
