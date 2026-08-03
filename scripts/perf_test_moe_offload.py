#!/usr/bin/env python3
"""
Performance test: TTFT, TPOP, End2End latency with MoE expert CPU/GPU migration.

Uses the same expert offload/prefetch scenario as experiment_moe_offload_latency.py:
- Experts start on CPU, fetch to GPU when needed
- When GPU memory insufficient, offload previous layer's experts
- GPU budget: half experts per layer + non-expert weights + KV cache

Measures TTFT, TPOP, End2End (avg, P95, P99) across different batch sizes.
Each request from ShareGPT, fixed 512 tokens (pad/truncate).
10 batches per batch size.

Example:
  python scripts/perf_test_moe_offload.py --model /path/to/Qwen3-30B-A3B \\
    --batch-sizes 1 2 4 --max-new-tokens 64 --output results/moe_offload_perf.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger("perf_test_moe_offload")

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "ShareGPT_V3_unfiltered_cleaned_split.json"
FIXED_PROMPT_LENGTH = 512
BATCHES_PER_SIZE = 10


@dataclass
class ExpertID:
    layer: int
    idx: int

    def __hash__(self):
        return hash((self.layer, self.idx))

    def __eq__(self, other):
        return isinstance(other, ExpertID) and self.layer == other.layer and self.idx == other.idx


def _get_layer_experts_info(model: nn.Module) -> Tuple[Dict[int, List[ExpertID]], Dict[ExpertID, nn.Module], int, int, int]:
    """Extract expert structure from model."""
    layer_experts: Dict[int, List[ExpertID]] = {}
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
                    layer_experts[layer_idx].append(expert_id)
                    expert_id_to_module[expert_id] = experts[expert_idx]
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
                        layer_experts[layer_idx].append(expert_id)
                        expert_id_to_module[expert_id] = experts[expert_idx]
                    if experts_per_layer == 0:
                        experts_per_layer = len(experts)
                except (TypeError, AttributeError):
                    pass

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
    if num_unique_modules and experts_per_layer:
        if num_unique_modules < len(layer_experts) * experts_per_layer:
            expert_size_bytes = total_expert_bytes // num_unique_modules
        else:
            expert_size_bytes = total_expert_bytes // (len(layer_experts) * experts_per_layer)
        if expert_size_bytes <= 0:
            expert_size_bytes = total_expert_bytes // max(1, len(expert_id_to_module))
    return layer_experts, expert_id_to_module, expert_size_bytes, experts_per_layer, num_layers


def compute_gpu_memory_limit_bytes(model: nn.Module) -> Tuple[int, Dict]:
    """GPU limit = half experts per layer + non-expert + KV cache for 2048 tokens."""
    layer_experts, _, expert_size_bytes, experts_per_layer, num_layers = _get_layer_experts_info(model)
    total_model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_expert_bytes = len(layer_experts) * experts_per_layer * expert_size_bytes if experts_per_layer else 0
    non_expert_bytes = total_model_bytes - total_expert_bytes
    num_layers_with_experts = len(layer_experts)
    expert_budget_bytes = num_layers_with_experts * (experts_per_layer // 2) * expert_size_bytes

    config = getattr(model, 'config', None)
    kv_cache_bytes = 0
    if config is not None:
        num_hidden_layers = getattr(config, 'num_hidden_layers', num_layers)
        num_kv_heads = getattr(config, 'num_key_value_heads', getattr(config, 'num_attention_heads', 32))
        head_dim = getattr(config, 'head_dim', None) or (
            getattr(config, 'hidden_size', 2048) // getattr(config, 'num_attention_heads', 32))
        kv_cache_bytes = num_hidden_layers * 2048 * num_kv_heads * head_dim * 2 * 2

    metadata = {
        "non_expert_bytes": non_expert_bytes,
        "expert_budget_bytes": expert_budget_bytes,
        "kv_cache_bytes": kv_cache_bytes,
        "experts_per_layer": experts_per_layer,
    }
    return expert_budget_bytes, metadata


class ExpertMemoryManager:
    """Manages expert CPU<->GPU migration: experts on CPU, fetch when needed, offload when full."""

    def __init__(self, model: nn.Module, device: torch.device, gpu_memory_limit_bytes: int):
        self.model = model
        self.device = device
        self.gpu_memory_limit_bytes = gpu_memory_limit_bytes
        layer_experts, expert_id_to_module, _, self.experts_per_layer, self.num_layers = _get_layer_experts_info(model)
        self.layer_experts = layer_experts
        self.expert_id_to_module = expert_id_to_module
        self.expert_residency: Dict[ExpertID, str] = {}
        self._gpu_expert_bytes = 0
        self._expert_sizes: Dict[ExpertID, int] = {}
        for expert_id, mod in expert_id_to_module.items():
            sz = sum(p.numel() * p.element_size() for p in mod.parameters())
            sz += sum(b.numel() * b.element_size() for b in mod.buffers())
            self._expert_sizes[expert_id] = sz
        self._initialize_expert_residency()
        LOGGER.info("ExpertMemoryManager: GPU limit %.2f GB", gpu_memory_limit_bytes / 1e9)

    def _initialize_expert_residency(self):
        for expert_id in self.expert_id_to_module:
            mod = self.expert_id_to_module[expert_id]
            for param in mod.parameters():
                if param.device.type == "cuda":
                    param.data = param.data.cpu()
            for buffer in mod.buffers():
                if buffer.device.type == "cuda":
                    buffer.data = buffer.data.cpu()
            self.expert_residency[expert_id] = "cpu"
        self._gpu_expert_bytes = 0

    def _get_layer_with_gpu_experts(self, exclude_layer: int) -> Optional[int]:
        start = (exclude_layer - 1) if exclude_layer > 0 else (self.num_layers - 1)
        for offset in range(self.num_layers):
            layer_idx = (start - offset) % self.num_layers
            if layer_idx == exclude_layer:
                continue
            if layer_idx in self.layer_experts:
                for expert_id in self.layer_experts[layer_idx]:
                    if self.expert_residency.get(expert_id) == "gpu":
                        return layer_idx
        return None

    def _offload_layer_experts(self, layer_idx: int) -> float:
        total_ms = 0.0
        if layer_idx not in self.layer_experts:
            return 0.0
        for expert_id in self.layer_experts[layer_idx]:
            if self.expert_residency.get(expert_id) == "gpu":
                total_ms += self._offload_expert_to_cpu(expert_id)
        return total_ms

    def _offload_expert_to_cpu(self, expert_id: ExpertID) -> float:
        if expert_id not in self.expert_residency or self.expert_residency[expert_id] == "cpu":
            return 0.0
        mod = self.expert_id_to_module[expert_id]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.time()
        for param in mod.parameters():
            if param.device.type == "cuda":
                param.data = param.data.cpu()
        for buffer in mod.buffers():
            if buffer.device.type == "cuda":
                buffer.data = buffer.data.cpu()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        ms = (time.time() - start) * 1000.0
        self.expert_residency[expert_id] = "cpu"
        self._gpu_expert_bytes -= self._expert_sizes.get(expert_id, 0)
        return ms

    def _load_expert_to_gpu(self, expert_id: ExpertID) -> float:
        if expert_id not in self.expert_residency or self.expert_residency[expert_id] == "gpu":
            return 0.0
        mod = self.expert_id_to_module[expert_id]
        sz = self._expert_sizes.get(expert_id, 0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        current_layer = expert_id.layer
        attempts = 0
        while self._gpu_expert_bytes + sz > self.gpu_memory_limit_bytes and attempts < self.num_layers + 1:
            offload_layer = self._get_layer_with_gpu_experts(current_layer)
            if offload_layer is None:
                break
            self._offload_layer_experts(offload_layer)
            attempts += 1
        start = time.time()
        for param in mod.parameters():
            if param.device.type == "cpu":
                param.data = param.data.to(self.device)
        for buffer in mod.buffers():
            if buffer.device.type == "cpu":
                buffer.data = buffer.data.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        ms = (time.time() - start) * 1000.0
        self.expert_residency[expert_id] = "gpu"
        self._gpu_expert_bytes += sz
        return ms

    def ensure_experts_on_gpu(self, activated_experts: Set[ExpertID], current_layer: int) -> float:
        total_ms = 0.0
        for expert_id in activated_experts:
            if expert_id in self.expert_residency and self.expert_residency[expert_id] == "cpu":
                total_ms += self._load_expert_to_gpu(expert_id)
        return total_ms


_captured_router_outputs: List[Tuple[int, List[int]]] = []


def setup_router_hooks(model: nn.Module, mem_manager: ExpertMemoryManager, thres_expert: float):
    """Setup hooks for router output to trigger expert load/offload."""
    global _captured_router_outputs

    def make_router_hook(layer_idx: int, num_experts: int, top_k: int):
        def hook_fn(module, input, output):
            activated_indices = []
            if isinstance(output, tuple) and len(output) >= 1:
                topk_idx = output[0]
                if isinstance(topk_idx, torch.Tensor) and topk_idx.numel() > 0:
                    activated_indices = [int(x) for x in topk_idx.flatten().unique().cpu().tolist()]
            elif isinstance(output, torch.Tensor) and output.dim() >= 2:
                n_exp = num_experts or output.shape[-1]
                router_logits = output.float().view(-1, n_exp)
                _, selected = torch.topk(router_logits, min(top_k, n_exp), dim=-1)
                activated_indices = [int(x) for x in selected.flatten().unique().cpu().tolist()]
            activated_experts = {ExpertID(layer=layer_idx, idx=idx) for idx in activated_indices}
            _captured_router_outputs.append((layer_idx, activated_indices))
            mem_manager.ensure_experts_on_gpu(activated_experts, layer_idx)
            return output
        return hook_fn

    def make_mlp_router_hook(layer_idx: int):
        def hook_fn(module, input, output):
            activated_indices = []
            if isinstance(output, tuple) and len(output) >= 2:
                router_indices = output[1]
                if isinstance(router_indices, torch.Tensor) and router_indices.numel() > 0:
                    activated_indices = [int(x) for x in router_indices.flatten().unique().cpu().tolist()]
            elif isinstance(output, tuple) and len(output) >= 1:
                topk_idx = output[0]
                if isinstance(topk_idx, torch.Tensor) and topk_idx.numel() > 0:
                    activated_indices = [int(x) for x in topk_idx.flatten().unique().cpu().tolist()]
            activated_experts = {ExpertID(layer=layer_idx, idx=idx) for idx in activated_indices}
            _captured_router_outputs.append((layer_idx, activated_indices))
            mem_manager.ensure_experts_on_gpu(activated_experts, layer_idx)
            return output
        return hook_fn

    config = getattr(model, "config", None)
    num_experts_default = getattr(config, "num_experts", 0) or getattr(config, "num_local_experts", 0)
    top_k_default = getattr(config, "num_experts_per_tok", 8) or getattr(config, "num_experts_per_token", 8) or 8
    hooks = []
    registered = set()

    for name, module in model.named_modules():
        if hasattr(module, "gate") and hasattr(module, "experts") and "mlp" in name.lower():
            m = re.search(r"layers\.(\d+)", name)
            if m and int(m.group(1)) not in registered:
                layer_idx = int(m.group(1))
                n_exp = getattr(module, "num_experts", num_experts_default)
                tk = getattr(module, "top_k", top_k_default)
                hooks.append(module.gate.register_forward_hook(make_router_hook(layer_idx, n_exp, tk)))
                registered.add(layer_idx)

    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx in registered:
            continue
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            if hasattr(mlp, "router"):
                hooks.append(mlp.router.register_forward_hook(make_mlp_router_hook(layer_idx)))
                registered.add(layer_idx)
            elif hasattr(mlp, "gate"):
                n_exp = getattr(mlp, "num_experts", num_experts_default)
                tk = getattr(mlp, "top_k", top_k_default)
                hooks.append(mlp.gate.register_forward_hook(make_router_hook(layer_idx, n_exp, tk)))
                registered.add(layer_idx)

    LOGGER.info("Registered %d router hooks", len(hooks))
    return hooks


def load_sharegpt_prompts(
    dataset_path: Path,
    tokenizer: AutoTokenizer,
    target_length: int,
    max_requests: int,
) -> List[Tuple[str, List[int]]]:
    """Load prompts from ShareGPT, fix each to target_length tokens. Returns (sample_id, token_ids)."""
    LOGGER.info("Loading ShareGPT from %s (target=%d, max=%d)", dataset_path, target_length, max_requests)
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    samples: List[Tuple[str, List[int]]] = []
    for item in data:
        if len(samples) >= max_requests:
            break
        if "conversations" not in item:
            continue
        text = None
        for conv in item["conversations"]:
            if conv.get("from") == "human" and "value" in conv:
                text = conv["value"]
                break
        if not text or not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < target_length:
            ids = ids + [pad_id] * (target_length - len(ids))
        else:
            ids = ids[:target_length]
        sample_id = str(item.get("id", f"req_{len(samples):05d}"))
        samples.append((sample_id, ids))
    LOGGER.info("Loaded %d prompts (fixed to %d tokens)", len(samples), target_length)
    return samples


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0,1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo))


def run_batch_generation(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    batch_token_ids: List[List[int]],
    device: torch.device,
    max_new_tokens: int,
    model_kwargs_base: Dict,
) -> Tuple[float, float, float]:
    """
    Run batched prefill + decode. Returns (ttft_ms, tpop_ms, end2end_ms).
    All sequences in batch have same TTFT/TPOP/End2End (processed together).
    """
    global _captured_router_outputs
    _captured_router_outputs = []

    batch_size = len(batch_token_ids)
    input_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    def maybe_sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    model_kwargs = {**model_kwargs_base, "use_cache": True, "return_dict": True}

    maybe_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)
    maybe_sync()
    ttft_ms = (time.perf_counter() - t0) * 1000.0

    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
    generated = next_tokens.clone()
    eos_id = tokenizer.eos_token_id

    decode_times: List[float] = []
    for _ in range(max_new_tokens - 1):
        maybe_sync()
        t_step = time.perf_counter()
        with torch.no_grad():
            step_out = model(
                input_ids=next_tokens,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                **model_kwargs_base,
            )
        maybe_sync()
        decode_times.append(time.perf_counter() - t_step)
        past_key_values = step_out.past_key_values
        logits = step_out.logits[:, -1, :]
        next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tokens], dim=1)
        if eos_id is not None and (next_tokens == eos_id).all():
            break

    num_decode_tokens = len(decode_times) + 1
    total_decode_ms = sum(decode_times) * 1000.0
    tpop_ms = (total_decode_ms / num_decode_tokens) if num_decode_tokens > 0 else 0.0
    end2end_ms = ttft_ms + total_decode_ms

    return ttft_ms, tpop_ms, end2end_ms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoE expert migration perf: TTFT, TPOP, End2End")
    p.add_argument("--model", help="Model path (single model)")
    p.add_argument("--models", type=str, nargs="+", help="Multiple models: --models m1 m2")
    p.add_argument("--devices", type=str, nargs="+", default=["cuda:0", "cuda:1"], help="Devices for each model, e.g. cuda:0 cuda:1")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="ShareGPT JSON path")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1,2,4,8,16,32], help="Batch sizes to test")
    p.add_argument("--prompt-length", type=int, default=FIXED_PROMPT_LENGTH)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--batches-per-size", type=int, default=BATCHES_PER_SIZE)
    p.add_argument("--device", default="cuda:0", help="Device when using single --model")
    p.add_argument("--output", type=Path, help="Output JSON")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _run_one_model(
    model_path: str,
    device_str: str,
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    samples: List[Tuple[str, List[int]]],
    batch_sizes_override: Optional[List[int]] = None,
    partial_save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run perf test for one model on one device. Returns result dict."""
    import gc
    use_auto_device_map = device_str.strip().lower() == "auto"
    if use_auto_device_map:
        device = torch.device("cuda:0")
        device_map = "auto"
    else:
        device = torch.device(device_str)
        device_map = device_str
    batch_sizes = batch_sizes_override if batch_sizes_override is not None else args.batch_sizes

    LOGGER.info("Loading model from %s on %s", model_path, device_map if use_auto_device_map else device_str)
    load_kwargs = dict(
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()

    expert_budget_bytes, gpu_meta = compute_gpu_memory_limit_bytes(model)
    experts_per_layer = gpu_meta.get("experts_per_layer", 0)
    thres_expert = experts_per_layer // 2 if experts_per_layer else 1

    mem_manager = ExpertMemoryManager(model, device, expert_budget_bytes)
    hooks = setup_router_hooks(model, mem_manager, thres_expert)

    model_kwargs_base = {}
    model_type = getattr(model.config, "model_type", "").lower()
    # Skip output_router_logits for DeepSeek and Phi-MoE (Phimoe triggers load_balancing_loss bug)
    if "deepseek" not in model_type and "phi" not in model_type and "phimoe" not in model_type:
        try:
            sig = inspect.signature(model.forward)
            if "output_router_logits" in sig.parameters:
                model_kwargs_base["output_router_logits"] = True
        except (TypeError, AttributeError):
            pass

    results: List[Dict[str, Any]] = []
    sample_idx = 0

    try:
        for batch_size in sorted(batch_sizes):
            LOGGER.info("=== Batch size %d ===", batch_size)
            ttft_list: List[float] = []
            tpop_list: List[float] = []
            e2e_list: List[float] = []

            for b in range(args.batches_per_size):
                batch_samples = [
                    samples[(sample_idx + i) % len(samples)][1]
                    for i in range(batch_size)
                ]
                sample_idx += batch_size

                ttft_ms, tpop_ms, end2end_ms = run_batch_generation(
                    model, tokenizer, batch_samples, device, args.max_new_tokens, model_kwargs_base
                )
                ttft_list.append(ttft_ms)
                tpop_list.append(tpop_ms)
                e2e_list.append(end2end_ms)

            summary = {
                "batch_size": batch_size,
                "num_batches": args.batches_per_size,
                "ttft_avg_ms": (sum(ttft_list) / len(ttft_list)) if ttft_list else None,
                "ttft_p95_ms": percentile(ttft_list, 0.95),
                "ttft_p99_ms": percentile(ttft_list, 0.99),
                "tpop_avg_ms": (sum(tpop_list) / len(tpop_list)) if tpop_list else None,
                "tpop_p95_ms": percentile(tpop_list, 0.95),
                "tpop_p99_ms": percentile(tpop_list, 0.99),
                "end2end_avg_ms": (sum(e2e_list) / len(e2e_list)) if e2e_list else None,
                "end2end_p95_ms": percentile(e2e_list, 0.95),
                "end2end_p99_ms": percentile(e2e_list, 0.99),
            }
            results.append(summary)
            if partial_save_path:
                partial_save_path.parent.mkdir(parents=True, exist_ok=True)
                partial_save_path.write_text(json.dumps({
                    "model": model_path, "device": device_str,
                    "prompt_length": args.prompt_length, "max_new_tokens": args.max_new_tokens,
                    "batches_per_size": args.batches_per_size, "gpu_metadata": gpu_meta,
                    "batch_sizes": results,
                }, indent=2), encoding="utf-8")

            LOGGER.info(
                "batch_size=%d | TTFT: avg=%.1f p95=%.1f p99=%.1f ms | "
                "TPOP: avg=%.1f p95=%.1f p99=%.1f ms | "
                "End2End: avg=%.1f p95=%.1f p99=%.1f ms",
                batch_size,
                summary["ttft_avg_ms"] or 0,
                summary["ttft_p95_ms"] or 0,
                summary["ttft_p99_ms"] or 0,
                summary["tpop_avg_ms"] or 0,
                summary["tpop_p95_ms"] or 0,
                summary["tpop_p99_ms"] or 0,
                summary["end2end_avg_ms"] or 0,
                summary["end2end_p95_ms"] or 0,
                summary["end2end_p99_ms"] or 0,
            )
    finally:
        for h in hooks:
            h.remove()
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "model": model_path,
        "device": device_str,
        "prompt_length": args.prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "batches_per_size": args.batches_per_size,
        "gpu_metadata": gpu_meta,
        "batch_sizes": results,
    }


def main() -> int:
    import gc
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    if args.models:
        model_list = args.models
        device_list = (args.devices + [args.devices[-1]] * (len(model_list) - len(args.devices)))[: len(model_list)]
    elif args.model:
        model_list = [args.model]
        device_list = [args.device]
    else:
        raise ValueError("Provide --model or --models")

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    all_results: List[Dict[str, Any]] = []

    for model_path, device_str in zip(model_list, device_list):
        LOGGER.info("Loading tokenizer from %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        max_requests = max(args.batch_sizes) * args.batches_per_size + 32
        samples = load_sharegpt_prompts(
            args.dataset, tokenizer, args.prompt_length, max_requests
        )

        batch_sizes_override = None
        if "80B" in model_path:
            batch_sizes_override = [1, 2, 4]

        try:
            out = _run_one_model(
                model_path, device_str, args, tokenizer, samples, batch_sizes_override, None
            )
            all_results.append(out)

            single_out_path = args.output
            if single_out_path and len(model_list) > 1:
                stem = "moe_offload_" + ("qwen30b" if "30B" in model_path else "qwen80b")
                single_out_path = single_out_path.parent / f"{stem}.json"
            if single_out_path:
                single_out_path.parent.mkdir(parents=True, exist_ok=True)
                single_out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
                LOGGER.info("Results saved to %s", single_out_path.resolve())
        except Exception as e:
            LOGGER.exception("Failed for %s: %s", model_path, e)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.output and len(all_results) > 1:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"models": all_results}, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Combined results saved to %s", args.output.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
