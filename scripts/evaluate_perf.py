#!/usr/bin/env python3
"""
python scripts/evaluate_perf.py \
    --model-path /home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507 \
    --model-path-low /home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound \
    --dataset-dir calibration_datasets/requests \
    --max-samples 200 \
    --output results_qwen3_30B_int4.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path so dynaexq_new can be imported (when not installed as package)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from dynaexq_new.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

LOGGER = logging.getLogger("perf_eval")

def _load_state_dict_from_path(path: str) -> Dict[str, torch.Tensor]:
    """Load full state dict from a checkpoint dir (safetensors or pytorch_model.bin)."""
    path_obj = Path(path)
    state_dict: Dict[str, torch.Tensor] = {}
    # Try safetensors shards first
    st_files = sorted(path_obj.glob("*.safetensors"))
    if not st_files and (path_obj / "model.safetensors").exists():
        st_files = [path_obj / "model.safetensors"]
    if st_files:
        try:
            from safetensors.torch import load_file
            for f in st_files:
                if f.name == "model.safetensors.index.json":
                    continue
                state_dict.update(load_file(str(f)))
        except ImportError:
            pass
    if not state_dict:
        bin_path = path_obj / "pytorch_model.bin"
        if not bin_path.exists():
            bin_path = path_obj / "model.safetensors"
        if bin_path.exists():
            if str(bin_path).endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(str(bin_path))
            else:
                state_dict = torch.load(
                    str(bin_path), map_location="cpu", weights_only=True
                )
    return state_dict


def load_model_and_tokenizer(
    model_path: str,
    *,
    device: str,
    torch_dtype: Optional[str],
    model_path_low: Optional[str] = None,
) -> Tuple[Any, Any]:
    LOGGER.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = None
    if torch_dtype:
        dtype = getattr(torch, torch_dtype)
    elif device.startswith("cuda"):
        dtype = torch.float16

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)
    device_map: Any
    if device == "auto":
        device_map = "auto"
    else:
        device_map = {"": device}

    _load_kw = {"trust_remote_code": True, "device_map": device_map}
    if dtype is not None:
        _load_kw["torch_dtype"] = dtype

    LOGGER.info("Loading local Qwen3 MoE model from %s", model_path)
    model = Qwen3MoeForCausalLM.from_pretrained(model_path, **_load_kw)
    model.eval()

    bind_moe_precision_managers(model)
    # if model_path_low:
    #     bind_moe_precision_managers_low(model, model_path_low)
    return model, tokenizer


def bind_moe_precision_managers(model: Any) -> None:
    """
    After loading a model (possibly with mixed precision), bind each MoE layer's
    ExpertPrecisionManager to the layer's expert weights: copy gate_proj, up_proj, down_proj
    from each expert MLP into the manager's high/low CPU buffers and GPU buffer.
    """
    device = "cuda:0"
    bound = 0
    for mod in model.modules():
        pm = getattr(mod, "precision_manager", None)
        experts = getattr(mod, "experts", None)
        if pm is None or experts is None:
            continue
        if not isinstance(experts, (list, torch.nn.ModuleList)):
            continue
        # Experts are Qwen3MoeMLP with gate_proj, up_proj, down_proj (nn.Linear)
        if not all(
            (hasattr(e, "gate_proj") and hasattr(e, "up_proj") and hasattr(e, "down_proj") for e in experts)
        ):
            continue
        try:
            gate_proj = torch.stack([e.gate_proj.weight.data for e in experts], dim=0)
            up_proj = torch.stack([e.up_proj.weight.data for e in experts], dim=0)
            down_proj = torch.stack([e.down_proj.weight.data for e in experts], dim=0)
        except Exception as e:
            gate_proj = torch.stack([e.gate_proj.qweight.data for e in experts], dim=0)
            up_proj = torch.stack([e.up_proj.qweight.data for e in experts], dim=0)
            down_proj = torch.stack([e.down_proj.qweight.data for e in experts], dim=0)
        pm.load_from_parameters(gate_proj, up_proj, down_proj, is_high=True)
        pm.set_device(device)
        pm.start_background_sync()
        bound += 1
    if bound:
        LOGGER.info("Bound %d MoE layer(s) ExpertPrecisionManager to expert weights.", bound)


def bind_moe_precision_managers_low(model: Any, low_checkpoint_path: str) -> None:
    """
    Fill each MoE layer's ExpertPrecisionManager low-precision buffers from a second
    checkpoint (e.g. quantized/low-precision), without loading a second full model.
    State dict keys: {name}.experts.{i}.gate_proj.weight, .up_proj.weight, .down_proj.weight.
    """
    low_state_dict = _load_state_dict_from_path(low_checkpoint_path)
    if not low_state_dict:
        LOGGER.warning("No state dict loaded from %s; skipping low-precision bind.", low_checkpoint_path)
        return
    bound = 0
    for name, mod in model.named_modules():
        pm = getattr(mod, "precision_manager", None)
        if pm is None:
            continue
        n = pm.num_experts
        gate_list, up_list, down_list = [], [], []
        prefix = f"{name}.experts."
        prefix_alt = prefix.replace("model.model.", "model.", 1)
        use_alt = (f"{prefix}0.gate_proj.weight" not in low_state_dict) and (
            f"{prefix_alt}0.gate_proj.weight" in low_state_dict
        )
        p = prefix_alt if use_alt else prefix
        for i in range(n):
            gk = f"{p}{i}.gate_proj.weight"
            uk = f"{p}{i}.up_proj.weight"
            dk = f"{p}{i}.down_proj.weight"
            if gk not in low_state_dict or uk not in low_state_dict or dk not in low_state_dict:
                break
            gate_list.append(low_state_dict[gk])
            up_list.append(low_state_dict[uk])
            down_list.append(low_state_dict[dk])
        if len(gate_list) != n:
            continue
        gate_proj = torch.stack(gate_list, dim=0)
        up_proj = torch.stack(up_list, dim=0)
        down_proj = torch.stack(down_list, dim=0)
        pm.load_from_parameters(gate_proj, up_proj, down_proj, is_high=False)
        bound += 1
    if bound:
        LOGGER.info(
            "Bound %d MoE layer(s) low-precision weights from %s.",
            bound,
            low_checkpoint_path,
        )


if __name__ == "__main__":
    prompt = "What is the capital of France?"
    model_path = "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"
    model_path_low = "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int2-mixed-AutoRound"
    device = "cuda:0"
    torch_dtype = "float16"

    model, tokenizer = load_model_and_tokenizer(
        model_path,
        device=device,
        torch_dtype=torch_dtype,
        model_path_low=model_path_low,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
