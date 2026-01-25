#!/usr/bin/env python3
"""
Summarise expert activations for Qwen3-30B-A3B on calibration prompts.

Usage
-----
python scripts/summarize_expert_activation.py \
    --dataset calibration_datasets/requests/mmlu_pro_200.jsonl \
    --model-id Qwen/Qwen3-30B-A3B \
    --max-prompts 50 \
    --output activation_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

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


LOGGER = logging.getLogger("expert_activation")


def load_prompts(
    path: Path,
    prompt_key: str = "prompt",
    limit: Optional[int] = None,
) -> List[str]:
    """Load prompts from .jsonl/.json or plain-text files."""
    prompts: List[str] = []
    suffix = path.suffix.lower()

    if suffix in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON line: {line[:120]}") from exc

                if prompt_key not in entry:
                    raise KeyError(
                        f"Key '{prompt_key}' missing in JSON object: {entry.keys()}"
                    )

                prompt = str(entry[prompt_key]).strip()
                if prompt:
                    prompts.append(prompt)
                    if limit and len(prompts) >= limit:
                        break
    else:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                prompts.append(line)
                if limit and len(prompts) >= limit:
                    break

    if not prompts:
        raise ValueError(f"No prompts found in {path}")

    LOGGER.info("Loaded %d prompts from %s", len(prompts), path)
    return prompts


def collect_expert_statistics(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    top_k: int,
    max_length: int,
    device: torch.device,
    generate_first_token: bool = False,
) -> List[Counter]:
    """
    Run prefill for each prompt and accumulate expert activations per layer.
    """
    num_layers = model.config.num_hidden_layers
    counters: List[Counter] = [Counter() for _ in range(num_layers)]

    model.eval()
    model.config.output_router_logits = True

    for idx, prompt in enumerate(prompts, start=1):
        try:
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )

            encoded = {key: value.to(device) for key, value in encoded.items()}
            attention_mask = encoded.get("attention_mask")

            LOGGER.info("Processing prompt %d/%d (length: %d)", idx, len(prompts), encoded["input_ids"].shape[1])
            
            import time
            forward_start = time.time()
            with torch.no_grad():
                outputs = model(
                    **encoded,
                    use_cache=False,
                    output_router_logits=True,
                    return_dict=True,
                )
            forward_time = time.time() - forward_start
            LOGGER.info("Forward pass completed for prompt %d in %.2f seconds", idx, forward_time)

            router_logits = outputs.router_logits
            if router_logits is None:
                LOGGER.warning(
                    "Model did not return `router_logits` for prompt %d. Skipping.",
                    idx
                )
                continue
            
            LOGGER.info("Got router_logits for prompt %d: type=%s, length=%s", 
                       idx, type(router_logits), len(router_logits) if router_logits else None)

            # Pre-compute mask once for all layers
            seq_len = encoded["input_ids"].shape[1]
            mask_cpu = (
                attention_mask.bool().cpu()
                if attention_mask is not None
                else torch.ones(
                    encoded["input_ids"].shape[:2], dtype=torch.bool
                )
            )
            if mask_cpu.dim() == 1:
                mask_cpu = mask_cpu.unsqueeze(1)
            if mask_cpu.size(1) != seq_len:
                mask_cpu = mask_cpu.expand(-1, seq_len)
            
            # Process all layers
            for layer_id, layer_logits in enumerate(router_logits):
                try:
                    # Move to CPU immediately to avoid GPU-CPU transfer overhead
                    if layer_logits.is_cuda:
                        layer_logits = layer_logits.cpu()
                    
                    flat_logits = layer_logits.real if torch.is_complex(
                        layer_logits) else layer_logits
                    
                    if flat_logits.dim() == 3:
                        # Shape: [batch, seq_len, num_experts] - already on CPU
                        top_indices = torch.topk(flat_logits, k=top_k, dim=-1).indices
                    elif flat_logits.dim() == 2:
                        # Shape: [batch, num_experts] - need to expand to seq_len
                        tile = flat_logits.unsqueeze(1).repeat(1, seq_len, 1)
                        top_indices = torch.topk(tile, k=top_k, dim=-1).indices
                    else:
                        LOGGER.warning(
                            "Unexpected router_logits dimensions for layer %d: %s. Skipping layer.",
                            layer_id, flat_logits.shape
                        )
                        continue

                    # top_indices is already on CPU
                    batch_size, seq_len, _ = top_indices.shape
                    # Convert to numpy for faster iteration
                    top_indices_np = top_indices.numpy()
                    mask_np = mask_cpu.numpy()
                    
                    for b in range(batch_size):
                        row = mask_np[min(b, mask_np.shape[0] - 1)]
                        for t in range(min(seq_len, len(row))):
                            if not row[t]:
                                continue
                            expert_ids = top_indices_np[b, t]
                            for expert_id in expert_ids:
                                counters[layer_id][int(expert_id)] += 1
                except Exception as e:
                    LOGGER.warning(
                        "Error processing layer %d for prompt %d: %s. Skipping layer.",
                        layer_id, idx, e
                    )
                    continue

            if generate_first_token:
                gen_outputs = model.generate(
                    **encoded,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                new_tokens = gen_outputs[:, encoded["input_ids"].shape[1]:]
                generated_text = tokenizer.decode(
                    new_tokens[0], skip_special_tokens=True
                )
                LOGGER.info(
                    "Generated first token for prompt %d: %s",
                    idx,
                    generated_text or "<EOS>",
                )

            if idx % 10 == 0 or idx == len(prompts):
                LOGGER.info("Processed prompt %d/%d", idx, len(prompts))
            else:
                LOGGER.debug("Processed prompt %d/%d", idx, len(prompts))
                
        except Exception as e:
            LOGGER.error(
                "Error processing prompt %d/%d: %s. Skipping.",
                idx, len(prompts), e, exc_info=True
            )
            continue

    return counters


def format_summary(counters: Sequence[Counter]) -> Dict[str, List[Dict[str, int]]]:
    """Convert counters to sorted dictionaries for serialisation."""
    summary: Dict[str, List[Dict[str, int]]] = {}
    for layer_id, counter in enumerate(counters):
        entries = sorted(
            counter.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        summary[f"layer_{layer_id}"] = [
            {"expert_id": expert_id, "activations": count} for expert_id, count in entries
        ]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise Qwen3 MoE expert activations over calibration prompts."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to JSONL/JSON/TXT calibration dataset (e.g. calibration_datasets/requests/mmlu_pro_200.jsonl).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-30B-A3B",
        help="Hugging Face model id or local path to Qwen3-30B-A3B weights.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional limit on number of prompts to process.",
    )
    parser.add_argument(
        "--prompt-key",
        type=str,
        default="prompt",
        help="Key to read from JSON objects (default: prompt).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top-k experts to consider per token (default: 8).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenisation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "awq-int4", "autoround-int4", "autoround-int2"],
        default="none",
        help=(
            "Quantisation strategy for loading the model "
            "(none, awq-int4, autoround-int4, autoround-int2)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path to save JSON summary.",
    )
    parser.add_argument(
        "--generate-first-token",
        action="store_true",
        help="After prefill, generate exactly one token (to validate end-to-end flow).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Reserved for future batching support (currently 1).",
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

    prompts = load_prompts(
        args.dataset, prompt_key=args.prompt_key, limit=args.max_prompts
    )

    device = torch.device(args.device or "cuda:0")
    if device.type != "cuda":
        raise ValueError(
            "MoE models require a CUDA device. Please specify --device cuda:<id>."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine.")

    LOGGER.info("Using device %s", device)
    
    # Set memory optimization environment variable
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # Clear GPU cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        LOGGER.info("Cleared GPU cache before model loading")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True
    )

    device_str = (
        device.type if device.index is None else f"{device.type}:{device.index}"
    )

    if args.quantization == "awq-int4":
        try:
            # type: ignore[import-not-found]
            from autoawq.modeling import AutoAWQForCausalLM
        except ImportError as exc:
            raise ImportError(
                "autoawq is required for --quantization awq-int4. Install via `pip install autoawq`."
            ) from exc

        model = AutoAWQForCausalLM.from_quantized(
            args.model_id,
            device_map={"": device_str},
            trust_remote_code=True,
        )
    elif args.quantization in {"autoround-int4", "autoround-int2"}:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map={"": device_str},
        )
    else:
        # Use auto device_map to allow transformers to distribute across GPUs/CPU
        # Use auto dtype to let model decide the appropriate dtype
        # Try to use the specified device first, with memory limits
        device_index = device.index if device.index is not None else 0
        max_memory = {device_index: "40GiB"}
        if torch.cuda.device_count() > 1:
            # Reserve some memory on other GPUs
            for i in range(torch.cuda.device_count()):
                if i != device_index:
                    max_memory[i] = "5GiB"
        
        LOGGER.info("Loading model with max_memory: %s", max_memory)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map={"": device_str},
            low_cpu_mem_usage=True,
            max_memory=max_memory if len(max_memory) > 1 else None,
        )
        # Clear cache after loading
        torch.cuda.empty_cache()

    LOGGER.info("Loaded %s with quantization mode %s",
                args.model_id, args.quantization)
    
    # Log model info
    LOGGER.info("Model config - num_hidden_layers: %s", 
                getattr(model.config, 'num_hidden_layers', 'N/A'))
    LOGGER.info("Model device: %s", next(model.parameters()).device if list(model.parameters()) else 'N/A')

    LOGGER.info("Starting to collect expert statistics for %d prompts", len(prompts))
    counters = collect_expert_statistics(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        top_k=args.top_k,
        max_length=args.max_length,
        device=device,
        generate_first_token=args.generate_first_token,
    )

    summary = format_summary(counters)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOGGER.info("Saved summary to %s", args.output)
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
"""
pip install --upgrade autoawq transformers accelerate

python scripts/summarize_expert_activation.py \
  --dataset calibration_datasets/requests/mmlu_pro_200.jsonl \
  --model-id Qwen/Qwen3-30B-A3B \
  --device cuda:0 \
  --max-prompts 50 \
  --max-length 1 \
  --top-k 2 \
  --output activation_summary.json

# AutoRound INT4 example
python scripts/summarize_expert_activation.py \
  --dataset calibration_datasets/requests/mmlu_pro_200.jsonl \
  --model-id Intel/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound \
  --device cuda:0 \
  --quantization autoround-int4 \
  --max-prompts 50 \
  --max-length 1 \
  --top-k 2 \
  --output activation_summary_autoround.json
"""
