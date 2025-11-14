#!/usr/bin/env python3
"""Boot an LLM with DynaExQ-managed dual precision experts and run a query."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from dynaexq.runtime import (
    Bitwidth,
    DualPrecisionWeights,
    ExpertID,
    ExpertMonitor,
    MemoryManager,
    PrecisionController,
    PrefetchPlanner,
    SwapConfig,
    SwapEngine,
    random_precision_selector,
)
from dynaexq.runtime.memmgr import PoolConfig
from dynaexq.runtime.weights import InMemoryWeightStore


LOGGER = logging.getLogger("dynaexq.infer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w4", required=True,
                        help="Path to W4/FP16 weights (file or directory)")
    parser.add_argument(
        "--w2",
        help="Optional path to W2 weights (file or directory) to preload into memory",
    )
    parser.add_argument(
        "--config",
        help="Model configuration directory (defaults to --w4)",
    )
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer directory (defaults to --w4)",
    )
    parser.add_argument(
        "--prompt",
        default="你好",
        help="Prompt to run through the model",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Target device (auto|cpu|cuda|cuda:0|...).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Torch dtype to cast model parameters to (auto|float32|float16|bfloat16)",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling during generation (defaults to greedy).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (only used when --do-sample).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (only used when --do-sample).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of remote code for custom model implementations.",
    )
    parser.add_argument(
        "--downgrade-prob",
        type=float,
        default=0.5,
        help="Probability of assigning an expert to the lower precision when bootstrapping.",
    )
    parser.add_argument(
        "--hot-slots",
        type=int,
        default=8,
        help="Maximum number of experts kept in high precision (W4) simultaneously.",
    )
    parser.add_argument(
        "--no-low-precision",
        action="store_true",
        help="Disable random downgrades (keep all experts at high precision).",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    normalized = dtype_arg.lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype argument: {dtype_arg}")


def load_dual_precision_repository(args: argparse.Namespace) -> DualPrecisionWeights:
    LOGGER.info("Loading weights into memory (W4: %s, W2: %s)",
                args.w4, args.w2 or "<none>")
    return DualPrecisionWeights.from_files(
        args.w4,
        args.w2,
        prefer_non_expert=Bitwidth.W4,
    )


def build_model(
    repo: DualPrecisionWeights,
    args: argparse.Namespace,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.device, torch.dtype]:
    config_path = args.config or args.w4
    tokenizer_path = args.tokenizer or args.w4

    LOGGER.info("Loading config from %s", config_path)
    config = AutoConfig.from_pretrained(
        config_path,
        trust_remote_code=args.trust_remote_code,
    )

    LOGGER.info("Instantiating model from config")
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=args.trust_remote_code,
    )

    selector = None
    if not args.no_low_precision and args.w2:
        selector = random_precision_selector(args.downgrade_prob)

    state_dict = repo.materialize_state_dict(
        expert_precision=selector,
    )
    LOGGER.info("Applying in-memory state_dict (%d tensors)", len(state_dict))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        LOGGER.warning("Missing keys during load: %s", missing)
    if unexpected:
        LOGGER.warning("Unexpected keys during load: %s", unexpected)

    LOGGER.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    LOGGER.info("Moving model to %s (%s)", device, dtype)
    model.to(device=device, dtype=dtype)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer, device, dtype


def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    args: argparse.Namespace,
) -> str:
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generation_kwargs.update({
            "temperature": args.temperature,
            "top_p": args.top_p,
        })

    LOGGER.info("Generating completion for prompt: %s", args.prompt)
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def warm_up_experts(
    repo: DualPrecisionWeights,
    args: argparse.Namespace,
) -> SwapEngine:
    LOGGER.info("Initializing DynaExQ runtime components")
    monitor = ExpertMonitor()
    controller = PrecisionController(
        tau_h=0.65,
        tau_c=0.45,
        max_w4_slots=args.hot_slots,
    )

    pools = PoolConfig(
        hot_capacity_bytes=8 * 1024 * 1024 * args.hot_slots,
        cold_capacity_bytes=32 * 1024 * 1024,
        transient_capacity_bytes=4 * 1024 * 1024,
    )
    memory = MemoryManager(pools)

    store = InMemoryWeightStore(repo, Bitwidth.W4)
    swap_engine = SwapEngine(memory, store, SwapConfig(max_workers=4))
    prefetch = PrefetchPlanner(swap_engine, controller, monitor)

    LOGGER.info("Simulating initial routing to stage experts")
    available_experts = list(repo.indices())
    expert_ids = sorted(
        {index.expert for index in available_experts if index.expert is not None},
        key=lambda e: (e.layer, e.idx),
    )

    for expert in expert_ids[: args.hot_slots]:
        swap_engine.upgrade(expert)
        swap_engine.wait_ready(expert)

    if len(expert_ids) > args.hot_slots:
        for expert in expert_ids[args.hot_slots: args.hot_slots * 2]:
            swap_engine.downgrade(expert)
            swap_engine.wait_ready(expert)

    LOGGER.info("Warm-up complete (%d experts staged)", len(expert_ids))
    return swap_engine


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")
    args = parse_args()

    repo = load_dual_precision_repository(args)
    swap_engine = warm_up_experts(repo, args)
    model, tokenizer, device, _ = build_model(repo, args)
    response = run_inference(model, tokenizer, device, args)

    print("Prompt:", args.prompt)
    print("Response:", response)

    swap_engine.close()


if __name__ == "__main__":
    main()
