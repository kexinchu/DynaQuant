"""Version-safe helpers for manual cached autoregressive decoding."""

from __future__ import annotations

import inspect

import torch


def prepare_one_token_decode(
    model,
    *,
    generated: torch.Tensor,
    next_token: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
) -> dict:
    """Prepare exactly one uncached token across Transformers API versions.

    Transformers 5.x no longer infers the uncached suffix length from the
    cache. Without ``next_sequence_length=1``, passing the accumulated token
    sequence replays the full prefix at every decode step. Older and custom
    helpers may return full-length position tensors as well, so the result is
    normalized and checked before it reaches the timed forward.
    """
    if past_key_values is None:
        raise RuntimeError(
            "checkpoint returned no KV cache; one-token decode is unavailable"
        )
    try:
        parameters = inspect.signature(
            model.prepare_inputs_for_generation
        ).parameters
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "cannot inspect prepare_inputs_for_generation"
        ) from error
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs = {
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
        "use_cache": True,
    }
    if "next_sequence_length" in parameters or accepts_kwargs:
        kwargs["next_sequence_length"] = 1
    prepared = dict(
        model.prepare_inputs_for_generation(
            generated,
            **kwargs,
        )
    )
    # A permissive custom helper may forward this control-only argument.
    prepared.pop("next_sequence_length", None)
    input_ids = prepared.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.shape[-1] != 1:
        prepared["input_ids"] = next_token
        prepared.pop("inputs_embeds", None)
    for name in ("position_ids", "cache_position", "token_type_ids"):
        value = prepared.get(name)
        if isinstance(value, torch.Tensor) and value.shape[-1] != 1:
            prepared[name] = value[..., -1:]
    if prepared["input_ids"].shape[-1] != 1:
        raise RuntimeError("decode preparation did not produce one token")
    prepared["past_key_values"] = past_key_values
    prepared["attention_mask"] = attention_mask
    prepared["use_cache"] = True
    return prepared


def last_logit_only_kwargs(model) -> dict[str, int]:
    """Return the causal-LM argument that suppresses full-sequence logits."""
    candidate = model
    visited: set[int] = set()
    while candidate is not None and id(candidate) not in visited:
        visited.add(id(candidate))
        try:
            parameters = inspect.signature(candidate.forward).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "logits_to_keep" in parameters:
            return {"logits_to_keep": 1}
        if "num_logits_to_keep" in parameters:
            return {"num_logits_to_keep": 1}
        candidate = getattr(candidate, "model", None)
    raise RuntimeError(
        "model cannot request last-position-only logits; refusing a "
        "memory-ambiguous generation run"
    )
