from __future__ import annotations

import torch

from dynaexq.integration.generation_utils import (
    last_logit_only_kwargs,
    prepare_one_token_decode,
)


class _ModernHelper:
    def __init__(self):
        self.next_sequence_lengths: list[int | None] = []

    def prepare_inputs_for_generation(
        self,
        generated,
        next_sequence_length=None,
        **kwargs,
    ):
        self.next_sequence_lengths.append(next_sequence_length)
        length = next_sequence_length or generated.shape[-1]
        return {
            "input_ids": generated[:, -length:],
            "position_ids": torch.arange(generated.shape[-1])[None, :],
            **kwargs,
        }


class _LegacyHelper:
    def prepare_inputs_for_generation(
        self,
        generated,
        past_key_values=None,
        attention_mask=None,
        use_cache=True,
    ):
        return {
            "input_ids": generated,
            "position_ids": torch.arange(generated.shape[-1])[None, :],
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }


class _ModernLM(torch.nn.Module):
    def forward(self, input_ids, logits_to_keep=0):
        del input_ids, logits_to_keep


class _Wrapped(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _ModernLM()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def _decode(helper):
    generated = torch.tensor([[1, 2, 3]])
    next_token = generated[:, -1:]
    attention_mask = torch.ones_like(generated)
    past = object()
    prepared = prepare_one_token_decode(
        helper,
        generated=generated,
        next_token=next_token,
        attention_mask=attention_mask,
        past_key_values=past,
    )
    assert prepared["input_ids"].tolist() == [[3]]
    assert prepared["position_ids"].shape == (1, 1)
    assert prepared["past_key_values"] is past
    assert "next_sequence_length" not in prepared


def test_modern_generation_api_receives_one_token_length():
    helper = _ModernHelper()
    _decode(helper)
    assert helper.next_sequence_lengths == [1]


def test_legacy_generation_api_is_normalized_to_one_token():
    _decode(_LegacyHelper())


def test_last_logit_argument_is_found_through_wrapper():
    assert last_logit_only_kwargs(_ModernLM()) == {"logits_to_keep": 1}
    assert last_logit_only_kwargs(_Wrapped()) == {"logits_to_keep": 1}
