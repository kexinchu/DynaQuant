"""Phi-3.5-MoE model package (Phase 3 port from dynaexq_new)."""

from .configuration_phimoe import PhimoeConfig
from .modeling_phimoe import (
    PhimoeBlockSparseTop2MLP,
    PhimoeForCausalLM,
    PhimoeModel,
    PhimoePreTrainedModel,
    PhimoeSparseMoeBlock,
)

__all__ = [
    "PhimoeConfig",
    "PhimoeBlockSparseTop2MLP",
    "PhimoeForCausalLM",
    "PhimoeModel",
    "PhimoePreTrainedModel",
    "PhimoeSparseMoeBlock",
]
