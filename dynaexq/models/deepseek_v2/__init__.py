# Copyright 2025 The HuggingFace Team. All rights reserved.
# DynaExq: direct exports (no _LazyModule) so package works outside transformers tree.

from .configuration_deepseek_v2 import DeepseekV2Config
from .modeling_deepseek_v2 import (
    DeepseekV2ForCausalLM,
    DeepseekV2ForSequenceClassification,
    DeepseekV2Model,
    DeepseekV2PreTrainedModel,
)

__all__ = [
    "DeepseekV2Config",
    "DeepseekV2ForCausalLM",
    "DeepseekV2ForSequenceClassification",
    "DeepseekV2Model",
    "DeepseekV2PreTrainedModel",
]
