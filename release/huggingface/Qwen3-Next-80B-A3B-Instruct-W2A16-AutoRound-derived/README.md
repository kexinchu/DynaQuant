---
license: apache-2.0
base_model: Kris2017/Qwen3-Next-80B-A3B-Instruct-W4A16-AutoRound
pipeline_tag: text-generation
library_name: transformers
tags:
  - autoround
  - quantized
  - moe
  - int2
---

# Qwen3-Next-80B-A3B-Instruct W2A16 AutoRound-derived

This experimental checkpoint was deterministically derived from the DynaExQ W4A16 AutoRound checkpoint by converting eligible expert weights to packed INT2 while preserving the mixed-precision fallbacks described in `quantization_config.json`. The original base model is [`Qwen/Qwen3-Next-80B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct).

The repository includes the conversion provenance and all indexed `safetensors` shards. This is a research artifact; backend support for its mixed INT2 packing must be verified before deployment.

Code, conversion scripts, verification manifests, experiment results, and paper sources are available in [DynaQuant](https://github.com/kexinchu/DynaQuant). The original model license and usage restrictions continue to apply.
