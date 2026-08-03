---
license: apache-2.0
base_model: Qwen/Qwen3-Next-80B-A3B-Instruct
pipeline_tag: text-generation
library_name: transformers
tags:
  - autoround
  - quantized
  - moe
---

# Qwen3-Next-80B-A3B-Instruct W4A16 AutoRound

This is the verified mixed-precision INT4/W4A16 AutoRound checkpoint used in the DynaExQ paper experiments. It is derived from [`Qwen/Qwen3-Next-80B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct).

The archive includes 11 primary `safetensors` shards, one auxiliary tensor shard, the weight index, tokenizer assets, quantization configuration, and provenance information.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Kris2017/Qwen3-Next-80B-A3B-Instruct-W4A16-AutoRound"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)
```

Code, experiment scripts, manifests, and paper sources are available in [DynaQuant](https://github.com/kexinchu/DynaQuant). The original model license and usage restrictions continue to apply.
