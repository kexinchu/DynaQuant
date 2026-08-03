---
license: apache-2.0
base_model: Qwen/Qwen3-30B-A3B-Instruct-2507
pipeline_tag: text-generation
library_name: transformers
tags:
  - autoround
  - quantized
  - moe
---

# Qwen3-30B-A3B-Instruct-2507 W4A16 AutoRound

This is the mixed-precision INT4/W4A16 AutoRound checkpoint used in the DynaExQ paper experiments. It is derived from [`Qwen/Qwen3-30B-A3B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507).

The checkpoint uses symmetric expert-weight quantization with mixed higher-precision fallbacks recorded in `quantization_config.json`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Kris2017/Qwen3-30B-A3B-Instruct-2507-W4A16-AutoRound"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)
```

Code, experiment scripts, manifests, and paper sources are available in [DynaQuant](https://github.com/kexinchu/DynaQuant). The original model license and usage restrictions continue to apply.
