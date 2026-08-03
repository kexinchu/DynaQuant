---
license: mit
base_model: microsoft/Phi-3.5-MoE-instruct
pipeline_tag: text-generation
library_name: transformers
tags:
  - autoround
  - quantized
  - moe
---

# Phi-3.5-MoE-instruct W4A16 AutoRound

This is the verified W4A16 AutoRound checkpoint used in the DynaExQ paper experiments. It is derived from [`microsoft/Phi-3.5-MoE-instruct`](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct).

The checkpoint contains 11 `safetensors` shards, its weight index, tokenizer files, PhiMoE modeling code, and quantization provenance. Loading custom PhiMoE code may require `trust_remote_code=True`; review remote code before enabling it.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Kris2017/Phi-3.5-MoE-instruct-W4A16-AutoRound"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
```

Code, experiment scripts, manifests, and paper sources are available in [DynaQuant](https://github.com/kexinchu/DynaQuant). The original model license and usage restrictions continue to apply.
