import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
import transformers
from auto_round import AutoRound

# model_name = "/workspace/Models/Qwen3-30B-A3B-Instruct-2507"
model_name = "/workspace/Models/Qwen3-Next-80B-A3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="cpu", torch_dtype="auto")

tokenizer = AutoTokenizer.from_pretrained(model_name)

layer_config = {}
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        # if "mlp.gate" in n:  # vllm only support 16 bit for this layer
        #     layer_config[n] = {"bits": 16}
        # elif isinstance(m, torch.nn.Linear) and (not "expert" in n or "shared_experts" in n) and n != "lm_head":
        #     layer_config[n] = {"bits": 8, "group_size": 128}
        # elif "expert" in n and "shared_experts" not in n:
        #     layer_config[n] = {"bits": 2}
        #     print(n, 2)
        if "expert" in n and "shared_experts" not in n:
            layer_config[n] = {"bits": 2}
            print(n, 2)
        elif n != "lm_head":
            layer_config[n] = {"bits": 8}
            print(n, 8)

autoround = AutoRound(model, tokenizer, iters=0,
                      group_size=64, layer_config=layer_config)
output_dir = "/workspace/Models/Qwen3-80B-A3B-Instruct-int2-mixed-AutoRound"
autoround.quantize_and_save(output_dir)
