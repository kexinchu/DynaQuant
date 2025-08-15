# test_pruned_qwen.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_dir = "/dev/shm/Qwen3-30B-A3B"  # 修改成你的新目录

print(f"[load] Loading tokenizer from {model_dir}...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

print(f"[load] Loading model from {model_dir}...")
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,  # 如果是 FP8/BF16 都可以改
    device_map="auto"
)

prompt = "Hello, this is a quick test for the pruned Qwen model."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("[gen] Running generation...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False
    )

print("[out] " + tokenizer.decode(output_ids[0], skip_special_tokens=True))
