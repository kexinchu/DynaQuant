from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "../Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))