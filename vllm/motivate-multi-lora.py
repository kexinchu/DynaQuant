from typing import List

import time
import ray
import random

import vllm
from transformers import AutoTokenizer
from tests.utils import fork_new_process_for_each_test
from vllm.lora.request import LoRARequest

from ..utils import multi_gpu_test
from read_dataset import sample_requests

MODEL_PATH = "../models/llama2-7b"

REQUESTS = []

def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int, prompts: str) -> List[str]:
    sampling_params = vllm.SamplingParams(temperature=0,
                                          max_tokens=256,
                                          stop=["[/assistant]"])
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
        if lora_id else None)
    
    # Print the outputs.
    generated_texts: List[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts

# main function
def generate_and_test(llm, sql_lora_files, max_loras, warmup):
    if len(REQUESTS) == 0:
        return 
    # create adapters
    if warmup:
        print("lora adapter created")
        for lora_id in range(max_loras):
            do_sample(llm, sql_lora_files, lora_id=0, prompts=REQUESTS[0])

    # random test loras
    for prompts in REQUESTS:
        lora_id = random.randint(0, max_loras)
        do_sample(llm, sql_lora_files, lora_id=lora_id, prompts=prompts)

    print("removing lora")

@multi_gpu_test(num_gpus=1)
@fork_new_process_for_each_test
def test_llama_lora(sql_lora_files, max_loras):
    # initiate
    llm = vllm.LLM(MODEL_PATH,
                   enable_lora=True,
                   max_num_seqs=16,
                   max_loras=max_loras,
                   tensor_parallel_size=1,
                   enable_chunked_prefill=True)
    
    # warmup
    generate_and_test(llm, sql_lora_files, max_loras, True)

    # test_latency
    start_time = time.time()
    generate_and_test(llm, sql_lora_files, max_loras, False)
    latency = time.time() - start_time

    print("Single GPU with " + max_loras + ", the latency is: " + latency)


if __name__ == "__main__":
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    # read datasets
    requests = sample_requests("../../../datasets/ShareGPT_V3_unfiltered_cleaned_split.json", 2048, tokenizer, 256)
    for prompt, prompt_len, output_len in requests:
        REQUESTS.append(prompt)
    
    # test diff loras, lora-rank = 8
    for max_loras in [2, 4, 8, 16, 32, 64]:
        test_llama_lora(None, max_loras)