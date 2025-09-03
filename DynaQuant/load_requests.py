import os
import json
import random
from typing import List, Tuple
import csv
import pandas as pd

# default output length for LLM
default_max_output_length = 256
default_min_prompt_length = 4

# get current abs path
current_file_path = os.path.dirname(__file__)
# print(f"当前文件的绝对路径是: {current_file_path}")
request_root_dir = current_file_path + "/"

def read_chatGPT(file_path):
    # Load the dataset.
    requests = []
    session_id = 0
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        for data in dataset:
            context = data["conversations"][0]["value"]
            answer = data["conversations"][1]["value"]
            requests.append((context, answer, session_id))
            session_id += 1
            if session_id > 32:
                break
    return requests

def read_txt(file_path, max_num=-1):
    # 从txt文件中读取请求；每行一个request
    requests = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            req = line.strip()
            if req:
                requests.append((req, ""))
    return requests

def read_chatgpt_paraphrases(file_path, max_num=-1):
    requests = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            prompt = row.get("text", "").strip()
            paraphrase = row.get("paraphrases", "").strip()
            if prompt:
                requests.append((prompt, paraphrase, i))
    return requests

def read_multiturn_chat(file_path, max_num=-1):
    requests = []
    session_id = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            full_context = data.get("instruction", "").strip()
            final_answer = data.get("output", "").strip()

            # 如果context为空，跳过
            if not full_context:
                continue

            # 分割对话轮次
            turns = full_context.split("\n")
            current_context = ""

            # 处理每一轮对话
            for i in range(0, len(turns), 2):
                if i + 1 >= len(turns):  # 如果是最后一轮且没有回答，使用final_answer
                    if turns[i].startswith("Human:"):
                        current_context += turns[i] + "\n"
                        requests.append((current_context.strip(), final_answer, session_id))
                else:
                    # 添加当前轮次的对话
                    current_context += turns[i] + "\n"
                    if i + 1 < len(turns):
                        # 提取当前轮次的回答
                        current_answer = turns[i + 1].replace("Assistant:", "").strip()
                        requests.append((current_context.strip(), current_answer, session_id))
                        current_context += turns[i + 1] + "\n"

            session_id += 1

    return requests

def read_configurable_system_prompt_multitask(file_path, max_num=-1):
    requests = []
    df = pd.read_parquet(file_path)
    for idx, row in df.iterrows():
        context = str(row.get("system", "")).strip() + "\n" + str(row.get("prompt", "")).strip()
        answer = str(row.get("chosen", "")).strip()
        if context:
            requests.append((context, answer, idx))
    return requests

def load_jsonl_dataset(path, sample_n=1000, seed=42):
    # 读取jsonl文件
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 打乱并采样
    random.seed(seed)
    random.shuffle(lines)
    lines = lines[:sample_n]
    # 解析
    texts = []
    labels = []
    for line in lines:
        item = json.loads(line)

        # 判断是否有PII
        if "new_prompts" in path:
            texts.append(item["rewritten"])
            labels.append(item["label"])
        elif "after_level" in path or "warmup" in path or "actual" in path:
            texts.append(item["prompt"])
            labels.append(item["label"])
        else:
            texts.append(item["source_text"])
            bio_labels = item["mbert_bio_labels"]
            if isinstance(bio_labels, str):
                bio_labels = eval(bio_labels)  # 兼容字符串格式
            label = 1 if any(l != "O" for l in bio_labels) else 0
            labels.append(label)
    return texts, labels

if __name__ == "__main__":
    requests = read_chatGPT("/root/code/ShareGPT_V3_unfiltered_cleaned_split.json")
    print(len(requests))
    print(requests[0:5])

