#!/usr/bin/env python3
# gen_expert_lines.py
import argparse, json, re, sys

def iter_param_names(index_path: str):
    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    if isinstance(idx, dict) and "weight_map" in idx:
        # 常见的 safetensors index 结构
        for name in idx["weight_map"].keys():
            yield name
    elif isinstance(idx, dict) and "tensors" in idx:
        # 有些工具导出的结构
        for t in idx["tensors"]:
            name = t.get("name")
            if name:
                yield name
    else:
        raise ValueError("Unrecognized index schema: expect 'weight_map' or 'tensors'")

def main():
    ap = argparse.ArgumentParser("Generate expert precision lines from safetensors index")
    ap.add_argument("index", help="path to model.safetensors.index.j--ison")
    ap.add_argument("--precision", default="FP8", help='value on the right, e.g. FP8 / gptq_int4')
    ap.add_argument("--indent", type=int, default=0, help="spaces to indent each line")
    args = ap.parse_args()

    pat = re.compile(r"experts?")  # 匹配 expert 或 experts
    count = 0
    for name in sorted(set(iter_param_names(args.index))):
        if not pat.search(name):
            continue
        # 只要 expert 下的常见权重；如果你想更严格，可加 endswith(".weight")
        print(" " * args.indent + f"\"{name}\": \"{args.precision}\"")
        count += 1

    if count == 0:
        print("# No expert params found in index.", file=sys.stderr)

if __name__ == "__main__":
    main()
