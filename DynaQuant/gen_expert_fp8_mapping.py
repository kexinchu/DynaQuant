#!/usr/bin/env python3
# python3 gen_expert_fp8_mapping.py /dcar-vepfs-trans-models/Qwen3-235B-A22B-FP8/model.safetensors.index.json --precision FP8 --indent 4 > ./sglang-0.4.7/mixed_precision_config.yaml
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
    ap.add_argument("index", help="path to model.safetensors.index.json")
    ap.add_argument("--precision", default="FP8", help='value on the right, e.g. FP8 / gptq_int4')
    ap.add_argument("--indent", type=int, default=0, help="spaces to indent each line")
    args = ap.parse_args()

    print("# 混合精度配置文件\n\
# 支持先加载低精度模型，再替换指定层为高精度\n\
# 支持safetensors索引文件和默认策略\n\
# 兼容不同量化精度的权重名称差异\n\
\n\
mixed_precision:\n\
  # 基础模型路径（通常是低精度模型，用于初始加载以避免OOM）\n\
  base_model_path: \"/dcar-vepfs-trans-models/Qwen3-235B-A22B-FP8\"\n\
\n\
  # 不同精度的模型路径\n\
  fp16_path: \"/dcar-vepfs-trans-models/Qwen3-235B-A22B\"\n\
  fp8_path: \"/dcar-vepfs-trans-models/Qwen3-235B-A22B-FP8\"\n\
  gptq_int4_path: \"/dcar-vepfs-trans-models/Qwen3-235B-A22B-GPTQ-Int4\"\n\
\n\
  # 权重映射配置 - 基于实际权重名称\n\
  # 系统会自动处理不同量化精度的权重名称差异\n\
  weight_mapping:\n\
    # 专家层使用GPTQ-Int4（低精度，节省内存）\n\
    # 注意：权重名称需要与实际模型结构匹配")

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

    print("\n\
# 加载策略配置\n\
loading_strategy:\n\
  # 是否启用混合精度加载\n\
  enabled: true\n\
\n\
  # 基础模型精度（用于初始加载）\n\
  base_precision: \"fp8\"\n\
\n\
  # 内存优化策略\n\
  memory_optimization:\n\
    # 是否启用压缩权重缓存\n\
    enable_cache: true\n\
    # 缓存大小限制（MB）\n\
    cache_size_limit: 1024\n\
    # 是否启用动态反量化\n\
    enable_dynamic_dequantization: true\n\
\n\
# 推理配置\n\
inference:\n\
  max_seq_length: 4096\n\
  max_batch_size: 32\n\
  dtype: \"bfloat16\"\n\
  device_map: \"auto\"\n\
\n\
  # 混合精度特定配置\n\
  mixed_precision:\n\
    use_cache: true  # 是否使用权重缓存\n\
    cache_size_mb: 1024  # 缓存大小限制\n\
    enable_dynamic_dequantization: true  # 启用动态反量化\n\
\n\
# 服务器配置\n\
server:\n\
  host: \"127.0.0.1\"\n\
  port: 8080\n\
  max_workers: 4\n\
")

if __name__ == "__main__":
    main()
