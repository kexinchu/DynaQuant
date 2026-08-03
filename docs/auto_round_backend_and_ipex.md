# 为什么用 intel/auto-round 量化的模型会用到 IPEX？

## 简短结论

**不是你的操作问题。**
intel/auto-round 支持多种**推理后端**；IPEX 是其中一种（面向 Intel CPU/XPU）。
后端是在**加载模型时**根据「目标设备」自动选的；选 CPU 时库会**优先用 IPEX**，所以会引入 IPEX。

## 1. AutoRound 和 IPEX 的关系

- **AutoRound**（intel/auto-round）负责的是**量化算法**和**权重量化**，保存的检查点格式（如 GPTQ）是**后端无关**的。
- **推理时**需要把量化权重交给某个「后端」来算：
  - **IPEX**（Intel Extension for PyTorch）：CPU/XPU，优先在 Intel 上使用。
  - **auto_round:torch**：纯 PyTorch，支持 CPU / CUDA / XPU。

所以：
「用 intel/auto-round 量化」≠「必须用 IPEX」。
IPEX 是**加载时**可选的推理后端之一，不是量化阶段强制的。

## 2. 后端是在「加载」时怎么被选中的？

加载 AutoRound 量化模型时（例如 `from_pretrained(..., device_map=...)`）：

1. HuggingFace 的 AutoRound quantizer 会调用
   `infer_target_device(device_map)` → 得到 `"cpu"` 或 `"cuda"` 等。
2. 然后调用 `convert_hf_model(model, target_device)`，在 AutoRound 的 `backend.py` 里：
   - 若 `backend == "auto"`，会在一批兼容的后端里按 **priority** 排序，选**优先级最高**的。
   - 对 **CPU**：`ipex_gptq_cpu` 的 priority=5，`auto_round:torch` 的 priority=0 → **会选 IPEX**。
   - 对 **CUDA**：IPEX 不支持 CUDA，只会选到 `auto_round:torch` 等支持 CUDA 的后端。

因此：
**只要加载时 `target_device` 被推断成 CPU（例如 `device_map="cpu"`），就会自动选到 IPEX。**
这是库的默认策略，不是用户配置错了。

## 3. 在我们脚本里具体发生了什么？

在 `evaluate_mmlu_perplexity_mixed.py` 的 `load_mixed_precision_model` 里：

- 为了先在 CPU 上做「FP16 + INT4 专家」的合并，INT4 模型是用
  `from_pretrained(int4_path, ..., device_map="cpu")` 加载的。
- 因此 `infer_target_device("cpu")` → `target_device="cpu"` → `convert_hf_model(..., "cpu")` → 在 CPU 兼容后端里选**优先级最高**的 → **IPEX**。
- 之后整机再 `.to(cuda:1)` 时，IPEX 的层只支持 CPU/XPU，就会出现「部分在 CPU、部分在 CUDA」的报错。

所以：**不是因为你用 intel/auto-round 量化错了，而是我们脚本在「加载 INT4」时用了 `device_map="cpu"`，触发了「选 IPEX 做 CPU 推理」这条路径。**

## 4. 你可以怎么做？

### 方案 A：希望用 CUDA 跑混合精度（推荐思路）

要让推理后端选成 **torch**（支持 CUDA），需要**加载 INT4 时**就让 `target_device` 变成 `"cuda"`，例如：

- 加载 INT4 时不用 `device_map="cpu"`，而用 `device_map="cuda:1"`（或你实际用的 GPU），这样
  `infer_target_device(device_map)` → `"cuda"` → 会选 `auto_round:torch` 等 CUDA 后端，**不会选 IPEX**。
- 合并时如果显存紧张，可以在选完后端、做完 `convert_hf_model` 之后，再把 INT4 模型 `.to("cpu")` 做合并，最后再把合并后的模型移到 CUDA（因为此时专家已经是 torch 后端，支持 CUDA）。

（脚本里可以在「CUDA 且要加载 INT4」时，对 INT4 的 `from_pretrained` 使用 `device_map=str(device)`，以强制走 CUDA 后端选择。）

### 方案 B：量化/导出时固定后端（治本）

若你在**量化或导出**阶段能改配置，可以在保存的 `quantization_config` 里**显式指定后端**，例如：

- `backend="auto_round:torch"`（或你使用的库/CLI 里对应的选项），这样即使用 `device_map="cpu"` 加载，也会用 torch 后端而不是 IPEX。

具体参数名要看你用的 AutoRound 版本（例如 `autoround run_quant` / 导出脚本）的文档。

### 方案 C：接受只用 CPU，且环境匹配 IPEX

若你坚持用 IPEX 后端（例如在 Intel CPU 上跑）：

- 需要满足 IPEX 的要求：**PyTorch 2.8.x** 和 **intel-extension-for-pytorch**。
- 当前若用 PyTorch 2.10，会和 IPEX 的 ABI 不兼容，需要降级 PyTorch 或等 IPEX 支持新版本。

## 5. 为什么需要 “repacking to CPU/XPU format”？不能直接 `.to('cpu')` 吗？

**原因：repacking 改的是权重的「内存布局」，不是设备位置。**

- **`.to('cpu')` 只做一件事**：把 tensor 挪到 CPU 设备上，**不改变**数据在内存里的排布方式。
- **量化权重在 checkpoint 里**是某一种「通用」打包格式（例如 GPTQ 的 block 排布），适合保存和跨后端。
- **IPEX 的 INT4 推理内核**要求权重的另一种排布（`weight_only_qlinear_prepack_int4` 等），才能在 CPU/XPU 上高效做 INT4 matmul。
- 所以 **repacking** = 对每个量化层调用 `layer.post_init()`，内部用 `torch.ops.ipex_prepack.weight_only_qlinear_prepack_int4(...)` 把 `qweight + scales + zero_points` 转成 IPEX 需要的 **op_context**（另一种内存格式）。
  若跳过这一步，只做 `.to('cpu')`，权重仍是「保存格式」，IPEX 内核要么算错要么报错。

**结论**：不能省掉 repacking 只靠 `.to('cpu')`；repacking 是「为 IPEX 内核准备数据布局」，不是「把数据搬到 CPU」。

**若想避免 repacking**：让加载 INT4 时走 **CUDA 后端**（见上文方案 A），即 `device_map="cuda:0"`，这样会选 `auto_round:torch` 而不是 IPEX，就不会触发这 1.8 万多次的 CPU/XPU repacking（但合并时显存占用会更高）。

## 6. 总结

| 问题 | 答案 |
|------|------|
| 为什么用 intel/auto-round 量化会引入 IPEX？ | 量化本身不绑定 IPEX；是**加载时**按 `target_device`（由 `device_map` 等推断）选后端，CPU 下库**优先选 IPEX**。 |
| 是我操作不对吗？ | 不是。这是 AutoRound 的默认优先级设计（CPU 上 IPEX priority 高于 torch）。 |
| 想在 CUDA 上跑混合精度怎么办？ | 加载 INT4 时让 `device_map` 指向 CUDA，使后端选为 torch；或量化/导出时显式设 `backend="auto_round:torch"`。 |
| 为什么必须 repacking，不能直接 .to('cpu')？ | repacking 是把权重的**内存布局**改成 IPEX 内核要求的格式；.to('cpu') 只改**设备**，不改布局，无法替代。 |
