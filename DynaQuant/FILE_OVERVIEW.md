# DynaQuant 文件功能与依赖速查表

本文档汇总 `DynaQuant/` 仓库的主要代码、脚本和数据产出位置，并用模块间的调用关系帮助你区分「当前主线实现」与「历史/待清理脚本」。

## 项目概览
- **核心能力**：`dynaexq/` 提供 DynaExQ 动态专家量化运行时，支持 W4A16/W2A16 MoE 专家热度感知、异步权重交换与三层存储。
- **量化组件**：`quant/awq_w2/` 实现 2-bit AWQ 权重量化、激活采样与推理层；`tools/`、`scripts/` 中的量化/评估脚本围绕这些组件组织。
- **文档**：`README.md` 为整体说明，`FINAL_SUMMARY.md` 汇总里程碑，`DynaExQ.md` 着重运行时设计。

## 运行时核心 (`dynaexq/`)

### 架构要点
- `DynaExQRuntime` 组装监控、调度、内存管理、交换、预取和遥测模块，驱动实时精度切换。

```183:316:dynaexq/integration/hooks_base.py
    def __init__(self, config: Dict[str, Any]):
        ...
        self.monitor = ExpertMonitor(...)
        self.controller = PrecisionController(...)
        self.memory_manager = MemoryManager(...)
        self.swap_engine = SwapEngine(...)
        self.prefetch_planner = PrefetchPlanner(...)
        self.telemetry = TelemetryCollector(...)

    def on_router_output(...):
        self.monitor.update_batch(...)
        active_experts = [...]
        targets = self.controller.plan(...)
        current = {... self.memory_manager.get_residency ...}
        diff = self.controller.get_diff(targets, current)
        for expert in diff["upgrades"]:
            self.swap_engine.upgrade(...)
        ...
        self.prefetch_planner.lookahead(layer_id)
        self.prefetch_planner.update_pattern(layer_id, active_experts)
```

- `PrecisionController` 根据 EWMA 热度 + 滞后策略决定升级/降级，并生成差异列表交给 `SwapEngine`。

```60:171:dynaexq/runtime/controller.py
    def plan(...):
        scores = monitor.get_all_scores()
        layer_experts = ...
        for layer_id, experts in layer_experts.items():
            target_precision.update(
                self._plan_layer(layer_id, experts, scores)
            )
        return target_precision

    def _plan_layer(...):
        expert_scores.sort(...)
        for expert, score in expert_scores:
            if score > self.tau_h and w4_count < self.max_w4_slots:
                target = "W4"
            elif score < self.tau_c:
                target = "W2"
            elif current == "W4" ...:
                target = "W4"
            else:
                target = "W2"
            layer_targets[expert] = target
        self.current_precision.update(layer_targets)
        return layer_targets

    def get_diff(...):
        if target == "W4" and current == "W2":
            upgrades.append(expert)
        elif target == "W2" and current == "W4":
            downgrades.append(expert)
```

- `MemoryManager` 管理 HBM 热池/冷池 + DRAM/SSD，必要时触发 LRU 逐出并维护驻留映射。

```197:308:dynaexq/runtime/memmgr.py
    def reserve_hot(...):
        if self.hot_pool.contains(expert):
            self.hot_pool.touch(expert)
            return True
        residency = Residency(bitwidth="W4", location="HBM", bytes=nbytes)
        if self.hot_pool.allocate(...):
            self.residency_map[expert] = residency
            return True
        evicted = self.hot_pool.evict_lru()
        if evicted:
            self._demote_to_dram(evicted)
            if self.hot_pool.allocate(...):
                self.residency_map[expert] = residency
                return True
        return False

    def reserve_cold(...):
        ...
    def place(...):
        if residency.location == "HBM":
            ...
        elif residency.location == "DRAM":
            self.dram_experts[expert] = residency
```

- `SwapEngine` 通过后台优先队列线程执行权重升级/降级，复用 `MemoryManager` 与可选权重加载器，提供 `wait_ready` 以阻塞等待。

```138:270:dynaexq/runtime/swap_engine.py
    def upgrade(...):
        if expert in self.pending_swaps: ...
        residency = self.memory_manager.get_residency(expert)
        if residency and residency.bitwidth == "W4": return
        task = SwapTask(...)
        self.pending_swaps[expert] = PendingSwap(...)
        self.task_queue.put(task)
        self.upgrade_count += 1

    def wait_ready(...):
        pending = self.pending_swaps.get(expert)
        if not pending:
            residency = self.memory_manager.get_residency(expert)
            if residency and residency.location == "HBM":
                self.ready_before_use += 1
                return True
            self.miss_count += 1
            return False
        if pending.event.wait(timeout):
            self.ready_before_use += 1
            return True
        self.miss_count += 1
        return False
```

- 其它运行时代码：
  - `monitor.py`: `ExpertMonitor` 维护 EWMA 热度、epoch 轮换。
  - `prefetch.py`: `PrefetchPlanner` 根据热度/历史模式高优先级预取。
  - `telemetry.py`: `TelemetryCollector` 记录 swap/性能指标并可写 jsonl。
  - `ssd_index.py`: SSD 专家权重索引与 mmap 访问。
  - `runtime/types.py`: `ExpertID`、`Residency`、`SwapTask` 数据结构。

### 集成与测试
- `dynaexq/config.py`: `DynaExQConfig` 读取 `dynaexq/configs/default.yaml` 并导出 runtime 参数字典。
- `dynaexq/integration/hooks_base.py`: 提供抽象 Hook 基类及 `DynaExQRuntime` 主实现（见上）。
- `dynaexq/scripts/demo_simple.py`: 构造模拟 MoE 负载演示完整管线。
- `dynaexq/tests/`: 多个 pytest 脚本覆盖监控、控制器、内存管理、遥测及端到端流程。
  - `test_end_to_end.py` 内置 `ExpertWeightLoader`，支持从 W4/W2 模型权重目录加载或生成 mock 权重，验证 swap/驻留逻辑。

```47:196:dynaexq/tests/test_end_to_end.py
class ExpertWeightLoader:
    ...
    def load_expert_weights(...):
        if cache_key in self.weight_cache:
            return self.weight_cache[cache_key]
        if bitwidth == "W4":
            model_path = self.w4a16_path
        elif bitwidth == "W2":
            model_path = self.w2a16_path
        ...
        for file_path in safetensors_files:
            weight = self._load_expert_weights_from_file(...)
            if weight is not None:
                self.weight_cache[cache_key] = weight
                return weight
        logger.warning(... creating mock weight ...)
        mock_weight = torch.randn(...)
        self.weight_cache[cache_key] = mock_weight
        return mock_weight
```

## AWQ W2A16 量化模块 (`quant/awq_w2/`)
- `quantize.py`: 实现 2-bit 对称量化、解量化、误差计算、`QuantizationConfig`。
- `calib.py`: 采集激活、网格搜索 `alpha`、按层校准。
- `runtime.py`: `W2AWQLinear`/`W2AWQLinearFused` 运行时层，直接集成到 Transformers。
- `pack.py`: 2-bit 打包/解包。
- `__init__.py`: 导出高层 API。

```16:145:quant/awq_w2/runtime.py
class W2AWQLinear(nn.Module):
    ...
    def load_weights(...):
        if packed:
            self.weight_packed.copy_(weight_q)
        else:
            weight_packed = pack_2bit(weight_q)
            self.weight_packed.copy_(weight_packed)
        self.scale.copy_(scale)

    def forward(...):
        x = x.to(self.dtype)
        weight = self.unpack_and_dequantize()
        output = F.linear(x, weight, self.bias)
        return output
```

## 量化/评估脚本 (`scripts/`)
| 文件 | 作用 | 依赖 | 备注 |
| --- | --- | --- | --- |
| `calibrate.py` | 运行 AWQ W2 校准数据采集与量化元信息存储 | `quant.awq_w2`、HF Transformers | 主线 |
| `bench_eval.py` | 统一入口批量评估量化模型困惑度/准确率 | HuggingFace、`tools/eval_ppl.py` | 主线 |
| `evaluate_model.py` | 通用量化模型多数据集评估 | HF Transformers、`datasets` | 主线 |
| `load_w2a16_model.py` | 加载 W2A16 模型并冒烟推理 | `quant.awq_w2.W2AWQLinear` | 主线 |
| `motivation_test.py` | 旧版混合精度 MoE 实验，直接替换 experts | HuggingFace | 见 Legacy |
| `collect_expert_activation.py` | 统计 Qwen MoE 专家激活分布，输出 JSON | HF Transformers、Parquet | 分析辅助 |
| `analyze_expert_activation.py` | 读取激活 JSON，生成统计/图表 | `matplotlib`、`numpy` | 分析辅助 |
| `motivation_test.py` & `analyze_motivation_test.py` | 对比 FP16/INT4/混合专家激活 | HF Transformers、`matplotlib` | Legacy 分析流水线 |
| `evaluate_dynamic_quant.py` | **旧实现**：依赖缺失的 `dynamic_quant_moe` 类执行动态量化评估 | 未提供模块 | Legacy |
| `evaluate_parallel_dynamic_quant.py` | **旧实现**：多 GPU 版本，复用上方类 | 未提供模块 | Legacy |
| `evaluate_expert_activation.py` 系列 | 数据清洗、可视化脚本 | `numpy`、`matplotlib` | 分析辅助 |
| `example_workflow.sh` | 串联量化→评估→报告（shell） | 各 Python 脚本 | 主线 |
| `generate_all_calibration_datasets.sh` | 批量生成校准数据 | `calibration_datasets/` | 主线 |

## 工具 (`tools/`)
- `bench_mem.py`: 对指定模型三件套 Benchmark 磁盘/MEM/吞吐。

```81:152:tools/bench_mem.py
def benchmark_inference(...):
    model.eval()
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(...)
    ...
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.generate(...)
    tokens_per_second = max_new_tokens / avg_time
    return {...}
```

- `eval_ppl.py`: 针对文本集合计算困惑度、生成 JSON 报告。

## 配置、实验与数据
- `dynaexq/configs/default.yaml`: DynaExQ runtime 默认配置。
- `configs/quantization_config_schema.json`: W2/W4 量化配置 JSON-Schema。
- `experiments/config_ptq_qat.yaml`: 综合配置模板（PTQ/QAT/Serving/Benchmark）。
- `calibration_datasets/`: 量化校准样本文本（按数据集划分）。
- `benchmark_results/`, `plots/`, `telemetry.jsonl`: 工具/脚本运行生成的结果目录。

## 顶层文档与提示
- `README.md`: 全量项目介绍、环境依赖、典型流程。
- `FINAL_SUMMARY.md`: 清理总结、下一步计划（中/英混合）。
- `DynaExQ.md`: 运行时代码设计说明。
- `dynaexq_cursor_prompt_todo.md`: 旧的 Cursor 工作笔记。

## 旧版本 / 待清理脚本
| 路径 | 状态说明 | 背景 |
| --- | --- | --- |
| `scripts/evaluate_dynamic_quant.py` | **Legacy**：导入不存在的 `dynamic_quant_moe`，与现有 DynaExQ runtime 脱节 | 早期动态量化评估框架 |
| `scripts/evaluate_parallel_dynamic_quant.py` | **Legacy**：依赖上方缺失模块的多 GPU 评估器 | 旧版多机评估流水线 |
| `scripts/motivation_test.py` & `scripts/analyze_motivation_test.py` | **Legacy**：直接交换 FP16/INT4 experts，未使用新 runtime | 动态量化调研阶段实验 |
| `quant/AutoRound-W2A16.sh` | **Legacy**：面向 AutoRound CLI 的 W2A16 方案，流程未接入 DynaExQ | 兼容性脚本 |
| `dynaexq_cursor_prompt_todo.md` | **Notes**：历史开发待办 | 仅文档 |

> 判断标准：若脚本引用缺失模块、实现与 `dynaexq` 当前管线重复且未在最新 README/FINAL_SUMMARY 中提及，则标记为 Legacy，便于后续迁移或归档。

## 如何定位调用链
1. 推理集成：外部框架 → `DynaExQRuntime.on_router_output()` → `PrecisionController` / `MemoryManager` / `SwapEngine` → `Telemetry`。
2. 权重量化：数据准备 (`calibration_datasets/`) → `scripts/calibrate.py` (调用 `quant/awq_w2`) → 生成含 W2AWQLinear 的模型 → `load_w2a16_model.py`/`tools/bench_mem.py` 冒烟与性能评估。
3. 测试验证：`dynaexq/tests` 通过模拟权重（或真实 safetensors）驱动 runtime，覆盖热度跟踪、内存调度、预取与遥测。

建议在扩展代码前，依据上述调用关系查看对应模块及测试覆盖情况，确保不误用 Legacy 脚本。


