# DynaExQ 快速入门指南

> **DynaExQ**: 动态专家量化运行时 - 工作负载感知的MoE推理系统

---

## 📦 安装

### 1. 安装依赖

```bash
cd /workspace/DynaQuant/DynaQuant

# 安装基础依赖
pip install pyyaml

# (可选) 安装完整依赖
pip install -r dynaexq/requirements.txt
```

### 2. 验证安装

```bash
# 运行单元测试
bash dynaexq/scripts/run_tests.sh

# 运行演示
python dynaexq/scripts/demo_simple.py
```

---

## 🚀 快速开始

### 方式1：使用完整运行时（推荐）

```python
from dynaexq.config import load_config
from dynaexq.integration.hooks_base import DynaExQRuntime
import numpy as np

# 1. 加载配置
config = load_config("dynaexq/configs/default.yaml")

# 2. 创建运行时
runtime = DynaExQRuntime(config.to_dict())
runtime.start()

# 3. 在MoE推理循环中集成
num_layers = 8
batch_size = 4
top_k = 2

for batch_idx in range(100):
    for layer_id in range(num_layers):
        # 模拟路由器输出
        topk_indices = np.random.randint(0, 32, size=(batch_size, top_k))
        logits = np.random.rand(batch_size, top_k)
        
        # DynaExQ钩子
        runtime.on_layer_start(layer_id)
        runtime.on_router_output(layer_id, topk_indices, logits)
        runtime.ensure_experts_ready(layer_id, topk_indices)
        
        # 这里调用实际的MoE layer forward
        # output = moe_layer(input, topk_indices)
        
        runtime.on_layer_end(layer_id)

# 4. 获取统计信息
stats = runtime.get_statistics()
print(f"Ready ratio: {stats['swap_engine']['ready_ratio']:.2%}")
print(f"W4 experts: {stats['controller']['current_w4_experts']}")

runtime.stop()
```

### 方式2：分别使用各个组件

```python
from dynaexq.runtime.monitor import ExpertMonitor
from dynaexq.runtime.controller import PrecisionController
from dynaexq.runtime.memmgr import MemoryManager
from dynaexq.runtime.types import ExpertID
import numpy as np

# 创建组件
monitor = ExpertMonitor(
    ewma_alpha=0.2,
    epoch_duration=300.0,
    num_layers=8,
    num_experts_per_layer=32
)

controller = PrecisionController(
    tau_h=0.65,
    tau_c=0.45,
    max_w4_slots=16,
    num_layers=8,
    num_experts_per_layer=32
)

# 更新热度
topk_idx = np.array([[0, 1], [2, 3]])
logits = np.array([[0.7, 0.3], [0.6, 0.4]])
monitor.update_batch(layer=0, topk_idx=topk_idx, logits=logits)

# 触发epoch更新
monitor.epoch_tick()

# 规划精度
active_experts = [ExpertID(layer=0, idx=i) for i in range(8)]
targets = controller.plan(active_experts, monitor)

print(f"Expert 0 target: {targets[ExpertID(layer=0, idx=0)]}")
```

---

## ⚙️ 配置

### 修改配置文件

编辑 `dynaexq/configs/default.yaml`:

```yaml
# 调整阈值
thresholds:
  tau_h: 0.70  # 提高热阈值，更严格的W4升级
  tau_c: 0.40  # 降低冷阈值，更积极的W2降级

# 调整内存池
pool:
  hot_w4_slots: 20     # 增加W4专家容量
  hot_pool_gb: 15.0    # 增加热池大小
  cold_pool_gb: 8.0

# 调整预取策略
prefetch:
  lookahead_layers: 2  # 提前2层预取
  prefetch_top_k: 12   # 预取更多专家
```

### 代码中覆盖配置

```python
config = load_config("dynaexq/configs/default.yaml")

# 覆盖部分配置
config.update({
    "thresholds.tau_h": 0.70,
    "pool.hot_w4_slots": 20,
    "hotness.window": 600.0  # 10分钟epoch
})

runtime = DynaExQRuntime(config.to_dict())
```

---

## 📊 监控和调试

### 1. 实时统计

```python
# 获取所有统计
stats = runtime.get_statistics()

# 监控器统计
print("Monitor:")
print(f"  Epoch: {stats['monitor']['current_epoch']}")
print(f"  Tracked experts: {stats['monitor']['total_experts_tracked']}")
print(f"  Mean hotness: {stats['monitor']['mean_hotness']:.4f}")

# 控制器统计
print("Controller:")
print(f"  W4 experts: {stats['controller']['current_w4_experts']}")
print(f"  W2 experts: {stats['controller']['current_w2_experts']}")

# 内存统计
print("Memory:")
print(f"  HBM pressure: {stats['memory']['hbm_pressure']:.2%}")
print(f"  Evictions: {stats['memory']['eviction_count']}")

# 交换引擎统计
print("Swap Engine:")
print(f"  Upgrades: {stats['swap_engine']['upgrade_count']}")
print(f"  Ready ratio: {stats['swap_engine']['ready_ratio']:.2%}")
```

### 2. 导出遥测数据

```python
# 导出详细遥测
runtime.telemetry.export_summary("telemetry_summary.json")

# 查看事件
events = runtime.swap_engine.get_telemetry()
for event in events[-10:]:  # 最近10个事件
    print(f"{event.event_type}: {event.expert} in {event.duration_ms:.1f}ms")
```

### 3. 调试日志

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 或只启用特定模块
logging.getLogger("dynaexq.runtime.monitor").setLevel(logging.DEBUG)
logging.getLogger("dynaexq.runtime.swap_engine").setLevel(logging.INFO)
```

---

## 🧪 测试

### 运行所有测试

```bash
cd /workspace/DynaQuant/DynaQuant
bash dynaexq/scripts/run_tests.sh
```

### 运行特定测试

```bash
# 测试监控器
python -m unittest dynaexq.tests.test_monitor

# 测试控制器
python -m unittest dynaexq.tests.test_controller

# 测试内存管理
python -m unittest dynaexq.tests.test_memmgr

# 测试特定用例
python -m unittest dynaexq.tests.test_monitor.TestExpertMonitor.test_ewma_decay
```

---

## 🔧 性能调优

### 1. 调整阈值以平衡精度和吞吐

```yaml
# 更高精度（更多W4专家）
thresholds:
  tau_h: 0.55  # 更容易升级到W4
  tau_c: 0.35
  
# 更高吞吐（更多W2专家）
thresholds:
  tau_h: 0.75  # 更难升级到W4
  tau_c: 0.55
```

### 2. 优化内存使用

```yaml
pool:
  hot_w4_slots: 12     # 减少W4容量以节省HBM
  hot_pool_gb: 8.0
  transient_mb: 1024   # 减少transient buffer
```

### 3. 提高ready-before-use率

```yaml
prefetch:
  lookahead_layers: 2  # 增加预取距离
  prefetch_top_k: 16   # 增加预取专家数

streams:
  memcpy_h2d: 4        # 增加传输streams
```

---

## 📝 常见问题

### Q1: Ready ratio太低（< 99%）

**A:** 增加预取或降低交换频率：
```yaml
prefetch:
  lookahead_layers: 2
  prefetch_top_k: 12

thresholds:
  tau_h: 0.70  # 增大迟滞带宽
  tau_c: 0.40
```

### Q2: HBM压力过高

**A:** 减少hot pool大小或降低max_w4_slots：
```yaml
pool:
  hot_w4_slots: 12  # 减少每层W4专家数
  hot_pool_gb: 8.0
```

### Q3: 专家交换太频繁

**A:** 增大epoch window或增大迟滞带宽：
```yaml
hotness:
  window: 600  # 10分钟epoch

thresholds:
  tau_h: 0.70
  tau_c: 0.35  # 增大 tau_h - tau_c
```

---

## 📚 更多资源

- **完整文档**: [`dynaexq/README.md`](dynaexq/README.md)
- **API参考**: 查看各模块的docstring
- **示例代码**: `dynaexq/scripts/demo_simple.py`
- **测试用例**: `dynaexq/tests/`

---

## 🎯 下一步

1. ✅ 运行demo验证安装
2. ✅ 阅读完整文档了解架构
3. ✅ 根据你的模型调整配置
4. ✅ 集成到你的推理框架
5. ✅ 监控性能指标并调优

**祝你使用愉快！** 🚀

如有问题，请查看 [`dynaexq/README.md`](dynaexq/README.md) 或提交 Issue。

