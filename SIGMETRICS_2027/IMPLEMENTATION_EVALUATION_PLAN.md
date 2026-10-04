# DynaByte 代码实现与评测计划（目标 9/10）

> 本文件是 `STORYLINE_WRITING_OUTLINE.md` 第 8 节所推迟的 Evaluation 规划，以及支撑它的实现规划。
> 上游依据：`03_design.tex`（byte lease / exchange 机制）、`DESIGN_NOVELTY_REWRITE_PLAN.md` 第 4 节（9/10 所需证据）、`STORYLINE_RESTRUCTURING.md` 第 7 节（实验组织）与 0.6 节（模拟审稿意见）。
> 快照时间：2026-10-03。所有硬件与 checkpoint 状态在开跑前需重新核验。

---

## 0. 执行摘要

### 0.1 本计划要解决的核心矛盾

Design 已经从「monitor + valuator + coordinator + executor」的通用组件叙事，重写为一个具体的分配原语：**byte lease 的可行交换（feasible exchange），其价值由同一条 deadline replay 的前后差计算，质量风险只作为可行性账目**。

但评测部分还没有跟上这次重写。因此 9/10 的瓶颈不在再多写一页 design，而在三件事：

1. **机制必须被单独测量。** 新颖性在「counterfactual 估值 + 显式 donor」，所以必须有一个消融把它换成独立估值和固定切分，并显示它真的会选错 donor 或重复计价。
2. **必要性必须对抗强基线。** 共享同一预算、充分调优的顺序优化和静态联合配置是生死线；`naive composition` 这种自己和自己冲突的对照不能当 headline。
3. **模型必须被验证。** 这是 POMACS，$\widehat L$ 和 $\widehat R$ 的预测误差本身就是可发表的结果；只报一个 speedup 会被按 systems 短文标准打低分。

### 0.2 三个必须在动手前接受的事实

| 事实 | 证据 | 对计划的影响 |
|---|---|---|
| 当前 `05_evaluation.tex` 与 design 不同构 | 该文件仍以 Phase I/II/III、$\Phi$、horizon $S$、$n_{\mathrm{hi}}$ 组织，H1--H4 全部是关于 $S^\star$ 和 pin 配额的断言，不含 lease、donor、exchange、counterfactual | §4 给出 claim 重映射表，整节需重写而不是补表格 |
| Design 缺少 use predictor 的定义 | $\omega_j$、$d_j$、$\hat U(S)$ 在 `03_design.tex` 中被使用，但预测器只在 evaluation 的消融里以 "pregate-or-frequency predictor" 一句出现 | 预测器升级为一等实现模块（§3.3.4），并在 §5 E6 中单独测准确率；design 需补一小段 |
| 本机没有 H20 / Ascend 910B，MoE checkpoint 已不在盘上 | `nvidia-smi` 只有 2× RTX A6000 48GB（GPU0 被他人 35.5GB/100% 占用）；`/home/kec23008/Models/` 仅剩 granite/piiranha；根分区 231GB 可用，`/dev/shm` 499GB，内存 1007GB | 模型集收缩到 3（+1 可选）；$C$ 的变化用受控带宽节流扫描代替三设备，并在正文明确标为 emulation；正文的三设备承诺必须删除或降级 |

### 0.3 从 3.8/10 到 9/10 的分数路径

`STORYLINE_RESTRUCTURING.md` 0.7 节给当前 PDF 的判断是 **3.8/10（Strong Reject，决定性缺陷是实验证据 1.0）**，并把「重构完成 + P0/P1 成立」的潜力估为 7.5--8.0。要再往上走到 9/10，差的不是更多模型和更多预算点，而是下面四类在该文件里尚未规划的证据：

| 额外证据 | 为什么能加分 | 对应实验 |
|---|---|---|
| 在线决策与**离线最优交换序列**的差距 | 把「greedy 是近似」从免责声明变成被量化的结论，这是 POMACS 口味 | E7 |
| $\widehat L$、$\widehat R$、use predictor 的**双向误差** | 模型论文的核心产出；含「代理误判」案例而非只报 MAE | E6 |
| **exchange 取证**：每类交换的次数、它换掉了什么、实际省了多少 | 直接回答审稿问题 5（收益来自策略还是底层实现） | E4 |
| 明确标注的**无收益区间与负面结果** | 可证伪性，审稿人对自曝边界的论文给信任分 | E0、E9 |

---

## 1. 审稿人最关心的数据类型

SIGMETRICS/POMACS 的评审口味与 MLSys/ASPLOS 不同：**分布优先于均值，模型验证优先于加速比，边界刻画优先于覆盖面**。下面七类数据按审稿人追问的顺序排列，每类后面标注它反击的是哪条模拟审稿意见。

### D1 耦合矩阵（联合最优是否在内点）
固定 $B$，二维扫描「高精度字节份额 × lookahead 字节份额」，每格记录 p95 exposed wait、p95 TPOT 与质量。
审稿人要看的不是曲面高度，而是**最优点是否落在内点、以及它是否随 workload 移动**。若最优点总在坐标轴上，联合分配不成立，论文必须收缩。
→ 反击 weakness 5（决定性 motivation 缺失）、weakness 8（H2 的生死条件下错）。

### D2 质量—延迟 Pareto 前沿（而非单点加速）
同一 $B$ 下，每个系统扫出自己的 (p95 TPOT, task score) 曲线。主结论只在相同 $\Delta Q \le \epsilon$ 的水平线上读取。
→ 反击 `STORYLINE_RESTRUCTURING.md` §7.2 明确禁止的「只给相对 uniform-low 的单点加速」。

### D3 真实峰值显存与账目一致性
每个点同时给出 runtime 的 live+reserved 计数器最大值与 NVML 进程峰值（2ms 采样）。
审稿人默认怀疑「收益来自悄悄多用了显存」。有一张计数器与 NVML 并列的时间线图，这个怀疑就被关闭。
→ 反击审稿问题 5 与 weakness 7（基线预算违规导致的虚假收益）。

### D4 模型预测误差（双向）
$\widehat L$ vs 实测 exposed wait 的散点与 MAE；$\widehat R$ vs 实测 $\Delta Q$；use predictor 的 recall@S。
必须包含两类错误案例：代理拒绝了实际可行的配置，代理放行了实际越界的配置。
→ 反击 weakness 4（最优性推导不充分）、weakness 6（质量代理链条断裂）。

### D5 机制归因（收益从哪来）
交换日志的分类统计 + 收益分解：避免的 miss 字节、被隐藏的传输时间、被避免的无效迁移、质量约束下的再分配。
→ 反击审稿问题 5：「收益来自更好的配置选择，还是不同的底层实现」。

### D6 非稳态下的适应动态
workload 切换点前后的时间序列：applied 状态、$\widehat R$ 账目、迁移字节、p95 滑窗。报告**适应期损失**，不只报稳态收益。
→ 反击 weakness 2/3（joint 实际是阶段切换）。

### D7 开销与可扩展性
控制器 CPU 时间分布（fast/slow 分开）相对 $T_\ell$ 的比例；$K$、$W$ 增大时的实测复杂度曲线，与 $O(K^2W)$ 界对照。
→ 反击 implementation 中那条尚未被测量的复杂度界。

**不被 SIGMETRICS 审稿人认可的数据**：单一均值加速比、只在一个 batch size 一个设备上的胜出、把自己实现的简化策略称作原系统、把 hotness 变化直接写成质量收益、没有置信区间的百分位数。

---

## 2. 现有代码资产盘点

`dynaexq/` 已有 103 个 Python 文件、约 11.4k 行核心代码，来自 ICCAD/TC 的 DynaExQ 工作。它不是从零开始，但也**不能直接支撑 lease/exchange 机制**，因为它的控制结构是「慢精度调度器 + 薄预取器」，正是本文要反对的那种分解。

| 现有模块 | 行数 | 可复用度 | 缺口 |
|---|---:|---|---|
| `core/memory_pool.py`（`PoolAllocator` 按层按 tier 的定长块池） | 269 | **高，直接用** | staging pool 需并入 $B$ 并暴露为 `M_reserved` |
| `core/budget_tracker.py`（`try_reserve/commit/release`，pending/committed 双账） | 305 | **高** | 缺「整笔交换的全有全无预留」；缺按时间索引的 peak 核算 |
| `core/transition_engine.py`（reserve→copy→publish→reclaim，event fence，repatriate） | 804 | **高，是最值钱的资产** | 只接受单个 `TransitionReq`；需支持一次提交多目的地的 schedule |
| `core/hotness_tracker.py`（EMA 路由权重） | 148 | **高** | 即为 Eq. (6) 的 $w_i$，需补 router weight $p_{\ell,t,e}$ 加权 |
| `core/router_observer.py` | 166 | 中 | 需产出 lookahead 预测所需的跨层路由历史 |
| `core/scheduler.py`（`PrecisionScheduler.plan`，慢环） | 314 | **低，将被替换** | 它是「先精度后其他」的词典序策略，与 exchange 语义冲突；保留为 baseline `SEQ-PQ` |
| `runtime/prefetch.py`（`PrefetchPlanner.lookahead`） | 47 | **低，将被替换** | 无 deadline、无排队、无 donor；保留为 baseline 的固定 horizon 预取器 |
| `runtime/controller.py`（`PrecisionController`，带 `_enforce_cap`） | 74 | 低 | 同上 |
| `core/weight_store.py`（双 tier pinned host 物化、逐层释放源权重） | 1048 | **高** | 已实现 `04_implementation.tex` 描述的逐层物化语义 |
| `core/quant.py` / `autogptq.py` / `quant_autoround.py` / `int2_kernel.py` | 1148 | **高** | INT4/INT2 kernel 与 AutoRound 流水线齐备 |
| `experiments/eval_perf.py` / `metrics.py` / `gpu_memory.py` | 871 | **高** | TTFT/TPOT/p50/p95/p99 与 NVML 采样已有；缺 exposed wait 与 bootstrap CI |
| `experiments/eval_dynamic.py`（统一 CLI：quality/perf/ablation/sensitivity/overhead/calibrate/...） | 1287 | **高，作为实验入口** | 需新增 exchange trace、policy 选择、Φ 参数化 |
| `baselines/lru_offload.py` | 137 | 中 | 作为 residency-only 下界 |
| `baselines/moe_infinity.py`（official checkout 校验 + prefetch telemetry） | 220 | **高** | 已有「跑真实外部系统」的纪律，照此模式扩展 |
| `baselines/expertflow.py` | **12（空壳）** | 无 | 需实现或删除，不能以空壳充当对照 |
| `tests/`（74 个测试，含 `test_transition_no_leak`、`test_event_fence`、`test_budget_tracker`） | — | **高** | 不变量测试文化已建立，新机制照此补测 |

**结论**：执行层（池、预留、发布、回收、量化 kernel、host 物化）可复用约 70%；**决策层必须重写**，因为现有决策层的结构本身就是论文要反对的对象。这是好消息：审稿人最在意的硬显存安全与可复现纪律已经有代码和测试支撑。

---

## 3. 实现计划

### 3.1 总原则：一个执行栈，多个策略插件

公平比较的前提是**所有被比较的系统共用同一个 dispatcher、同一个池、同一个预留门、同一套量化 kernel 与同一个 NVML 采样器**，只替换策略对象。这同时服务三件事：隔离策略净收益（审稿问题 5）、让基线能自然使用全部驻留能力（`STORYLINE_RESTRUCTURING.md` §7.1）、让「失败的预留」成为可计数事件而不是崩溃。

```python
# dynaexq/policy/base.py
class ExpertMemoryPolicy(Protocol):
    name: str
    def on_control_event(self, state: RuntimeState, event: ControlEvent) -> list[Exchange]: ...
    def on_miss(self, key: ExpertKey, deadline: float) -> MissAction: ...
    def frozen_constants(self) -> dict: ...          # 写入 result artifact
```

`RuntimeState` 对所有策略暴露相同的可观测量（路由历史、预测 uses、$C$、$T_\ell$、lease 账目、风险账目）。策略之间的差别只能来自它用了哪些字段、怎么排序、以及是否构造 donor。任何需要额外底层能力的基线（例如 HOBBIT 的 miss-path 低精度取回）都以执行层的**开关**实现，而非另写一套运行时。

### 3.2 目录结构（新增部分）

```
dynaexq/
  lease/
    ledger.py        # ByteLease, LeaseLedger, M_live/M_reserved/B_slack, 时间索引 peak
    risk.py          # q_i 表, w_i EMA, R_hat, eps_hat, repair
    replay.py        # PredictedUse, EDF replay, r_j, L_hat, C/T_l 估计器
    exchange.py      # Request/Donor 构造, 四项可行性, G, eta, 贪心接纳
    predictor.py     # use predictor: router-lookahead / pregate / frequency prior
  policy/
    base.py
    dynabyte.py      # Algorithm 1 的两时间尺度协调器
    uniform.py  static_mixed.py  precision_only.py  horizon_only.py
    hobbit_matched.py  dymoe_matched.py  promoe_matched.py  mxmoe_static.py
    seq_pq.py  seq_qp.py  static_joint.py  fixed_partition.py
    oracle_offline.py
  runtime/
    executor.py      # 多目的地全有全无 schedule, read-lease 引用计数, demand 路径
    trace.py         # exchange / ledger / replay 的结构化日志
  offline/
    sensitivity.py   # q_i 流水线（局部代理 + 真值抽样校验）
    calibrate.py     # eps_hat, 常量冻结, static baseline 的校准侧调优
```

### 3.3 必须新建的模块与接口契约

#### 3.3.1 `lease/ledger.py` — byte lease 与时间索引账目

对应 `03_design.tex` §3.3 与 Eq. (4)(5)。

```python
@dataclass(frozen=True, slots=True)
class ByteLease:
    use: Literal["fid", "res", "look"]
    key: ExpertKey
    nbytes: int
    t_start: float
    t_finish: float            # 控制承诺，不是占用保证
    owner_epoch: int           # 用于最小租期与 hysteresis
```

`LeaseLedger` 必须提供：
- `live_bytes()` / `reserved_bytes()`：live 包含仍有 reader 的旧块；reserved 包含所有未 publish 的目的地。
- `slack()` = $B - M_{\text{live}} - M_{\text{reserved}}$。
- `peak_over(schedule)`：给定一个完整转换 schedule（含源与替换共存区间），返回时间索引峰值。**这是 §3.7「全有全无预留」的判据，现有 `BudgetTracker` 只有瞬时账，必须补这一层。**
- `assert_invariant()`：任何时刻 live+reserved ≤ B，违反即抛错而不是降级。

不变量测试（照 `test_transition_no_leak.py` 的风格）：
- `test_ledger_peak_counts_coexistence`：demote 期间源与替换共存，peak 必须同时计入两者。
- `test_ledger_rejects_double_spend`：两笔交换不能花同一段未来字节。
- `test_lease_expiry_does_not_free_early`：lookahead lease 在 reader 结束前不可回收。

#### 3.3.2 `lease/risk.py` — 质量风险账目

对应 Eq. (6)(7)。$q_i$ 由离线流水线给出（§3.4），运行时只读。需实现：
- `update_routing_weight(layer, router_weights, routed_mask)`：Eq. (6) 的 $c_{\ell,e}$ 与 EMA。复用 `hotness_tracker.py`，但现有实现按 dispatch 计数，需改为 **router weight 加权**。
- `risk(state) -> float`：$\widehat R(X) = \sum_i w_i q_i (1 - h_i(X))$。
- `repair(state) -> list[Exchange]`：workload 漂移导致 $\widehat R > \widehat\epsilon$ 时，按「单位风险削减的最小延迟机会成本」排序的 promotion 交换序列。这条规则在正文中被明确写出（可能出现 fidelity 合并效应），所以必须可被日志验证。

#### 3.3.3 `lease/replay.py` — deadline 排序的反事实重放

对应 Eq. (8)(9) 与 §3.6。这是全系统的计算热点，也是消融的靶心。

```python
def replay(state: LeaseState, uses: list[PredictedUse], *,
           bandwidth_C: float, queued_bytes: int) -> ReplayResult:
    """EDF 顺序重放，尊重 earliest issue time，返回每个 use 的 r_j 与总 L_hat。"""
```

实现要点：
- uses 在控制事件开始时**按 deadline 排序一次**，每个 request 与其 donor set 的评估复用这份排序（`04_implementation.tex` 已承诺这一点）。
- 增量重算：维护「request → 受影响的 use 区间」依赖集；只有 resident set、表示大小或队列位置变化的 request 才重放。实现一个 `dirty` 标记并用一个开关 `--replay-full` 强制全量重放，用于验证增量实现无偏差（这是一个必须有的自检测试）。
- 两个带宽估计器并存：`C_overlap`（只统计与 expert compute 重叠的 copy 的 bytes/event-duration）与 `C_idle`（启动时一次 H2D microbenchmark）。后者**只在 E6 的模型误差实验中作为对照出现**，不参与在线决策。

#### 3.3.4 `lease/predictor.py` — use predictor（设计中缺失的一等模块）

$\omega_j$、$d_j$、$\hat U(S)$ 都依赖它，必须定义清楚，并在 §5 E6 中单独测量。提供三档，可在配置里切换：

| 档位 | 机制 | 用途 |
|---|---|---|
| `freq` | 按层 EMA 频率先验取 top-$n$ | 下界对照；也是多数已有预取器的隐含假设 |
| `router-lookahead` | 用当前层的 hidden state 过**下 $S$ 层的 router**（权重常驻，开销 = $S$ 次小 GEMM）得到候选集与 $\omega_j$ | **默认**；给出有校准意义的概率 |
| `pregate` | 轻量 MLP/随机森林，输入 (token id, layer, 当前激活集) | 消融项；只有当 E2 显示残余等待主要来自 miss-set 误差时才纳入推荐系统 |

$d_j$ 由在线测得的 $T_\ell$ 累积给出：use $j$ 在第 $\ell+s$ 层的 deadline = 当前时刻 + $\sum_{u=1}^{s} T_{\ell+u}$。$\omega_j$ 由 predictor 的校准概率给出，并用 reliability diagram 验证（E6）。

#### 3.3.5 `lease/exchange.py` — 交换构造、估值与接纳

对应 §3.5 与 Algorithm 1。

```python
@dataclass
class Exchange:
    request: ByteLease
    donors: tuple[ByteLease, ...]
    schedule: TransitionSchedule     # 供 ledger.peak_over 使用
    g_value: float                   # Eq. (10)
    eta: float                       # Eq. (11)
    risk_after: float
    provenance: dict                 # 为 E4 取证保留：donor 选择过程与被拒备选
```

四项可行性检查按正文顺序逐条实现并各配一个单测：lease 不在同一 expert 上冲突；$\widehat R(X \oplus x) \le \widehat\epsilon$；整个转换期间满足 Eq. (4)；每个预测传输不早于其 release time 且按 deadline 排序。

donor 构造按「counterfactual loss 递增」扫描并在凑够字节后停止，扫描上限写进配置并记录在 artifact 里（搜索被刻意限界，这点正文已声明，评测必须给出实测的候选数分布）。

**`provenance` 字段是 E4 的唯一数据来源**，必须在第一版就写进去，不能等实验阶段再补。

#### 3.3.6 `policy/dynabyte.py` — 两时间尺度协调器

直接对应 Algorithm 1。$T_{\text{fast}}$（routed layers）与 $T_{\text{slow}}$（forward steps）、最小 fidelity 租期、替换 hysteresis margin 四个常量在校准集上冻结，写入 artifact，**不按设备重拟**。

需要提供的开关（每个都对应一个消融）：
`--valuation {counterfactual,independent}`、`--donors {explicit,slack-only}`、`--partition {exchange,fixed}`、`--risk {feasibility,shadow-price}`、`--timescale {two,fast-only,slow-only}`、`--hysteresis {on,off}`、`--bandwidth {overlap,idle}`、`--predictor {router-lookahead,freq,pregate}`。

#### 3.3.7 `runtime/executor.py` — 全有全无的安全执行

在 `transition_engine.py` 之上加一层：
- `reserve_all(schedule) -> ReservationSet | None`：整笔交换的所有目的地一次预留，任一失败则全部回滚，published state 不变。
- read-lease 引用计数：forward path 在启动 expert kernel 前取 handle 的读租约，回收必须等所有 reader 与 last-use event 结束。现有 `_fence_before_reclaim` 已有 event fence 基础。
- demand 路径：lookahead 迟到的 expert，其 token 等待剩余 copy，同层已 publish 的 expert 继续算；迟到字节与 exposed wait 反馈到下一周期。
- `failed_reservations` 计数器：供 `naive composition` 与紧预算场景计数，而不是让它溢出。

### 3.4 离线 sensitivity $q_i$ 流水线

这是 weakness 6（质量代理链条断裂）的直接修复点，也是容易被低估工作量的地方。Qwen3-30B 有 48×128 = 6144 个专家，逐个做完整校准集前向不可行。

两级方案：

1. **局部代理（全量，便宜）**：对每层捕获一次 expert 输入分布，$\hat q_i = \mathbb{E}_x \|f_i^{\mathrm{hi}}(x) - f_i^{\mathrm{lo}}(x)\|_2^2 \cdot \bar p_i$。整模型一次前向即可拿到全部 6144 个值。
2. **真值抽样校验（分层抽样 ~128 个专家）**：真的只把该专家降精度，在 128×2048 token 的独立校准集上测 loss 增量，得到 $q_i^{\text{true}}$。

报告 $\hat q$ 与 $q^{\text{true}}$ 的 **Spearman 秩相关**，并与「按 dispatch 频率排序」「按 $\|W\|$ 排序」对照。这张表直接回答审稿问题 4（hotness 与 marginal quality gain 的秩相关是多少，与离线 sensitivity 比如何）。若秩相关不足（例如 < 0.5），必须按 `STORYLINE_RESTRUCTURING.md` §10.3 的决策规则收缩「动态 identity 选择」主张，而不是继续声称。

校准数据用 **Pile-10k**（已在本机 HF cache 中：`datasets--NeelNanda--pile-10k`），与 WikiText-2 测试集和所有任务集严格不相交。

### 3.5 Instrumentation 与 artifact 契约

每个 run 产出一个 JSON + 三个 JSONL：

| 文件 | 内容 | 服务于 |
|---|---|---|
| `result.json` | 冻结常量、checkpoint revision、量化配方、GPU UUID、NVML 2ms 峰值、runtime 预留计数器最大值、failed reservations、policy name 与全部开关 | D3，复现 |
| `exchange.jsonl` | 每笔接纳/拒绝的交换：request、donors、$G$、$\eta$、risk 前后、peak bytes、被拒备选 | D5（E4） |
| `ledger.jsonl` | 控制事件级的 live/reserved/slack 时间线 | D3 |
| `replay.jsonl` | 预测 $r_j$、$\widehat L$ 与对应的实测 exposed wait、实测 $C$、$T_\ell$ | D4（E6） |

沿用现有 `results/paper/manifest.json` 的 group 登记纪律与 `scripts/audit_paper_results.py`、`tests/test_paper_audit.py`。**规则不变：runtime 计数器一旦超过 $B$ 就是 bug 不是数据点**；没有足够信息核验 $B$、$\epsilon$ 与所选 expert 状态的结果不得进表。

### 3.6 里程碑与工作量估计

| 里程碑 | 内容 | 估计 | 阻塞关系 |
|---|---|---:|---|
| **M0** | 环境与 checkpoint 重建（§4.2 清单）、GPU 独占性与噪声门限验证 | 3--5 天 | **P0 阻塞全部** |
| **M1** | 共用执行栈：policy 插件接口、`executor.reserve_all`、read-lease 计数、failed-reservation 计数、不变量测试 | 5--7 天 | M0 |
| **M2** | `ledger` + `risk` + `replay` + `predictor` 四模块及其单测；离线 $q_i$ 流水线与秩相关表 | 8--12 天 | M1 |
| **M3** | `exchange` + `policy/dynabyte`（Algorithm 1）、两时间尺度、provenance 日志 | 7--10 天 | M2 |
| **M4** | 基线层 B/C：matched 重实现 4 个 + 组合基线 4 个 + oracle 2 个 | 8--12 天 | M1（与 M2/M3 可并行） |
| **M5** | 实验驱动：Φ 参数化扫描、bootstrap CI、artifact 登记、图表渲染 | 4--6 天 | M1 |
| **M6** | 实验战役 E0→E9 | 见 §7 | M3, M4, M5 |

M4 与 M2/M3 并行是关键：**组合基线只需要 M1 的执行栈**，所以 E0 的耦合扫描（Go/No-Go）可以在 DynaByte 本体写完之前就跑出来。这正是 `STORYLINE_RESTRUCTURING.md` §0.8 建议的顺序：先用网格和四个操作点回答「联合是否必要」，不要先实现完整在线控制器。

---

## 4. claim 重映射：旧 H1--H6 → 新机制

`05_evaluation.tex` 必须整节重写。下表是对应关系，避免重写时丢掉已经想清楚的东西。

| 旧 | 旧内容 | 处置 | 新归属 |
|---|---|---|---|
| H1 | $\Phi \ge 1$ 时 $S=0$ 最优 | **降级为边界结论**，不再是 hypothesis | §6.2 Boundary policies，由 E9 给数据 |
| H2 | $S^\star$ 随 $C$ 与 $m$ 单调移动，且是「联合论文的存在条件」 | **拆分**。argmin 不移动不等于无耦合（weakness 8）。存在条件改为 D1 的内点性与可移动性 | E0（内点性）+ E7（$C$、$m$ 敏感性） |
| H3 | feasible set 在 prefill/decode 上取值不同 | 保留为**非稳态证据**，不再是独立假设 | E5 |
| H4 | 最大可行 $n_{\mathrm{hi}}$ 等于 slack 界 | **替换**为 lease 账目的正确性与 peak 安全 | E8 |
| H5 | hotness 排序的 pin 在固定延迟带内提升任务分 | 保留，但必须接 §3.4 的秩相关前提 | E3 + E6 |
| H6 | naive composition 越界或更慢 | **降级为诊断项**，不作 headline（weakness 7） | E2 的一个弱对照行 |

新的假设集（在开跑前冻结进配置，artifact 可见）：

- **J1（必要性/内点性）** 在目标工作区间内，D1 耦合矩阵的联合最优落在内点，且其位置在至少两个 workload slice 之间移动超过一格。
  *拒绝条件*：最优点始终在坐标轴上或不随 workload 移动 → 停止「必须在线联合」主线，论文收缩为预算相位图 + deadline 模型。
- **J2（超越强组合基线）** 在相同 $B$ 与 $\Delta Q \le \epsilon$ 下，DynaByte 的 p95 TPOT 优于 SEQ-PQ、SEQ-QP 与 STATIC-JOINT-CAL 中的最好者，差距超过 seed 间范围。
  *拒绝条件*：被 STATIC-JOINT-CAL 在噪声内追平 → 按 §10.3 收缩在线控制主张，突出配置模型与操作区间。
- **J3（counterfactual 估值的必要性）** 把估值换成独立动作分数会导致可测量的退化，且退化可被归因为重复计价或 donor 选错。
  *拒绝条件*：独立估值在所有 slice 上等效 → 反事实重放不是贡献，退回「共享账目与执行机制」的较弱主张。
- **J4（质量代理有效）** $\widehat R \le \widehat\epsilon$ 的配置在测试集上满足 $\Delta Q \le \epsilon$ 的比例 ≥ 95%，且 $\hat q$ 对 $q^{\text{true}}$ 的秩相关显著高于频率排序。
  *拒绝条件*：代理双向误差大 → 质量约束改为保守静态配额，删除在线 identity 主张。
- **J5（决策质量）** 在线交换序列相对离线最优交换序列的 p95 exposed wait 差距 ≤ 15%。
- **J6（安全与开销）** 全部 run 的 runtime 计数器从不超过 $B$；fast 周期控制器时间的 p99 < $0.1 T_\ell$。
- **J7（三类交换真实发生）** fidelity→residency、fidelity→lookahead、residency→lookahead 三类交换各至少在一个 workload 上被接纳，且其预测收益与实测收益相关。

J7 直接对应 `DESIGN_NOVELTY_REWRITE_PLAN.md` §4 第 4 条。若某类交换从未发生，说明机制比声称的简单，正文必须相应缩写。

---

## 5. 评测设置

### 5.1 模型集（含盘与内存核算）

正文现在列了 6 个模型。**建议收缩到 3 主 + 1 可选**：审稿人更认 3 个被研究透的模型，而不是 6 个浅尝；而且当前盘上一个 MoE checkpoint 都没有，231GB 可用空间装不下 6 个模型的双 tier。

| 模型 | 结构 | tier 对 | 盘占用（双 tier） | 优先级 | 入选理由 |
|---|---|---|---:|---|---|
| Qwen3-30B-A3B-Instruct-2507 | 48L / 128E / top-8 | FP16 + INT4 | ~78 GB | **P0** | 细粒度多专家，$\Phi$ 可扫满三个区间；既有路由 trace |
| DeepSeek-V2-Lite | 27L / 64E / top-6 + shared expert + MLA | FP16 + INT4 | ~40 GB | **P0** | 架构异质（shared expert 走非 expert reserve、MLA 改变 KV 预留），迭代快 |
| Qwen3-Next-80B-A3B | 48L / 512E / top-10 | **INT4 + INT2** | ~65 GB（不下 BF16） | **P1** | 唯一能把 $\Phi<1$ 压到极紧的模型；tier 对不同，检验机制不依赖具体 bit 宽 |
| Phi-3.5-MoE | 32L / 16E / top-2 | FP16 + INT4 | ~107 GB | P2 | 粗粒度稀疏的反向检验；仅在盘与时间允许时做，可用 `/dev/shm` 轮转 |

P0 合计 118GB，P0+P1 合计 183GB，在 231GB 内；加 Phi-3.5 需要 staging 轮转。
Host pinned 需求：最大单模型约 78GB，1007GB 内存充裕；`weight_store.py` 的逐层物化+释放已经避免同时持有未打包 checkpoint 与双 tier。

**删除 Qwen1.5-MoE 与 Qwen2-MoE**（正文现在列了）：它们不提供新的结构维度，只增加表格宽度。

### 5.2 预算点：用 $\Phi$ 参数化而不是用 GB

正文现在扫 $B \in \{12,16,20,24,32,40,48\}$ GB，这让跨模型结果不可比。改为按 $\Phi = B/P_{\mathrm{lo}}$ 取点，$P_{\mathrm{lo}}$ 用**实测打包尺寸**而非名义参数量：

$$\Phi \in \{0.5,\ 0.7,\ 0.85,\ 1.0,\ 1.3,\ \rho\},\qquad \rho = P_{\mathrm{hi}}/P_{\mathrm{lo}}$$

主结果用 $\{0.7, 0.85, 1.3\}$ 三点（三个区间各一），$\{0.5, 1.0, \rho\}$ 用于边界与退化（E9）。这同时是一个小的建模贡献：$\Phi$ 让「机制在哪个区间有用」成为可跨模型陈述的结论。

### 5.3 设备与带宽 $C$：把三设备承诺换成受控扫描

**正文当前的三设备声明（A6000 + H20 + 910B）在本机无法兑现，必须修改**，否则是审稿可见的 overclaim。

替代方案（按可信度排序，建议同时做前两项）：

1. **迁移流上的令牌桶节流（主）**：把 copy 切成固定 chunk 并按配速发射，扫 $C \in \{4, 8, 12, 16, \text{unthrottled}\}$ GB/s。每个设定都用在线 `C_overlap` 估计器反测，验证实际达到的带宽与目标一致。这比两台真实设备**更能隔离 $C$**，因为 kernel、软件栈、量化实现全部不变（正文 §6.7 已承认跨设备比较混入软件差异）。在正文中明确标为 bandwidth emulation。
2. **pinned vs pageable host 内存（锚点）**：给出两个真实、未人工节流的 $C$ 点，用来证明节流扫描落在真实可达区间内。
3. **真实第二设备（可选加分）**：若能借到 H20 或 910B，只跑 E7 的一个确认点；拿不到就在 Threats to Validity 中写明，而不是留着未兑现的承诺。

GPU 调度纪律：GPU0 当前被他人占满（35.5GB，100% util）。**所有计时实验必须在独占 GPU 上跑**，run 期间采样 `nvidia-smi` 并把其他进程的存在写进 artifact；控制配置的 run-to-run p95 变异系数 > 5% 的数据点作废重跑。

### 5.4 数据集

#### 延迟/服务侧

| 数据集 | 角色 | 为什么审稿人在意 |
|---|---|---|
| **ShareGPT**（`ShareGPT_Vicuna_unfiltered`，已有 `scripts/fetch_sharegpt_benchmark.sh`） | 主延迟语料，按 prompt 长度分桶，桶内长度变异 < 5%，每桶 ≤ 50 请求 | 社区标准，可比 |
| **Azure LLM Inference Trace**（2023/2024 版） | 提供**真实到达过程**；时间戳取自 trace，请求体取自 ShareGPT | SIGMETRICS 审稿人对生产 trace 的 burstiness 特别买账；突发会同时压缩 overlap window 和抬高 KV 预留，是机制最该发光的地方 |
| **长上下文切片**（ShareGPT >8K 子集 或 LongBench 单任务） | KV 预留变大 → $B$ 变小 | 直接测试 §6.6 承认的集成失败模式：KV 挤占导致 ledger 合规但进程越界 |
| **任务切换流**（WikiText → GSM8K → HumanEval → MMLU-Pro 分段，切换点受控） | E5 的非稳态输入 | 这是在线控制相对静态联合配置的**唯一**合法优势来源 |

#### 质量侧

| 数据集 | 协议 | 规模 |
|---|---|---|
| WikiText-2 perplexity | ≤ 128 个不重叠 2048-token 窗口 | 主质量指标，与 `02_background.tex` 的 motivation 曲线同构 |
| MMLU-Pro | 固定 200 题子集，按完整答案标签的条件对数似然打分 | 知识 |
| GPQA-Diamond | 全集 | 难推理 |
| GSM8K | 固定 200 题 | 数学，greedy |
| HumanEval | 全集，pass@1，执行官方测试 | 代码，长生成 |

质量 run 关闭 continuous batching、用 greedy 解码，一道题的所有候选在一个 unpadded batch 内打分，保证精度变化不会发生在同一题的候选之间（`04_implementation.tex` 已有此纪律）。准确率用 Wilson 95% 区间，**不把任务池成单一显著性检验**。

#### 校准侧（与测试严格不相交）

Pile-10k 子集用于 $q_i$、$\widehat\epsilon$、控制器常量与所有静态基线的调优；WikiText 的独立切片用于 perplexity 侧校准。`scripts/build_independent_calibration.py` 与 `tests/test_build_independent_calibration.py` 已有相应纪律。

#### $\epsilon$ 的预注册

必须在跑之前写定，不能事后选：

$$\epsilon:\quad \Delta\mathrm{PPL}_{\text{WikiText-2}} \le 0.5 \ \wedge\ \Delta\text{score}_{\text{task}} \ge -1.0\ \text{pp（每个任务）}$$

并同时报告一个宽松档（$\Delta\mathrm{PPL} \le 1.5$）和一个严格档（$\le 0.2$），用来显示结论对 $\epsilon$ 的敏感性。$\widehat\epsilon$ 与 $\epsilon$ 的映射在校准集上确定一次，测试集上不重拟。

### 5.5 Baseline 矩阵

分四层。**层 A 是可信度，层 C 是生死线，层 D 是新颖性。**

#### 层 A：可运行的已发表系统（原样跑，不改策略）

| 系统 | 引用 | 状态与做法 |
|---|---|---|
| MoE-Infinity | `xue2024moe`, `moeinfinityrepo` | 已有 `baselines/moe_infinity.py` 做 official checkout 校验与 prefetch telemetry，沿用 |
| Mixtral-offloading | `eliseev2023fast` | LRU + speculative prefetch，公开实现 |
| ProMoE | `song2024promoe` | 若 artifact 可用则原样跑；否则降级到层 B 的 `ProMoE-M` |
| HOBBIT | `tang2024hobbit` | 若 artifact 可用则原样跑；否则降级到层 B 的 `HOBBIT-M` |
| vLLM / SGLang（参考线） | `kwon2023vllm`, `zheng2024sglang` | 不是 offload 对照，只用来给出「同机同模型的常规服务延迟量级」，避免审稿人怀疑整个栈慢 |

纪律（正文已声明，必须执行）：**跑不起某个 checkpoint 的格子留空，不近似**；自己实现的简化策略一律标 `-M` 后缀，绝不称作执行了原系统。

#### 层 B：同栈 matched 重实现（隔离策略净收益）

| 名称 | 对应 SOTA | 固定了什么决策 |
|---|---|---|
| `Uniform` | 统一 PTQ（`frantar2022gptq`, `lin2024awq`, `autoround`） | $B$ 内能装下的最高统一 tier，无在线变化。$\Phi \ge \rho$ 区间的最优 |
| `MxMoE-static` | `duanmu2025mxmoe` | 离线混合精度配额与身份，全部冻结 |
| `MoPEQ-static` / `MoQE-static` | `chitty2025mopeq`, `kim2023mixturequantizedexpertsmoqe` | 敏感度/频率排序的离线 bit 分配 |
| `LRU` / `LFU` | `baselines/lru_offload.py` | residency-only 下界，单 tier |
| `AdapMoE-M` | `zhong2024adapmoe` | 自适应 gating + 预取，单一表示 |
| `ProMoE-M` | `song2024promoe` | 自适应 horizon，expert 尺寸固定；**允许它自然用满全部驻留能力**（修正 §8 列出的「horizon-only 人为驱逐」缺陷） |
| `ExpertFlow-M` | `shen2025expertflow` | 预测路由 + 调度，固定表示。注意区分同名的 `he2024expertflow`（改 token 分配） |
| `Pre-gated-M` | `hwang2024pre` | 结构化预取 |
| **`HOBBIT-M`** | `tang2024hobbit` | **必做正面对照**：miss-path 低精度取回 + 多级 cache + 自适应预取 |
| **`DyMoE-M`** | `huang2026dymoe` | **必做正面对照**：在线 importance + depth-aware 混合精度 + cache + lookahead。用其 **4/2 不跳过专家** 配置，使执行语义与我们一致 |

HOBBIT 与 DyMoE 的对照是 `DESIGN_NOVELTY_REWRITE_PLAN.md` §4 第 1 条，也是 weakness 9 的唯一解法。差异必须落在**优化目标与决策语义**上，并被结果钉住：它们把精度当 miss-path 表示或重要性启发式，不把三类用途放进同一笔显式的机会成本账。

#### 层 C：强组合基线（必要性的生死线）

| 名称 | 构造 | 回答什么 |
|---|---|---|
| `SEQ-PQ` | 先在校准集上定精度配额（复用现有 `core/scheduler.py`），再在剩余预算上调 horizon；**共享同一个 $B$，充分调优** | 「先精度后预取」够不够 |
| `SEQ-QP` | 反向顺序 | 「先预取后精度」够不够 |
| `STATIC-JOINT-CAL` | 在**校准集**上做 D1 二维网格搜索，取最优格并冻结 | **最关键的对照**：静态联合配置是否已足够 |
| `FIXED-PARTITION` | 三类用途各给固定私有配额，内部各自最优 | 固定切分是否够 |
| `NAIVE-COMPOSITION` | 两个控制器各自按完整 $B$ 申请，预留门拦截，计 failed reservation | **仅作诊断行**，不作 headline（weakness 7） |

#### 层 D：上界与消融（新颖性的定位器）

| 名称 | 构造 | 回答什么 |
|---|---|---|
| `ORACLE-GRID-TEST` | 在**测试集**上搜 D1 网格的最优格 | 非稳态的价格 = 它与 `STATIC-JOINT-CAL` 的差 |
| `ORACLE-OFFLINE-EXCHANGE` | 对回放的 trace 解一个离线交换序列最优化（每事件的 lease 背包 + 短 trace 上的全时域 MIP） | **决策质量差距**，J5。这是 9/10 的差异化证据 |
| `IND-VALUE` | 估值换成独立动作分数（residency 与 lookahead 各自计功） | J3：是否出现重复计价与 donor 选错 |
| `SLACK-ONLY` | 只在有 slack 时接纳请求，不构造 donor | 显式 donor 是否必要 |
| `RISK-AS-LATENCY` | 质量风险用影子价格折成毫秒（即旧版 design） | 可行性框架相对影子价格的收益 |
| `FAST-ONLY` / `SLOW-ONLY` | 单时间尺度 | 两时间尺度是否必要 |
| `NO-HYSTERESIS` / `NO-MIN-TENURE` | 关掉 churn 控制 | 迁移震荡的代价 |
| `IDLE-C` | 用 idle microbenchmark 带宽 | 支撑「必须用 in-overlap $C$」 |
| `FREQ-PRED` / `PREGATE-PRED` | 换 use predictor | 收益归因于预测器还是分配器 |

### 5.6 指标与统计方法

**主指标**：p95 / p99 TPOT（Eq. (1) 的目标）、p95 TTFT。
**机制指标**：exposed wait（层在缺失专家上停等、扣除与该层已驻留专家计算的重叠部分）、miss-path time（迁移流时长，允许大于 exposed wait）、miss bytes、late bytes、迁移字节、命中率、每类交换计数。
**约束指标**：NVML 进程峰值（2ms 采样）、runtime live+reserved 计数器峰值、failed reservations、各任务分数与 Wilson 区间、WikiText PPL。
**服务指标**：给定 SLO（例如 TPOT ≤ 50ms）下的 attainment 比例与 goodput。审稿人会问「百分位改善是否转化成了可部署的收益」，这一行能直接回答。
**开销指标**：fast/slow 控制器 CPU 时间分布、候选数与 donor 扫描长度分布。

统计纪律：
- 5 seeds；**所有系统跑同一条请求流**（paired），差值用 Wilcoxon signed-rank。
- 百分位数用 bootstrap（10k 重采样）给 95% CI；seed 间用中位数 + range 汇总。
- 每配置 5 warmup + 100 measured iterations（沿用现有协议）。
- 区间重叠就报告为重叠，不做「略优于」的叙述。
- 假设 J1--J7 与 $\epsilon$ 在开跑前冻结进配置文件，artifact 可见。

---

## 6. 实验战役

每个实验给出：问题 → 输入 → 输出图表 → 通过/失败判据。

### E0 —— 统一栈上的现象复现与耦合矩阵（Go/No-Go）

**问题**：在一个统一运行栈上，三类用途是否真的互相挤压，联合最优是否在内点并随 workload 移动？
**输入**：Qwen3-30B + DeepSeek-V2-Lite，$\Phi \in \{0.7, 0.85\}$，二维网格「高精度字节份额 × lookahead 字节份额」（各 5--6 档），3 个 workload slice（ShareGPT 短/长桶、Azure burst 段、GSM8K 段）。只需 M1 的执行栈，**不需要 DynaByte 本体**。
同时重测 `02_background.tex` 的三个现象：stage-level demand、Layer-15 hot-set 跨任务重叠、latency-vs-lookahead 的 U 形。
**输出**：每个 slice 一张热力图（exposed wait）+ 一张等质量线叠加；最优格位置的迁移箭头图。
**判据**：J1。最优格在内点，且至少两个 slice 之间移动 > 1 格 → Go。否则按 §10.3 收缩主线。E0 **完整报告，包含负面结果**。

### E1 —— 端到端 Pareto 与相位图（headline）

**问题**：相同 $B$ 与 $\Delta Q \le \epsilon$ 下，用户可见延迟改善多少？
**输入**：3 个模型 × $\Phi \in \{0.7, 0.85, 1.3\}$ × {层 B 全部、层 C 全部、DynaByte}；ShareGPT + Azure 到达过程。
**输出**：主图是 (p95 TPOT, task score) 的 Pareto 前沿，每系统一条；相位图把 p95 TPOT 与 PPL 对 $\Phi$ 作图，横轴标注实测 $\Phi=1$ 与 $\Phi=\rho$；数值表按区间分行而不是给单一 speedup。
**判据**：J2。

### E2 —— 必要性：对抗强组合基线

**问题**：充分调优、共享预算的顺序优化与静态联合配置能否达到同样效果？
**输入**：层 C 全部 + `ORACLE-GRID-TEST`。
**输出**：差距分解柱状图——把 DynaByte 与最佳组合基线的 p95 差值拆成：避免的 miss 字节、被隐藏的传输时间、被避免的无效迁移、质量约束下的再分配。`NAIVE-COMPOSITION` 只作一行诊断（failed reservations 计数）。
**判据**：J2。若 `STATIC-JOINT-CAL` 在噪声内追平 DynaByte，**在线联合的必要性不成立**，必须收缩标题与贡献。

### E3 —— 与近邻系统的正面比较

**问题**：相对已经组合了混合精度、cache 与 prefetch 的系统，差异是否可测量？
**输入**：`HOBBIT-M`、`DyMoE-M`（4/2 不跳过专家配置）、`MxMoE-static`、`ProMoE-M` + 层 A 中所有能跑起来的原系统。
**输出**：相同 $B$、相同质量约束下的延迟—质量表；外加一张「决策语义对照表」（精度是 resident state / miss-path 表示 / 可跳过计算；质量是显式约束还是启发式；三类动作是否共享显式字节机会成本）。
**判据**：差异必须由结果或严格 matched ablation 支撑。跑不起来的格子留空。

### E4 —— 机制取证：交换日志

**问题**：收益是否真的来自声称的机制？
**输入**：E1 的 `exchange.jsonl`。
**输出**：
(a) 三类交换（fid→res、fid→look、res→look）的计数、字节量与实测收益；
(b) 每类至少一条完整案例时间线（request、donor、$G$、$\eta$、风险前后、实际省下的等待）；
(c) `IND-VALUE` 对照下的错误分类：重复计价次数（同一次避免的传输被 residency 与 lookahead 同时计功）与 donor 选错次数；
(d) 预测 $G$ 与实测收益的相关性。
**判据**：J7 + J3。若某类交换从未被接纳，正文必须相应缩写机制描述。

### E5 —— 非稳态适应

**问题**：在线调整是否胜过充分调优的固定配置？
**输入**：任务切换流、prefill↔decode 边界、batch 变化（1/8/32）、Azure burst 段。
**输出**：切换点对齐的时间序列（applied 状态、$\widehat R$、迁移字节、滑窗 p95）；**适应期损失**（收敛到新最优格 1 格内所需的控制周期数与期间迁移字节）；与「一个周期直接跳到最优」的消融对比，验证 hysteresis 的代价确实更低。
**判据**：适应期损失小于静态配置在切换后的稳态损失。若 workload 本身稳态为主，必须明说收益有限。

### E6 —— 模型验证（POMACS 的核心产出）

**问题**：$\widehat L$、$\widehat R$ 和 use predictor 各自有多准，错在哪？
**输出**：
(a) $\widehat L$ vs 实测 exposed wait 散点 + MAE，按区间分层；`IDLE-C` 与 `C_overlap` 两个估计器并列；
(b) $\widehat R$ vs 实测 $\Delta Q$；混淆矩阵形式给出**两类代理误判**：拒绝了实际可行的配置、放行了实际越界的配置；
(c) use predictor 的 recall@S 与 $\omega_j$ 的 reliability diagram，三档预测器对比；
(d) $\hat q$ vs $q^{\text{true}}$ 的 Spearman 秩相关，与频率排序、权重范数排序对照（§3.4）；
(e) Eq. (14) 的聚合 overlap 判据作为剪枝器的误差：它放过/漏掉了多少逐层 deadline 实际会违约的情形——直接回答审稿问题 3。
**判据**：J4 + in-overlap $C$ 的误差显著小于 idle $C$。

### E7 —— 带宽、表示尺寸与决策质量

**问题**：机制对 $C$ 与 $m$ 的依赖是否与模型一致？在线决策离最优有多远？
**输入**：§5.3 的 $C$ 节流扫描 × {$m^{\mathrm{lo}}$, $m^{\mathrm{hi}}$} × 2 个模型；`ORACLE-OFFLINE-EXCHANGE`。
**输出**：最优 lookahead 字节份额对 $C$、$m$ 的响应面；在线 vs 离线最优的 p95 差距曲线。
**判据**：J5。注意：份额不随 $C$、$m$ 移动**不**直接否定耦合（修正旧 H2），但会削弱「必须在线」的论证强度，需如实报告。

### E8 —— 安全与开销

**输出**：
(a) ledger 时间线与 NVML 峰值叠加图，覆盖全部 run，声明「0 次计数器越界」或列出每一次拒绝的预留；
(b) fast/slow 控制器 CPU 时间的 CDF 及其占 $T_\ell$ 的比例；
(c) 候选数 $K$ 与窗口 $W$ 增大时的实测时间曲线，与 $O(K^2W + W\log W)$ 界对照，并给出依赖跟踪带来的实际削减；
(d) failed reservations 在紧预算与 `NAIVE-COMPOSITION` 下的计数。
**判据**：J6。

### E9 —— 边界与负面结果

**问题**：哪些情况下 DynaByte 不提供收益？
**输入**：$\Phi \ge \rho$（全高精度可驻留）、$\Phi = 0.5$（极紧）、$\epsilon$ 宽松档与严格档、单任务稳态 workload、`FREQ-PRED`（弱预测器）、迁移带宽受限。
**输出**：一张「操作区间图」，标出机制退化为 uniform / 低精度 cache / deadline 预取 / 静态混合精度的四个区域（对应 `06_discussion.tex` §6.2），以及收益不显著的区域。
**判据**：退化行为与 §6.2 的声明一致。这一节是**主动自曝边界**，审稿人据此给可信度分。

---

## 7. 时间表与 Go/No-Go

| 阶段 | 内容 | 周 | 门限 |
|---|---|---:|---|
| **P0** | M0 环境与 checkpoint；M1 执行栈；**E0 耦合矩阵**（1 模型 × 2 预算 × 3 slice） | 1--3 | **Go/No-Go：J1**。不过则停止在线联合主线，改写为相位图 + deadline 模型论文，或拆回两篇 |
| **P0.5** | 层 C 组合基线 + `STATIC-JOINT-CAL` + `ORACLE-GRID-TEST`，在 E0 网格上构造四个操作点 | 3--4 | **第二道门：J2 的早期信号**。若静态联合已追平，立即收缩主张，不要先把六模型三设备补齐 |
| **P1** | M2 + M3（lease/risk/replay/predictor/exchange/coordinator）+ 离线 $q_i$ 与秩相关 | 4--8 | **第三道门：J4**。秩相关不足则删除动态 identity 主张 |
| **P1.5** | M4 层 B matched 基线（HOBBIT-M、DyMoE-M 优先）+ M5 实验驱动 | 7--10 | — |
| **P2** | E1--E4 主战役（3 模型 × 3 预算 × 全基线） | 10--14 | J2、J3、J7 |
| **P2.5** | E5--E9（非稳态、模型验证、带宽、开销、边界） | 13--17 | J5、J6 |
| **P3** | 重写 `05_evaluation.tex`、填表、artifact 登记与审计、Phi-3.5 可选补充 | 16--19 | 全部 claim 有 manifest 条目 |

关键顺序判断：**E0 与层 C 基线必须在 DynaByte 本体之前完成**。这是 `STORYLINE_RESTRUCTURING.md` §0.8 的建议，它能在第 4 周就回答论文是否值得继续，而不是在第 15 周发现联合收益不存在。

---

## 8. 分数映射：每一分要靠哪份证据

| 审稿维度 | 当前（0.7 节） | 目标 | 需要的证据 |
|---|---:|---:|---|
| 问题重要性 / SIGMETRICS fit | 8.0 | 9.0 | E1 的 Pareto + Azure 真实到达过程下的 SLO attainment |
| 故事统一性 | 5.0 | 9.0 | design 已重写；E4 的三类交换取证证明统一账目真的在工作 |
| 技术新颖性 | 4.5 | 8.5 | E3（HOBBIT-M / DyMoE-M 正面对照）+ E4c（`IND-VALUE` 的重复计价与 donor 选错） |
| 模型严谨性 | 5.0 | 9.0 | E6 全部五项，特别是 (b) 的双向误判与 (e) 对审稿问题 3 的正面回答 |
| 实验设计 | 7.0 | 9.5 | 层 C 强组合基线 + 预注册 J1--J7 + paired 统计 + E9 负面结果 |
| 实验证据 | **1.0** | 8.5 | E0--E8 全部完成，所有表格填满或明确标注留空原因 |
| 写作与组织 | 6.0 | 9.0 | `05_evaluation.tex` 按 J1--J7 重写，删除未兑现的三设备与六模型承诺，去掉未来时 |
| 可复现性 | 7.0 | 9.5 | 四文件 artifact 契约 + manifest 审计 + `exchange.jsonl` 使决策过程可审 |

9/10 的门槛不是上面每一格都满分，而是**没有任何一格低于 8**。目前拉低总分的是「实验证据 1.0」和「技术新颖性 4.5」，它们分别由 E0--E8 的完成度和 E3/E4 的对照质量决定。

---

## 9. 风险登记册

| 风险 | 影响 | 概率 | 缓解 |
|---|---|---|---|
| **checkpoint 全部不在盘上，231GB 装不下计划模型** | P0 阻塞 | 已发生 | 先下 DeepSeek-V2-Lite（40GB）启动开发；Qwen3-30B 其次；80B 的 INT4/INT2 不下 BF16；Phi-3.5 走 `/dev/shm` 轮转 |
| **无 H20/910B，正文已承诺三设备** | 审稿可见 overclaim | 已发生 | 改为带宽节流扫描并标明 emulation；正文删除三设备声明 |
| GPU0 被他人长期占用，只有一张卡做计时 | 吞吐减半 | 高 | 计时实验排队独占 GPU1；非计时（质量、trace 采集、离线 $q_i$）可在共享卡上跑；每个 run 记录同卡其他进程 |
| HOBBIT / DyMoE 无公开 artifact | 正面对照退化为重实现 | 中--高 | 以 `-M` 后缀的 matched 重实现 + 决策语义对照表；明确不声称跑了原系统；对照口径在正文逐条列出 |
| E0 显示最优点总在坐标轴上（J1 被拒） | 主线崩塌 | 中 | 第 3 周就知道；按 §10.3 改写为预算相位图 + deadline-aware 模型论文 |
| `STATIC-JOINT-CAL` 追平在线控制（J2 被拒） | 在线主张崩塌 | 中 | 第 4 周就知道；收缩为「配置模型 + 操作区间」，这仍是一篇可发表的 measurement 论文 |
| $\hat q$ 秩相关不足（J4 被拒） | 质量侧主张崩塌 | 中 | 退回保守静态精度配额，删除在线 identity claim，只保留 residency/lookahead 的联合分配 |
| 增量 replay 实现有偏 | 结果不可信 | 中 | `--replay-full` 自检开关 + 等价性单测，强制每个实验配置至少跑一次全量重放校验 |
| 控制器开销吃掉收益 | J6 被拒 | 中 | 限界搜索的扫描上限可配；E8c 的 $K$/$W$ 曲线提前暴露；必要时把 slow 环搬到独立线程 |
| 实验资源不够跑满 3 模型 × 3 预算 × 全基线 | 范围失控 | 高 | 严格按 P0→P2 顺序；层 A/P2 模型为可选；宁可 3 模型做透也不要 6 模型做浅 |

---

## 10. 本计划明确不做的事

- 不做多 GPU / expert-parallel（改变每设备的 $B$ 与 $C$，`06_discussion.tex` 已划为范围外）。
- 不做 SSD tier。
- 不在线合成新 bit 宽；只在两个离线预制表示之间选择。
- 不改 top-$k$、router 参数或 expert 架构；不跳过被路由到的专家。
- 不做 KV cache 与 expert memory 的联合预算（只报告 ledger 与进程峰值的分离，使集成失败可见）。
- 不把任务分数池成单一显著性检验。
- 不用空壳基线（`baselines/expertflow.py` 当前 12 行空文件，要么实现要么删除）。
- 不在 `05_evaluation.tex` 保留任何未兑现的设备、模型或数据集承诺。

---

## 11. 下一步具体动作（按序）

1. 确认 checkpoint 获取路径与配额，先落 DeepSeek-V2-Lite 双 tier（约 40GB），打通 M0。
2. 把 `05_evaluation.tex` 的旧框架按 §4 的映射表重写为 J1--J7，并删除三设备与六模型声明；同时在 `03_design.tex` 补一小段 use predictor 的定义。
3. 实现 M1 的 policy 插件接口与 `executor.reserve_all`，补三个 ledger 不变量测试。
4. 用 M1 的执行栈跑 E0 的耦合矩阵（1 模型 × 2 预算 × 3 slice），在第 3 周做 Go/No-Go。
5. 在同一网格上构造 `SEQ-PQ`、`SEQ-QP`、`STATIC-JOINT-CAL`、`ORACLE-GRID-TEST` 四个操作点，第 4 周做 J2 早期判断。
6. 两道门都通过后，才开始写 `lease/` 与 `policy/dynabyte.py`。
