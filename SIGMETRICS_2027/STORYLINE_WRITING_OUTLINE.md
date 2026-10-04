# DynaByte 写作主线与章节大纲

> 当前阶段只固定论文主线和非实验章节的论证顺序。  
> Evaluation 暂不展开；待 Introduction、Background、Design、Discussion 和 Related Work 稳定后，再根据这些章节提出的 claim 反向规划。

## 1. 核心主线

### 一句话问题

在显存受限的 MoE 推理中，同一份 expert-memory budget 可以用于三种目的：

- **Fidelity：** 提高部分专家的表示精度，以满足模型质量要求；
- **Residency：** 保留更多专家，避免未来的 expert miss；
- **Lookahead：** 提前装入即将使用的专家，将不可避免的传输隐藏在计算窗口中。

三种用途竞争同一批显存字节，而每个字节的价值会随专家敏感度、访问概率、复用距离、使用 deadline、带宽和请求阶段变化。

### 一句话方法

> **DynaByte 在质量约束和硬显存预算下，估计 fidelity、residency 与 lookahead 动作的边际收益，并通过统一协调器动态决定显存字节的用途，以降低 MoE 推理尾延迟。**

### 一句话 novelty

现有方法通常分别回答“专家应使用什么精度”“哪些专家应留在 GPU”“应提前多远预取”；DynaByte 回答的是它们共享同一显存预算以后，**下一个可用字节此刻应该分配给哪一种动作**。

因此，novelty 不能写成“同时具有 quantization、cache 和 prefetch”，而应落在：

1. 三种用途具有共同且动态的机会成本；
2. 固定切分或固定优先级无法表达这种机会成本；
3. DynaByte 用统一的 marginal value per byte 协调三种动作；
4. 所有动作共享同一个质量约束和显存账目。

### 全文逻辑链

> MoE 的 sparse computation 没有消除 expert memory pressure<br>
> → 压缩、驻留和预取都能缓解压力，但消耗同一显存预算<br>
> → 三种用途的单位字节价值随运行状态变化<br>
> → 独立控制或固定优先级无法表达动态机会成本<br>
> → DynaByte 统一比较动作的质量价值、延迟价值和迁移成本<br>
> → 在质量约束与硬显存预算下选择当前最有价值的动作。

全文应固定以下表述：

- **目标：** tail latency；
- **约束：** peak expert-memory budget 和 quality degradation bound；
- **决策：** expert precision、resident set 和 prefetch timing；
- **原则：** marginal value per byte；
- **执行：** shared admission 与安全状态转换。

## 2. Introduction

Introduction 的任务是让读者接受“一字节三用途”是一个独立且重要的问题，不要从两篇旧工作的拼接开始。

### 第 1 段：部署矛盾

MoE 通过稀疏激活降低计算量，但庞大的 expert pool 仍造成显存压力。显存不足迫使系统在模型精度、专家驻留和 host-device transfer 之间取舍，最终影响质量、TTFT 和 TPOT。

### 第 2 段：一字节三用途

同一个空闲显存块可以用于：提升一个专家的精度、保留更多低精度专家，或者暂存即将使用的专家。任何一种选择都会排斥另外两种用途，因此三者不是可以独立调优的参数。

### 第 3 段：现有工作的缺口

Precision-oriented 方法通常在固定 residency 假设下分配 bit width；cache/prefetch-oriented 方法通常在固定 expert size 下决定保留和搬运。部分工作已经组合 mixed precision、cache 与 prefetch，因此本文不能以“首次组合这些机制”为贡献。缺口应表述为：现有方法没有显式比较三种用途在同一质量约束和 byte budget 下的动态机会成本。

### 第 4 段：核心洞察

三种动作的单位字节价值随状态变化：precision upgrade 取决于质量敏感度与未来使用概率；residency 取决于复用和可避免的 miss；lookahead 取决于 deadline、overlap window 和缓存污染。因此不存在对所有阶段都正确的固定切分或固定优先级。

### 第 5 段：DynaByte

DynaByte 将 precision、retain/evict 和 prefetch 表示成同一状态空间中的候选动作，估计每个动作的质量影响、延迟收益、字节成本和迁移代价，再由统一协调器在约束下选择动作。短期 residency/lookahead 和长期 precision adaptation 可以采用不同更新频率，但必须共享同一本账和同一个 admission gate。

### 第 6 段：贡献

贡献建议保持三项：

1. 将有限 expert memory 表述为 fidelity、residency 与 lookahead 的统一资源分配问题；
2. 提出以 marginal value per byte 为核心、受质量和显存约束的联合协调方法；
3. 实现共享 byte ledger 和安全状态转换，使三类动作能够在硬预算下协同执行。

最终结果句留待 Evaluation 规划完成后再补，不在当前阶段预写数值结论。

## 3. Background and Motivation

本节只建立统一问题所需的概念，不提前展开算法。

### 3.1 MoE expert data path

解释 router、expert execution、cache miss 和 host-device transfer。明确区分：

- residency 通过保留 expert 消除一次 transfer；
- lookahead 不消除 transfer，而是争取在 deadline 前完成它；
- precision 同时改变质量、resident capacity 和 transfer bytes。

### 3.2 统一状态空间

每个 expert 只需要四类状态：absent、resident-low、resident-high，以及尚未 publish 的 in-transition 状态。Prefetch 不是新的永久状态，而是让 absent expert 在 deadline 前进入 resident state 的动作。

这一定义把 precision、cache 和 prefetch 放进同一个 expert state machine，避免将它们写成三个松散子系统。

### 3.3 为什么固定切分不够

围绕三种变化说明：

- **Quality value changes：** 不同专家和任务对量化误差的敏感度不同；
- **Reuse value changes：** routing locality 随任务、batch 和请求阶段变化；
- **Urgency changes：** 不同 expert 的 deadline 和可用 overlap window 不同。

因此，“先保留完整低精度池”“先满足预取再用 slack 提升精度”或“固定比例分给 cache”都只能作为特定条件下的边界策略，不能预设为一般最优原则。

### 3.4 问题定义与范围

统一目标是：

\[
\min_{\pi}\operatorname{P95}(\mathrm{TPOT};\pi)
\quad
\text{s.t.}
\quad M_{\mathrm{peak}}(\pi)\le B,
\quad \Delta Q(\pi)\le\epsilon,
\]

其中策略 \(\pi\) 同时决定 precision、residency 和 prefetch timing。

范围需要明确：

- \(B\) 是扣除 non-expert weights、KV cache、activation 和 workspace 后的 expert budget；
- quality constraint 相对固定 reference configuration 定义；
- 在线统计只是质量风险代理，不直接构成质量保证；
- 当前系统限定为 single-GPU expert offloading。

本节最后导出三个设计要求：统一比较动作、响应运行状态、在并发迁移下保持硬预算。

## 4. Design

Design 按“统一决策如何形成”组织，不再先讲 precision controller、再讲 prefetch controller。

### 4.1 System overview

DynaByte 包含四个逻辑部分：

1. **State monitor：** 观察 routing demand、reuse、deadline、带宽、计算窗口和质量代理；
2. **Action valuator：** 估计 promote、demote、retain、evict 和 prefetch 的边际价值；
3. **Joint coordinator：** 在质量和显存约束下选择兼容动作；
4. **Safe executor：** 负责 reserve、copy、publish 和 reclaim。

### 4.2 Unified action model

所有控制都写成 expert state transition：

| 动作 | 主要收益 | 主要成本 |
|---|---|---|
| Promote | 降低质量风险 | 额外字节和迁移成本 |
| Demote | 释放显存 | 增加质量风险 |
| Retain | 避免未来 miss | 持续占用显存 |
| Evict | 释放显存 | 增加未来 miss 风险 |
| Prefetch | 隐藏预计传输 | staging、带宽和污染风险 |

统一接口至少比较 quality impact、latency impact、byte cost 和 transition cost。Marginal value per byte 是决策原则，但正文不应在算法尚未确定时声称全局最优。

### 4.3 Joint coordination

每个控制周期遵循同一逻辑：生成候选动作，排除违反质量、deadline 或显存约束的组合，比较剩余动作的边际价值，处理动作间冲突，再提交给统一 admission gate。

这里最重要的叙事是：协调器直接比较“升级 A”“保留 B”和“预取 C”，而不是先给三个模块划定固定配额。

### 4.4 Two-timescale control

Residency/lookahead 对当前请求变化更敏感，可以快速更新；precision transition 成本更高、质量统计更慢，可以低频更新。两个时间尺度只是联合决策的实现近似：二者仍共享状态、机会成本和显存账目，不能重新变成两个独立控制器。

### 4.5 Safe execution

执行层维护一个不变量：published blocks、in-flight destinations、尚不可回收的旧版本和 staging 的总和始终不超过 \(B\)。所有状态转换使用 reserve--copy--publish--reclaim 协议。

必须区分：可选 precision migration 不应阻塞已有可用版本；真正 absent expert 的 demand fetch 仍可能阻塞 forward path。

## 5. Discussion

### 5.1 Coupling 的有效区域

DynaByte 的价值主要存在于中间预算区间：系统能够在部分高精度专家、更多低精度 residency 和有限 lookahead 之间作真实选择。全高精度可驻留、质量约束不可满足等极端情况，应被视为自然边界而不是主要贡献。

### 5.2 退化到简单策略

统一方法应自然退化：全量目标精度可驻留时关闭迁移；质量约束宽松且复用高时接近低精度 cache；复用弱而 overlap 充分时接近 deadline-aware prefetch；质量敏感专家长期稳定时接近静态 mixed precision。

### 5.3 Quality proxy 的边界

Hotness 不等于 quality sensitivity。合理定位是：离线 sensitivity 表示量化某专家的潜在损失，在线 routing 表示这种损失在当前 workload 中出现的概率，两者共同形成运行时风险估计。论文不能声称在线信号直接保证真实任务质量。

### 5.4 模型误差与系统范围

讨论 routing prediction、bandwidth contention、deadline estimation、workload shift 和 transition amortization 的误差。当前 claim 限定在 single-GPU、host-device expert offloading 和有限个实际 packed representations；multi-GPU、SSD tier 和联合 KV budgeting 留作扩展。

## 6. Related Work

Related Work 按“每类方法固定了哪个决策”组织。

### 6.1 Precision assignment

讨论 uniform PTQ、MoE mixed precision 和 sensitivity/frequency-based bit allocation。这类工作主要优化 fidelity，通常把 residency 和 transfer schedule 视为给定条件。DynaByte 使用其表示或 sensitivity，但进一步考虑 precision bytes 对另外两种用途的机会成本。

### 6.2 Expert caching and residency

讨论 routing-aware cache、locality、memory hierarchy 和 expert placement。这类工作主要决定保留谁，通常假设 expert precision 和 size 固定。DynaByte 将 retain/evict 与 precision transition 放入同一 byte ledger。

### 6.3 Expert prefetching

讨论固定或自适应 horizon、look-ahead routing 和 pregate prediction。这类工作主要决定何时搬运，通常在固定 representation 下权衡 overlap 与 pollution。DynaByte 同时考虑 representation 对 transfer time、capacity 和 quality risk 的影响。

### 6.4 Mixed-precision offloading systems

HOBBIT、DyMoE 等已经同时包含 mixed precision、cache 和 prefetch，必须单独正面比较。比较维度应是：

- precision 是 resident state、miss-path representation，还是允许跳过的计算；
- quality 是显式约束还是启发式保护规则；
- 三类动作是否共享显式的 byte opportunity-cost model；
- 系统是否根据运行状态比较三种动作的边际价值。

DynaByte 的差异必须落在优化目标和决策语义上，不能只说“我们也有动态精度和预取”或“不 drop experts”。

### 6.5 Serving and memory management

简要定位 KV paging、continuous batching、dense-model offload 和通用 allocator。DynaByte 管理预先划定的 expert budget，与上层 serving engine 互补。

## 7. 章节交接关系

- **Introduction → Background：** 要比较一字节的三种用途，必须先把表示、驻留和 deadline 写成同一状态空间。
- **Background → Design：** 固定切分的问题不在单个控制器，而在三类动作从未在同一质量、延迟和字节账目下比较。
- **Design → Discussion：** 联合分配只在存在真实机会成本时有价值，因此要明确退化条件、代理误差和范围。
- **Discussion → Related Work：** 方法边界提供了公平比较标准：相关系统是统一状态空间中的边界策略，或使用了不同优化目标。
- **Design → Evaluation：** 后续只需验证三个高层问题——联合分配是否必要、边际价值是否可估计、在线实现是否值得其开销；具体规划留到下一阶段。

## 8. 当前不展开的内容

- Evaluation 的研究问题、baseline 和实验矩阵；
- 模型、设备、预算点、数据集和图表；
- 数值结果与摘要结果句；
- reviewer score 和投稿判断。

后续应先据此改写 Introduction、Background、Design、Discussion 和 Related Work，再从实际保留下来的 claim 反向生成 Evaluation。
