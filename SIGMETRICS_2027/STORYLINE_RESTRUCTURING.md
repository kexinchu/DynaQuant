# DynaByte 合并论文：故事线、Observation 与 Novelty 重构

> 当前写作主线以 [STORYLINE_WRITING_OUTLINE.md](STORYLINE_WRITING_OUTLINE.md) 为准；本文保留为详细论证、图表 provenance 与后续 evaluation 规划参考。

> 日期：2026-10-02  
> 状态：研究定位与写作大纲建议，待联合实验验证。  
> 目标：按接近 9/10 的大纲标准组织问题、因果链、贡献和证据；不代表已达到相应论文评分。  
> 本笔记整理本轮讨论，不修改论文正文，不将预期收益写成已验证结果。

## 0. 执行版结论：先改变论文在讲什么

### 0.1 一句话主线

> **DynaByte 在给定 GPU 专家显存预算和质量损失上限时，联合决定每个专家的表示精度、驻留状态和预取时机，使有限显存字节在 fidelity、residency 与 lookahead 三种用途之间按边际收益动态分配，从而降低 MoE 推理的尾延迟。**

这句话与当前 PDF 的核心区别不是措辞，而是优化问题不同：

- 当前 PDF 实际采用“先满足 latency，再用 slack 提升 precision”的词典序规则；quality 只是 slack 的用途。
- 重构后的论文必须把质量写进可行域：比较策略时固定 \(\Delta Q\le\epsilon\)，再最小化 P95 TPOT／TTFT。
- 当前 PDF 是按 \(\Phi=B/P_{\mathrm{lo}}\) 切换两个边界策略；重构后的论文要证明，在目标预算区间内，三个动作确实存在不可忽略的机会成本，联合协调优于共享预算且充分调优的组合。

若二维扫描不能证明最后一点，不应继续把论文包装成“联合控制”；应退回为一个更窄的 phase-boundary／configuration model 论文。

### 0.2 审稿人应该在 30 秒内理解的冲突

同一个空闲显存块只能做三件事之一：

1. **Fidelity：** 把一个已驻留专家从低精度提升到高精度，争取质量收益；
2. **Residency：** 多保留一个低精度专家，避免未来一次或多次 miss；
3. **Lookahead：** 提前放入即将使用的专家，把不可避免的传输藏进计算窗口。

三者不是固定优先级。一次 precision upgrade 是否值得，取决于它挤掉的 resident/prefetched expert 会产生多少 deadline miss；一次更深的 prefetch 是否值得，取决于它占用的字节是否比用于质量恢复更有价值。任务、batch、prefill/decode、带宽和路由局部性变化时，这个相对价值也变化。

这才是两项旧工作合并后产生的新问题。若正文仍写成“\(\Phi\ge1\) 只做 precision；\(\Phi<1\) 先做 prefetch、剩余空间做 precision”，审稿人会把它理解为 regime switch，而不是 joint optimization。

### 0.3 推荐的因果链

全文只保留下面一条链，不再让“动态精度”和“动态预取”各讲一遍：

> **有限 expert-byte budget**<br>
> \(\rightarrow\) 同一字节在质量、驻留和提前量之间竞争<br>
> \(\rightarrow\) 每种用途的边际收益随请求阶段与工作负载改变<br>
> \(\rightarrow\) 静态切分和顺序优化会落在次优操作点<br>
> \(\rightarrow\) 需要一个共享成本模型和联合 admission 机制<br>
> \(\rightarrow\) 用快 residency/prefetch 环和慢 precision 环近似联合决策<br>
> \(\rightarrow\) 在相同峰值显存与质量约束下改善尾延迟。

每个箭头都需要一项证据：

| 因果箭头 | 最小证据 | 失败后的处理 |
|---|---|---|
| 字节竞争真实存在 | precision quota × horizon 二维扫描，显示 feasible boundary 和 latency surface | 若只有边界变化而无性能差距，收缩 novelty |
| 收益随状态变化 | 同一运行内的 miss bytes、overlap capacity、late bytes 时间线 | 若固定点已覆盖主要状态，删除在线控制主张 |
| 顺序优化次优 | precision-first、prefetch-first、最佳静态联合、test oracle 四者同预算比较 | 若差距在噪声内，改写为静态配置模型 |
| hotness 值得用于 precision | promotion 的 marginal quality gain 与 hotness/sensitivity 的相关性 | 若相关性弱，加入 sensitivity 或冻结 identity |
| 两时间尺度合理 | hot-set persistence 对比 promote+demote break-even | 若不能摊销，删除慢环迁移 |
| 系统收益成立 | 同 \(B\)、同 \(\epsilon\) 的 P95 TPOT、TTFT、吞吐及迁移字节 | 若只改善内部 wait，不声称用户可见收益 |

### 0.4 推荐标题、摘要中心句和贡献

**推荐标题**

> **DynaByte: Quality-Constrained Joint Precision, Residency, and Prefetch Control for MoE Inference**

若实验最终只支持配置规律而不支持在线控制，则改为：

> **DynaByte: Mapping the Precision--Residency Tradeoff in Memory-Constrained MoE Inference**

不建议继续使用 *The Optimal Expert Prefetch Horizon Is Set by Precision*。它同时有三个风险：因果关系过度单一、`optimal` 需要更强证明、且会让 precision 看起来只是 prefetch 的输入参数。

**摘要的四句骨架**

1. MoE expert memory 在有限 GPU 上既决定表示质量，也决定 miss 数量和可用于隐藏 miss 的 lookahead。
2. 我们测得，单独调优 precision quota 或 prefetch horizon 后再组合，会因为共享字节和逐层 deadline 落在较差操作点；该差距随 prefill/decode、batch 和带宽变化。
3. DynaByte 用统一的 byte/deadline/value 模型协调 precision、residency 和 prefetch，并通过两时间尺度控制器与 reserve--copy--publish--reclaim 协议安全执行。
4. 在相同峰值显存与质量损失约束下，DynaByte 相对最强共享预算基线降低 P95 TPOT/TTFT **[结果]**，且距离 test oracle **[结果]**。

在正式结果出现前，第 2 和第 4 句只能作为待验证模板，不能写成完成时。

**贡献应压成三项**

1. **Finding：** 定量发现 quality-constrained precision、residency 与 prefetch 在一段实际预算区间内存在显著耦合，并界定耦合消失的边界区间。
2. **Method：** 提出包含 expert bytes、逐层 deadline、预测 miss、质量价值和迁移摊销的联合决策方法，并用两时间尺度在线近似实现。
3. **System and evidence：** 实现硬预算下的原子表示切换和预取 admission；在统一栈上以强共享预算基线、静态联合搜索和 oracle 验证延迟—质量 Pareto 改善。

内存池、versioned handle、EMA、hysteresis 和 \(\pm1\) 更新是实现机制，不应各自列为论文级 contribution。

### 0.5 重新组织后的正文顺序

| 章节 | 叙事任务 | 必须留下的核心内容 | 建议删除／后移 |
|---|---|---|---|
| 1. Introduction | 建立“一字节三用途”的冲突和研究问题 | 质量约束下的目标、现有组合为何不够、三项贡献、关键结果 | 不在引言推导三阶段和控制器细节 |
| 2. Scope and Background | 固定部署语义与比较边界 | single GPU、两种 packed representations、host/device path、\(B\) 的完整口径、质量参考 | 大段 related-work taxonomy 后移 |
| 3. Characterizing the Coupling | 用数据证明论文为什么必须存在 | O1 transfer pressure；O2 二维 surface；O3 quality value 与 persistence | 旧稿中不能直接支持 coupling 的零散 observation |
| 4. Problem Formulation | 定义联合优化而非预设策略 | P95 TPOT objective、quality/memory/deadline constraints、动作与迁移成本 | 把 \(\Phi\) 从“决定算法”降为 boundary descriptor |
| 5. DynaByte | 解释如何近似求解与安全执行 | value/cost estimates、快慢环分解依据、shared admission、fallback | 不先验写“prefetch first, precision from slack”为最优 |
| 6. Implementation | 让系统 claim 可复现 | packed sizes、streams、pool、atomic publish、demand miss、instrumentation | 泛化到未实现的 multi-GPU |
| 7. Evaluation | 按审稿问题而非组件顺序组织 | necessity、end-to-end、adaptation、quality validity、overheads/boundaries | 弱 naive composition 不作 headline baseline |
| 8. Related Work and Discussion | 精确界定与 HOBBIT/DyMoE 等差异 | 优化目标、状态语义、在线信号和证据上的差异 | “我们也有量化+预取”式 novelty |
| 9. Conclusion | 只回答被数据支持的问题 | 操作区间、收益、边界 | 不重复未证实的 optimality |

### 0.6 对当前 PDF 的模拟审稿意见

#### Summary

论文研究单 GPU、显存受限的 MoE inference，提出用 \(\Phi=B/P_{\mathrm{lo}}\) 把部署划为三阶段，并用快 prefetch loop 与慢 precision loop 管理 expert bytes。问题契合 SIGMETRICS 的系统测量、建模与性能分析范围；内存账目、可证伪假设和复现实验协议写得认真。然而，当前稿件尚没有联合系统结果，核心优化目标没有纳入质量约束，所谓 joint policy 实际是一个词典序的 regime-switching policy，关键 baseline 也偏弱。因此，当前版本无法支撑标题、摘要和贡献中的 optimality/coupling 主张。

#### Strengths

- 问题重要且及时：MoE expert footprint、offload latency 和质量之间确有现实冲突。
- \(B\) 的口径、staging reserve、publish-before-reclaim、runtime counter 与 NVML 双重核验，体现了较好的系统严谨性。
- 论文主动写出 falsification conditions，区分 mean miss、p95 miss、idle bandwidth 与 in-overlap bandwidth，这一点符合 measurement paper 的气质。
- 评测协议覆盖 phase boundary、request 内变化、模型误差、控制开销和质量，不只是报一个 speedup。
- 文字总体清楚，限制条件也比多数早期系统稿交代得充分。

#### Major weaknesses

1. **当前不是一篇完成的实证论文。** 结果表为空，正文明确说“this paper is a model, a controller, and a protocol, not a performance result”。SIGMETRICS 要求 principled solution 之外还要 empirical validation；仅凭协议不能评估有效性。
2. **优化目标与故事线不一致。** 故事线希望求解 \(\min \mathrm{P95(TPOT)}\) s.t. \(\Delta Q\le\epsilon\)，但 PDF 明确写“quality enters only as the use of slack”。因此当前方法没有联合优化 latency 和 quality，只是在 latency-first 规则后使用剩余空间。
3. **“joint”目前更像阶段切换。** Phase II 固定 \(S=0\) 后只调 precision；Phase III 先找可行 \(S\)，再从 slack 分配 pins。方法没有比较一个 precision byte 与一个 residency/prefetch byte 的共同边际价值。
4. **核心最优性推导不充分。** \(M_{\mathrm{miss}}(S)\le C S T_\ell\) 是窗口总量条件，不保证每个专家在自己的逐层 deadline 前到达；feasible-set 为空也不能推出 \(S_{\min}\) 一定最小化残余等待。标题中的 “optimal” 因而过强。
5. **决定性 motivation 缺失。** activation density、任务间 hot set 变化和 horizon 的 U 形曲线分别支持两个旧子问题，却没有直接证明共享预算下独立或顺序优化造成显著损失。
6. **质量代理链条断裂。** top-10 hot experts 随任务变化，不等于对这些 experts 提升精度能改善任务质量；还缺 expert-level marginal quality sensitivity 和迁移成本的摊销时间。
7. **主要组合基线是稻草人。** `Naive composition` 让两个控制器都按完整 \(B\) 独立申请，必然冲突。真正有说服力的基线应共享预算并充分调优，包括两种顺序优化、最佳静态预算切分和 deployable static joint policy。
8. **H2 被赋予了错误的生死条件。** 即便精度变化后 argmin \(S\) 不移动，精度仍可能通过 miss bytes、miss identity、cache residency、curve height 或 quality-feasible region 产生耦合。反之，\(S\) 移动也不自动证明在线联合控制值得其复杂度。
9. **与近邻工作的差异仍不够硬。** HOBBIT 已联合 mixed-precision loading、prefetch 与 cache；DyMoE 也同时包含动态精度、look-ahead prefetch 和 cache management。差异必须落在优化目标、byte opportunity cost、质量约束和强基线结果上，而不是组件清单。
10. **范围过大而主证据尚无。** 六个模型、三类设备、多个质量任务和七档预算会消耗大量实验资源，却未优先回答“联合控制是否必要”。应先完成最小 P0 扫描，再决定扩展范围。

#### Questions I would ask the authors

1. 在相同 \(B\) 和 \(\Delta Q\le\epsilon\) 下，DynaByte 相对最佳静态联合配置与两种顺序优化究竟改善多少 P95 TPOT？置信区间是多少？
2. 如果允许牺牲一部分 low-precision residency 来满足更严格的质量要求，为什么 \(\Phi\ge1\) 仍必然有 \(S^*=0\)？这是由 workload 决定，还是被 latency-first 目标预设出来的？
3. aggregate overlap predicate 如何处理第一个急需 expert 已过 deadline、但整个窗口总 bytes 仍满足不等式的情况？
4. hotness 对 expert promotion 的 marginal quality gain 的 rank correlation 是多少？与离线 sensitivity 相比如何？
5. DynaByte 的收益来自更好的配置选择，还是来自不同的底层 cache/pool/issue-order 实现？共享执行机制后的策略净收益是多少？

### 0.7 分数

以下采用两套尺度，避免“故事潜力”和“当前可投稿性”混淆。

#### 当前 `main.pdf`，按今天直接投稿

| 维度 | 分数（10 分） | 判断 |
|---|---:|---|
| 问题重要性／SIGMETRICS fit | 8.0 | 强，属于 systems for ML、measurement 和 resource control |
| 故事统一性 | 5.0 | 有统一 byte budget，但方法仍是两个边界策略的切换 |
| 技术新颖性 | 4.5 | 相对 HOBBIT/DyMoE 的差异尚未被机制或结果钉牢 |
| 模型严谨性 | 5.0 | 账目清楚；deadline、排队和最优性仍有缺口 |
| 实验设计 | 7.0 | 协议全面、可证伪，但 baseline 仍需加强 |
| 实验证据 | 1.0 | 联合结果为空，这是决定性缺陷 |
| 写作与组织 | 6.0 | 清楚但 thesis 与 objective 不一致，且 future-tense 过多 |
| 可复现性准备 | 7.0 | artifact 规划好，尚未被实际结果验证 |

**综合：3.8/10。**<br>
**五档推荐：1/5，Strong Reject。**<br>
决定该分数的不是想法差，而是它目前还是研究计划；即使忽略空结果，中心 claim 与实际算法也没有完全对齐。

#### 仅评价当前故事线构想，不考虑实验尚未完成

**6.5/10，Borderline。** 问题和统一资源视角成立，但“为什么必须联合”仍是一项待发现的 empirical result，而不是已由现有 observations 推出的事实。

#### 完成上述重构且 P0/P1 结果成立后的潜力

**7.5--8.0/10；五档推荐约 3.5--4/5（Weak Accept 到 Accept）。** 达到这一档至少需要：

- 二维扫描显示强共享预算基线与联合决策之间有稳定、非噪声级差距；
- 相同质量约束和真实峰值显存下有用户可见的 P95 TPOT/TTFT 改善；
- hotness/sensitivity 与质量收益的链条成立，或诚实删除动态 identity claim；
- deadline-aware 模型能解释主要趋势，在线策略接近 calibration static joint 和 test oracle；
- 与 HOBBIT、DyMoE 的差异由可运行对照或严格 matched-policy ablation 支撑。

### 0.8 Go / No-Go 顺序

1. **先跑一个模型、一个 GPU、2--3 个 Phase III 预算的 quota × horizon 网格。** 同时记录质量、P95 TPOT、miss bytes、late bytes 和峰值显存。
2. **从同一网格构造四个操作点：** precision-first、prefetch-first、calibration static joint、test oracle。不要先实现完整在线控制器。
3. **Go 条件：** 联合点在多个 workload slice 上形成稳定 Pareto 改善，且收益不是由 baseline 的预算违规或底层实现差异造成。
4. **若 Go：** 再验证 hotness/sensitivity 和 persistence，决定是否需要慢 precision loop。
5. **若 No-Go：** 停止“joint online control”主线，把论文收缩为预算 phase map、deadline-aware horizon model，或拆回两篇独立工作。

这个顺序能最早回答论文是否值得继续，而不是先把六模型、三设备和所有 provenance 补齐后才发现联合收益不存在。

## 1. 核心判断

两篇论文可以围绕一个统一问题合并，但中心需要从“动态精度 + 动态预取”提升为：

> **在固定 GPU 显存预算与模型质量要求下，联合选择专家的精度、驻留集合和预取时机，以降低推理尾延迟。**

共享显存提供了合并的交点，但仅有共享预算还不足以构成强贡献。新论文需要证明：分别优化精度与预取，会在目标工作区间内产生可量化的损失；联合决策能够超过充分调优、共享预算的组合基线。

### 1.1 两篇原工作的关系

| 工作线 | 原始问题 | 原始决策 |
|---|---|---|
| 动态精度 | 专家全量驻留时，有限高精度容量应该给谁？ | 哪些专家保持高精度，何时调整 |
| 动态预取 | 专家不能全量驻留时，如何减少等待和缓存污染？ | 哪些专家提前搬入，提前多少层 |

若新稿只是“显存足够运行动态精度，显存不足运行动态预取”，能够说明适用范围扩大，却无法充分证明新的研究贡献。

真正值得研究的是两者同时有价值的区域：

> 显存不足以让所有专家保持目标精度，却足以在精度、驻留覆盖和预取之间做出有意义的选择。改变精度会改变缓存与传输成本；改变预取会改变可用于质量恢复的空间。

全部高精度驻留、全部低精度驻留适合作为边界情况，不宜承担全文主要 novelty。

### 1.2 工作标题

**DynaByte: Joint Precision and Residency Control for Memory-Constrained MoE Inference**

这里的 residency 包含当前保留什么，以及通过预取何时让专家驻留。

暂不建议继续采用 *The Optimal Expert Prefetch Horizon Is Set by Precision*：

- Precision 是影响 horizon 的变量之一，带宽、计算窗口、预测误差和驻留集合也很重要。
- “Optimal”需要比当前近似模型更强的论证。
- 该标题使动态精度成为预取的辅助变量，削弱原工作的质量目标。

## 2. 统一问题与目标

### 2.1 三个相互影响的决策

- **精度：** 每个专家占多少空间、搬运多少字节，以及相应质量代价。
- **驻留：** 哪些传输可以避免。
- **预取：** 无法避免的传输能否在使用前完成。

建议使用一个明确的主要优化目标：

$$
\min_{\pi}\ \operatorname{P95}(\mathrm{TPOT})
\quad\text{s.t.}\quad
M_{\mathrm{peak}}(\pi)\le B,
\qquad
\Delta Q(\pi)\le\epsilon.
$$

其中，$\pi$ 为运行时策略；若 $B$ 指专家显存预算，$M_{\mathrm{peak}}$ 必须包括专家驻留、预取和迁移瞬时副本，非专家权重、KV cache、activation 等另行预留。进程峰值与设备容量也需独立核验。

TTFT 单独报告，并明确相应约束或适用范围。质量损失相对固定参考配置、按任务定义，不能随意把不同任务分数合成一个保证。

### 2.2 质量约束的边界

质量约束首先是离线评估标准。在线 hotness 等信号只是代理；没有额外证明时，不能声称运行时保证真实任务质量。

当前“先满足 overlap，再把剩余空间给高精度”的规则，可以作为待验证策略，但不能预先当作全局最优原则。

## 3. Observation：重排为三步因果链

现有激活密度、热专家变化、预取 U 形曲线分别支持两个子系统。合并论文需要一个直接证明耦合的关键观察。

| Observation | 要说明的问题 | 对设计的要求 | 当前证据状态 |
|---|---|---|---|
| O1：传输压力随执行阶段变化 | 同一请求内，待搬运字节与可用计算窗口共同变化 | 不能固定使用一种驻留／预取策略 | 有密度和 horizon 材料，需同栈重测 |
| O2：精度改变预取与驻留的收益 | 提升精度可能挤掉有价值的驻留或预取；降精度可能消除 miss | 精度与预取必须共享成本模型 | **关键新实验，尚缺** |
| O3：精度收益与实现成本具有不同时间尺度 | 专家重要性变化，但精度迁移需要时间和带宽偿还成本 | 调整需要考虑持续时间与迁移代价 | 热集变化仅提供部分支持 |

### 3.1 O1：传输压力变化，而不只是激活密度变化

Prefill 访问更多专家，但也可能提供更长的计算窗口。更有解释力的量是：

$$
\text{transfer pressure}
=
\frac{\text{需要搬运的专家字节}}
{\text{有效带宽}\times\text{可用重叠时间}}.
$$

稿件中的 94% 与 26% 激活率差异，只能说明工作集变化，不能直接推出“prefill 不可隐藏、decode 可以隐藏”。这些数值仍应按正式证据要求重测。

**建议动机图：** 同时展示实际 miss bytes、可用 overlap window、最终 exposed wait。图要解释同一策略为何失效，而不只是展示工作负载变化。

### 3.2 O2：同一份显存的三种竞争用途

新增一个低精度驻留专家，可以避免未来 miss；提升一个专家的精度，可以改善质量；提前驻留未来专家，可以减少等待。这三种用途的收益随当前工作负载变化。

**说明性例子，不是实验结果：** 若高精度版本大小是低精度版本的四倍，一次精度提升新增的空间相当于三个低精度专家的驻留空间。只有被牺牲的驻留／预取价值足够低时，这次提升才值得做。

**决定性图：二维响应面。**

- 横轴：高精度配额。
- 纵轴：预取深度。
- 颜色：尾延迟。
- 标注：质量可行区、内存不可行区及不同策略的操作点。

在同一图中比较：

1. 先确定精度，再调预取。
2. 先确定预取，再分配精度。
3. 充分搜索的静态联合配置。
4. 在线联合策略。

应当验证的发现是：**分别调好的配置，组合后会偏离质量约束下的好操作点。**

仅证明“精度变化让最佳 $S$ 移动”还不够。即使 $S$ 没变，miss 数量、迁移流量或质量可行区也可能改变。因此，不应把当前 H2 作为联合论文唯一成立条件。

### 3.3 O3：频繁访问不等于值得提升精度

热专家变化说明静态访问统计可能过时，不能直接证明动态提升这些专家能够改善质量。

需要分别验证：

1. Hotness 能否预测精度提升的质量收益。
2. 热集保持多久，是否足以摊销提升和降级成本。

第二条成立，才为两时间尺度设计提供依据。第一条若不成立，应考虑加入离线敏感度，或收缩在线身份调整的主张。

### 3.4 两篇原稿的图：哪些可以复用

完整的源图号、文件路径、SHA-256、实验注册状态和补注册动作见 [FIGURE_PROVENANCE.md](FIGURE_PROVENANCE.md)。这里的“已有图”一律先作为 provenance 输入登记；只有 `EVIDENCE_REGISTERED` 才表示其底层实验 claim 也完成注册。`ASSET_REGISTERED / EVIDENCE_MISSING` 只证明现有 PDF／PNG 的身份，不能据此进入最终论文。

这里需要区分两种“复用”：

- **原图直接复用：** 原 PDF／PNG 及其数据口径不变，只改编号和 caption。
- **证据或构图复用：** 保留原问题、原始数据或视觉形式，但在统一运行栈上重测、重画，不能把旧图当作新稿的联合证据。

#### 3.4.1 Motivation 相关图

| 原稿图 | 原图表达的事实 | 复用结论 | 在新稿中的安全用途 | 主要限制／改动 |
|---|---|---|---|---|
| 动态精度稿 Fig. 2 `expert-active`：WikiText／GSM8K／HumanEval 的 Layer-15 expert dispatch | 不同任务的热专家身份变化，三组 top-10 不重合 | **有条件直接复用；当前仅 asset 已登记** | 作为 O3 的前半步：静态身份可能过时 | 只能说明访问频率变化，不能说明提升这些专家会改善质量，也不能证明变化持续到足以摊销迁移；正式稿前需补齐 3 个 `routing_hotset` claim |
| 动态精度稿 Fig. 3 `ppl`：低精度专家比例与 WikiText perplexity | 冻结 ranking 时，精度配额与质量存在连续权衡 | **有条件直接复用；当前仅 asset 已登记** | 放在 Background／Scope，建立质量约束不是离散开关 | 它不是联合控制结果，也不证明 hotness ranking 最优；正式稿前需补齐 2 个 `perplexity_curve` claim |
| 动态预取稿 Fig. 2 `M1`：horizon 与 waiting/cache-miss latency | horizon 存在 overlap--pollution tradeoff，较大 $S$ 可能引起 miss cliff | **只复用问题和构图，不建议原图进入最终稿** | 作为 O1／O2 新扫描的绘图原型 | 模型、预算和运行栈与新稿不一致；现图把 waiting 与 cache-miss 分开画但 caption 合并解释，也不能展示精度对曲线的影响；当前仓库未发现其 raw-artifact manifest |
| 动态精度稿 Table I `expert_activation_ratio` | prefill／decode 的 active set 密度不同 | **可复用表格结构；数值需统一协议复核** | 作为 O1 的 workload 描述或 appendix sanity check | 密度不是 transfer pressure；不能由 94% vs. 26% 直接推出哪一阶段更难隐藏传输；当前 manifest 仅完成 2/6 个 model-stage claim |
| 动态精度稿 Fig. 1 `waiting-vs-prompt` | 串行 demand load 的 blocking waiting 随 prompt length 增长 | **成图与底层 claim 已登记，但不作为新稿核心图直接复用** | 最多作为 appendix 中的 offload 下界／压力背景 | Qwen 是直接测量，Phi 是 cold-cache trace replay；它不是带 overlap 的统一系统结果，也没有同时展示 miss bytes 和 overlap window |
| 动态预取稿 Fig. `M3`：固定 $S$ 下 batch size 与 latency components | batch 改变 waiting／cache behavior | **不直接复用** | 重测后可并入 O1 | 图中横轴范围和正文所称 batch 1--32 不一致；没有质量维度或精度维度，且旧模型／旧栈无法支撑联合 claim |
| 动态预取稿 Fig. `M5`：batch／embedding distance 与 activated experts | request composition 改变 active set | **不直接复用** | 若 routing diversity 最终进入控制器，可重画为补充分析 | 图中没有直接画出 caption 所称的端到端 latency variation；代理量、active-set 和 latency 之间的链条过长，不适合承担核心 motivation |
| 动态预取稿 Fig. `overview`：四类 prefetch 方法 | 固定与动态 horizon 的机制区别 | **不直接复用** | 可提取“何时发起预取”的小图标语言 | 没有 precision／residency budget；“跟踪 per-period optimum”等原 claim 尚不能迁移到联合系统 |

因此，最终稿 motivation 中可以原样保留的候选只有 `expert-active` 和 `ppl`，而且都必须满足 provenance gate，并分别限定为“热身份变化”和“配额—质量关系”两个窄结论。它们是背景证据，不是新论文的决定性证据。`M1` 最有复用价值，但复用的是实验设计和 U 形构图，不是旧文件本身。

#### 3.4.2 其余系统图与结果图

| 图组 | 复用结论 | 处理建议 |
|---|---|---|
| 动态精度稿 architecture／promotion procedure | **改造复用** | 保留 versioned handle、reserve--copy--publish--reclaim 机制；加入 absent state、prefetch admission、统一 byte budget 和 deadline，不能继续只画 `Pool_hi/Pool_lo` |
| 动态预取稿 architecture／predictor-controller co-design | **改造复用** | 保留 horizon、predictor、prefetch path；与 precision/residency coordinator 合并为一张统一架构图，避免两个旧系统左右拼接 |
| 两篇原稿的 end-to-end、throughput、ablation、predictor、cache-policy 等结果图 | **不直接复用为新稿主结果** | 可用于确定实验范围和 sanity check；新稿主结果必须来自同模型、同硬件、同预算、同质量约束和同基础执行机制的联合协议 |

当前合并稿的 `fig:phase` 也不应作为 motivation 的经验证据。它是模型示意图，而且 caption 中“先取最小可行 horizon、剩余空间再给高精度”的顺序已经预设了待验证策略。若 O2 表明最优点不是这种词典序分配，应重画或删除该图，而不是用示意图代替耦合实验。

### 3.5 Motivation 需要补充的图

建议把 Section 3 控制在三张核心实证图，加一张可选 teaser。每张图只承担一个可证伪结论。

| 优先级 | 建议图 | 推荐面板与坐标 | 必须支撑的结论 | 验收／否决条件 |
|---|---|---|---|---|
| **P0，决定论文是否成立** | **O2：Precision quota × prefetch horizon 二维耦合图** | 每个代表性预算一组：横轴 high-precision bytes 或 quota，纵轴 $S$；主 heatmap 为 P95 TPOT／exposed wait；叠加质量可行边界、显存不可行 mask，以及 precision-first、prefetch-first、最佳静态联合、在线联合四类操作点 | 在相同 $B$ 与 $\Delta Q\le\epsilon$ 下，单独或顺序调优会偏离较好的联合操作点；精度变化通过 expert bytes、miss set 或 slack 改变预取收益 | 若顺序优化与联合搜索的差距落在噪声内，收缩“必须联合在线控制”的主张；不得只展示最佳 $S$ 是否移动 |
| **P1** | **O1：同一请求内的 transfer-pressure timeline** | 对齐 prefill→decode 或若干 control windows；面板 (a) actual miss bytes，(b) $C\times T_{\mathrm{overlap}}$ 与逐层 deadline，(c) exposed wait／late bytes；同图比较同一固定策略与可适应策略 | active-set density 本身不足；真正决定 stall 的是需搬运字节、可用 overlap window 和到达 deadline 的共同变化 | 三个量必须来自同一次运行、同一时间轴；不能把 replay、串行 load 和 overlapping runtime 混在同一因果图中 |
| **P1** | **O3：质量收益与迁移摊销的两时间尺度图** | 面板 (a) expert promotion 的实测边际质量收益或 loss/NLL 变化，对比 hotness、离线 sensitivity、二者组合的排序质量；面板 (b) hot-set／top-$k$ overlap 随窗口间隔衰减，并标出一次 promote+demote 的 byte/time break-even | hotness 是否能预测“值得升精度”，以及该身份是否保持得足够久以覆盖迁移成本 | 若 hotness 与质量收益相关性弱，方法必须加入 sensitivity 或降低在线身份调整 claim；若保持时间短于 break-even，不能把慢精度环作为收益来源 |
| **P2，可选 teaser** | **One byte, three uses 概念＋代表性 operating points** | 左：同一显存 byte 用于 low-precision residency、precision upgrade、prefetch staging 的三种用途；右：从 O2 抽取一个代表性切片，标出三种选择的机会成本 | 让读者一眼看懂“两个旧系统相加”为什么不是新问题的答案 | 左侧只能是概念图，不能伪装成测量；右侧必须来自 O2 正式数据，否则暂不放 teaser |

推荐版面顺序：

1. Introduction 放 P2 teaser（仅在 O2 数据完成后）。
2. Characterization 首先放 O1，说明状态为何变化。
3. 随后放 O2，证明两个控制维度必须在同一预算下考虑；这是全篇最重要的 motivation 图。
4. 最后放 O3，解释为什么需要快预取／慢精度的两时间尺度实现。

原有 `expert-active` 与 `ppl` 若保留，应缩为 O3 或 Background 的 supporting panels，避免与三张新核心图并列成四个互不相连的 observation。

### 3.6 图复用与新图的统一审计门槛

任何进入最终稿的 motivation 图都应记录：模型与 checkpoint、两种 packed representation 的实际 bytes、GPU 与有效 H2D 带宽、专家预算 $B$、KV／activation reserve、batch／prompt／decode 协议、预测器与缓存策略、运行 commit、raw samples 和绘图命令。除此之外：

- 直接复用旧图时，caption 必须明确它是 preliminary、blocking-only、trace replay 或 frozen-scheduler 中的哪一种，不能扩大成联合系统结论。
- 用旧数据重画时，必须保留原协议边界；换了模型、预算、量化格式或运行时后，应视为新实验而非“美化旧图”。
- O1--O3 的主图必须来自统一栈；不同原稿中的点不能拼成一个 joint curve 或 joint heatmap。
- 图中 quality-feasible、memory-infeasible、oracle 和 deployable operating point 必须用不同视觉语义标记，避免把 oracle 搜索结果当作在线可得结果。
- 每张图在开跑前先写一句“若结果相反，删除或收缩哪条 claim”，防止只保留支持当前故事线的切片。

#### 3.6.1 Provenance 的双层注册

不要把“图文件存在”与“实验结论可审计”混为一谈：

1. **Asset registry：** 在 [FIGURE_PROVENANCE.md](FIGURE_PROVENANCE.md) 登记源论文、原图号、文件路径与 SHA-256。两篇原稿当前正文引用的图已完成这一层盘点。
2. **Evidence registry：** 实证图还必须在 `results/paper/manifest.json` 或后续 joint manifest 中登记 raw artifact、claim ID、复现实验命令、commit 和输入／输出哈希。概念架构图只需第一层，但重画时仍需做语义来源审计。

当前 motivation 的 evidence 状态：

| 项目 | Asset | Evidence | 下一动作 |
|---|---|---|---|
| Fig. 1 `waiting-vs-prompt` | 已登记 | 已登记，但协议很窄 | 保留原限制；不升级 claim |
| Fig. 2 `expert-active` | 已登记 | 缺 3 个 `routing_hotset` claim | 补 raw dispatch bundle 后再直接复用 |
| Fig. 3 `ppl` | 已登记 | 缺 2 个 `perplexity_curve` claim | 补 raw window／单点 artifact 后再直接复用 |
| Table I activation density | LaTeX 已定位 | 仅 2/6 claim | 若保留，补 4 个 Qwen model-stage claim，并在统一栈重测 |
| DynaMoE Fig. 2--4 motivation plots | 已登记 | 当前仓库未发现 claim manifest | 优先统一栈重测，不把旧 PDF 当联合证据 |

#### 3.6.2 执行顺序

1. **先跑 P0/O2 二维扫描。** 它决定“联合控制”是否是论文主线，优先级高于修复所有旧结果图。
2. **并行补两张可直接复用图的 evidence provenance。** 即 `expert-active` 的 3 个 hot-set claim 和 `ppl` 的 2 个 curve claim；补不齐就从正文候选降为内部参考。
3. **再跑 O1。** 所有面板必须来自同一次 overlapping-runtime 运行，禁止拼接 blocking replay 与其他运行。
4. **再跑 O3。** 先验证 hotness 是否预测质量收益，再决定是否保留慢精度身份调整。
5. **最后才重画 teaser 与统一架构图。** Teaser 的实证切片来自 P0；架构图只表达最终被实验支持的控制关系。

### 3.7 图的生产路线与 S2 范围

Motivation 图按内容类型锁定生产方式：

| 图 | 生产方式 | S2 是否生成 | 约束 |
|---|---|---:|---|
| O1 transfer-pressure timeline | 统一运行栈 telemetry + 确定性绘图脚本 | 否 | 同一运行、同一时间轴，注册 raw samples 与绘图命令 |
| O2 precision quota × horizon surface | 联合网格实验 + 确定性绘图脚本 | 否 | infeasible mask、quality boundary 和 operating points 均来自真实 artifact |
| O3 quality value／migration amortization | sensitivity／persistence artifact + 确定性绘图脚本 | 否 | hotness、sensitivity、break-even 分开注册 |
| 原 Fig. 2 `expert-active`、Fig. 3 `ppl` | evidence 补注册后复用原 PDF | 否 | 不由模型重画，不扩大原 claim |
| 可选 `One byte, three uses` teaser 的纯概念部分 | S2 概念图候选 | 是 | 禁止 axes、curves、heatmaps、numbers、伪 operating points 或任何实验结论 |

S2 的 C01--C08 只探索概念 teaser 的读者路径和空间语法，包括 linear backbone、funnel/hourglass、单流程 conflict bridge、radial relation、deadline-centered、memory-bin、two-timescale 和 constraint-first 八种方向。它们不是 O1--O3 的替代品，也不能作为实验图占位。对应策略、脚本数据契约和 prompt-index 位于 `figure-studio-runs/dynabyte-sigmetrics-2027-storyline/outputs/`。

## 4. Novelty 定位

### 4.1 先承认近邻工作的覆盖范围

[HOBBIT](https://arxiv.org/html/2411.01433v1) 已结合动态精度加载、混合精度预取和缓存管理，因此不能声称“首次组合量化与预取”。

[DyMoE](https://arxiv.org/html/2603.19172v1) 包含动态精度、look-ahead prefetch 与混合精度缓存；既有 4/0 配置，也有不跳过专家的 4/2 配置。因此，“不 drop expert”可以作为执行语义约束，但不足以成为主要 novelty。

以上是本轮对这两篇原文的定向核对，不代表已完成全面相关工作检索。

### 4.2 三个贡献层次

以下均为需要实验支撑的目标，而非已完成发现。

| 层次 | 要建立的贡献 | 成立所需证据 |
|---|---|---|
| 发现 | 明确操作区间内，精度与预取存在影响性能的耦合，独立优化损失可量化 | 二维扫描与强组合基线 |
| 方法 | 利用耦合协调专家表示、驻留与预取，并显式计入迁移成本 | 超过充分调优的顺序优化与静态联合策略 |
| 系统 | 并发预取和精度切换在硬预算内执行，瞬时副本及回收都有正确账目 | 峰值内存、迁移开销、正确性和端到端收益 |

版本化 handle、内存池、异步 copy 支撑系统可信度，但不宜单独包装为最强 novelty。

两个控制器、EMA、$\pm1$ 更新、hysteresis 属于实现选择。贡献需要落在它们解决的可测问题上。

## 5. 设计逻辑

设计应组织为统一决策过程，避免两个原系统依次登场。

1. **估计需求。** 预测近期专家需求与使用时点，测量实际表示大小、计算时间及重叠带宽。
2. **评估动作。** 保留、驱逐、提升、降级、预取分别改变多少驻留字节、传输和质量代理收益。
3. **协调资源。** 在统一预算下选择可行配置；精度调整计入对近期需求的影响。
4. **限制迁移。** 只有预期收益能覆盖迁移成本时才调整，避免短暂热度引起反复搬运。
5. **安全执行。** 先预留瞬时空间，复制完成后发布新 handle，旧读者结束后回收。

当前“快预取、慢精度”的结构可以保留，但需要解释：这种分解为什么在目标工作负载上有效，与小规模联合搜索相比损失多少。

### 5.1 模型需要覆盖逐层 deadline

总传输量满足

$$
M_{\mathrm{miss}}(S)\le C\cdot S T_\ell
$$

并不保证每个专家都能在自己的使用 deadline 前到达。第一层急需的专家可能已经迟到，即使整个窗口的总量能够搬完。

当前公式应作为近似判据，并解释逐层 deadline、传输排队和预测错误的影响，不宜直接作为充分条件或全局最优证明。

## 6. 推荐论文大纲

以下按约 19 页正文分配写作空间；这是结构建议，不是会务规则确认。

| 章节 | 建议篇幅 | 必须回答的问题 |
|---|---|---|
| 1. Introduction | 2 页 | 固定显存与质量要求下，为什么独立控制精度和预取会损失性能？ |
| 2. Background and Scope | 1 页 | 单 GPU、两种表示、expert offload、预算范围、质量参考和执行语义是什么？ |
| 3. Characterizing the Coupling | 3 页 | O1 传输压力变化；O2 二维耦合；O3 收益持续时间与迁移成本 |
| 4. Problem Formulation and Model | 2 页 | 统一目标、逐层需求／deadline、内存与迁移约束、模型边界 |
| 5. Joint Runtime Control | 3 页 | 候选动作、预算协调、两时间尺度、噪声与预测错误处理 |
| 6. Implementation | 1 页 | 统一池、瞬时副本、发布回收、kernel 与控制开销 |
| 7. Evaluation | 5 页 | 联合收益、近邻比较、机制解释、动态适应、边界与开销 |
| 8. Discussion and Related Work | 1.5 页 | 适用区域、失效条件、与混合精度 offload 的具体差异 |
| 9. Conclusion | 0.5 页 | 回答研究问题，陈述被实验支持的发现 |

### 6.1 Introduction 的六段结构

1. **部署矛盾：** Sparse compute 并未消除专家存储需求；压缩与 offload 都有代价。
2. **已有进展：** 精度分配和预取分别缓解质量损失与传输等待，已有系统也组合两者。
3. **具体缺口：** 组合后共享显存与传输资源，局部最优可能互相破坏。
4. **关键观察：** 相同显存字节在不同状态下，对质量恢复与等待消除具有不同价值。
5. **方法：** 统一预算与成本估计，协调精度、驻留与预取，并限制迁移震荡。
6. **贡献与结果：** 写可验证发现、方法和端到端结果；新实验完成前保留占位。

读完引言，审稿人应能复述：为什么需要合并、合并解决什么新问题、用什么实验判断其必要性。

## 7. 实验组织

| 实验 | 主问题 | 最关键的对照 |
|---|---|---|
| E1：耦合扫描 | 精度与 horizon 的相互影响是否足够大？ | 高精度配额 × horizon 二维扫描 |
| E2：组合基线 | 认真调优的组合能否达到相同效果？ | 两种顺序优化、最佳静态预算切分、静态联合搜索 |
| E3：近邻与端到端 | 相同质量约束下，用户延迟是否改善？ | 可执行的 HOBBIT／DyMoE 等近邻及统一栈策略对照 |
| E4：动态变化 | 在线调整是否胜过充分调优的固定配置？ | 阶段、batch、任务变化；报告适应期损失 |
| E5：质量分配 | 动态身份选择是否值得迁移成本？ | 固定身份、hotness、敏感度及组合排名 |
| E6：开销与边界 | 哪些情况下无收益，系统代价多少？ | 全驻留、极紧预算、低预测准确率、迁移受限 |

### 7.1 强组合基线是必要条件

不应把“两个控制器分别假设自己能使用整个 $B$，随后发生冲突”作为主要对照。该基线过于容易击败。

强组合基线必须：

- 共享同一个预算。
- 允许正常缓存命中和驻留复用。
- 得到充分调优。
- 与新策略使用相同的基础执行机制，以隔离策略收益。

### 7.2 公平比较与指标

- 使用同一模型、量化表示、GPU 与显存预算。
- 主结论在相同质量要求下比较。
- 区分 miss-path 时间、exposed wait 和端到端延迟。
- 不把自行实现的简化策略称为执行了完整原系统。
- 展示质量—延迟 Pareto 曲线，避免只给相对 uniform-low 的单点加速。
- 静态策略的可部署参数由校准数据确定；测试集上的离线搜索明确标为 oracle。

## 8. 当前稿件优先修正的论证

| 当前论断 | 问题 | 建议处理 |
|---|---|---|
| $\Phi\ge1$，所以最优 $S=0$ | 只说明全低精度驻留可行，未解决质量约束下的全局最优 | 限定保留全驻留集合、无需其他表示加载等条件；作为边界结论 |
| 没有完全隐藏传输的 horizon，所以 $S_{\min}$ 最优 | 无法完全隐藏不代表更长 horizon 不能减少部分等待 | 直接比较残余等待与缓存代价，不作未经证明的最优断言 |
| 全低精度池放得下时，horizon-only 仍然驱逐 | 正常缓存策略可以保留专家，人为驱逐会削弱基线 | 允许基线自然利用全部可用驻留能力 |
| Forward path never waits on migration | 可选精度升级与真正缺失专家的 demand fetch 不同 | 限定为已有可用版本的升级；明确缺失专家仍可能阻塞 |

此外，不能把 hotness 变化直接写成质量收益，也不能把最优 $S$ 不变视为耦合必然无效。

## 9. 接近 9/10 的大纲检查点

1. **统一性：** 删除任一原工作的核心决策，都会损失对新问题的解释或解决能力。
2. **必要性：** 联合策略超过共享预算、充分调优的组合，而非只胜过弱对照。
3. **可解释性：** 能指出收益来自避免哪些 miss、释放哪些空间、减少哪些无效迁移。
4. **可证伪性：** 若静态联合配置或顺序优化已足够好，就收缩“必须在线联合”的主张。

## 10. 证据状态与下一步

### 10.1 本轮检查到的证据状态

- [合并稿实验部分](05_evaluation.tex) 仍是待执行协议，联合结果表尚未填入。
- [结果清单](../results/paper/manifest.json) 中，质量显著性、性能、消融、运行时开销等组尚无注册条目。
- 清单存在 activation-density 和 offload-waiting 条目，不代表本轮已独立审计其有效性，也不能替代联合系统结果。
- [原稿结果溯源说明](../ICCAD_2026_DynExq/RESULT_PROVENANCE.md) 记录了正式结果所需的证据要求和既有数据问题。
- [源图 provenance 台账](FIGURE_PROVENANCE.md) 已登记两篇原稿正文引用图的原图号、路径、SHA-256、复用方式与证据缺口；该台账是 asset registry，不替代实验 claim manifest。

上述状态来自本轮讨论中的检查，是 2026-10-02 的快照；后续写作前应重新核验。

### 10.2 优先级最高的工作

- [ ] 在一个统一运行栈上完成 O2 二维扫描：高精度配额 × 预取深度。
- [ ] 加入“先精度后预取”和“先预取后精度”两种顺序优化。
- [ ] 加入静态联合搜索，区分校准配置与测试 oracle。
- [ ] 在相同显存和质量约束下检查性能差距。
- [ ] 将差距分解为 miss、exposed wait、迁移字节和预测误差。
- [ ] 根据结果决定主张，再改写标题、摘要和引言。

### 10.3 决策规则

| 结果 | 对论文主线的影响 |
|---|---|
| 联合配置明显优于顺序优化，在线控制还能适应变化 | 支持以精度—驻留耦合及在线联合控制为中心 |
| 静态联合配置有收益，在线调整收益很小 | 收缩在线控制主张，突出配置模型与操作区间 |
| 强组合基线已达到相同效果 | 联合必要性未建立；不应靠强化标题维持合并叙事 |
| Hotness 无法带来质量收益 | 保留预算／预取研究，收缩动态精度身份选择主张 |

## 11. 材料入口

- [动态预取原稿引言](../MOE_ICPP/01_introduction.tex)
- [动态精度原稿引言](../ICCAD_2026_DynExq/01_introduction.tex)
- [合并稿引言](01_introduction.tex)
- [合并稿背景与模型](02_background.tex)
- [合并稿设计](03_design.tex)
- [合并稿实验协议](05_evaluation.tex)
- [HOBBIT 原文](https://arxiv.org/html/2411.01433v1)
- [DyMoE 原文](https://arxiv.org/html/2603.19172v1)
