# Background and Motivation 修改计划

## 1. 修改目标

第二章需要完成一条可由图和公式共同验证的推理链：

1. 单卡 MoE 运行时管理的不是一个抽象 cache，而是处于不同精度、驻留状态和到达时刻的 expert representations。
2. 同一个显存字节可以用于 fidelity、residency 或 lookahead，三种用途的收益随 workload state 改变。
3. 单独优化任意一项都会忽略另外两项的机会成本。
4. 因而系统必须在质量约束和硬显存约束下，按边际收益联合选择 promote、retain、evict 和 prefetch。

第二章只负责证明问题存在、给出问题定义和导出设计要求。它不提前宣称系统已经取得端到端收益，也不承担完整 evaluation。

## 2. 当前版本的审稿人判断

当前文字主线已经统一，但证据强度约为 6.5/10。

- 优点：三种显存用途的定义清楚；质量约束、显存约束和动作集合已经形式化；对 frequency 与 sensitivity 的区别交代得较谨慎。
- 主要缺口：章节用三段文字分别描述三类动态性，却没有把“共享预算”和“机会成本”画出来。审稿人仍可能把工作理解为动态量化与动态预取的简单拼接。
- 次要缺口：旧稿中的 horizon、batch 和 diversity 数字出现在正文中，但相应旧图不能按现有口径直接支持这些陈述。
- 版面问题：独立的 capacity-regime 图信息密度较低；六行 activation table 占据较多空间，却没有显示 demanded bytes、overlap window 或 exposed wait。
- 论证顺序问题：目前在读者看到联合耦合的直接证据前就进入优化公式，形式化是合理的，但说服力不足。

## 3. 旧图复用审计

“直接搬”分为两个标准：版式上可以原样嵌入，以及证据达到投稿标准。旧资产能够满足前者的不一定满足后者。

| 旧资产 | 结论 | 在第二章中的用途 | 投稿前条件 |
|---|---|---|---|
| `ICCAD_2026_DynExq/figures/wiki_ppl_qwen30b.pdf` 和 `wiki_ppl_qwen80b.pdf` | 可组合后复用 | 证明 fidelity 的收益具有连续的质量代价，并给出 quality constraint 的直觉 | 补齐每个点的 raw NLL、token count、window boundary、frozen ranking、命令和版本登记。caption 必须限定为 frozen precision-quota sweep，不能称为在线策略结果 |
| `ICCAD_2026_DynExq/figures/wikitext_thinking_on_layer_15.pdf`、`gsm8k_thinking_off_layer_15.pdf`、`humaneval_thinking_on_layer_15.pdf` | 可组合后复用 | 证明 workload 改变时 hot expert identity 会变化，从而说明静态 fidelity/residency map 会过时 | 补齐三组 raw dispatch counts 和 workload/checkpoint/scheduler 登记；正文只解释 access frequency，不把它写成 quality sensitivity |
| `ICCAD_2026_DynExq/figures/waiting_latency_vs_prompt_length.pdf` | 不进入主叙事；可放 appendix | 只能说明串行 blocking transfer 会暴露等待 | 虽然窄口径证据已登记，但 Qwen direct measurement 与 Phi cold-cache replay 混在同图，且不能说明 overlap-aware runtime，更不能证明三用途耦合 |
| `MOE_ICPP/figures/M1_numstep_latency.pdf` | 不能直接搬；重测并重画 | 可作为新 joint scan 的构图原型 | 旧图没有质量轴和 precision quota，且 provenance 缺失。新实验需扫描 precision quota、resident budget 和 horizon，并从同一运行记录 latency、quality、traffic 和 peak bytes |
| `MOE_ICPP/figures/M3_batchsize_latency.pdf` | 不能直接搬 | 仅保留“demand 与 overlap 必须同测”的实验问题 | 旧图横轴范围与正文口径不一致，waiting/cache-miss 定义也不足以映射到新主线 |
| `MOE_ICPP/figures/M5_DeepSeek_Qwen1.5_Qwen2.pdf` | 不建议搬 | 最多作为 appendix 的 predictor 特征分析 | embedding dispersion 是间接 proxy，不能独立支撑 latency 因果关系；新稿应直接测 miss bytes、deadline slack 和 exposed wait |
| 当前 `fig:phase` TikZ 图 | 删除独立浮动图，内容并入新概念图或正文 | 仅解释容量边界，不提供实验证据 | 保留三种 regime 的定义和必要公式，但不再占用一张主图 |

结论：没有一张旧的 empirical figure 可以在不补 provenance 或不改变论证口径的情况下无条件进入主线。两组 DynaExQ 图可以保留绘图结果并重新组合；DynaMoE 的三张 motivation 图都应重测。

## 4. 目标章节结构

### 2.1 Expert Memory Is a Shared Resource

用约半页建立最少背景：router、host/device、published representation、in-flight transition、deadline。随后定义 fidelity、residency、lookahead。避免在这一节展开算法或 related work。

配图为 Figure 2：`One byte, three uses`。左侧画 expert 的 host low/high representations，右侧画 device 上 absent、low、high 和 in-flight 状态；中间用统一的 budget bar 表示 promotion、retention 和 prefetch staging 从同一容量中取字节。图中只出现论文统一术语，不使用 quality pool、cache pool 或 prefetch quota 等会暗示静态分区的词。

### 2.2 The Value of Each Use Changes with Runtime State

按同一模板写三段，每段回答两个问题：这个用途的收益是什么，以及哪一个状态量会改变该收益。

- Fidelity：质量收益取决于 quantization sensitivity 与 execution probability。
- Residency：延迟收益取决于 reuse probability 与 avoided transfer cost。
- Lookahead：延迟收益取决于 deadline slack、overlap bandwidth 与 speculative occupancy。

本节不使用三个彼此独立的 quota 来描述问题。段末明确指出三个收益都以 byte 为成本，因此必须比较 marginal value per byte。

### 2.3 Measurements That Expose the Coupling

按照“单项价值会变化”到“联合决策不可分”的顺序安排四组证据。

**Figure 3：Quality and demand are workload dependent。**

- Panel (a)：复用两张 perplexity 曲线并统一字体、legend 和轴范围，说明 fidelity 存在可控但非零的质量代价。
- Panel (b)：将三张 Layer-15 activation 图重组为紧凑的 workload × expert heatmap。保留原始 counts，明确标出各 workload 的 top-10，而不是并排放三张稀疏柱状图。
- 这张图只能支持两个有限结论：precision quota 影响质量；hot expert identity 随 workload 改变。它不声称 frequency 等于 sensitivity。

**Figure 4：Demanded bytes and overlap capacity evolve differently。**

- 新测一条代表性 request 或一个短时间窗，将 prefill 与 decode 对齐。
- 共享横轴，依次画 demanded expert bytes、resident-hit/miss bytes、available overlap bytes，以及最终 exposed wait。
- 同一运行中记录这些量，避免用 activation ratio 间接推出 transfer pressure。
- 当前 activation-density table 移到 appendix，或只在正文保留一个最能说明阶段差异的数字。若 Figure 4 已覆盖三个模型和 batch trend，则删除该表。

**Figure 5：Independent tuning misses the joint operating point。**

- 这是第二章最重要的新图。固定模型、设备、workload 和总 expert-memory budget，扫描 high-precision allocation 与 lookahead/residency allocation。
- 主面板画 P95 TPOT 或 exposed wait 的二维 heatmap；叠加 quality-feasible boundary、memory-infeasible region 和 measured joint optimum。
- 在边缘标出三个对照点：fidelity-first、residency-first、lookahead-first。对照点必须来自同一套运行，不能拼接旧稿数字。
- 如果二维图无法同时表达 residency 与 horizon，则将横轴定义为可复现的 transient/lookahead byte allowance，而不是抽象的 horizon；horizon 作为调度结果或副轴报告。
- caption 的结论应是：改变 fidelity allocation 会移动最优 residency/lookahead 配置，因此三者不能独立调参。不要在这里报告系统相对 baseline 的最终 speedup。

### 2.4 Constrained Allocation Problem

在读者看到 Figure 5 后再引入目标函数。保留 P95 TPOT、peak expert-memory budget 和 quality-degradation limit。先用一段 prose 定义 policy 与五种动作，再给主优化式。

`M_miss(S)`、aggregate overlap bound 和 hard fit invariant 保留，但将 `S^star_agg` 降为正文中的 feasibility diagnostic，或移至 appendix。它不是联合目标的解，正文不应让这一 horizon-specific 公式获得与主目标同等视觉权重。

### 2.5 Boundary Cases and Design Requirements

将当前 `Capacity Regimes as Boundary Cases`、`Design Requirements` 和 `Limits of Runtime Signals` 压缩为一个小节。

- 用三句话交代 full high pool fits、full low pool fits、full low pool does not fit 三个边界，不保留独立 phase 图。
- 保留“某些紧预算下不存在 quality-feasible policy”这一诚实限制。
- 以三个 requirements 结束第二章：comparable action values、state-responsive estimates、single admission path。
- 最后一段只说明 runtime observes proxies 及其保守处理，不预告设计细节的摘要列表。

## 5. 图的最终编排

| 新编号 | 图 | 来源 | 状态 | 目标信息 |
|---|---|---|---|---|
| Fig. 2 | One byte, three uses | 新画概念图 | 不需要实验 | 一眼建立共享预算和统一状态空间 |
| Fig. 3 | Quality and demand vary | PPL 曲线与 activation counts 重组 | 有条件复用 | 说明 fidelity value 与 demand map 都非静态 |
| Fig. 4 | Demand versus overlap over time | 新实验 | 必须新增 | 用直接量连接 routing、residency、deadline 和 exposed wait |
| Fig. 5 | Joint allocation surface | 新实验 | 必须新增 | 直接证明独立优化会错过联合最优点 |

主文不放 `waiting-vs-prompt`、旧 M1、旧 M3、旧 M5 和独立 capacity-regime 图。

## 6. 实施顺序

1. 冻结术语表：只使用 fidelity、residency、lookahead、expert-memory budget、published representation、in-flight transition、quality limit 和 marginal value per byte。
2. 先画 Figure 2 的黑白草图，据此重写 2.1 和 2.2，删除重复定义。
3. 补齐 PPL 与 activation 两组 provenance；在不改变数据的前提下重新排版 Figure 3。
4. 设计并运行 Figure 4 的同源测量，保证 demanded bytes、overlap capacity、miss bytes 和 wait 来自同一次 execution trace。
5. 运行 Figure 5 的 joint scan。先确定 quality-feasible region，再在其中比较 latency，避免把违反质量约束的最低延迟点标成 optimum。
6. 根据 Figure 4 和 Figure 5 的实际证据重写 Preliminary Measurements，删除无法由图支持的 2.5x 和 300% 等旧数字。
7. 后移并压缩形式化部分；合并 boundary cases、requirements 和 limitations。
8. 编译后逐项审计 caption、正文、公式和图例中的术语与变量，确保同一概念只有一个名称。

## 7. 9/10 验收标准

- 读者只看 Figure 2 和 Figure 5 就能复述论文主问题：同一显存预算下，三个用途必须按边际收益联合分配。
- 每张 empirical figure 的每个点都能追溯到 raw artifact、固定 protocol、command、commit 和 plot script。
- 第二章中的数值结论全部能在相邻图表中直接读出；没有由 proxy 跳到 latency 或 quality 的因果越级。
- fidelity、residency、lookahead 在 Introduction、Background、Design 和 captions 中保持同一含义和拼写。
- Background 不介绍具体 controller 规则，Design 不重复 motivation，Evaluation 不再重新定义问题。
- 主文第二章控制在约 3.5 至 4 页、4 张图以内；每张图至少支撑一个后续设计决定。
- 交给不了解项目的系统审稿人试读后，对“为什么不是两个旧系统拼接”的回答能够直接指向 Figure 5，而不依赖作者口头解释。

## 8. 当前落地状态

- 旧 PPL 与 Layer-15 activation PDFs 已直接嵌入 `02_background.tex`，未重新生成或改写数据。
- `scripts/build_background_motivation_data.py` 校验已有 measured artifacts，并生成 stage-level transfer-pressure 与 joint-allocation replay JSON。
- `scripts/plot_background_motivation.py` 只从上述派生 JSON 绘制论文 PDF，不在绘图阶段重新计算结果。
- `dynaexq/tests/test_background_motivation_figures.py` 覆盖 byte accounting、quality/memory feasibility 和两张图的无界面渲染。
- 派生数据位于 `results/paper/background/`，图片位于 `SIGMETRICS_2027/figures/`，证据边界和哈希记录在 `FIGURE_PROVENANCE.md`。

复现命令：

```bash
python scripts/build_background_motivation_data.py
python scripts/plot_background_motivation.py
pytest -q dynaexq/tests/test_background_motivation_figures.py dynaexq/tests/test_render_paper_figures.py
```

当前 joint surface 是使用 measured inputs 的 trace replay，不是统一 runtime 的 end-to-end measurement。Evaluation 阶段应以相同 JSON schema 接入真实 joint executor 的扫描结果，再决定是否替换 Background 中的 replay 图。
