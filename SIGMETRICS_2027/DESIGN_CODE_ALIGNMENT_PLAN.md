# 新 Design 与代码的差距，以及下一轮实现与实验计划

日期：2026-10-03。对照 `03_design.tex`（byte exchange、自适应 envelope、单一 arena）和当前决策代码（`dynaexq/lease/`、`dynaexq/policy/online.py`、`scripts/run_a6000_wikitext_replay.py`）。上一轮结论在 `A6000_WIKITEXT_REPLAY_CONCLUSIONS.md`。

这一轮仍然只做 **一个数据集、一张 A6000**：WikiText-103 的已测路由 trace，外加同一张卡上已登记的带宽、块大小和 WikiText-2 perplexity。不新测 kernel 计时，除非执行时有一张空闲且独占的 A6000，并且 checkpoint 已经在盘上。

## 1. 差距

上一轮代码实现的是当时的设计：固定 horizon、按层预算、只预取低精度、用高精度专家个数满足 perplexity 曲线。新设计改了决策单位、搜索范围和内存所有权。下面按「会不会改变上一轮结论」排序。

| 新设计 | 现在的代码 | 对上一轮结论的影响 |
|---|---|---|
| 交换是三元组 \((Z^{+}, D, \mathcal{T})\)：一组要创建的 lease、一组 donor、一份源和替换共存的 schedule（Eq. 1，Table 1） | `Exchange` 只有一个 expert、可选一个 donor，没有 schedule。`lookahead` 把目的地记成 prefetch；`promote` / `demote` 在 `_apply` 里有分支，在线循环没有用 | 中。单 donor 覆盖不了「一个高精度增量加一个驻留」同时出资 |
| **一个 arena**，free list 不按层、精度或用途切配额（Safe Executor） | `Placement.total_bytes` 按层和 `budget` 比较。Φ = 1.0 时每层 30 个高精度加 15 个低精度就把该层填满，别的层的空位不能借给它 | **高。** 上一轮「Φ = 1.0 时 slot 挤不进去」是按层配额的结果，不是新设计的结果 |
| 自适应 envelope \(S_t\)：冷启动 Eq. 8，之后由 stall \(\widetilde s\) 减 pollution \(\widetilde p\) 做带滞回的 ±1（Eq. 9–10、Eq. 14） | `run_online(..., horizon=1)`，脚本里 `HORIZON = 1`。没有 \(s_t\)、\(p_t\) | **高。** 上一轮长 prefill 选 0 个 slot、短窗口选 16 个 slot，正是 \(S_t\) 应该自己走出来的行为，代码却写死了 |
| Lookahead 的低精度和高精度是互斥候选；价值里包含更大的传输（Eq. 12） | 在线候选的 `tier` 恒为 `"lo"`。Demand miss 也恒为低精度 | 高。高精度专家一旦不在显存里，设计要比较「搬高精度」和「搬低精度并另付 fidelity」 |
| Donor 先扫可回收 lease：没开始的 lookahead、envelope 之外的驻留。envelope 内的驻留受保护，除非 replay 把随后的 reload 计入 \(G\) 仍为正 | `_donor` 先避开预测集，没有就退回任意低精度驻留。不取消 lookahead，不把 fidelity 当 donor | 高。上一轮在线策略几乎只做 residency→lookahead，而且预测集宽到 105，保护失效后退回驱逐热专家 |
| \(G\) 减去迁移的暴露成本 \(C_{\mathrm{move}}\)（Eq. 12） | `counterfactual_gain` 只做两次 exposed wait 的差 | 中。短窗口上，交换自己引入的拷贝可能把 \(G\) 打成负的 |
| 重放按最早截止时间；同一 expert、同一 tier 只有一次就绪（continuous batching） | `simulate_forward` 按层顺序发 demand，同一层内专家串行，没有跨请求合并 | 低，对这条 trace。Trace 是每层的唯一专家集合，没有逐 token、也没有 batch。合并就绪要有单测，但这条数据集上看不到 |
| Resident-first：已发布的 expert 组先算，只有错过 deadline 的组等待 | 层在所有 miss 到齐之前不开始计算 | 中。同层既有命中又有 miss 时，现行模型把整层都算作等待 |
| 质量是 \(\widehat R=\sum w_i q_i(1-h_i)\)，超限时按「单位风险的最小延迟代价」提升；允许 fidelity 合并 | `_enforce_quality` 把高精度个数补到曲线要求的下限，按频率选人，不看 \(q_i\) | 中，但这条数据上做不到真的 \(q_i\)。仓库里没有逐专家 sensitivity。这轮不编造 \(q_i\) |
| 快环 \(T_{\mathrm{fast}}\) 刷新 deadline / residency / lookahead / \(S_t\)；慢环 \(T_{\mathrm{slow}}\) 才动 fidelity，并有最短任期 | 每个 trial 开始时就把高精度个数补满 | 中。Fidelity 没有任期，短 prompt 上会过度搬迁 |
| 保护索引和可回收索引；LRU 只是同一索引内的次序 | 没有索引。`staging_slots` 仍是静态策略的固定 lookahead 配额 | 高，和 arena 是同一件事。静态基线不能再靠「每层预留 K 个 slot」代表新设计 |
| 版本化 handle、读租约、按 compute stream 记录 last-use、extent 绑定失败则拒绝 | `LeaseLedger` 只检查字节和。`dynaexq.core` 仍是按层按 tier 的 `PoolAllocator` | 低，对 replay。这是执行器，不是这一轮要改的结论来源。`04_implementation.tex` 仍写着 per-layer pool，和设计不一致，本轮不改论文 |
| 预测器：有 pregate 就用 logits，否则用路由转移频率；低置信度扩大专家集合而不是加长 \(S_t\)；可选 CPU 修正模型单独评估 | 用 warmup 频率的 top-\(W\)，\(W\) 等于平均活跃集大小（2048 token 上是 105） | 高。上一轮结论已经写了：集合太宽，donor 经常是马上要用的专家 |

没有变、而且代码已经对齐的部分：lease 的三种用途、live+reserved ≤ \(B\)、全有全无预留、反事实重放不为已驻留专家的 prefetch 记功、质量不折成毫秒、lookahead 命中后转成 residency。这些单测留着，改代码时不能退回去。

## 2. 这一轮改什么

改 replay 里的控制器，让它执行新设计里会改变等待时间的那几条规则。不改 CUDA 执行器，不改论文。

### 2.1 一个预算，而不是 48 个层预算

`Placement` 不再各自对着 `budget` 做可行性判断。全模型一个 `LeaseLedger`，容量是现在的「每层预算 × 48」。高精度、低精度、在途 lookahead 都占这个账。一层的空位可以资助另一层的交换。

静态基线也在这本账上重算。上一轮 Φ = 1.0 / 1.3 / 1.931 的毫秒数不能直接拿来比，要在新账本上重跑。

### 2.2 交换覆盖 Table 1

每个候选是下面之一，并且同一专家的低精度 lookahead 和高精度 lookahead 互斥：

- absent → 在途低精度，或 absent → 在途高精度
- 低精度驻留 → 高精度（取得 fidelity 增量）
- 高精度驻留 → 低精度（释放 fidelity，且替换发布前增量不能再花）
- 取消一个还没开始的 lookahead
- 驱逐一个驻留（释放 residency，以及它身上的 fidelity）

\(D\) 可以有多个 donor。扫描顺序按设计：先取消无用 lookahead，再拿 envelope 之外的驻留，最后才考虑 envelope 之内的驻留。最后这一类只有在重放已经加上它的 reload、并且 \(G-C_{\mathrm{move}}>0\) 时才保留。

\(C_{\mathrm{move}}\) 用这次交换自己新发出的拷贝里、落在截止时间之后的那一段。不把「本来就要发生的 demand」再算一遍。

### 2.3 自适应 \(S_t\)

冷启动用 Eq. 8，\(N_e\) 和 \(E_s\) 来自 warmup 的缺页个数和所选 tier 的字节。之后每个快周期用 Eq. 14。

\(s_t\) 是本周期暴露等待除以路由层时间。\(p_t\) 的分子是未使用的 lookahead 字节，加上驱逐后同一 trial 内又被 demand 装回的字节。分母是本周期接纳的 lookahead 字节，至少为一个专家块。\(\beta\)、\(\lambda\)、\(\theta_h\)、\(S_{\min}\)、\(S_{\max}\) 在 warmup 上选一次，写进结果，不在 measured trial 上重选。

\(S_t\) 只限制候选层，不预留字节。Eq. 16 再剪掉「预测缺页字节超过 \(S \cdot T_\ell \cdot C\)」的前缀。

低置信度时扩大该层的专家集合，不增加 \(S_t\)。这条 trace 没有 pregate，预测器就是层间转移频率。置信度用 warmup 上该转移的命中率；命中率低时把集合扩到频率前缀的下一档。不训练 CPU 修正模型。设计把该模型留给 Evaluation，单独评估；这轮没有逐 token 标签，训了也不诚实。

### 2.4 质量和时间尺度

可行性仍用已测的 WikiText-2 曲线：高精度专家个数对应的 ΔPPL ≤ \(\epsilon\)。没有逐专家 \(q_i\)，就不计算 Eq. 7，也不声称算了。

修复顺序改成设计的规则：缺的高精度名额，优先给「提升后预测等待增加最小」的层和专家，而不是每层按频率各补到同一个个数。超额的 fidelity 可以当 donor。Fidelity 一旦接受，至少保留 \(T_{\mathrm{slow}}\) 个 trial 才允许 demote。快环每个层边界更新 \(S_t\)、residency 和 lookahead。

### 2.5 Resident-first 的等待

同一层里，已发布专家的计算可以和 miss 的剩余拷贝重叠。这条 trace 没有逐 token 计数，所以重叠上限取 \(T_\ell \times |\text{hits}|/|\text{active}|\)，并且在结果里写成近似。层的暴露等待是 miss 队列超出这段重叠的部分，不是「整层在第一个 miss 处停住」。单测用一个构造出来的两专家层核对这个定义。

## 3. 调试

沿用 `dynaexq/tests/test_lease_invariants.py` 的风格，每条新规则一个测试，不依赖 GPU。

| 测试 | 必须成立的行为 |
|---|---|
| 跨层资助 | 层 0 有空位、层 1 已满时，用层 0 的字节给层 1 的 lookahead，全局账本通过；按层账本会拒绝 |
| 互斥 tier | 同一专家的低精度 lookahead 和高精度 lookahead 不能同时留下 |
| 滞回 | \(\widetilde s-\lambda\widetilde p\) 落在 \(\pm\theta_h\) 内时 \(S_t\) 不变；高于上界只加 1；低于下界只减 1 |
| 污染来自误驱逐 | 预取未使用，或驱逐后同一请求又装回，\(p_t\) 上升且下一步 \(S_t\) 不允许因此变长 |
| 受保护 donor | envelope 内的驻留被驱逐时，\(G\) 含它的 reload；reload 大于预取收益则拒绝。取消一个未使用的 lookahead 优先于驱逐它 |
| 迁移成本 | 交换自己的拷贝错过截止时间时，\(G\) 比「只看前后等待差」更小，可以为负 |
| 任期 | 慢环刚接受的 fidelity，在 \(T_{\mathrm{slow}}\) 之内不能当 donor |
| 重叠 | 同层一个命中、一个 miss，暴露等待小于整段 miss 拷贝 |
| 旧不变量 | 超预算的预留整笔回滚；已驻留专家的 lookahead 反事实得分为 0，独立估值得分为正 |

调试顺序：先让上表全绿，再跑下面的实验。实验脚本如果和单测对同一组合成输入给出不同的等待，以单测的定义为准，改脚本。

## 4. 实验

数据和上一轮相同，便于看到「设计改了之后数字怎么变」而不是换了一条 trace。

- 模型与路由：Qwen3-30B，`results/paper/qwen30b_routing_active_set_trace.json`
- 带宽与块：`qwen30b_offload_waiting.json`、`qwen30b_dynaexq_bs1.json` 的 pool 块大小
- 质量：`results/perplexity/Qwen3-30B/30b-*.txt`
- 主长度 2048，层窗口用实测 TTFT/48（38.58 ms）
- 128 和 512 仍按 token 比例缩放窗口，结果里继续标明是假设
- Φ 取 1.0、1.3、以及已部署的 1.931。预算是**全局** \(48 \times \Phi \times P_{\mathrm{lo,layer}}\)
- \(\epsilon\) 仍是预注册的 0.5 和 2.1。0.5 预期仍然不可行，只报告可行性，不拿它比速度

策略，全部跑在新的全局账本和 resident-first 等待上：

| 名称 | 作用 |
|---|---|
| precision-floor | 只满足 ΔPPL 的最少高精度专家，其余字节按频率做驻留，\(S=0\) |
| horizon-fixed | 不提升精度，\(S\) 在 warmup 上从 \(\{0,1,2,4\}\) 里选定后冻结 |
| seq-qp | warmup 上先在零精度时锁定 \(S\)，再补质量下限，不重新打开 envelope |
| static-joint | warmup 上联合选高精度个数和 \(S\)，然后冻结 |
| oracle-grid | 同一网格在 measured trial 上选，单独标出 |
| dynabyte-fixed-S | 上一轮控制器：\(S=1\)、只预取低精度、按层思路的 donor。迁到全局账本上重跑，作为「旧控制律」 |
| dynabyte-adaptive | 第 2 节的控制器 |

不重跑 independent 估值。上一轮它在端到端上和反事实差 3 ms，新设计已经把独立记分排除在控制律之外。单测仍保留「已驻留专家不得记两次功」。

每个格子记录：平均和 p95 exposed wait、ΔPPL、是否越过 \(\epsilon\)、全局 live+reserved 的峰值、\(S_t\) 的轨迹、三类交换的笔数（fidelity、residency、lookahead）、低精度 lookahead 和高精度 lookahead 各多少、被拒绝的受保护 donor 数、\(p_t\) 的均值。峰值超过 \(B\) 的运行是 bug，不进表。

## 5. 结论里允许写的话

跑完之后写 `SIGMETRICS_2027/A6000_WIKITEXT_ADAPTIVE_CONCLUSIONS.md`，并明确它替代不了上一轮文件：上一轮是按层配额、固定 \(S=1\)。新文件回答四件事。

1. **全局账本有没有改变可行性边界。** 若 Φ = 1.0、ΔPPL ≤ 2.1 时，跨层借用让 lookahead 变得可行，上一轮「这个预算点 slot 挤不进去」就只对按层池成立，对新设计不成立。
2. **\(S_t\) 会不会自己分开长短窗口。** 期望是 2048 的实测窗口上 \(S_t\) 停在较小值，128 的缩放窗口上 \(S_t\) 升高。若两边 \(S_t\) 一样，自适应 envelope 在这条 trace 上没有信息，论文里就把它写成未被这条数据支持的控制律。
3. **自适应控制器是否优于 static-joint 和 seq-qp。** 比较的是同一本全局账本上的平均 exposed wait。差不过 measured trial 的范围，就沿用上一轮的收缩：保留联合配置，不声称在线 envelope 更好。
4. **高精度 lookahead 有没有被选中。** 若笔数为 0，Table 1 里的互斥高精度装载在这条数据集上是空分支，正文不应把它写成观察到的行为。

优化分析只写控制器下一步，不写新的模型或新的 GPU。优先顺序以这轮数字为准，预先只定原则：如果 \(S_t\) 不随窗口变化，先查 \(s_t\) 和 \(p_t\) 的尺度，而不是加大 \(S_{\max}\)；如果 oracle 的 \(S\) 仍明显大于自适应控制器，先收窄预测集合，而不是再加一种估值。

## 6. 执行顺序

1. 改账本和交换，补第 3 节的前四个测试。
2. 加 \(S_t\)、donor 保护和 \(C_{\mathrm{move}}\)，补剩下的测试。全绿之前不跑 trace。
3. 把 `scripts/run_a6000_wikitext_replay.py` 换成第 4 节的策略表，重跑。静态网格先跑；自适应控制器只在 2048 上默认打开，128 和 512 若单次 trial 超过几秒再决定是否只跑 Φ = 1.3。
4. 对照单测定义抽查一个格子的交换日志，然后写结论文件。

不在这轮做的事：改 `dynaexq.core` 的按层池、实现版本化 handle、下载 checkpoint、在被占用的 GPU 上计时、把 \(q_i\) 换成频率冒充 sensitivity、修改 `04_implementation.tex`。最后一项和设计矛盾，留到有独占 GPU 的执行器实现时一起改。
