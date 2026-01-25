## Idea

### 解决的问题：
LLM model增加parameter size导致依赖昂贵的GPU(尤其是MoE)，为了能够在有限资源下部署，量化是一个可行的方案，但是现在的PTQ量化方案都是静态量化(选择request负载校准)；但是这种量化方案：1，无法避免造成精度损失，导致量化尾款受到限制；2，没有充分利用MoE的稀疏激活特性，在inference中，只有一部分expert被依赖，大部分expert并不对输出产生决定影响；3，但是workload是变化的，会给expert的hot/cold带来变化 (有Motivation Test) 

### Motivation: 
- 1，为了在customer-level GPU上部署先进的MoE model，采取激进的量化策略(W2A16)； 
- 2，为了保护精度不受激进量化策略的影响，使用动态模型调度，为重要的expert使用较高精度的量化 W4A16 
- 3，考虑负载变化，动态管理expert的精度 (以及相应的Group GEMM kernel)

### Challenges
- 克服现有量化策略“静态、输入无关”问题，使量化与 workload 动态耦合；
- 在有限显存下支持专家动态切换而不造成推理暂停；实现不阻塞推理的expert切换
- 不同精度的专家占用显存不同，动态切换可能造成显存碎片化或带宽抖动。
- 通过系统 runtime 层实现资源，workflow感知的动态量化阈值调度

### Designs
- 轻量级 Expert 热度探测
    - 在每次 forward pass 后，收集当前 batch 的 gating 分布（logits 或 top-k index） —— 注意简单基于激活次数来评估hot/cold可能不准确，最好是讲logits打分考虑进去。
    - 通过时间窗口 计算专家热度分数 $S_i$；为了简化实现，可以通过全局epoch的方式来定义时间窗口，全局epoch 每5分钟 +1
    - 热度阈值 $\tau_h, \tau_c$ 控制升/降精度切换（例如 $S_i>\tau_h$ → W4A16，$S_i<\tau_c$ → W2A16）。
    - 所有统计都在 CPU 侧异步完成，GPU 只保留状态寄存器。
- 异步 Expert 精度切换 Pipeline
    - 首先需要实现混合精度expert的group GEMM优化GPU计算效率
    - 将高精度和低精度版本存储在：
        - GPU HBM：当前热专家；
        - Host DRAM：候选专家/不同精度；
        - SSD：如果DRAM资源不足，可以构建index，将全量expert存储在SSD上，通过方位index快速获取所需权重的address，然后加载到DRAM
    - 当专家状态变化时：
        - 由异步线程预取下一批可能“热”的专家；
        - 使用 pinned memory 进行双缓冲传输；
        - 在 CUDA stream 中 overlap compute 与 transfer。
        - 利用expert的层计算特性，实现layer-wise pipeline；避免阻塞
- 显存动态分配策略
    - 显存划分为三个 pool：
        - Hot Pool（高精度区）
        - Cold Pool（低精度区）
        - Transient Buffer（用于 swap 的中间缓存）
    - 切换时不直接 realloc，而采用固定窗口替换策略（FIFO 或 LRU），防止碎片化。
- 资源-精度感知调整
    - 最开始最大化HBM的利用，尽可能保留hot expert；
    - 随着系统执行，结合系统workload 和 expert激活的集中情况，适时调整阈值(更多W4的expert还是更少？)