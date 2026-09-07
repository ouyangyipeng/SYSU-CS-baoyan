# AI Infrastructure、GPU 与端侧推理

用于复习基础概念与口头简答。S 为优先掌握，A 为常见知识，B 为扩展内容；标记表示复习建议，不是出题概率统计。

## GPU 执行与内存层次

### [S] Kernel

- Kernel 是在 GPU 上由大量线程并行执行的函数。
- CUDA 中线程组织为 thread → block → grid；同一 block 可共享 shared memory并同步。

### [S] 内存层次

- Register：每线程，最快但容量有限。
- Shared memory：每 block，低延迟、显式管理。
- Global memory：容量大、延迟高；需要合并访问与缓存利用。
- 过多寄存器/shared memory 会降低 occupancy。

### [S] 吞吐与延迟

- Latency：单个请求完成时间。
- Throughput：单位时间完成量。
- 批处理通常提高吞吐但可能增加单请求等待。

### [A] 算术强度与 Roofline

- 算术强度 = 运算量 / 数据搬运量。
- 低于机器平衡点更可能 memory-bound，高于则更可能 compute-bound。

## 并发 Kernel 与融合

### [S] 融合收益与风险

- 收益：减少启动、同步和中间数据读写，共享数据。
- 风险：寄存器/共享内存压力增大、并行度下降、指令/分支复杂，可能变慢。
- 是否融合需要合法性与收益判断，不能“一律融合”。

## Prefill 与 Decode

### [S] 区别

- Prefill：一次处理整个输入序列，矩阵乘规模大，通常更 compute-bound、并行度高。
- Decode：逐 token 生成，复用 KV Cache，单步矩阵较小，常受内存带宽和调度延迟限制。

## KV Cache

### [S] 是什么

- 保存历史 token 在每层 attention 的 Key/Value，生成新 token 时无需重算全部历史。
- 显存占用随层数、序列长度、batch、KV heads、head dimension和精度增长。

### [A] 优化

- 量化、分页/PagedAttention、淘汰/压缩、GQA/MQA、分层存储。
- 优化需检查精度、碎片、传输开销与端到端延迟。

## 量化

### [S] 基本原理

- 用更低位宽表示权重/激活，降低存储和带宽，并可能利用低精度硬件加速。
- 常见 PTQ 与 QAT；per-tensor 与 per-channel；对称与非对称量化。
- 位宽降低不等于必然加速：还取决于硬件支持、反量化和 Kernel 实现。

## MoE

### [S] 原理

- 多个专家网络 + router；每个 token 只激活 top-k 专家，所以总参数量大但单 token 计算量可控。
- 难点：专家负载不均、路由通信、专家权重存储/加载、训练稳定性。

### [S] 端侧 MoE

- 优势：稀疏激活允许按需加载专家。
- 瓶颈：存储到内存的数据搬运可能压过计算；需要预取、缓存、内存映射和调度。

## Triton 与 NPU 调优

### [S] Triton

- 用 Python-like DSL 编写并行张量 Kernel，由编译器生成目标代码；核心仍需选择 tile、并行度、内存布局等。
- Triton 写法简洁不等于自动获得最优性能。

### [S] 正确性门禁

- 自动生成候选必须先编译、再与参考结果比较数值误差，通过后才能测性能。
- 浮点比较用容差，不做简单逐位相等。

### [A] Profiling

- 先定位 compute、memory、launch、同步或并行度瓶颈，再定向优化。
- 单个微指标改善不代表端到端变快，必须用相同工作负载复测。
