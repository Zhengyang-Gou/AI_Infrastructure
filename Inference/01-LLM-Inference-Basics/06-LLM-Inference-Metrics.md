# LLM 推理流程与性能指标

## 1. 为什么不能只看 Tokens/s

LLM Serving 同时关心用户体验、系统容量和成本。一个配置可能提高整体吞吐，却让首 Token 等待更久；也可能平均延迟很好，但少数请求非常慢。

因此至少要同时观察：

```text
延迟：用户等多久
吞吐：系统单位时间完成多少工作
Goodput：满足 SLO 的有效吞吐
资源：GPU 和显存是否被有效使用
成本：完成相同工作需要多少钱
```

## 2. 从请求链路理解指标

### 2.1 一次请求的完整流程

```text
用户请求 → API Server → 拼装完整 Prompt → Tokenizer
  → Scheduler 排队、调度、组 Batch → 分配 KV Cache
  → Prefill → 首次采样 → 首 Token 转文本并流式返回
  → Decode（输入上次采样的 Token，更新 KV Cache）
  → 采样下一个 Token → 转文本并流式返回
  → 未结束则继续 Decode；结束则释放请求占用的缓存资源
```

启用前缀缓存时，可复用的缓存块可能被保留，而非请求结束后立即清空。

真实输入通常包含系统提示词、历史对话、工具信息和聊天模板，不只是用户最后一句话。这些内容都会计入输入长度，影响 Prefill 开销和 KV Cache 占用。例如：

```text
<system>你是一个 helpful assistant</system>
<user>请解释一下 KV Cache 的作用。</user>
<assistant>
```

Tokenizer 将文本转成 Token ID，通常在 CPU 上执行。例如，“KV Cache 很重要”可能对应一串整数；具体切分和 ID 取决于使用的 Tokenizer。输出端由 Detokenizer 将生成的 Token 转回文本。

### 2.2 调度器如何组织请求

多个请求往往共享 GPU，输入长度和输出长度可能差异很大：

```text
Request A：输入 200 Tokens，计划生成 100 Tokens
Request B：输入 1000 Tokens，计划生成 50 Tokens
Request C：输入 50 Tokens，计划生成 500 Tokens
```

调度器需要决定请求顺序、Batch 组成、KV Cache 分配、是否等待更多请求、是否抢占或换出请求，以及如何安排 Prefill 与 Decode。这些决策同时影响排队时间、吞吐和尾延迟。

### 2.3 Prefill 与 Decode 各自产生什么

**Prefill** 并行处理 Prompt 中的 Token，为各层构建 KV Cache，并利用最后一个有效位置的 Logits 采样首个输出 Token。Logits 是未归一化分数，经过 Softmax 才得到概率分布。长 Prompt 的处理也可以被拆成多个调度块，并非总在一次调度中完成。

**Decode** 将上次采样得到的 Token 输入模型。在每层计算当前 Token 的 Q、K、V，将当前 K、V 加入该层缓存，再用 Q 关注历史及当前 K、V；经过 Attention、MLP 和后续层后得到新的 Logits，采样下一个 Token。普通自回归生成中，后一个输出依赖前一个输出，不能把整段未知回答一次性并行计算。

Prefill 常有较大的矩阵乘法，更容易受算力限制；小 Batch Decode 需要反复读取权重和增长中的 KV Cache，更容易受显存带宽限制。实际瓶颈也取决于 Batch、上下文长度、Kernel 和硬件。

优化方法影响的环节不同：MQA/GQA/MLA 减少 KV 存储与读取需求，PagedAttention 改善缓存管理，FlashAttention 优化 Attention 数据搬运，Prefix Cache / RadixAttention 利用共享前缀减少重复 Prefill。缓存局部性、Attention Kernel 和调度效率也会影响 Decode 延迟。

### 2.4 将链路映射到时间指标

```text
端到端延迟 ≈ 排队与调度 + Tokenization + Prefill
           + 首次采样 + 后续 Decode 与采样
           + Detokenization + 网络传输等开销
```

这是定位开销的拆解；流式服务中的部分步骤会重叠，不能把各模块独立计时结果机械相加。以下延迟指标应使用同一测量端、同一时钟的时间戳。

## 3. 延迟指标

以客户端观测为例，设发送请求时间为 `t₀`，第一个输出 Token 到达时间为 `t₁`，第 `N` 个（最后一个）输出 Token 到达时间为 `tₙ`。若采用服务端口径，则应统一改用服务端时间戳，并注明不包含哪些网络与客户端开销。

### TTFT：Time To First Token

```text
TTFT = t₁ - t₀
```

表示用户等待首 Token 的时间，包含排队、调度、Prefill、首次采样和传输等开销。它对聊天类应用尤其重要。首 Token 通常由 Prefill 的输出直接采样得到，不应额外加上一次常规 Decode 前向计算。

### ITL：Inter-Token Latency

```text
ITLᵢ = tᵢ - tᵢ₋₁
```

表示相邻输出 Token 的时间间隔，可以发现生成过程中的抖动或卡顿。

### TPOT：Time Per Output Token

常见定义是首 Token 之后的平均生成时间：

```text
TPOT = (tₙ - t₁) / (N - 1)
```

该公式要求 `N > 1`；只有一个输出 Token 时没有 Token 间隔，应记为不适用。比如输出 101 个 Token，首个到最后一个耗时 5 秒，则 TPOT 为 `5 / 100 = 50 ms/Token`，对应单请求约 `20 Tokens/s`。

TPOT 主要反映 Decode 性能。不同工具可能使用不同分母，比较数据前必须确认定义。

### E2E Latency

```text
E2E Latency = tₙ - t₀
```

这里用最后一个 Token 到达作为结束点；若工具统计到最终完成事件，需另行注明。它受输入长度、输出长度和排队时间共同影响，不能脱离请求长度直接比较。

流式响应的一个数据块可能包含多个 Token，因此客户端观测的块间隔未必等于模型逐 Token 的 ITL，测量时需说明统计粒度。

## 4. 吞吐指标

- **Request Throughput**：每秒完成的请求数，单位 `requests/s`。
- **Input Throughput**：每秒处理的 Prompt Token 数。
- **Output Throughput**：每秒生成的 Token 数。
- **Total Token Throughput**：输入与输出 Token 的总和除以测试时间。

```text
Request Throughput = 完成请求数 / 测试持续时间
Input Throughput = 处理的输入 Token 数 / 测试持续时间
Output Throughput = 所有请求生成的 Token 数 / 测试持续时间
Total Token Throughput = (输入 Token 数 + 输出 Token 数) / 测试持续时间
```

请求吞吐也常用 RPS 或 QPS 表示，但需区分发送速率与完成速率。Input Throughput 主要反映 Prefill 处理能力，Output Throughput 主要反映生成能力；单请求的 `1 / TPOT` 不等于整个系统的输出吞吐。

只报告 Tokens/s 容易误导：Prefill Token 和 Decode Token 的计算特征不同，输入/输出长度分布也会显著改变结果。

## 5. Goodput 与 SLO

吞吐统计所有完成的工作，Goodput 只统计满足服务目标的工作：

```text
Goodput = 满足 SLO 的请求数 / 测试持续时间
```

SLO 可以组合多个条件，例如：

```text
单请求达标条件：TTFT < 1 s、TPOT < 100 ms、E2E < 10 s
服务整体目标示例：TTFT P99 < 1 s、错误率 < 0.1%
```

SLO 是服务质量目标，SLA 是服务协议中的承诺。计算请求级 Goodput 时，应先明确单请求的达标条件，不能直接用总体 P99 判断某个请求是否达标。例如每秒完成 100 个请求，其中 40 个未满足条件，则 Goodput 为 60 requests/s。

某配置即使吞吐更高，如果大量请求超过 SLO，其 Goodput 反而可能更低。

## 6. 分位数与尾延迟

- **P50**：中位数，约 50% 的观测值不超过它，表示典型用户体验。
- **P90/P95/P99**：分别约有 90% / 95% / 99% 的观测值不超过它们，用于观察慢请求和尾延迟。
- **Max**：有助于发现极端异常，但容易受单个噪声影响。

在线系统不能只看平均值。排队、长 Prompt、调度抢占、内存不足和网络抖动都可能只影响部分请求，却显著恶化 P99。

## 7. 系统与资源指标

性能异常需要通过资源指标继续定位：

| 层级 | 常用指标 | 主要用途 |
| --- | --- | --- |
| 调度 | Queue Time、Running/Waiting Requests、Batch Tokens | 判断排队和组批问题 |
| KV Cache | 使用率、Block 数、Prefix Cache Hit Rate | 判断显存容量和缓存效率 |
| GPU | SM Utilization、HBM Bandwidth、Kernel Time | 判断计算或访存瓶颈 |
| 服务 | 并发数、Arrival Rate、错误率、取消率 | 判断负载和稳定性 |

### 并发、Batch 与 Continuous Batching

**Concurrency** 是同时在途的请求数；报告时应区分客户端在途请求、服务端等待请求和正在运行的请求。增加并发可能提高吞吐，但接近饱和后通常会增加排队和尾延迟。

**Batch Size** 可以指一次 Prefill 的请求数，也可以指一次 Decode 的活跃请求数。由于请求长度不同，还应报告 **Batch Tokens**，即一次调度处理的 Token 数。

固定 Batch 中，请求长短不一，先结束的请求可能留下闲置槽位。**Continuous Batching** 在调度步之间移除已完成请求、加入符合资源条件的新请求，使 Batch 动态变化，提高资源利用率。它仍受 Token 预算、KV Cache 容量和调度策略约束。

### KV Cache 容量与命中率

对于普通 MHA/MQA/GQA，每个请求的逻辑 KV Cache 大小约为：

```text
KV Cache 字节数 ≈ 2 × num_layers × seq_len
                × num_kv_heads × head_dim × bytes_per_element
```

`2` 对应 K 和 V，`seq_len` 包含已缓存的输入与输出 Token。多个请求可按各自长度求和；分页分配、共享前缀和预留空间会让物理占用与简单求和不同。MLA 等特殊结构需要使用对应的缓存布局估算。

KV Cache 会影响最大上下文、最大并发、Batch 容量和 Decode 读取成本。权重能装入显存，并不代表剩余显存足以支持目标负载。

前缀缓存可以复用相同系统提示词等内容，减少重复 Prefill，改善 TTFT。**Prefix Cache Hit Rate** 必须注明口径，例如：

```text
Token 命中率 = 命中的前缀 Token 数 / 查询的输入 Token 总数
```

有些实现按请求、Block 或可复用 Token 统计，不能直接横向比较。还应同时观察命中 Token 数与实际节省的 Prefill 时间。

### 权重显存、动态显存与峰值显存

权重占用约为 `参数量 × 每参数字节数`。仅计算权重时，7B FP16 约为 14 GB，70B FP16 约为 140 GB（十进制单位）。INT8/INT4 可以降低权重存储，但还需计入量化元数据等开销。

KV Cache 的需求随并发数、输入输出长度、层数、KV Head 数、Head 维度和缓存精度变化；引擎可能预先分配缓存池，因此已分配显存不一定随每个请求实时增减。

**Peak Memory** 还包括激活临时 Buffer、Attention Workspace、CUDA Graph Buffer、通信 Buffer 和框架管理开销。容量规划应观察实测峰值并留出余量，而非只加权重与逻辑 KV 大小。

### GPU 利用率与计算、访存瓶颈

GPU Utilization 高不等于有效吞吐高。忙碌时间不能说明 GPU 达到了多少有效算力，应结合以下指标分析：

- SM Utilization、Tensor Core Utilization：计算资源使用情况。
- HBM / 显存带宽利用率：数据读取与写入压力。
- Kernel Time、Kernel Occupancy：耗时热点和执行资源利用情况；Occupancy 高也不保证性能好。

大 Batch GEMM、MLP 和较长 Prompt 的 Prefill 往往更偏 Compute-bound；小 Batch Decode、长上下文的 KV 读取往往更偏 Memory-bound。需要通过实际测量判定，不能只凭阶段名称下结论。

**MFU（Model FLOPs Utilization）** 比较有效模型计算速率与硬件理论峰值：

```text
MFU = (模型有效 FLOPs / 测量耗时) / GPU 理论峰值 FLOPs/s
```

多 GPU 时分母使用参与计算的 GPU 峰值之和，并匹配精度与稀疏性口径。MFU 越高，表示该口径下有效计算速率越接近理论峰值；受带宽限制的 Decode 可能 MFU 较低，仍需结合延迟和 Goodput 评价。

### 多 GPU / 多机通信

张量并行（TP）、流水线并行（PP）、数据并行（DP）和专家并行（EP）具有不同的通信模式。常见操作包括 AllReduce、AllGather、ReduceScatter、Broadcast、Send / Recv，以及专家并行中的 All-to-All。

观察通信延迟、有效带宽、通信耗时占比，以及通信与计算的重叠程度。NVLink、PCIe 和 InfiniBand 等互连的性能会影响扩展效率：例如 TP 中频繁的集合通信可能拉长每个 Decode Step。总通信耗时不等于额外延迟，应关注未被计算覆盖、落在关键路径上的部分。

## 8. 成本指标

常见表达方式：

- 每百万输入/输出 Token 的成本。
- `tokens / GPU-second`、`tokens / $`、`requests / GPU-hour`。
- 每个满足 SLO 请求的成本。
- 单位 Token 的能耗。

成本比较必须固定模型、精度、硬件、负载分布和质量要求。量化虽然可能提高吞吐，但若质量不满足要求，就不能算有效优化。

## 9. 如何做公平的 Benchmark

至少固定并记录：

- 模型、精度、并行配置、软件版本和 GPU 型号。
- 输入/输出长度分布、请求数量、并发或 Arrival Rate。
- Open-loop 还是 Closed-loop 压测模型。
- 预热方式、测试时长、随机种子和采样参数。
- 是否包含 Tokenization、网络传输和客户端开销。

测试时逐步增加负载并绘制 **吞吐—延迟曲线**。当 Arrival Rate 接近系统最大处理能力时，队列会持续增长，尾延迟通常急剧恶化，这就是饱和点。

## 10. 面试快速回答

**TTFT 和 TPOT 分别受什么影响？** TTFT 主要受排队与 Prefill 影响；TPOT 主要受 Decode 调度、Batch、权重读取和 KV Cache 访问影响。

**吞吐越高是否代表系统越好？** 不一定。更大的 Batch 能提高吞吐，却可能增加排队时间和尾延迟，应在目标 SLO 下比较 Goodput。

**为什么必须报告输入和输出长度？** Prefill 与 Decode 的开销不同；长度分布改变，两种阶段占比也会改变，测试结果无法直接横向比较。

## 总结

评估 LLM 推理系统时，应以 **TTFT、TPOT、P99、Throughput 和 Goodput** 为核心，再结合 KV Cache、GPU 与调度指标定位原因。最有意义的结论不是“峰值 Tokens/s 最高”，而是“在指定负载和 SLO 下，有效吞吐与成本最优”。
