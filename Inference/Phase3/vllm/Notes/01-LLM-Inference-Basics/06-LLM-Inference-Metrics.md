# LLM 推理性能指标

> 状态：已完成初稿 ｜ 更新：2026-09-04

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

## 2. 延迟指标

设请求到达时间为 `t₀`，第一个输出 Token 到达时间为 `t₁`，最后一个输出 Token 到达时间为 `tₙ`。

### TTFT：Time To First Token

```text
TTFT = t₁ - t₀
```

表示用户等待首 Token 的时间，包含排队、调度、Prefill、首次采样和传输等开销。它对聊天类应用尤其重要。

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

TPOT 主要反映 Decode 性能。不同工具可能使用不同分母，比较数据前必须确认定义。

### E2E Latency

```text
E2E Latency = tₙ - t₀
```

它受输入长度、输出长度和排队时间共同影响，不能脱离请求长度直接比较。

## 3. 吞吐指标

- **Request Throughput**：每秒完成的请求数，单位 `requests/s`。
- **Input Throughput**：每秒处理的 Prompt Token 数。
- **Output Throughput**：每秒生成的 Token 数。
- **Total Token Throughput**：输入与输出 Token 的总和除以测试时间。

```text
Output Throughput = 所有请求生成的 Token 数 / 测试持续时间
```

只报告 Tokens/s 容易误导：Prefill Token 和 Decode Token 的计算特征不同，输入/输出长度分布也会显著改变结果。

## 4. Goodput 与 SLO

吞吐统计所有完成的工作，Goodput 只统计满足服务目标的工作：

```text
Goodput = 满足 SLO 的请求数 / 测试持续时间
```

SLO 可以组合多个条件，例如：

```text
TTFT P99 < 1 s
TPOT P99 < 50 ms
错误率 < 0.1%
```

某配置即使吞吐更高，如果大量请求超过 SLO，其 Goodput 反而可能更低。

## 5. 分位数与尾延迟

- **P50**：典型用户体验。
- **P95/P99**：慢请求和尾延迟。
- **Max**：有助于发现极端异常，但容易受单个噪声影响。

在线系统不能只看平均值。排队、长 Prompt、调度抢占、内存不足和网络抖动都可能只影响部分请求，却显著恶化 P99。

## 6. 系统与资源指标

性能异常需要通过资源指标继续定位：

| 层级 | 常用指标 | 主要用途 |
| --- | --- | --- |
| 调度 | Queue Time、Running/Waiting Requests、Batch Tokens | 判断排队和组批问题 |
| KV Cache | 使用率、Block 数、Prefix Cache Hit Rate | 判断显存容量和缓存效率 |
| GPU | SM Utilization、HBM Bandwidth、Kernel Time | 判断计算或访存瓶颈 |
| 服务 | 并发数、Arrival Rate、错误率、取消率 | 判断负载和稳定性 |

GPU Utilization 高不等于有效吞吐高：可能存在低效 Kernel、同步等待或重复计算，需要结合 Kernel 时间和业务指标判断。

## 7. 成本指标

常见表达方式：

- 每百万输入/输出 Token 的成本。
- `tokens / GPU-second`。
- 每个满足 SLO 请求的成本。
- 单位 Token 的能耗。

成本比较必须固定模型、精度、硬件、负载分布和质量要求。量化虽然可能提高吞吐，但若质量不满足要求，就不能算有效优化。

## 8. 如何做公平的 Benchmark

至少固定并记录：

- 模型、精度、并行配置、软件版本和 GPU 型号。
- 输入/输出长度分布、请求数量、并发或 Arrival Rate。
- Open-loop 还是 Closed-loop 压测模型。
- 预热方式、测试时长、随机种子和采样参数。
- 是否包含 Tokenization、网络传输和客户端开销。

测试时逐步增加负载并绘制 **吞吐—延迟曲线**。当 Arrival Rate 接近系统最大处理能力时，队列会持续增长，尾延迟通常急剧恶化，这就是饱和点。

## 9. 面试快速回答

**TTFT 和 TPOT 分别受什么影响？** TTFT 主要受排队与 Prefill 影响；TPOT 主要受 Decode 调度、Batch、权重读取和 KV Cache 访问影响。

**吞吐越高是否代表系统越好？** 不一定。更大的 Batch 能提高吞吐，却可能增加排队时间和尾延迟，应在目标 SLO 下比较 Goodput。

**为什么必须报告输入和输出长度？** Prefill 与 Decode 的开销不同；长度分布改变，两种阶段占比也会改变，测试结果无法直接横向比较。

## 总结

评估 LLM 推理系统时，应以 **TTFT、TPOT、P99、Throughput 和 Goodput** 为核心，再结合 KV Cache、GPU 与调度指标定位原因。最有意义的结论不是“峰值 Tokens/s 最高”，而是“在指定负载和 SLO 下，有效吞吐与成本最优”。
