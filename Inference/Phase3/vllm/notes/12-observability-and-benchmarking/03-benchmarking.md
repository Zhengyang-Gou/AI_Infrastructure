# Benchmarking

## 1. 文件定位

- CLI：`vllm/entrypoints/cli/benchmark/`。
- 实现：`vllm/benchmarks/latency.py`、`throughput.py`、`serve.py`、`datasets/`。
- 职责：提供离线和在线工作负载生成、请求发送、指标计算与结果保存。

## 2. Benchmark 类型

| 类型 | 主要问题 | 关键控制变量 |
| --- | --- | --- |
| latency | 单批/单请求执行多快 | batch、输入输出长度、warmup |
| throughput | 离线尽快跑完固定请求集 | 请求数、长度分布、并行配置 |
| serve | 在线到达流下的延迟和吞吐 | request rate、burst、并发、endpoint |
| startup | 引擎加载和就绪需要多久 | 模型、量化、并行、缓存状态 |

在线 benchmark 同时报告 TTFT、TPOT、ITL 和 E2EL；goodput 则只统计满足指定 SLO 的请求或 token 工作量。

## 3. 在线执行流程

```text
加载并过滤 dataset
→ 生成 prompt/output length 请求
→ 按 request rate 产生到达时间
→ 异步调用 endpoint
→ 记录首 token 与后续 token timestamps
→ 计算成功率、吞吐、延迟分位数和 goodput
→ 保存参数与结果
```

请求生成器、客户端连接池和进度统计自身也可能成为瓶颈，压测前应确认客户端能够提供目标负载。

## 4. 关键代码解析

### `BenchmarkServingSubcommand.cmd()`

### `get_request()`

### `benchmark()`

### `calculate_metrics()`

### `parse_goodput()`

### `run_vllm()`

### `get_requests()`

### `main_async()`

### `BenchmarkLatencySubcommand.cmd()`

### `BenchmarkThroughputSubcommand.cmd()`

## 5. 实验纪律

- 固定 commit、模型、硬件、驱动、配置和环境变量。
- 保存请求数据集、随机 seed、输入输出 token 长度分布。
- 先 warmup，再采集多个稳定窗口，并报告分位数而非单次结果。
- 冷启动、热缓存、prefix cache 命中实验应分开进行。
- 比较系统时使用相同 tokenizer 口径与成功请求过滤规则。
- 同时记录 GPU 利用率、显存、排队和失败率，防止吞吐以不可接受长尾为代价。

## 6. 当前结论

一个有意义的 benchmark 必须明确工作负载和服务目标；“tokens/s 更高”只有在长度分布、延迟 SLO 和成功率一致时才可比较。
