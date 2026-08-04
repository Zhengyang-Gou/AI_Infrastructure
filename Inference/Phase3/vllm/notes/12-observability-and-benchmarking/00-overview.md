# Observability and Benchmarking

## 学习目标

本阶段建立从请求事件到指标、trace 和 profiler 的观测链，并学会用正确工作负载测量在线延迟、吞吐与 goodput，而不是只比较单个 kernel。

## 阅读顺序

| 顺序 | 笔记 | 主要内容 |
| --- | --- | --- |
| 1 | `01-metrics.md` | 请求事件、统计聚合与 Prometheus |
| 2 | `02-tracing-and-profiling.md` | OpenTelemetry spans 与 Worker profiler |
| 3 | `03-benchmarking.md` | latency、throughput、serve benchmark 和结果解释 |

## 观测链

```text
Scheduler / Engine / Request events
→ IterationStats
→ StatLoggerManager
→ log / Prometheus metrics

HTTP trace context
→ API / Engine spans
→ OpenTelemetry exporter

Worker execution
→ Torch / CUDA profiler
→ operator timeline 与 layer statistics
```

## 完成标准

- 能说明 TTFT、TPOT、ITL、E2EL、throughput 和 goodput 的定义与适用场景。
- 能从请求事件定位某项直方图指标的记录位置。
- 能区分持续监控、分布式 tracing 和短时间 profiling。
- 能设计具有固定数据集、到达率、并发和输出长度的可复现实验。
- 能识别 warmup、prefix cache、tokenizer、网络和客户端限速造成的偏差。

## 当前结论

性能学习必须形成闭环：指标发现现象，trace 定位跨组件耗时，profiler 下钻 GPU/CPU 热点，benchmark 再验证改动是否改善真实工作负载。
