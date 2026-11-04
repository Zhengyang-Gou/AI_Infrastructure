# 09 · Runtime Optimizations

**怎样在基础执行语义上减少开销、重叠工作或一次验证多个 token？**

状态：部分正文：batch queue 已有正文，异步调度已有概览，其余主题待写。

## 1. 职责边界

| 项目 | 契约 |
| --- | --- |
| 输入 | 01–08 的基础链路与启用的优化配置。 |
| 输出 | 优化后的执行时序、进度修正规则与等价结果。 |
| 本章负责 | torch.compile；CUDA Graph 捕获与重放；AsyncScheduler 的输出占位；多批在途的 batch queue；推测解码的 draft、验证、接受/拒绝与进度修正。 |

本章遇到其他模块只使用接口结果，不展开其实现。

| 内容 | 唯一归属 |
| --- | --- |
| 同步 step 与核心驱动 | 02 Engine Core |
| 基础预算、队列、结果更新与判停 | 03 Scheduler |
| 基础输入张量与前向调用 | 06 Model Execution |
| 多设备拓扑与跨 rank 协调 | 10 Distributed Execution |
| HTTP 和 asyncio 的请求交付 | 11 Online Serving |

## 2. 正文范围

1. batch queue 与多批在途（已有正文）。
2. AsyncScheduler 与输出占位（已有概览，函数分析待写）。
3. torch.compile 与 CUDA Graph。
4. 推测解码与进度修正。

已有正文：[01 · Batch Queue 与多批在途](Core-Modules/01-Batch-Queue.md)。

已有概览：[02 · 异步调度与输出占位](Core-Modules/02-Async-Scheduling.md)。

[返回模块索引](../README.md) · [源码纵览](../00-Overview.md)
