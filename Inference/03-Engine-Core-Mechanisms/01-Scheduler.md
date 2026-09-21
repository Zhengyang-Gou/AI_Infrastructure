# 01 · Scheduler

状态：待学习。

## 核心问题

- 等待队列和运行队列如何维护请求状态？
- 每轮调度受到哪些请求数、Token 数和 KV Cache 容量限制？
- Prefill 与 Decode 如何选择执行顺序？
- 资源不足时如何处理抢占、重算与公平性？

## 阅读入口

- [已有笔记：调度与批处理](../02-Mini-Inference-Engine/notes/04-Scheduler-and-Batching.md)
- [已有笔记：请求状态与生命周期](../02-Mini-Inference-Engine/notes/02-Sequence-and-Lifecycle.md)

## 手工推演

设定请求数与 Token 预算，列出三个不同到达时间的请求在每轮的队列状态、调度结果和资源占用。

## 学习记录

待补充：机制说明、源码位置与版本、推演过程、疑问及总结。

[返回学习目录](README.md)
