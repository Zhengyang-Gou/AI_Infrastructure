# 03 · Engine 核心机制

第三周学习入口：Scheduler、Continuous Batching、Paged KV Cache、Prefix Cache、Chunked Prefill。

学习目标：结合第二阶段的 Mini Inference Engine，理解请求调度、批处理与 KV Cache 管理如何协作。本阶段从源码的设计思路出发讲解机制，使用流程图、表格与手工推演连接五个主题。

## 学习顺序

| 顺序 | 主题 | 核心问题 |
| --- | --- | --- |
| 01 | [Scheduler](01-Scheduler.md) | 每轮选择哪些请求，分配多少 Token 和缓存资源？ |
| 02 | [Continuous Batching](02-Continuous-Batching.md) | 请求如何在迭代之间加入和退出 Batch？ |
| 03 | [Paged KV Cache](03-Paged-KV-Cache.md) | 如何分配、寻址和回收 KV Cache 块？ |
| 04 | [Prefix Cache](04-Prefix-Cache.md) | 如何识别和复用请求之间的公共前缀？ |
| 05 | [Chunked Prefill](05-Chunked-Prefill.md) | 如何拆分长 Prompt 的 Prefill 并与 Decode 协同调度？ |

## 学习方式

每个主题按「问题 → 机制 → 手工推演 → 协作关系 → 观察指标」组织。先复习 [Mini Inference Engine 笔记](../02-Mini-Inference-Engine/notes/README.md)，再沿着一条请求追踪计算进度、缓存状态和执行机会。当前阶段不展开代码、字段与函数细节，具体实现留到下一阶段学习。

[上一阶段：Mini Inference Engine](../02-Mini-Inference-Engine/README.md) · [下一阶段：vLLM 源码](../04-vLLM-Source/) · [学习路线](../Roadmap.md)
