# 03 · Engine 核心机制

第三周学习入口：Scheduler、Continuous Batching、Paged KV Cache、Prefix Cache、Chunked Prefill。

学习目标：结合第二阶段的 Mini Inference Engine，理解请求调度、批处理与 KV Cache 管理如何协作。以下笔记为待完成的学习模板。

## 学习顺序

| 顺序 | 主题 | 核心问题 |
| --- | --- | --- |
| 01 | [Scheduler](01-Scheduler.md) | 每轮选择哪些请求，分配多少 Token 和缓存资源？ |
| 02 | [Continuous Batching](02-Continuous-Batching.md) | 请求如何在迭代之间加入和退出 Batch？ |
| 03 | [Paged KV Cache](03-Paged-KV-Cache.md) | 如何分配、寻址和回收 KV Cache 块？ |
| 04 | [Prefix Cache](04-Prefix-Cache.md) | 如何识别和复用请求之间的公共前缀？ |
| 05 | [Chunked Prefill](05-Chunked-Prefill.md) | 如何拆分长 Prompt 的 Prefill 并与 Decode 协同调度？ |

## 学习方式

每个主题按「问题 → 机制 → 源码 → 手工推演 → 总结」补充笔记。先复习 [Nano-vLLM 笔记](../02-Mini-Inference-Engine/notes/README.md)，再对照 [本地 vLLM 源码](../04-vLLM-Source/vllm/)。记录所读代码版本与实现差异。

## 学习验收

- [ ] 画出请求从等待、执行到完成的调度流程。
- [ ] 用三个长度不同的请求推演 Continuous Batching。
- [ ] 手工追踪 KV Cache 块的分配、映射与释放。
- [ ] 推演公共前缀的缓存命中与未命中情况。
- [ ] 推演一个长 Prompt 与多个 Decode 请求共存的调度过程。
- [ ] 串起五种机制，说明需要观察的 TTFT、TPOT、吞吐与缓存占用指标。

[上一阶段：Mini Inference Engine](../02-Mini-Inference-Engine/README.md) · [下一阶段：vLLM 源码](../04-vLLM-Source/) · [学习路线](../Roadmap.md)
