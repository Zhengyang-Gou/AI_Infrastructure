# 05 · Chunked Prefill

状态：待学习。

## 核心问题

- 长 Prompt 的 Prefill 会如何影响正在 Decode 的请求？
- 如何按照每轮 Token 预算拆分 Prefill？
- 分块之间如何维护已计算 Token 数、位置与 KV Cache？
- Prefill 分块与 Decode 如何共同参与调度？
- 如何分析分块大小对 TTFT、TPOT 和吞吐的影响？

## 阅读入口

- [基础笔记：Prefill 与 Decode](../01-LLM-Inference-Basics/03-Prefill-and-Decode.md)
- [前置主题：Scheduler](01-Scheduler.md)
- [前置主题：Continuous Batching](02-Continuous-Batching.md)
- [本地 vLLM 源码](../04-vLLM-Source/vllm/)

## 手工推演

设每轮最多调度 8 个 Token，已有两个请求各需 Decode 1 个 Token，另有一个长度为 18 的 Prompt 等待 Prefill。明确调度策略后，逐轮记录预算分配、Prefill 进度与请求状态。

## 学习记录

待补充：机制说明、源码位置与版本、推演过程、疑问及总结。

[返回学习目录](README.md)
