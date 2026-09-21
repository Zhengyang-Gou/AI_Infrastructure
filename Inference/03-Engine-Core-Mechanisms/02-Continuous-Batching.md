# 02 · Continuous Batching

状态：待学习。

## 核心问题

- 固定 Batch 与 Continuous Batching 的执行过程有什么区别？
- 新请求何时加入，完成的请求何时退出？
- Batch 成员变化时如何维护各请求的执行位置和缓存？
- 如何观察吞吐、等待时间与单请求延迟之间的取舍？

## 阅读入口

- [已有笔记：调度与批处理](../02-Mini-Inference-Engine/notes/04-Scheduler-and-Batching.md)
- [已有笔记：一次 Step 的完整数据流](../02-Mini-Inference-Engine/notes/10-End-to-End-Step.md)

## 手工推演

使用三个到达时间和输出长度不同的请求，分别画出固定 Batch 和 Continuous Batching 的逐轮执行表。

## 学习记录

待补充：机制说明、源码位置与版本、推演过程、疑问及总结。

[返回学习目录](README.md)
