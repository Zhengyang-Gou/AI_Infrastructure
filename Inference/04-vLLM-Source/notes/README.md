# vLLM 源码笔记

沿一批请求的实际执行顺序，跟踪输入提交、逐轮计算、状态推进与结果返回。

固定配置与源码阅读约定见 [00 · 固定配置与阅读约定](00-Reading-Setup.md)。先读 [全局执行路线](Overview.md)，再按八章推进。

## 八章目录

| 顺序 | 章节 | 跟踪范围 |
| --- | --- | --- |
| 01 | [创建引擎与准备执行环境](01-Engine-Initialization/README.md) | 从创建 LLM 到具备接收和执行请求的环境。 |
| 02 | [提交一批离线请求](02-Offline-Request-Submission/README.md) | 从 LLM.generate(prompts) 到逐条提交核心请求。 |
| 03 | [请求进入核心，等待执行](03-Core-Request-Admission/README.md) | 从核心接收请求到进入第一次调度。 |
| 04 | [决定本轮执行哪些请求](04-Step-Scheduling/README.md) | 从未完成请求与资源状态到本轮 SchedulerOutput。 |
| 05 | [把执行计划变成模型输入](05-Model-Input-Preparation/README.md) | 从 SchedulerOutput 到准备好本轮 GPU 计算输入。 |
| 06 | [完成本轮前向与采样](06-Forward-and-Sampling/README.md) | 从本轮模型输入到 ModelRunnerOutput。 |
| 07 | [回写执行结果，推进下一轮](07-Request-Update-and-Next-Step/README.md) | 从模型执行结果到下一轮请求状态或核心收尾。 |
| 08 | [将增量还原成文本，返回离线结果](08-Output-and-Result-Collection/README.md) | 从核心增量输出到本次 generate 的最终结果。 |

## 文档组织

- 00：固定配置、初始化示例与源码阅读约定。
- 全局 README：八章导航。
- 全局 Overview：执行阶段、循环与输出返回关系。
- 各章 README：全引擎流程与本章定位、带简短用途说明的章内调用图，以及主题与前后导航。
- 章内笔记：后续按实际调用顺序展开，记录进入前状态、执行代码、状态变化与下一站。

第 01 章已完成初始化主线的七篇正文；第 02～08 章目前为章节骨架，正文待后续整理。原有按组件组织的笔记完整保留在 [旧版笔记归档](../notes-archive/README.md)。

[返回推理学习路线](../../Roadmap.md)
