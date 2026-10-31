# 02 · Engine Core

本章学习 vLLM 的核心执行框架：**请求准备好之后，谁决定这一轮执行什么，谁运行模型，又如何让请求继续到下一轮？**

上一章已经串起「准备输入 → 构造请求 → 提交核心 → 收集结果」。这一章进入核心内部，重点掌握 **接收请求 → 调度 → 执行 → 更新状态 → 返回增量** 的循环。

以仓库内 `04-vLLM-Source/vllm/` 的源码为依据，先沿普通文本、decoder-only、每条输入一个候选（`n=1`）、单核心的生成路径阅读。

## 1. 本章在学习路线中的位置

```text
Inference Entry：准备请求、提交、将输出还原为用户结果
    ↓ EngineCoreRequest                 ↑ EngineCoreOutputs
Engine Core：接收请求，组织一轮轮调度与执行             ← 本章
    ├─ Scheduler：选择请求，分配本轮 token 预算
    ├─ KV Cache 管理：为执行提供缓存块
    └─ Executor → Worker / Model Runner：组织模型执行
```

从 [第一章 Overview](../01-Inference-Entry/Overview.md) 接着读即可。入口部分只需记住 `LLMEngine.add_request()` 提交请求、`LLMEngine.step()` 取得并处理输出。

## 2. 章节框架

| 顺序 | 主题 | 要回答的问题 | 学习深度 |
| --- | --- | --- | --- |
| 01 | EngineCore 与初始化 | 核心持有哪些组件？执行器、KV Cache 配置、调度器怎样建立？ | 掌握组件关系与初始化顺序 |
| 02 | Request 与请求接收 | EngineCoreRequest 如何变成 Request？请求在哪里等待，怎样保存进度？ | 理解对象变化、入队与状态 |
| 03 | Step 与执行闭环 | 本轮怎样调度、执行、处理结果，并开始下一轮？ | 本章重点，沿调用链读透 |
| 04 | 核心客户端与进程循环 | 谁调用核心 step？同进程与后台进程怎样驱动同一个核心？ | 掌握调用路线与队列作用 |
| 05 | 完成、取消与清理 | 请求何时结束？怎样退出队列、回收资源并通知入口？ | 串起正常完成与取消两条路线 |

## 3. 文档组织与阅读方式

- [README](README.md)：说明本章范围、主题划分和学习目标。
- [Overview](Overview.md)：先用一条请求建立组件框架、数据流和执行路线。
- [Core-Modules](Core-Modules/)：按主题解析具体函数、数据结构与状态变化。

实现笔记按以下五个主题组织：

| 笔记 | 主要源码与关注点 |
| --- | --- |
| [01 · EngineCore 与初始化](Core-Modules/01-EngineCore-and-Initialization.md) | EngineCore.__init__、_initialize_kv_caches；组件与资源怎样准备 |
| [02 · Request 与请求接收](Core-Modules/02-Request-and-Admission.md) | preprocess_add_request、Request.from_engine_core_request、Scheduler.add_request；请求怎样进入核心 |
| [03 · Step 与执行闭环](Core-Modules/03-Step-and-Execution-Loop.md) | EngineCore.step、Scheduler.schedule、Executor.execute_model / sample_tokens、Scheduler.update_from_output；完整一轮怎样衔接 |
| [04 · 核心客户端与进程循环](Core-Modules/04-Core-Client-and-Process-Loop.md) | InprocClient、SyncMPClient、EngineCoreProc；调用、输入队列、后台循环和输出返回 |
| [05 · 完成、取消与清理](Core-Modules/05-Completion-and-Abort.md) | check_stop、EngineCore.abort_requests、Scheduler.finish_requests 与资源清理；请求怎样结束 |

先读 Overview，能用自己的话解释一轮执行，再按 01 → 05 展开。阅读每个函数时，先确认它接收什么、改变什么状态、把结果交给谁；与主线相关的关键调用再进入实现。

函数源码在代码块开头使用 `# 文件`、`# 类`、`# 函数` 标注位置；实际源码节选注明本地行范围，流程示意明确标注为示意。

## 4. 相关章节的分工

| 位置 | 本章关注点 | 对应章节的重点 |
| --- | --- | --- |
| [03 · Scheduler](../03-Scheduler/) | schedule 产生本轮执行计划，执行结果回来后更新请求 | token 预算、排队策略、抢占、Chunked Prefill、连续批处理 |
| [04 · KV Cache Management](../04-KV-Cache-Management/) | 调度需要缓存块；请求结束需要释放相应引用 | 块分配、引用计数、前缀复用、缓存不足时的处理 |
| [05 · Model Execution](../05-Model-Execution/) | Executor 接收调度计划，将执行任务交给 Worker / Model Runner | 输入张量、位置、block table、模型前向及采样调用 |
| [08 · Sampling and End-to-End](../08-Sampling-and-End-to-End/) | 核心取得采样后的 token，检查结束条件并返回增量 | 采样算法与完整执行链路复盘 |

先用 `EngineCore.step()` 理解顺序明确的一轮执行，再认识 `step_with_batch_queue()` 如何允许多批次在途。具体调度器类型与 `step_fn` 由本地配置选择。

## 5. 本章学习目标

- 区分 LLMEngine、EngineCoreClient、EngineCore、Scheduler 与 Executor 的职责。
- 解释 EngineCoreRequest 与核心 Request 的区别，以及 Request 为什么需要跨轮保存。
- 串起 `schedule → execute_model / sample_tokens → update_from_output`。
- 区分本轮参与计算的 token 数与本轮新生成的 token 数。
- 解释同进程调用与核心后台循环各自如何推进执行。
- 说明请求完成或取消后，核心状态清理怎样与入口输出处理衔接。

[开始阅读 Overview](Overview.md) · [上一章：Inference Entry](../01-Inference-Entry/README.md) · [下一章：Scheduler](../03-Scheduler/)
