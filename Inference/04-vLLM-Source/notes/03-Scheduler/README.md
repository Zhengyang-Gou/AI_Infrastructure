# 03 · Scheduler

本章学习 vLLM 的请求调度：**每一轮选哪些请求、各计算多少 token，资源不足时怎样取舍，执行结果回来后又如何继续？**

上一章已经串起 `schedule → execute_model / sample_tokens → update_from_output`。这一章进入 Scheduler 内部，将请求进度、token 预算、队列策略和 KV Cache 约束放到同一轮执行中理解。

固定场景见[顶层 README](../README.md) 第 1 节。先沿基础 `Scheduler` 和没有多批次在途的执行顺序阅读，再认识 `AsyncScheduler`、推测解码等扩展。

## 1. 本章在学习路线中的位置

```text
Engine Core：组织接收、调度、执行与结果返回
    ↓ schedule                         ↑ update_from_output
Scheduler：维护请求，决定本轮计算内容并更新执行状态       ← 本章
    ├─ Request / RequestQueue：保存进度，组织等待与运行请求
    ├─ KVCacheManager：检查缓存复用与块分配是否可行
    └─ SchedulerOutput：将请求、计算量与缓存信息交给执行侧
            ↓↑
        Executor → Worker / Model Runner：执行模型与采样
```

从 [第二章 Overview](../02-Engine-Core/Overview.md) 接着读即可。机制层可以回看 [Scheduler](../../../03-Engine-Core-Mechanisms/01-Scheduler.md)、[Continuous Batching](../../../03-Engine-Core-Mechanisms/02-Continuous-Batching.md) 和 [Chunked Prefill](../../../03-Engine-Core-Mechanisms/05-Chunked-Prefill.md)，本章把这些机制对应到本地源码的对象、字段与调用顺序。

## 2. 章节框架

| 顺序 | 主题 | 要回答的问题 | 学习深度 |
| --- | --- | --- | --- |
| 01 | Scheduler、配置与请求状态 | 调度器保存什么？请求表、等待队列与运行列表如何分工？ | 掌握对象关系与生命周期 |
| 02 | Token 进度与运行请求调度 | 如何从请求进度算出本轮需求？预算、长度和缓存怎样限制计算量？ | 本章重点，读透 RUNNING 路线 |
| 03 | 等待请求接纳与连续批处理 | 新请求与抢占请求何时进入运行集合？FCFS、Priority 和 Prompt 分块怎样协作？ | 本章重点，读透 WAITING 路线 |
| 04 | KV 约束、抢占与恢复 | 有 token 预算却分不到缓存块时怎么办？恢复为什么可能需要重算？ | 追踪选择、回退与重新接纳 |
| 05 | 调度计划与结果更新 | SchedulerOutput 传什么？何时推进进度、追加输出、判停并清理？ | 串起完整调度闭环 |
| 06 | 异步调度与扩展边界 | 结果尚未返回时怎样继续安排？哪些字段服务于推测解码和额外依赖？ | 先理解占位与在途状态，再按需展开 |

## 3. 文档组织与阅读方式

- **README** 说明本章范围、主题划分、源码入口与学习目标。
- **[Overview](Overview.md)** 建立组件框架、进度模型和执行路线，并用分轮示例复盘。
- **Core-Modules** 后续可按上面的六个主题展开，沿用前两章「职责与边界 → 源码定位与关键数据 → 实现分析 → 手工推演 → 设计原因与自测」的组织方式。

先读 Overview，再按下面的顺序进入源码；目前的源码阅读入口均列在这两份文档中。

| 顺序 | 主要源码 | 重点函数或结构 |
| --- | --- | --- |
| 01 | [scheduler.py](../../vllm/vllm/v1/core/sched/scheduler.py)、[request.py](../../vllm/vllm/v1/request.py)、[配置](../../vllm/vllm/config/scheduler.py) | Scheduler.__init__、Request / RequestStatus、SchedulerConfig |
| 02 | scheduler.py、request.py | schedule 的 RUNNING 循环、num_tokens / num_tokens_with_spec、num_computed_tokens |
| 03 | scheduler.py、[request_queue.py](../../vllm/vllm/v1/core/sched/request_queue.py) | schedule 的 WAITING 循环、_select_waiting_queue_for_scheduling、FCFSRequestQueue / PriorityRequestQueue |
| 04 | scheduler.py、[kv_cache_manager.py](../../vllm/vllm/v1/core/kv_cache_manager.py) | _preempt_request、get_computed_blocks、allocate_slots |
| 05 | scheduler.py、[output.py](../../vllm/vllm/v1/core/sched/output.py)、[utils.py](../../vllm/vllm/v1/core/sched/utils.py) | SchedulerOutput、_update_after_schedule、update_from_output、_update_request_with_output、check_stop、finish_requests |
| 06 | [async_scheduler.py](../../vllm/vllm/v1/core/sched/async_scheduler.py)、配置 | AsyncScheduler、num_output_placeholders、num_in_flight_tokens、get_scheduler_cls |

源码路径相对于 `04-vLLM-Source/vllm/`。函数源码节选沿用前两章的约定：代码块开头标注文件、类、函数和本地行范围；流程与公式示意单独说明。阅读时先确认本轮输入、持久状态与输出，再进入相关分支。

## 4. 相关章节的分工

| 位置 | 本章关注点 | 对应章节的重点 |
| --- | --- | --- |
| [02 · Engine Core](../02-Engine-Core/README.md) | schedule 与 update_from_output 怎样接入执行闭环 | 客户端、核心进程循环、执行器调用与结果返回 |
| [04 · KV Cache Management](../04-KV-Cache-Management/) | 查询命中、申请块、分配失败后的调度选择与资源释放接口 | 块池、引用计数、前缀匹配、缓存淘汰与实际回收时机 |
| [06 · Model Execution](../06-Model-Execution/) | 执行侧需要怎样的计划，结果怎样按请求 ID 对回 | 输入张量、位置、block table、模型前向与执行侧状态 |
| [08 · Sampling](../08-Sampling/) | 取得有效新 token，检查请求是否结束 | 采样算法与从请求到输出的整条链路 |
| [09 · Runtime Optimizations](../09-Runtime-Optimizations/) | 异步占位、推测 token 与调度进度如何衔接 | 批次重叠、推测解码等优化的完整执行机制 |

## 5. 本章学习目标

- 解释 `requests`、`waiting`、`skipped_waiting` 与 `running` 的职责，以及它们如何引用同一条 Request。
- 用 `num_tokens`、`num_computed_tokens` 与 `num_scheduled_tokens` 解释 Prefill、Prompt 分块和普通 Decode。
- 区分本轮 token 预算、运行请求数量限制、单请求长度限制与 KV Cache 容量约束。
- 沿「先运行请求、再等待请求」追踪一轮选择，并说明 FCFS / Priority 的作用范围。
- 用逐轮示例解释 Continuous Batching 与 Chunked Prefill 如何在同一个 schedule 中实现。
- 说明抢占、恢复与完成的区别，理解保留 token 历史后为什么仍可能重算 KV。
- 串起 `SchedulerOutput → 模型执行结果 → Request 更新 → 核心增量输出`，解释调度进度为什么会在结果返回前推进。
- 区分基础调度、AsyncScheduler 和 Engine Core 的 batch queue，认识在途进度与输出占位。

[开始阅读 Overview](Overview.md) · [上一章：Engine Core](../02-Engine-Core/README.md) · [下一章：KV Cache Management](../04-KV-Cache-Management/) · [返回推理学习路线](../../../Roadmap.md)
