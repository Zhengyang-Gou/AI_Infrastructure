# 03 · Scheduler

**本章回答：每轮谁执行、算多少 token，请求何时等待、抢占或结束？**

## 1. 职责边界

| 项目 | 契约 |
| --- | --- |
| 输入 | Request、调度配置、KV 管理接口结果、模型结果或外部结束指令 |
| 输出 | SchedulerOutput、更新后的 Request、EngineCoreOutputs 与资源交还指令 |
| 本章负责 | Request/RequestStatus、请求表与队列、预算、接纳、抢占对象选择、同步进度、结果更新、核心判停与请求退场 |
| 到此为止 | 用 KVCacheManager 返回的命中长度、块 ID、成功/失败形成调度决策；不进入块池和引用计数 |

| 内容 | 唯一归属 |
| --- | --- |
| 核心调用顺序、客户端与进程循环 | 02 Engine Core |
| 前缀匹配、块分配、淘汰与实际回收 | 04 KV Cache Management |
| 输入张量、执行侧请求状态与模型前向 | 06 Model Execution |
| logits 与采样算法 | 08 Sampling |
| 停止字符串与输出状态 | 01 Inference Entry |
| AsyncScheduler 占位、推测接受/拒绝、batch queue | 09 Runtime Optimizations |

EOS、停止 token 和长度等核心判停由本章负责，08 只产生采样结果。

## 2. 正文范围与当前状态

先读 [Overview](Overview.md)。基础调度的完整分轮推演只在 Overview §6 维护。

| 顺序 | 主题 | 状态 |
| --- | --- | --- |
| 01 | [Request 与调度队列](Core-Modules/01-Request-and-Queues.md) | 已有正文：初始状态、登记、队列归属 |
| 02 | Token 进度与 RUNNING 请求预算 | Overview 已有说明；独立正文待写 |
| 03 | WAITING 接纳与连续批处理 | Overview 已有说明；独立正文待写 |
| 04 | KV 接口约束、抢占选择与恢复 | Overview 已有说明；独立正文待写 |
| 05 | [结果更新、判停与请求退场](Core-Modules/05-Result-and-Completion.md) | 已有正文：输出追加、判停、退场与释放接口 |

01 负责状态起点与队列归属；02–04 负责形成计划；05 负责结果返回后的状态变化。每篇使用其他篇的结果，不重复展开算法。

## 3. 源码入口

路径相对于 `04-vLLM-Source/vllm/`；节选标注文件、类、函数与本地行范围。

| 文件 | 关注内容 |
| --- | --- |
| [request.py](../../vllm/vllm/v1/request.py) | Request、RequestStatus、token 历史与进度 |
| [scheduler.py](../../vllm/vllm/v1/core/sched/scheduler.py) | add_request、schedule、_preempt_request、_update_after_schedule、update_from_output、finish_requests |
| [request_queue.py](../../vllm/vllm/v1/core/sched/request_queue.py) | FCFSRequestQueue、PriorityRequestQueue |
| [output.py](../../vllm/vllm/v1/core/sched/output.py) | SchedulerOutput 及请求数据契约 |
| [utils.py](../../vllm/vllm/v1/core/sched/utils.py) | check_stop |
| [配置](../../vllm/vllm/config/scheduler.py) | 预算、队列策略与基础调度器配置 |

## 4. 验收

- 手算案例 A 的每轮预算、运行集合与输出增长。
- 解释 RUNNING 为什么可能仍在 Prefill。
- 区分 token 预算、运行名额、总长度与 KV 容量限制。
- 区分“请求被抢占”和“请求已经结束”。
- 指出调度时推进的字段与结果返回后才更新的字段。
- 解释采样结果到达后，谁判停、谁退出队列、谁接到释放指令。

[源码纵览](../00-Overview.md) · [下一章：KV Cache Management](../04-KV-Cache-Management/README.md)
