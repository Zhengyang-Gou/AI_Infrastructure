# 02 · 异步调度与输出占位

状态：概览说明；具体函数的连续源码分析待写。

## 1. 职责与边界

本篇只负责 AsyncScheduler 相对基础调度增加的进度与输出占位。基础请求选择结果已经给定，不重新推导预算、队列与核心判停。

| 项目 | 契约 |
| --- | --- |
| 输入 | 已安排的 token 位置、当前请求进度、异步返回的正式输出 |
| 输出 | 用于后续调度的进度和占位，以及结果到达后的结算状态 |
| 不展开 | batch queue 的入队与收取、GPU 前向、推测 token 的验证、跨 rank 协调 |

## 2. 为什么需要输出占位

异步调度允许在真实 token 尚未返回调度侧时形成后续计划。因此，不能只用已经收到的输出 token 数判断后续需求，还需要记录预期返回的位置。

| 状态 | 表示什么 |
| --- | --- |
| num_computed_tokens | 调度侧已经安排到的位置 |
| num_in_flight_tokens | 已安排但尚未处理返回结果的计算量 |
| num_output_placeholders | 为尚未返回的输出保留的位置数 |

占位表示预期输出，不等于已收到正式 token，也不表示对应 GPU 计算已完成。

## 3. 本地源码入口

路径相对于 `04-vLLM-Source/vllm/`。

- `vllm/config/scheduler.py` 的 get_scheduler_cls 按配置选择基础 Scheduler、AsyncScheduler 或自定义实现。
- `vllm/v1/core/sched/async_scheduler.py` 继承基础调度算法，主要扩展 _update_after_schedule 和 _update_request_with_output。
- 调度后增加相应占位；结果回来后结算占位与在途工作。抢占时还需区分重置前的在途结果，避免错误修改新进度。

这些是后续源码分析要验证的状态变化；本页不把基础调度或 batch queue 的实现再复制一遍。

## 4. 与其他优化分开

batch queue 保存计划与 Future，异步调度维护请求进度与输出占位。推测解码则增加 draft token 的验证与接受/拒绝规则；即使配置同时启用，也应独立成篇。

[章节目录](../README.md)
