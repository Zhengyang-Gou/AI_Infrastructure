# 01 · Batch Queue 与多批在途

## 1. 职责与边界

本篇负责多批次在途的计划/Future 配对与收取时机，并区分 batch queue 和异步调度。基础同步 step 的调用契约已给定；本篇不重讲预算选择、KV 算法或模型前向。

输入：执行计划和异步下发返回的 Future。输出：与原计划配对的模型结果，或“尚无可返回结果”。

## 2. 启用条件

核心读取 `VllmConfig.max_concurrent_batches`；大于 1 时建立 batch queue，step_fn 选择 step_with_batch_queue。PP 与异步调度会影响并发批次数，具体拓扑归 10，本篇只使用最终配置值。

## 3. 计划和结果一一配对

`step_with_batch_queue()` 使用相同的调度、执行和更新接口，同时允许多个批次在途。关键是把 Future 与产生它的调度计划放在一起：

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：step_with_batch_queue
# 源码范围：L679–L687（连续节选）
if not deferred_scheduler_output:
    # Add this step's future to the queue.
    batch_queue.appendleft((future, scheduler_output, exec_future))
    if len(batch_queue) < self.batch_queue_size and (
        model_executed or self.scheduler.has_requests()
    ):
        # Don't block on next worker response unless the queue is full
        # or there are no more requests to schedule.
        return None, model_executed
```

`future` 用于取得最终的 ModelRunnerOutput，`scheduler_output` 对应本轮计划，`exec_future` 用于保留模型执行的完成或异常信息。收取较早批次时，从队列取回这一组，再调用 `update_from_output()`。

队列还能接纳工作时，一次调用可能先下发新批次并返回 `None`，稍后才收取在途结果。调用方会继续检查 Scheduler 工作和 batch queue，因此暂时没有输出不等于所有请求结束。

## 4. 与异步调度的接口边界

batch queue 接收已经形成的计划，保存计划与 Future，决定何时取得并结算结果。AsyncScheduler 提供允许后续轮次继续形成计划的请求进度；本篇不展开它的占位增减与回退。

异步进度说明集中在 [02 · 异步调度与输出占位](02-Async-Scheduling.md)。同进程/多进程只决定请求与结果如何传递，不改变本篇的计划/Future 配对契约。

## 5. 阅读检查

- 为什么 Future 必须和生成它的 SchedulerOutput 一起保存？
- 一次调用返回 None，为什么仍可能有工作在途？
- batch queue、AsyncScheduler 与同进程/多进程各自改变什么？

[章节目录](../README.md)
