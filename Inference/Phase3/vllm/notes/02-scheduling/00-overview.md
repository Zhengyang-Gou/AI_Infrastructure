# Scheduling

## 学习目标

这一阶段关注 `Scheduler` 如何维护请求状态，并在每一轮中决定“执行哪些请求、每个请求执行多少 token”。

完成本阶段后，应该能够说明：

1. `Request` 保存了哪些调度状态和 token 进度。
2. 新请求如何从 waiting 队列进入 running 队列。
3. `token_budget` 如何限制一轮调度的总 token 数。
4. Prefill、Chunked Prefill 和 Decode 如何统一为“追赶未计算 token”的问题。
5. KV Cache 空间不足时如何触发抢占。
6. `SchedulerOutput` 如何描述本轮需要交给 Model Runner 的工作。
7. 模型输出如何写回请求，并触发停止、完成和资源释放。

本阶段只理解 Scheduler 对 KV Cache 的调用边界，不深入 block 分配和 Prefix Cache 的内部实现。

## 阅读顺序

| 顺序 | 文件 | 主要关注点 |
| --- | --- | --- |
| 1 | `vllm/v1/request.py` | 请求状态、token 进度以及完成状态如何保存 |
| 2 | `vllm/v1/core/sched/output.py` | 一轮调度结果如何表示并传给 Model Runner |
| 3 | `vllm/v1/core/sched/scheduler.py` | running、waiting、token budget、抢占和结果更新 |

## 整体结构

```text
Request
   保存请求输入、状态、已生成 token 和已计算 token 数
        ↓
Scheduler
   维护 waiting / skipped_waiting / running
   计算 token budget 并申请 KV Cache slots
        ↓
SchedulerOutput
   描述本轮新请求、缓存请求、token 数和 block IDs
        ↓
Model Runner
   执行模型并返回 ModelRunnerOutput
        ↓
Scheduler.update_from_output()
   写回 token、检查停止条件并释放完成请求
```

## 请求进入调度器

```text
EngineCore.add_request()
→ Scheduler.add_request()
→ requests[request_id] = Request
→ waiting 或 skipped_waiting
```

普通新请求进入 `waiting`。暂时被结构化输出语法、远端 KV 或流式输入阻塞的请求进入 `skipped_waiting`，等待依赖满足后再参加调度。

## 一轮调度主流程

```text
Scheduler.schedule()
→ 初始化 token_budget
→ 优先遍历 running 请求
→ 计算本轮 num_new_tokens
→ KVCacheManager.allocate_slots()
→ 空间不足时抢占低优先级请求
→ 在没有发生抢占时接纳 waiting 请求
→ 查询 Prefix Cache 命中并申请 slots
→ 更新 waiting / running / PREEMPTED 状态
→ 构造 SchedulerOutput
→ _update_after_schedule() 乐观推进计算进度
```

当前实现没有独立的 Prefill 队列和 Decode 队列。二者都由同一个差值统一描述：

```text
待调度 token 数
=
request.num_tokens_with_spec
+ request.num_output_placeholders
- request.num_computed_tokens
```

随后还会受到总 token budget、最大模型长度、长 Prefill 阈值、Encoder budget 和 KV Cache 可用空间等约束。

## 请求状态变化

```text
WAITING
   ↓ 首次被接纳
RUNNING
   ↓ KV Cache 不足，被抢占
PREEMPTED
   ↓ 重新进入 waiting 并再次被接纳
RUNNING
   ↓ stop / length / abort / error
FINISHED_*
```

部分请求还可能暂时处于：

- `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`：等待结构化输出语法准备完成。
- `WAITING_FOR_REMOTE_KVS`：等待远端 KV Cache 加载完成。
- `WAITING_FOR_STREAMING_REQ`：等待流式会话的下一段输入。

## 调度输出的数据边界

| 数据结构 | 作用 |
| --- | --- |
| `NewRequestData` | 向 Worker 发送第一次执行所需的完整请求数据 |
| `CachedRequestData` | 对 Worker 已缓存的请求只发送增量数据 |
| `SchedulerOutput` | 汇总本轮请求、token 数、block IDs、完成请求和附加元数据 |
| `ModelRunnerOutput` | 返回采样 token、logprobs、pooling 结果等模型执行结果 |
| `EngineCoreOutputs` | Scheduler 更新状态后返回给 Engine Core 前端的结果 |

## 模型输出回写流程

```text
ModelRunnerOutput
→ Scheduler.update_from_output()
→ 根据 speculative decoding 接受结果修正 num_computed_tokens
→ 将 sampled token 追加到 Request
→ check_stop()
→ 完成请求从 running 移除
→ 释放请求占用的 KV Cache blocks
→ 生成 EngineCoreOutput
→ 按 client_index 汇总为 dict[int, EngineCoreOutputs]
```

## 跨文件调用表

| 调用方 | 被调用方 | 作用 |
| --- | --- | --- |
| `EngineCore` | `Scheduler.add_request()` | 把内部 `Request` 加入调度器 |
| `Scheduler.add_request()` | waiting 队列 | 保存尚未运行的请求 |
| `EngineCore.step()` | `Scheduler.schedule()` | 生成本轮执行计划 |
| `Scheduler.schedule()` | `KVCacheManager.allocate_slots()` | 为本轮 token 获取 KV slots |
| `Scheduler.schedule()` | `NewRequestData.from_request()` | 构造首次发送给 Worker 的请求数据 |
| `Scheduler.schedule()` | `SchedulerOutput(...)` | 汇总本轮调度结果 |
| `ModelExecutor` | Model Runner | 执行 `SchedulerOutput` 描述的工作 |
| `EngineCore` | `Scheduler.update_from_output()` | 用模型结果更新请求状态 |
| `Scheduler.update_from_output()` | `Request.append_output_token_ids()` | 保存新生成的 token |
| `Scheduler` | KV Cache Manager | 请求结束或抢占时释放 blocks |

## 当前阶段检查清单

- [ ] 能说明 `num_computed_tokens` 与 `num_tokens_with_spec` 的区别。
- [ ] 能解释为什么当前 Scheduler 不需要显式区分 Prefill 和 Decode 队列。
- [ ] 能画出 waiting、running、preempted 和 finished 的状态变化。
- [ ] 能找到 running 请求和 waiting 请求各自被调度的位置。
- [ ] 能说明 `token_budget`、`max_num_seqs` 和 KV Cache 空间分别限制什么。
- [ ] 能区分 `NewRequestData` 与 `CachedRequestData`。
- [ ] 能说明 `SchedulerOutput` 如何进入 Model Runner。
- [ ] 能说明 sampled token 如何写回 `Request` 并触发资源释放。

## 一句话总结

`Scheduler` 在 token、请求数和 KV Cache 容量约束下选择本轮工作，用 `SchedulerOutput` 把执行计划交给 Model Runner，再根据模型结果推进或结束请求。
