# Scheduler Output Update

## 1. 学习目标

聚焦 `Scheduler.update_from_output()`，理解 GPU 生成的 token 如何写回 `Request`、触发停止判断、释放完成请求资源，并转换成前端能够接收的 `EngineCoreOutputs`。

## 2. 文件定位

- 文件路径：`vllm/v1/core/sched/scheduler.py`
- 所属层次：Engine Core 的请求状态更新层
- 核心职责：本阶段只关注执行结果回写；调度决策由同文件的 `schedule()` 完成，已属于调度阶段。
- 在调用链中的位置：位于 `ModelRunnerOutput` 与 `EngineCoreOutput` 之间。

## 3. 核心类与组件

| 类 / 组件 | 作用 |
| --- | --- |
| `Scheduler` | 持有所有活跃请求、队列和 KV Cache 管理器，并消费模型执行结果 |
| `SchedulerOutput` | 记录本轮各请求实际调度的 token 数及 speculative token 等信息 |
| `ModelRunnerOutput` | 携带 sampled token IDs、logprobs、pooling 输出与执行侧附加结果 |
| `Request` | 保存输出 token、已计算 token 数、状态、停止原因和客户端索引 |
| `EngineCoreOutput` | 表示单个请求本轮需要交给前端的新结果 |
| `EngineCoreOutputs` | 按前端 client index 聚合多个 `EngineCoreOutput` |
| `check_stop()` | 根据 token 与长度限制更新请求结束状态和停止原因 |

## 4. 主执行流程

### 标准结果回写路径

```text
Scheduler.update_from_output(scheduler_output, model_runner_output)
→ 遍历本轮 num_scheduled_tokens
→ 按 request_id 找到 Request 与采样结果
→ _update_request_with_output()
→ Request.append_output_token_ids()
→ check_stop()
→ 提取 logprobs、prompt logprobs 和 pooling output
→ 构造 EngineCoreOutput
→ 结束请求时 _free_request()
→ 从 running / waiting 队列移除
→ 按 client_index 聚合 EngineCoreOutputs
```

### 单个请求 token 更新

```text
new_token_ids
→ 逐个 append 到 Request.output_token_ids
→ 每加入一个 token 后执行 check_stop()
→ 若停止，裁掉同一轮中停止点之后的 token
→ 返回保留的 token 与 stopped 标记
```

实际函数还处理 speculative decoding、异步调度产生的 stale output、KV Connector、结构化输出和 pooling 等分支。第一轮学习先抓住普通文本生成路径。

## 5. 输入与输出

### 输入

- `scheduler_output`：本轮调度结果，提供每个请求执行的 token 数等上下文。
- `model_runner_output`：提供按请求索引组织的 sampled token IDs、logprobs 和其他模型输出。

### 输出

- 返回 `dict[int, EngineCoreOutputs]`，key 是前端 client index。
- 每个 `EngineCoreOutput` 可包含新 token IDs、finish reason、stop reason、logprobs、事件和缓存统计等。

### 状态变化

- 更新请求的 in-flight / computed token 相关状态。
- 将新 token 追加到 `Request`。
- 触发停止时更新 `RequestStatus` 与 `stop_reason`。
- 完成请求从队列和请求表移除，并释放 KV Cache 等资源。
- 更新 finished request 集合与 Scheduler 统计。

## 6. 关键代码解析

### `Scheduler.update_from_output()`

### `Scheduler._update_request_with_output()`

### `Scheduler._handle_stopped_request()`

### `Scheduler.finish_requests()`

### `Scheduler._free_request()`

### `Scheduler._free_blocks()`

## 7. 与其他文件的关系

- 上游调度信息：`vllm/v1/core/sched/output.py` 中的 `SchedulerOutput`。
- 上游执行结果：`vllm/v1/outputs.py` 中的 `ModelRunnerOutput`。
- 请求状态：`vllm/v1/request.py` 中的 `Request` 与 `RequestStatus`。
- 停止规则：`vllm/v1/core/sched/utils.py` 中的 `check_stop()`。
- 资源释放：`KVCacheManager`、Encoder Cache 与可选 Connector。
- 下游：`EngineCore` 将对应 client 的 `EngineCoreOutputs` 交给前端，随后由 `OutputProcessor` 处理。

## 8. 当前结论

在输出链路中，Scheduler 是 GPU 结果与核心请求状态之间的权威更新点。它接收 sampled tokens，把它们逐个写入 `Request`，完成 token 级停止判断和资源释放，再产生 `EngineCoreOutput`。
