# Scheduler

## 1. 文件定位

- 文件路径：`vllm/v1/core/sched/scheduler.py`
- 所属层次：核心调度层
- 核心职责：维护请求队列，在资源约束下构造每轮 `SchedulerOutput`，并用 `ModelRunnerOutput` 更新和结束请求。
- 在调用链中的位置：由 `EngineCore` 调用，位于请求状态、KV Cache Manager 和模型执行层之间。

当前 Scheduler 将 Prefill、Chunked Prefill、Decode 和 Speculative Decode 统一成 token 进度问题，而不是为它们建立互斥的执行阶段。

## 2. 核心类与组件

| 类 / 组件 | 作用 |
| --- | --- |
| `Scheduler` | 实现核心调度和结果更新逻辑 |
| `requests` | `request_id → Request` 的完整请求索引 |
| `waiting` | 可以参加接纳调度的请求队列 |
| `skipped_waiting` | 因依赖或约束暂时跳过的等待请求队列 |
| `running` | 已占用执行槽或 KV Cache 的请求列表 |
| `KVCacheManager` | 查询 Prefix Cache、分配 slots 和释放 blocks |
| `EncoderCacheManager` | 管理多模态或 Encoder-Decoder 的 Encoder Cache |
| `SchedulerOutput` | 本轮发送给模型执行层的计划 |

### 主要约束

| 约束 | 来源 | 作用 |
| --- | --- | --- |
| `max_num_scheduled_tokens` | `SchedulerConfig` | 限制一轮所有请求的 token 总数 |
| `max_num_running_reqs` | `max_num_seqs` | 限制同时处于运行集合的请求数 |
| `max_model_len` | `ModelConfig` | 限制单请求上下文长度 |
| KV Cache 可用 blocks | `KVCacheManager` | 决定请求是否能获得所需 slots |
| Encoder compute budget | 多模态配置 | 限制本轮 Encoder 输入计算量 |

## 3. 主执行流程

### 添加请求

```text
Scheduler.add_request()
→ 检查 request_id 是否已经存在
→ 新请求写入 requests
→ 普通请求加入 waiting
→ 被阻塞请求加入 skipped_waiting
→ 流式会话的重复 request_id 作为后续输入处理
```

### 调度 running 请求

```text
初始化 token_budget
→ 遍历 running
→ num_new_tokens = 未追平的 token 数
→ 应用长 Prefill 阈值、总 budget 和 max_model_len
→ 处理 Encoder 输入约束
→ KVCacheManager.allocate_slots()
→ 分配成功：记录请求及本轮 token 数
→ 分配失败：抢占最低调度优先级的 running 请求
```

FCFS 策略下通常从 running 尾部抢占；PRIORITY 策略会根据 `priority` 和 `arrival_time` 选择最低优先级请求。抢占会释放请求 blocks、把状态改为 `PREEMPTED`、把 `num_computed_tokens` 重置为 0，并把请求放回 waiting 队列头部。

### 调度 waiting 请求

```text
仅在本轮没有发生抢占时继续
→ 检查 max_num_seqs、LoRA 和依赖状态
→ 首次调度时查询本地 Prefix Cache 命中
→ 计算剩余 Prefill / Decode token 数
→ 应用 chunked prefill 与 token budget
→ KVCacheManager.allocate_slots()
→ waiting / PREEMPTED → RUNNING
→ 记录 NewRequestData 或恢复请求数据
```

### 构造调度输出

```text
scheduled_new_reqs
+ scheduled_running_reqs
+ scheduled_resumed_reqs
+ num_scheduled_tokens
+ KV Cache / Encoder / Connector 元数据
→ SchedulerOutput
→ _update_after_schedule()
→ 返回 EngineCore
```

`_update_after_schedule()` 在 GPU 结果返回前乐观增加 `num_computed_tokens` 和 `num_in_flight_tokens`，这样异步或流水线调度可以继续向前推进；若 speculative token 被拒绝，后续会在 `update_from_output()` 中回滚相应计数。

### 更新模型输出

```text
Scheduler.update_from_output()
→ 取得 sampled_token_ids、logprobs 和 pooling 输出
→ 减少 num_in_flight_tokens
→ 修正 speculative decoding 被拒绝的 token 数
→ _update_request_with_output()
→ Request.append_output_token_ids()
→ check_stop()
→ 完成请求从 running 移除并释放资源
→ 构造 EngineCoreOutput
→ 按 client_index 汇总 EngineCoreOutputs
```

## 4. 输入与输出

### 输入

- 初始化输入：`VllmConfig`、`KVCacheConfig`、结构化输出管理器和 block size。
- 请求输入：由 Engine Core 传入的 `Request`。
- 调度输入：waiting / running 状态、token budget、KV Cache 容量和其他资源约束。
- 执行输入：本轮 `SchedulerOutput` 对应的 `ModelRunnerOutput`。

### 输出

- `schedule()` 返回 `SchedulerOutput`。
- `update_from_output()` 返回 `dict[int, EngineCoreOutputs]`，按请求的 `client_index` 分组。
- `finish_requests()` 返回本次真正结束的 `Request` 列表。
- 统计接口返回请求数、KV Cache 使用率和 Scheduler 指标。

### 状态变化

- 新请求写入 `requests` 并进入 waiting 或 skipped_waiting。
- 接纳成功后进入 running，状态变为 `RUNNING`。
- 抢占请求释放 blocks、状态变为 `PREEMPTED` 并重新等待。
- 调度后乐观推进 `num_computed_tokens`；结果返回后处理回滚、追加 token 和停止条件。
- 完成请求从队列与 `requests` 中移除，其 KV Cache 和 Encoder Cache 被释放。

## 5. 关键代码解析

### `Scheduler.__init__()`

### `Scheduler.add_request()`

### `Scheduler.schedule()`

### `Scheduler._preempt_request()`

### `Scheduler._update_after_schedule()`

### `Scheduler._make_cached_request_data()`

### `Scheduler.update_from_output()`

### `Scheduler._update_request_with_output()`

### `Scheduler.finish_requests()`

### `Scheduler._free_request()`

### `Scheduler._free_request_blocks()`

### `Scheduler.get_num_unfinished_requests()`

### `Scheduler.has_requests()`

## 6. 与其他文件的关系

- 上游：`vllm/v1/engine/core.py` 调用 `add_request()`、`schedule()` 和 `update_from_output()`。
- 请求状态：`vllm/v1/request.py`。
- 调度输出：`vllm/v1/core/sched/output.py`。
- KV Cache：`vllm/v1/core/kv_cache_manager.py`。
- 停止判断：`vllm/v1/core/sched/utils.py` 中的 `check_stop()`。
- 下游：Executor 和 Model Runner 执行 `SchedulerOutput`，并返回 `ModelRunnerOutput`。
- 前端输出：本文件构造 `EngineCoreOutput`，再由 Engine Core 交回前端引擎。

## 7. 当前结论

`Scheduler` 是每轮推理的资源决策中心：它优先推进 running 请求，再接纳 waiting 请求，并把 token budget、请求数限制和 KV Cache 容量共同转化成可执行的 `SchedulerOutput`。
