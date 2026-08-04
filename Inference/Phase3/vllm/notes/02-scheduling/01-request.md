# Request

## 1. 文件定位

- 文件路径：`vllm/v1/request.py`
- 所属层次：核心请求数据层
- 核心职责：保存 Scheduler 处理一个请求所需的输入、状态、token 进度、Prefix Cache 哈希和完成信息。
- 在调用链中的位置：由 `EngineCoreRequest` 转换而来，随后被 `Scheduler` 持续更新，直到请求结束。

`Request` 是调度器中的可变状态对象。它不仅保存用户输入，还记录当前已经计算多少 token、已经生成哪些 token、是否被抢占以及是否已经完成。

## 2. 核心类与状态

| 类 / 状态 | 作用 |
| --- | --- |
| `Request` | Scheduler 内部维护的完整请求状态 |
| `StreamingUpdate` | 流式会话继续执行时需要追加的轻量数据 |
| `RequestStatus` | 描述等待、运行、抢占和各种完成状态 |

### 核心字段

| 字段 | 作用 |
| --- | --- |
| `request_id` | 请求的唯一标识 |
| `status` | 当前调度状态，初始通常为 `WAITING` |
| `prompt_token_ids` | 文本 prompt 对应的 token IDs |
| `output_token_ids` | 只读视图，保存已经生成的 token IDs |
| `all_token_ids` | prompt 与生成 token 的只读组合视图 |
| `num_prompt_tokens` | prompt 长度 |
| `num_computed_tokens` | Scheduler 认为已经计算或正在执行的 token 数 |
| `spec_token_ids` | speculative decoding 提供的 draft token IDs |
| `num_in_flight_tokens` | 已调度但输出尚未处理的 token 数 |
| `num_output_placeholders` | 异步调度中尚待回填的输出位置数 |
| `max_tokens` | 本请求允许生成的最大 token 数 |
| `block_hashes` | Prefix Cache 查找使用的块哈希链 |
| `num_preemptions` | 请求被 Scheduler 抢占的次数 |
| `stop_reason` | 请求停止时记录的 token ID 或字符串原因 |

## 3. 主执行流程

### 创建请求

```text
EngineCoreRequest
→ Request.from_engine_core_request()
→ Request.__init__()
→ 初始化状态为 WAITING
→ 建立 token 列表和只读视图
→ 计算已有完整块的 block_hashes
→ Scheduler.add_request()
```

### 推进请求

```text
Scheduler.schedule()
→ 读取 num_tokens_with_spec 与 num_computed_tokens
→ 决定本轮 num_scheduled_tokens
→ _update_after_schedule() 增加 num_computed_tokens
→ Model Runner 返回 sampled token
→ append_output_token_ids()
→ 更新 all_token_ids 与 block_hashes
```

### 完成请求

```text
check_stop()
→ 更新 Request.status 为 FINISHED_*
→ Request.is_finished()
→ Scheduler 释放请求资源
→ Request.get_finished_reason()
```

## 4. 输入与输出

### 输入

- `EngineCoreRequest` 中的 prompt token、prompt embedding 和多模态特征。
- `SamplingParams` 或 `PoolingParams`。
- LoRA、优先级、cache salt、trace headers 和流式会话配置。
- Scheduler 每轮写回的计算进度和 Model Runner 生成的 token。

### 输出

`Request` 不产生独立的用户输出。它向 Scheduler 暴露当前 token 数、完成状态、block hashes、事件和停止原因，供 Scheduler 构造调度计划及 `EngineCoreOutput`。

### 状态变化

- 初始状态通常是 `WAITING`；结构化输出请求可能从 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` 开始。
- 被接纳后进入 `RUNNING`。
- 抢占后进入 `PREEMPTED`，`num_computed_tokens` 被重置，随后等待重新调度。
- 停止、长度到限、取消、忽略、错误或重复检测会进入相应的 `FINISHED_*` 状态。
- 每次追加生成 token 后，`output_token_ids`、`all_token_ids` 和可用的 `block_hashes` 同步增长。

## 5. 关键代码解析

### `StreamingUpdate.from_request()`

### `Request.__init__()`

### `Request.from_engine_core_request()`

### `Request.append_output_token_ids()`

### `Request.update_block_hashes()`

### `Request.is_finished()`

### `Request.get_finished_reason()`

### `RequestStatus.is_finished()`

### `RequestStatus.get_finished_reason()`

## 6. 与其他文件的关系

- 上游：`vllm/v1/engine/core.py` 将 `EngineCoreRequest` 转换为 `Request`。
- 调度方：`vllm/v1/core/sched/scheduler.py` 读取并更新请求状态。
- 调度输出：`vllm/v1/core/sched/output.py` 从 `Request` 提取 Worker 所需数据。
- KV Cache：`KVCacheManager` 使用 `request_id`、token 数和 `block_hashes` 查询及分配 blocks。
- 停止判断：`vllm/v1/core/sched/utils.py` 根据请求状态和采样参数更新 `RequestStatus`。

## 7. 当前结论

`Request` 是 Scheduler 的状态中心。Prefill、Decode、抢占和完成并不是独立请求类型，而是同一个 `Request` 在 token 进度和 `RequestStatus` 上的不同状态。
