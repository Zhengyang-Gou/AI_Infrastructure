# Scheduler Output

## 1. 文件定位

- 文件路径：`vllm/v1/core/sched/output.py`
- 所属层次：Scheduler 与 Model Runner 之间的数据边界
- 核心职责：定义一轮调度计划的数据结构，区分首次发送的完整请求和 Worker 已缓存请求的增量更新。
- 在调用链中的位置：由 `Scheduler.schedule()` 构造，经 Executor 传给 Worker / Model Runner。

这个文件只描述“本轮要执行什么”，不负责选择请求，也不保存模型执行后的 sampled token。

## 2. 核心数据结构

| 数据结构 | 作用 |
| --- | --- |
| `NewRequestData` | 携带新请求第一次进入 Worker 所需的完整数据 |
| `CachedRequestData` | 携带已存在于 Worker 中请求的增量变化 |
| `ScheduledEncoderInputStats` | 记录本轮多模态 Encoder 输入统计 |
| `SchedulerOutput` | 汇总本轮完整执行计划 |
| `GrammarOutput` | 保存结构化输出请求 ID 及其 grammar bitmask |

### `NewRequestData` 主要字段

| 字段 | 作用 |
| --- | --- |
| `req_id` | 请求 ID |
| `prompt_token_ids` / `prompt_embeds` | 首次建立 Worker 侧请求状态所需输入 |
| `sampling_params` / `pooling_params` | 生成或池化配置 |
| `block_ids` | 按 KV Cache group 组织的 block table |
| `num_computed_tokens` | 请求进入本轮前的计算进度 |
| `mm_features` | 多模态特征 |
| `lora_request` | 请求使用的 LoRA 信息 |

### `CachedRequestData` 主要字段

| 字段 | 作用 |
| --- | --- |
| `req_ids` | 本轮已缓存请求的 ID 列表 |
| `resumed_req_ids` | 本轮从抢占状态恢复的请求 |
| `new_token_ids` | Pipeline Parallel 场景下需要补发的 token IDs |
| `new_block_ids` | 追加或替换到 Worker block table 的 block IDs |
| `num_computed_tokens` | 各请求进入本轮前的计算位置 |
| `num_output_tokens` | 各请求已有输出及占位 token 数 |

### `SchedulerOutput` 主要字段

| 字段 | 作用 |
| --- | --- |
| `scheduled_new_reqs` | 首次调度请求的完整数据 |
| `scheduled_cached_reqs` | 已缓存请求的增量数据 |
| `num_scheduled_tokens` | `request_id → 本轮 token 数` |
| `total_num_scheduled_tokens` | 本轮所有请求的 token 总数 |
| `scheduled_spec_decode_tokens` | speculative decoding 的 draft token IDs |
| `scheduled_encoder_inputs` | 本轮需要处理的 Encoder 输入索引 |
| `num_common_prefix_blocks` | 各 KV Cache group 的公共前缀 block 数 |
| `finished_req_ids` | 两轮之间已经结束、需要通知 Worker 清理的请求 |
| `preempted_req_ids` | 本轮被抢占的请求 ID |
| `kv_connector_metadata` | KV Connector 本轮传输元数据 |
| `new_block_ids_to_zero` | 使用前需要由 Worker 清零的新 blocks |
| `kv_cache_block_copies` | Prefix Cache partial hit 产生的 CoW block copy |

## 3. 主执行流程

### 新请求数据

```text
Request 第一次被调度
→ NewRequestData.from_request()
→ 携带 prompt、参数、block IDs 和当前计算位置
→ 加入 SchedulerOutput.scheduled_new_reqs
```

### 已缓存请求数据

```text
RUNNING 或 PREEMPTED 后恢复的 Request
→ Scheduler._make_cached_request_data()
→ 只整理增量 token、block IDs 和计数
→ SchedulerOutput.scheduled_cached_reqs
```

### 传递执行计划

```text
Scheduler.schedule()
→ SchedulerOutput
→ ModelExecutor.execute_model()
→ Worker.execute_model()
→ Model Runner 更新 Worker 侧请求与 block table
→ 模型前向和采样
```

## 4. 输入与输出

### 输入

- `Request` 中的 prompt、参数、token 进度和 LoRA 信息。
- `KVCacheManager` 返回的 block IDs。
- Scheduler 本轮计算出的 token 数、抢占集合、完成集合及 Encoder / Connector 元数据。

### 输出

`SchedulerOutput` 是本文件最主要的输出，它被传给 Model Runner。Model Runner 执行后返回的是定义在 `vllm/v1/outputs.py` 中的 `ModelRunnerOutput`，两者不要混淆。

### 状态变化

这些数据类主要是单轮快照，不直接修改 Scheduler 状态。新的实例会在每一轮 `schedule()` 中重新构造；`Scheduler._update_after_schedule()` 才负责推进 `Request` 的内部状态。

## 5. 关键代码解析

### `NewRequestData.from_request()`

### `NewRequestData.anon_repr()`

### `CachedRequestData.is_context_phase()`

### `CachedRequestData.make_empty()`

### `SchedulerOutput.make_empty()`

## 6. 与其他文件的关系

- 上游状态：`vllm/v1/request.py`。
- 构造方：`vllm/v1/core/sched/scheduler.py`。
- 下游消费者：Executor、Worker 和 GPU Model Runner。
- KV Cache 数据来源：`KVCacheManager` 返回的按 group 组织的 block IDs。
- 对应返回类型：`vllm/v1/outputs.py` 中的 `ModelRunnerOutput`。

## 7. 当前结论

`SchedulerOutput` 是 Scheduler 与执行层之间的执行契约：Scheduler 决定工作，输出对象把新请求、增量请求、token 数和 KV Cache block table 一次性描述给 Model Runner。
