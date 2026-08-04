# Output Processor

## 1. 学习目标

理解前端如何维护每个请求的输出状态，将 `EngineCoreOutput` 中的 token IDs、logprobs、结束信息和 pooling 结果转换成同步或异步调用方可以消费的 `RequestOutput`。

## 2. 文件定位

- 文件路径：`vllm/v1/engine/output_processor.py`
- 所属层次：Frontend Engine 输出处理层
- 核心职责：维护前端请求状态、执行增量 detokenize、更新 logprobs、组装输出对象并处理完成或中止。
- 在调用链中的位置：位于 `EngineCoreOutput` 与 `LLM.generate()` / AsyncLLM 请求流之间。

## 3. 核心类与组件

| 类 / 组件 | 作用 |
| --- | --- |
| `RequestOutputCollector` | 为异步请求暂存输出；DELTA 模式下可合并生产者积压的增量 |
| `OutputProcessorOutput` | 同步返回本轮 `request_outputs` 与需要通知核心中止的请求 IDs |
| `RequestState` | 保存 prompt、detokenizer、logprobs、输出模式、父子请求关系和统计状态 |
| `OutputProcessor` | 注册请求并批量处理 `EngineCoreOutput` |
| `IncrementalDetokenizer` | 将新增 token IDs 转成文本并检查 stop string |
| `LogprobsProcessor` | 累积并整理 sample / prompt logprobs |
| `ParentRequest` | 在 `n > 1` 时汇总同一外部请求的多个子序列 |

## 4. 主执行流程

### 注册前端请求状态

```text
OutputProcessor.add_request()
→ RequestState.from_new_request()
→ 创建 LogprobsProcessor
→ 创建 IncrementalDetokenizer
→ 保存 internal request ID 与 external request ID 映射
→ 保存可选 ParentRequest
```

### 处理核心输出

```text
OutputProcessor.process_outputs(engine_core_outputs)
→ 找到对应 RequestState
→ 更新本轮统计
→ detokenizer.update(new_token_ids)
→ 检查 stop string
→ logprobs_processor.update_from_output()
→ RequestState.make_request_output()
→ CompletionOutput / PoolingOutput
→ RequestOutput / PoolingRequestOutput
→ 同步加入返回列表，或写入异步 collector
→ 完成时清理前端 RequestState
```

如果 Detokenizer 检测到 stop string，而 `EngineCoreOutput` 尚未标记核心请求完成，处理器会把 internal request ID 放入 `reqs_to_abort`，由上层通知 Engine Core 停止后续执行。

### 输出模式

- `DELTA`：只返回自上次输出后的新增文本、token 与 logprobs。
- `CUMULATIVE`：返回当前累计结果。
- `FINAL_ONLY`：中间步骤不产生用户输出，仅在结束时返回完整结果。

## 5. 输入与输出

### 输入

- `EngineCoreRequest`：注册请求时用于创建前端 `RequestState`。
- `list[EngineCoreOutput]`：每轮新增 token、finish reason、logprobs 与其他核心结果。
- 可选的时间戳和迭代统计对象。

### 输出

- `OutputProcessorOutput.request_outputs`：同步 `LLMEngine` 路径消费的用户输出列表。
- `OutputProcessorOutput.reqs_to_abort`：前端检测到 stop string 后需核心中止的请求 IDs。
- 异步路径通过 `RequestOutputCollector` 将输出交给每个 generate task。

### 状态变化

- `request_states` 新增、更新或删除请求状态。
- `external_req_ids` 维护外部 ID 到内部 ID 列表的映射。
- Detokenizer 和 LogprobsProcessor 累积 token、文本与 logprobs。
- 并行采样时，`ParentRequest` 汇总多个 child outputs。

## 6. 关键代码解析

### `RequestOutputCollector.put()`

### `RequestOutputCollector.get()`

### `RequestState.from_new_request()`

### `RequestState.make_request_output()`

### `RequestState._new_request_output()`

### `RequestState._new_completion_output()`

### `OutputProcessor.add_request()`

### `OutputProcessor.process_outputs()`

### `OutputProcessor.abort_requests()`

### `OutputProcessor._finish_request()`

## 7. 与其他文件的关系

- 上游：`vllm/v1/engine/llm_engine.py` 调用 `add_request()` 与 `process_outputs()`。
- 核心输入结构：`vllm/v1/engine/__init__.py` 中的 `EngineCoreRequest` 和 `EngineCoreOutput`。
- 文本转换：`vllm/v1/engine/detokenizer.py`。
- logprobs：`vllm/v1/engine/logprobs.py`。
- 并行采样：`vllm/v1/engine/parallel_sampling.py`。
- 最终对象：`vllm/outputs.py` 中的 `CompletionOutput`、`RequestOutput` 与 pooling 输出类型。

## 8. 当前结论

`OutputProcessor` 是核心 token 输出到用户 API 的转换中心。它同时维护文本、logprobs、流式模式、父子候选关系和请求生命周期，并在字符串停止条件与核心状态不一致时负责发出 abort 请求。
