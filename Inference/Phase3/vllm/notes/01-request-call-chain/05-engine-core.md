# EngineCore

## 1. 文件定位

- 文件路径：`vllm/v1/engine/core.py`
- 所属层次：核心引擎层
- 核心职责：持有 Scheduler 和 ModelExecutor，接收内部请求，并完成每一轮调度、模型执行与状态更新。
- 在调用链中的位置：请求进入真正推理执行流程的核心边界。

本阶段只关注 `EngineCore` 的输入、输出和主循环。Scheduler、KV Cache、GPU 执行等内部机制分别留到后续阶段深入。

## 2. 核心类与组件

| 类 / 组件 | 作用 | 本阶段关注点 |
| --- | --- | --- |
| `EngineCore` | 核心推理循环 | 请求如何进入、一次 step 如何执行 |
| `Scheduler` | 决定本轮执行哪些请求和 token | 只关注调用边界 |
| `ModelExecutor` | 在执行设备上运行模型 | 只关注输入输出边界 |
| `StructuredOutputManager` | 管理结构化输出约束 | 只识别其初始化位置 |
| KV Cache 相关组件 | 初始化和管理 KV Cache | 留到 KV Cache 阶段 |
| `EngineCoreProc` | 在独立进程中运行 `EngineCore` | 理解多进程入口即可 |

## 3. 主执行流程

### 请求进入核心引擎

```text
EngineCoreRequest
→ preprocess_add_request()
→ Request.from_engine_core_request()
→ EngineCore.add_request()
→ Scheduler.add_request()
```

### 执行一轮推理

```text
EngineCore.step()
→ Scheduler.schedule()
→ SchedulerOutput
→ ModelExecutor.execute_model()
→ ModelOutput
→ Scheduler.update_from_output()
→ EngineCoreOutputs
```

如果 Scheduler 中没有请求，`step()` 会直接返回空输出。如果存在请求，它会完成一次调度和模型执行，然后用模型结果更新请求状态。

## 4. 输入与输出

### 输入

- `EngineCoreRequest`：从前端引擎收到、尚未转换的核心请求。
- `Request`：由 `preprocess_add_request()` 构造、供 Scheduler 使用的内部请求。
- `VllmConfig` 与 `executor_class`：初始化 Scheduler、KV Cache 和执行器所需配置。

### 输出

`step()` 返回按客户端或数据并行 rank 组织的 `EngineCoreOutputs`，以及本轮是否真正执行模型的标志。

### 状态变化

- 新请求被加入 Scheduler 的 waiting 队列。
- Scheduler 在每轮执行中改变 waiting、running 和 finished 状态。
- KV block、生成 token、完成状态等随模型结果更新。
- 已完成请求通过 `EngineCoreOutputs` 返回前端。

## 5. 关键代码解析

### `EngineCore.__init__()`

### `EngineCore.preprocess_add_request()`

### `EngineCore.add_request()`

### `EngineCore.step()`

### `EngineCore.post_step()`

### `EngineCoreProc.__init__()`

## 6. 与其他文件的关系

- 上游：`EngineCoreClient`。
- 下游：Scheduler、ModelExecutor、KV Cache 管理器和结构化输出管理器。
- 接收的数据：`EngineCoreRequest`，随后转换为 `Request`。
- 返回的数据：`EngineCoreOutputs`。

## 7. 当前结论

`EngineCore` 是 vLLM V1 的内部主循环。它把请求交给 Scheduler，按照调度结果调用 ModelExecutor，再把模型输出交回 Scheduler 更新状态，最终产生前端可消费的核心输出。

## 8. 后续阶段再研究

- Scheduler 如何计算 token budget 和处理抢占。
- KV Cache block 如何申请、复用和释放。
- `SchedulerOutput` 如何转换成 GPU Tensor。
- ModelExecutor 如何调用 GPU Worker 与 Model Runner。
- Sampling 结果如何更新请求并形成最终输出。
