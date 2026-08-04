# LLMEngine

## 1. 文件定位

- 文件路径：`vllm/v1/engine/llm_engine.py`
- 所属层次：前端引擎层
- 核心职责：处理输入、维护前端请求状态、向核心引擎提交请求，并把核心输出转换成用户输出。
- 在调用链中的位置：位于 `LLM` 与 `EngineCoreClient` 之间。

`LLMEngine` 自身不执行模型。它通过 `InputProcessor` 和 `OutputProcessor` 管理核心引擎边界两侧的数据转换。

## 2. 核心类与组件

| 类 / 组件 | 作用 | 输入 | 输出 |
| --- | --- | --- | --- |
| `LLMEngine` | 协调请求输入、核心执行和输出处理 | 用户请求 | `RequestOutput` |
| `Renderer` | 处理 tokenizer 与输入渲染相关能力 | prompt | 引擎输入 |
| `InputProcessor` | 将公开输入转换成核心请求 | `PromptType` / `EngineInput` | `EngineCoreRequest` |
| `EngineCoreClient` | 提交请求并取得核心输出 | `EngineCoreRequest` | `EngineCoreOutputs` |
| `OutputProcessor` | 跟踪前端状态并组装用户输出 | `EngineCoreOutputs` | `RequestOutput` |

## 3. 主执行流程

### 请求提交

```text
LLMEngine.add_request()
→ InputProcessor.process_inputs()
→ EngineCoreRequest
→ OutputProcessor.add_request()
→ EngineCoreClient.add_request()
```

当 `SamplingParams.n > 1` 时，一个父请求会被拆成多个子请求。前端输出处理器需要保存父子关系，最终再将候选结果组织到同一个用户输出中。

### 输出处理

```text
LLMEngine.step()
→ EngineCoreClient.get_output()
→ EngineCoreOutputs
→ OutputProcessor.process_outputs()
→ 终止满足 stop 条件的请求
→ RequestOutput 列表
```

## 4. 输入与输出

### 输入

- `request_id`：前端请求标识。
- `PromptType`、`EngineInput` 或兼容使用的 `EngineCoreRequest`。
- `SamplingParams` 或 `PoolingParams`。
- LoRA、优先级和 tokenization 参数等可选信息。

### 输出

- `add_request()` 返回最终采用的请求 ID。
- `step()` 返回本轮新产生的 `RequestOutput` 或 `PoolingRequestOutput`。

### 状态变化

- `OutputProcessor` 新增并跟踪请求状态。
- `EngineCoreClient` 将请求交给核心引擎。
- 每次 `step()` 后，前端状态根据核心输出、停止字符串和结束请求更新。

## 5. 关键代码解析

### `LLMEngine.__init__()`

### `LLMEngine.from_engine_args()`

### `LLMEngine.has_unfinished_requests()`

### `LLMEngine.add_request()`

### `LLMEngine.step()`

## 6. 与其他文件的关系

- 上游：`LLM` 与 `OfflineInferenceMixin`。
- 下游：`EngineCoreClient`。
- 输入侧依赖：`Renderer` 和 `InputProcessor`。
- 输出侧依赖：`OutputProcessor`。
- 核心边界数据：输入为 `EngineCoreRequest`，输出为 `EngineCoreOutputs`。

## 7. 当前结论

`LLMEngine` 是前端协调器。它把外部输入转换成核心请求，把请求交给 `EngineCoreClient`，再将核心输出转换成用户能够消费的结果。
