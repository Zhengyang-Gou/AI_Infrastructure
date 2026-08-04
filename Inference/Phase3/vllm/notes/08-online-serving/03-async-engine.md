# Async Engine

## 1. 文件定位

- 路径：`vllm/v1/engine/async_llm.py`、`core_client.py`。
- 职责：提供异步生成接口，把请求映射为独立输出流，并持续从 Engine Core 接收结果。

## 2. 主执行流程

```text
AsyncLLM.generate()
→ add_request()
→ 为 request 创建 AsyncStream
→ EngineCoreClient.add_request()
→ 后台 output handler 接收 EngineCoreOutputs
→ OutputProcessor
→ 对应 AsyncStream.put(RequestOutput)
→ API coroutine async for 消费
```

输出 handler 与 HTTP 请求 coroutine 分离，使多个请求可以共享同一 Engine Core 输出通道。

## 3. 输入与输出

- 输入：prompt、SamplingParams、request ID、priority、LoRA 和 trace headers。
- 输出：异步 `RequestOutput` 流。
- 状态：维护 request-to-stream 映射、后台 task、暂停和错误状态。

## 4. 关键代码解析

### `AsyncLLM.from_engine_args()`

### `AsyncLLM.add_request()`

### `AsyncLLM.generate()`

### `AsyncLLM._run_output_handler()`

### `AsyncLLM.abort()`

### `AsyncLLM.shutdown()`

## 5. 与其他文件的关系

- 上游：OpenAI serving、gRPC 或自定义异步 frontend。
- 下游：异步 MP Client 与 Engine Core process。
- 输出：复用 `OutputProcessor`、Detokenizer 和 RequestOutput。

## 6. 当前结论

`AsyncLLM` 用后台输出循环把共享 Engine Core 通道拆成按请求隔离的异步输出流。
