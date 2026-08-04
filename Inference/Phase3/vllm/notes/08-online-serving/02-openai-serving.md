# OpenAI Serving

## 1. 文件定位

- 路径：`vllm/entrypoints/openai/chat_completion/`、`completion/`、`responses/`。
- 职责：校验 OpenAI 协议，请求渲染，调用 AsyncLLM，并生成兼容的流式或完整响应。

## 2. 核心对象

| 对象 | 作用 |
| --- | --- |
| Protocol models | 定义请求和响应的 Pydantic schema |
| API Router | 把 HTTP endpoint 绑定到 serving object |
| `OpenAIServingChat` | 处理 Chat Completions 请求 |
| `OnlineRenderer` | 将 chat/completion 请求转换为引擎输入 |
| Stream Generator | 把增量 `RequestOutput` 转换成 SSE chunk |

## 3. 主执行流程

```text
ChatCompletionRequest
→ API Router
→ render_chat_request()
→ SamplingParams / EngineInput
→ AsyncLLM.generate()
→ AsyncIterator[RequestOutput]
→ stream generator 或 full generator
→ OpenAI response
```

## 4. 输入与输出

- 输入：messages、tools、response format、sampling 参数和 stream 标志。
- 输出：ChatCompletionResponse 或 Server-Sent Events。
- 状态：streaming 过程中保存 role、工具调用、reasoning 和 usage 增量。

## 5. 关键代码解析

### `attach_router()`

### `OpenAIServingChat.render_chat_request()`

### `OpenAIServingChat.create_chat_completion()`

### `OpenAIServingChat.chat_completion_stream_generator()`

### `OpenAIServingChat.chat_completion_full_generator()`

## 6. 当前结论

OpenAI serving 层只处理协议和响应语义，真正的 token 生成仍由 `AsyncLLM` 与 Engine Core 完成。
