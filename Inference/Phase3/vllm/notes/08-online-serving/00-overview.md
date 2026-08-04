# Online Serving

## 学习目标

本阶段把离线 `LLM.generate()` 主线扩展为在线异步服务，理解 HTTP 请求如何经过 OpenAI 协议、Renderer、`AsyncLLM` 和 Engine Core，再以流式或非流式响应返回客户端。

## 阅读顺序

| 顺序 | 笔记 | 主要内容 |
| --- | --- | --- |
| 1 | `01-server-startup.md` | CLI、FastAPI 和 Engine Client 启动 |
| 2 | `02-openai-serving.md` | Chat/Completion 协议与响应生成 |
| 3 | `03-async-engine.md` | 异步请求提交、输出 handler 和取消 |
| 4 | `04-input-rendering.md` | Chat template、tokenization 和 EngineCoreRequest |

## 主调用链

```text
vllm serve
→ ServeSubcommand
→ API Server / FastAPI
→ OpenAI API Router
→ OpenAIServingChat
→ OnlineRenderer
→ AsyncLLM.generate()
→ EngineCoreClient
→ EngineCore
→ RequestOutput stream
→ SSE / JSON response
```

## 完成标准

- 能区分 FastAPI server、OpenAI serving object、Renderer 和 AsyncLLM 的职责。
- 能说明 streaming response 如何与引擎输出异步并行。
- 能追踪 chat messages 变成 token IDs 和 `EngineCoreRequest`。
- 能说明客户端断开时请求如何 abort。

## 当前结论

在线服务不是在离线 API 外简单包一层 HTTP，而是引入协议转换、异步请求流、持续输出 handler、取消语义和跨进程 Engine Client。
