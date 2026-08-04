# Server Startup

## 1. 文件定位

- 路径：`vllm/entrypoints/cli/serve.py`、`vllm/entrypoints/openai/api_server.py`。
- 职责：解析 `vllm serve` 参数，创建 Engine Client、FastAPI app、路由和 HTTP server。

## 2. 主执行流程

```text
vllm serve MODEL
→ ServeSubcommand.cmd()
→ 创建 EngineArgs / ServerArgs
→ setup_server()
→ 启动 AsyncLLM Engine Client
→ build_app()
→ 注册 API routers 与 middleware
→ Uvicorn 接收连接
```

多 API Server 和 DP 模式会创建多个 frontend process，并让它们连接一个或多个 Engine Core。

## 3. 输入与输出

- 输入：CLI 参数、模型配置、网络地址、TLS、认证和插件配置。
- 输出：运行中的 FastAPI/Uvicorn server 与已初始化 Engine Client。
- 状态：启动期间完成 tokenizer、renderer、路由和 metrics 初始化。

## 4. 关键代码解析

### `ServeSubcommand.cmd()`

### `ServeSubcommand.subparser_init()`

### `run_multi_api_server()`

### `setup_server()`

### `build_app()`

### `validate_api_server_args()`

## 5. 与其他文件的关系

- 配置来自 `EngineArgs` 和 Server CLI 参数。
- API routers 调用 Chat、Completion、Responses 等 serving objects。
- Engine Client 通常是 `AsyncLLM`。

## 6. 当前结论

Server startup 负责把配置、异步引擎和 HTTP frontend 装配为可接收请求的服务进程。
