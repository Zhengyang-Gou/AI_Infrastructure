# EngineCoreClient

## 1. 文件定位

- 文件路径：`vllm/v1/engine/core_client.py`
- 所属层次：前端与核心引擎之间的通信层
- 核心职责：为进程内、同步多进程和异步多进程模式提供统一的请求与输出接口。
- 在调用链中的位置：位于 `LLMEngine` 和 `EngineCore` 之间。

这一层把“请求如何送到核心引擎”与“核心引擎做什么”分开，使上层不必了解核心引擎位于当前进程还是后台进程。

## 2. 核心类与接口

| 类 / 接口 | 作用 | 使用场景 |
| --- | --- | --- |
| `EngineCoreClient` | 定义统一客户端接口和客户端工厂 | 所有运行模式 |
| `InprocClient` | 直接持有并调用 `EngineCore` | 同进程同步执行 |
| `MPClient` | 封装多进程通信的公共逻辑 | 多进程客户端基类 |
| `SyncMPClient` | 通过进程间通信同步收发请求和输出 | `LLM` 多进程模式 |
| `AsyncMPClient` | 异步收发核心请求和输出 | 在线异步引擎 |

主要公共接口包括：

| 接口 | 数据方向 | 作用 |
| --- | --- | --- |
| `add_request()` | 前端 → 核心 | 提交 `EngineCoreRequest` |
| `get_output()` | 核心 → 前端 | 获取 `EngineCoreOutputs` |
| `abort_requests()` | 前端 → 核心 | 取消指定请求 |
| `get_supported_tasks()` | 核心 → 前端 | 查询引擎支持的任务 |
| `shutdown()` | 前端 → 核心 | 关闭客户端和核心资源 |

## 3. 主执行流程

### 客户端选择

```text
EngineCoreClient.make_client()
├─ 非多进程                 → InprocClient
├─ 多进程 + 同步            → SyncMPClient
└─ 多进程 + 异步            → AsyncMPClient 或 DP 异步客户端
```

### 进程内模式

```text
LLMEngine
→ InprocClient.add_request()
→ EngineCore.preprocess_add_request()
→ EngineCore.add_request()

LLMEngine.step()
→ InprocClient.get_output()
→ EngineCore.step_fn()
→ EngineCoreOutputs
```

### 同步多进程模式

```text
LLMEngine
→ SyncMPClient.add_request()
→ IPC 输入通道
→ EngineCoreProc

EngineCoreProc
→ IPC 输出通道
→ SyncMPClient.get_output()
→ LLMEngine
```

## 4. 输入与输出

### 输入

客户端接收 `EngineCoreRequest`、请求 ID、控制命令和引擎配置等数据。

### 输出

主要输出是 `EngineCoreOutputs`，此外还包括任务能力查询、缓存控制和 profiling 等管理操作的结果。

### 状态变化

- `InprocClient` 直接改变同进程 `EngineCore` 的状态。
- 多进程客户端将请求写入通信通道，由后台核心进程改变状态。
- 客户端选择在初始化时确定，`LLMEngine` 后续只依赖统一接口。

## 5. 关键代码解析

### `EngineCoreClient.make_client()`

### `InprocClient.__init__()`

### `InprocClient.add_request()`

### `InprocClient.get_output()`

### `SyncMPClient.__init__()`

### `SyncMPClient.add_request()`

### `SyncMPClient.get_output()`

## 6. 与其他文件的关系

- 上游：`LLMEngine`。
- 下游：`EngineCore` 或独立运行的 `EngineCoreProc`。
- 向下传递：`EngineCoreRequest` 和控制命令。
- 向上返回：`EngineCoreOutputs`。

## 7. 当前结论

`EngineCoreClient` 是通信适配层。`LLMEngine` 使用统一接口提交请求和获取输出，而具体调用方式由 `InprocClient`、`SyncMPClient` 或异步客户端决定。
