# Request Call Chain

## 学习目标

这一阶段只关注一次离线推理请求如何从用户 API 进入 `EngineCore`，以及结果如何返回给用户。

完成本阶段后，应该能够回答：

1. 用户调用 `LLM.generate()` 后，请求经过了哪些对象？
2. 原始 prompt 在哪里被转换为引擎可处理的请求？
3. `LLMEngine` 和 `EngineCore` 为什么需要通过 `EngineCoreClient` 连接？
4. 谁驱动 `EngineCore` 不断执行推理步骤？
5. 引擎输出如何最终变成 `RequestOutput`？

本阶段暂时不深入 Scheduler、KV Cache、模型执行和 Sampling 的内部实现。

## 阅读顺序

| 顺序 | 文件 | 主要关注点 |
| --- | --- | --- |
| 1 | `examples/basic/offline_inference/basic.py` | 用户如何创建 `LLM` 并调用 `generate()` |
| 2 | `vllm/entrypoints/llm.py` | `LLM.generate()` 如何接收参数并启动离线推理 |
| 3 | `vllm/entrypoints/offline_utils.py` | 请求提交和引擎循环之间的桥接逻辑 |
| 4 | `vllm/v1/engine/llm_engine.py` | 输入处理、请求提交和输出处理 |
| 5 | `vllm/v1/engine/core_client.py` | 前端引擎如何与核心引擎通信 |
| 6 | `vllm/v1/engine/core.py` | `EngineCore` 如何执行一次核心推理步骤 |

> `offline_utils.py` 虽然不在最初的五个目标文件中，但实际追踪 `LLM.generate()` 时会经过它，因此需要把它作为连接代码阅读。

## 整体结构

```text
用户代码
   ↓
LLM                         对外提供易用的离线推理 API
   ↓
OfflineInferenceMixin       批量添加请求并驱动推理循环
   ↓
LLMEngine                   处理输入、管理前端请求状态和处理输出
   ↓
EngineCoreClient            隔离进程内与多进程通信方式
   ↓
EngineCore                  调度请求并执行模型
```

## 主调用链

### 1. 提交请求

```text
用户代码
→ LLM.generate()
→ OfflineInferenceMixin._run_completion()
→ OfflineInferenceMixin._add_completion_requests()
→ LLMEngine.add_request()
→ EngineCoreClient.add_request()
→ EngineCore.add_request()
```

这一部分负责把用户传入的 prompt 和采样参数转换成核心引擎能够处理的请求，并把请求加入等待队列。

### 2. 驱动执行并返回结果

```text
OfflineInferenceMixin._run_engine()
→ LLMEngine.step()
→ EngineCoreClient.get_output()
→ EngineCore.step()
→ Scheduler.schedule()
→ ModelExecutor.execute_model()
→ Scheduler.update_from_output()
→ EngineCoreOutputs
→ LLMEngine.output_processor.process_outputs()
→ RequestOutput
→ 用户代码
```

这一阶段先记住边界即可：`EngineCore.step()` 内部包含调度、模型执行和调度状态更新，具体实现留到后续阶段学习。

## 请求对象的变化

```text
PromptType + SamplingParams
          ↓
   EngineCoreRequest
          ↓
       Request
          ↓
  Scheduler 内部状态
```

| 数据形态 | 所在位置 | 作用 |
| --- | --- | --- |
| `PromptType` | `LLM.generate()` | 用户输入的文本、token 或多模态 prompt |
| `SamplingParams` | `LLM.generate()` | 控制温度、最大生成长度等采样行为 |
| `EngineCoreRequest` | `LLMEngine.add_request()` | 经过输入处理、准备发送给核心引擎的请求 |
| `Request` | `EngineCoreClient` / `EngineCore` | 核心引擎和 Scheduler 使用的内部请求 |
| `EngineCoreOutputs` | `EngineCore` → `LLMEngine` | 核心引擎每一步产生的输出 |
| `RequestOutput` | `LLMEngine` → 用户 | 用户最终收到的生成结果 |

## 跨文件调用表

| 调用方 | 被调用方 | 作用 |
| --- | --- | --- |
| 示例代码 | `LLM.generate()` | 发起离线生成请求 |
| `LLM.generate()` | `_run_completion()` | 进入通用离线推理流程 |
| `_run_completion()` | `_add_completion_requests()` | 将一个或多个 prompt 添加到引擎 |
| `_add_completion_requests()` | `LLMEngine.add_request()` | 提交单个请求 |
| `LLMEngine.add_request()` | `EngineCoreClient.add_request()` | 将处理后的请求交给核心引擎 |
| `_run_engine()` | `LLMEngine.step()` | 循环取得推理结果，直到请求全部完成 |
| `LLMEngine.step()` | `EngineCoreClient.get_output()` | 获取核心引擎本轮输出 |
| `EngineCoreClient` | `EngineCore.step()` | 根据客户端实现直接调用或跨进程驱动核心引擎 |
| `EngineCore.step()` | Scheduler 与 ModelExecutor | 完成本轮调度、模型执行和状态更新 |
| `LLMEngine.step()` | `OutputProcessor` | 将核心输出转换为面向用户的结果 |

## `EngineCoreClient` 的意义

`LLMEngine` 不直接依赖某一种 `EngineCore` 部署方式，而是通过客户端抽象进行通信：

- `InprocClient`：`EngineCore` 与前端位于同一进程，可以直接调用。
- `MPClient`：`EngineCore` 位于独立进程，需要通过进程间通信提交请求和接收输出。

因此，无论底层运行方式如何变化，`LLMEngine` 都可以使用相对统一的接口。

## 当前阶段的检查清单

- [ ] 能从示例入口找到 `LLM.generate()`。
- [ ] 能说明 `_run_completion()` 做了什么。
- [ ] 能找到请求进入 `LLMEngine.add_request()` 的位置。
- [ ] 能说明 `EngineCoreClient` 的作用。
- [ ] 能找到 `EngineCore.step()` 的入口。
- [ ] 能区分“提交请求”和“驱动执行”两条路径。
- [ ] 能画出从 prompt 到 `RequestOutput` 的完整调用链。

## 一句话总结

`LLM.generate()` 负责接收用户输入，`LLMEngine` 负责前端请求与输出管理，`EngineCoreClient` 负责连接前端和核心引擎，`EngineCore.step()` 负责驱动真正的调度与模型执行。

## 补充专题

- `06-configuration-and-initialization.md`：补齐 `EngineArgs → VllmConfig → Platform → Engine` 的初始化链路。
