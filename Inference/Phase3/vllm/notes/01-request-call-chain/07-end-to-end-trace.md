# End-to-End Request Trace

## 1. 追踪目标

本文把前面各文件中的局部调用链串成一次完整的离线生成过程。追踪入口是：

```python
llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(prompts, sampling_params)
```

重点回答四个问题：引擎如何创建、prompt 如何进入核心引擎、谁反复驱动模型执行，以及结果如何恢复为用户可见的 `RequestOutput`。

## 2. 阶段一：初始化引擎

```text
LLM.__init__()
→ 整理并规范化用户参数
→ EngineArgs(...)
→ LLMEngine.from_engine_args()
→ EngineArgs.create_engine_config()
→ VllmConfig
→ Executor.get_class(vllm_config)
→ LLMEngine.__init__()
   ├─ Renderer
   ├─ InputProcessor
   ├─ OutputProcessor
   └─ EngineCoreClient.make_client()
      ├─ InprocClient → EngineCore
      └─ SyncMPClient → EngineCoreProc → EngineCore
```

初始化结束后，前端 `LLMEngine` 已经具备三类能力：把公开输入转换成核心请求、与 `EngineCore` 通信，以及把核心输出转换为用户输出。模型执行资源位于 `EngineCore` 下游的 Executor，而不是 `LLM` 对象本身。

## 3. 阶段二：提交请求

### 3.1 进入离线编排层

```text
LLM.generate(prompts, sampling_params)
→ 校验 runner_type == "generate"
→ 选择显式或默认 SamplingParams
→ OfflineInferenceMixin._run_completion()
→ OfflineInferenceMixin._add_completion_requests()
```

`_add_completion_requests()` 将单值或批量输入统一成等长序列，并逐个渲染 prompt。此时用户输入从 `PromptType` 变成 `EngineInput`。

### 3.2 进入前端引擎

```text
OfflineInferenceMixin._add_request()
→ 生成 request_id
→ LLMEngine.add_request()
→ InputProcessor.process_inputs()
→ EngineCoreRequest
→ OutputProcessor.add_request()
→ EngineCoreClient.add_request()
```

`OutputProcessor.add_request()` 先建立前端 `RequestState`，这样核心输出返回时能够找到 tokenizer、采样参数、父子请求关系和已生成内容。随后 `EngineCoreRequest` 才通过客户端边界发送给核心引擎。

当 `SamplingParams.n > 1` 时，`LLMEngine.add_request()` 会建立一个父请求并扇出多个子请求。核心引擎调度的是子请求，前端输出处理器负责把多个候选重新组合到同一个 `RequestOutput` 中。

### 3.3 进入核心引擎

进程内模式的路径为：

```text
InprocClient.add_request(engine_core_request)
→ EngineCore.preprocess_add_request()
→ Request.from_engine_core_request()
→ EngineCore.add_request()
→ Scheduler.add_request()
→ waiting queue
```

多进程模式中，请求会先经过 IPC 输入通道到达 `EngineCoreProc`，但进入 `EngineCore` 后的数据转换和调度器入口相同。`EngineCoreClient` 的价值就是让 `LLMEngine` 无需区分这两条传输路径。

## 4. 阶段三：同步驱动执行

所有请求提交后，控制流回到 `OfflineInferenceMixin._run_engine()`：

```text
while LLMEngine.has_unfinished_requests():
    LLMEngine.step()
```

一次 step 的向下调用链是：

```text
LLMEngine.step()
→ EngineCoreClient.get_output()
→ EngineCore.step()
→ Scheduler.schedule()
→ SchedulerOutput
→ ModelExecutor.execute_model()
→ ModelOutput
→ Scheduler.update_from_output()
→ EngineCoreOutputs
```

`Scheduler.schedule()` 决定本轮运行哪些请求以及为每个请求处理多少 token；`ModelExecutor.execute_model()` 执行实际模型计算；`Scheduler.update_from_output()` 保存生成 token、完成状态和 KV cache 状态。一次 `EngineCore.step()` 只推进一轮，而不是直接完成整批生成。

进程内客户端可以在 `get_output()` 内直接驱动 `EngineCore`。多进程客户端则从输出通道接收后台 `EngineCoreProc` 产生的结果。两种模式最终都向 `LLMEngine` 返回 `EngineCoreOutputs`。

## 5. 阶段四：输出回传

```text
EngineCoreOutputs
→ LLMEngine.step()
→ OutputProcessor.process_outputs()
→ 更新 RequestState / detokenize / stop 判断
→ RequestOutput 或 PoolingRequestOutput
→ OfflineInferenceMixin._run_engine()
```

`OutputProcessor` 把 token ID 增量写入前端请求状态，完成解码、停止条件判断和候选合并。若前端 stop 条件使请求结束，`LLMEngine` 还会通知核心引擎中止相应请求，避免继续计算。

`_run_engine()` 只收集 `finished=True` 的最终输出。由于短请求可能先完成，收集顺序不一定等于提交顺序；循环结束后按数字 request ID 排序，最终 `LLM.generate()` 返回与输入 prompt 顺序一致的 `list[RequestOutput]`。

## 6. 数据对象的完整变化

```text
用户输入
PromptType + SamplingParams
        │ Renderer / OfflineInferenceMixin
        ▼
EngineInput
        │ InputProcessor
        ▼
EngineCoreRequest
        │ EngineCore.preprocess_add_request()
        ▼
Request
        │ Scheduler 持有并更新
        ▼
SchedulerOutput → ModelOutput
        │ Scheduler.update_from_output()
        ▼
EngineCoreOutputs
        │ OutputProcessor
        ▼
RequestOutput
```

这些对象不是同一请求的重复命名，而是不同边界所需的数据表示：公开 API 关注易用性，核心请求关注可调度字段，Scheduler 的 `Request` 关注运行时状态，最终输出则重新组织为用户接口。

## 7. 控制权与状态归属

| 层次 | 控制权 / 状态 |
| --- | --- |
| `LLM` | 公开 API、默认参数和入口校验 |
| `OfflineInferenceMixin` | 批量提交顺序、同步 step 循环、最终结果排序 |
| `LLMEngine` | 前端请求状态、输入转换、输出处理 |
| `EngineCoreClient` | 进程内或跨进程通信方式 |
| `EngineCore` | 核心 step 边界、Scheduler 与 Executor 协调 |
| `Scheduler` | waiting/running/finished 状态和 KV cache 使用 |
| `ModelExecutor` | 模型前向及设备侧执行 |

关键区别是：`OfflineInferenceMixin` 驱动循环，`EngineCore` 执行一轮，`Scheduler` 决定这一轮做什么，`ModelExecutor` 完成计算。

## 8. 推荐调试断点

若要用调试器验证这条调用链，可按以下顺序设置断点：

1. `LLM.generate()`：确认公开输入和采样参数。
2. `OfflineInferenceMixin._add_request()`：观察 request ID 与 `EngineInput`。
3. `LLMEngine.add_request()`：观察 `EngineCoreRequest` 的形成。
4. `EngineCore.preprocess_add_request()`：观察核心 `Request`。
5. `EngineCore.step()`：观察一次调度和模型执行边界。
6. `LLMEngine.step()`：观察 `EngineCoreOutputs` 如何进入输出处理器。
7. `OfflineInferenceMixin._run_engine()`：观察完成顺序与最终排序。

多进程模式下断点会跨越 `EngineCoreProc`；第一次跟踪建议关闭 V1 多进程模式，先走 `InprocClient` 路径，以便在单进程中看清数据变化。

## 9. 一句话总结

一次 `LLM.generate()` 会先由 `OfflineInferenceMixin` 把 prompt 渲染并提交给 `LLMEngine`，再经 `EngineCoreClient` 进入 `EngineCore`；随后 Mixin 同步循环调用 step，由 Scheduler 和 ModelExecutor 逐轮推进，最后由 `OutputProcessor` 组装结果并按输入顺序返回。
