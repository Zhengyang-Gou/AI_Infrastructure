# 05 · LLMEngine 与核心边界

本篇将前面几篇连接起来：请求经校验后，如何注册输出状态并交给核心；核心返回结果后，入口又如何推进输出处理。

## 1. 职责与边界

**本篇负责：** LLMEngine 的组件组装、输入处理/状态登记/提交的调用顺序、取输出与取消转交。

**终点：** 调用到 processor/client 接口为止；内部 ID 算法归第 04 篇，输出状态与解码实现归第 06 篇，客户端内部归 02 章。

`LLMEngine` 是入口侧的协调组件，持有 renderer、输入处理器、输出处理器和核心客户端。

```text
add_request：输入校验与构造 → 内部 ID → 注册输出状态 → 提交核心

step：取得核心输出 → 处理输出 → 反馈取消 → 返回用户侧结果
```

这里的 `self.engine_core` 实际上是 `EngineCoreClient` 的某个实现对象。它的名字不能被直接理解为“当前进程中的 EngineCore 实例”；核心可以位于后台进程。

## 2. 源码定位与关键数据

| 文件 | 类 | 重点函数或结构 |
| --- | --- | --- |
| [v1/engine/llm_engine.py](../../../vllm/vllm/v1/engine/llm_engine.py) | `LLMEngine` | `__init__`、`add_request`、`step`、`has_unfinished_requests`、`abort_request` |
| [v1/engine/core_client.py](../../../vllm/vllm/v1/engine/core_client.py) | `EngineCoreClient` | 提交、取输出和取消的接口契约；不展开工厂 |
| [v1/engine/__init__.py](../../../vllm/vllm/v1/engine/__init__.py) | `EngineCoreRequest`、`EngineCoreOutput`、`EngineCoreOutputs` | 跨边界的请求、单请求增量结果与结果批次 |
| [v1/engine/parallel_sampling.py](../../../vllm/vllm/v1/engine/parallel_sampling.py) | `ParentRequest` | `n>1` 时的子请求与结果聚合 |

| 数据 | 生产方与消费方 |
| --- | --- |
| `EngineCoreRequest` | InputProcessor 构造，经客户端交给核心 |
| `RequestState` | OutputProcessor 在提交前建立，供多轮输出处理复用 |
| `EngineCoreOutput` | 核心产生的一条请求增量，通常带内部 ID、新 Token 与结束信息 |
| `EngineCoreOutputs` | 客户端取得的一批核心输出，还包含时间戳和统计等信息 |
| `OutputProcessorOutput` | 包含用户结果列表及需要向核心取消的内部 ID |
| `RequestOutput` | 入口返回给用户的对象，使用外部请求 ID |

## 3. 实现分析

### 3.1 初始化时组装入口侧组件

```python
# 文件：vllm/v1/engine/llm_engine.py
# 类：LLMEngine
# 函数：__init__
# 源码范围：L91–L113（连续节选）
self.renderer = renderer = renderer_from_config(self.vllm_config)

# Convert EngineInput --> EngineCoreRequest.
self.input_processor = InputProcessor(self.vllm_config, renderer)

# Converts EngineCoreOutputs --> RequestOutput.
self.output_processor = OutputProcessor(
    renderer.tokenizer,
    log_stats=self.log_stats,
    stream_interval=self.vllm_config.scheduler_config.stream_interval,
    tracing_enabled=tracing_endpoint is not None,
)

# EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
self.engine_core = EngineCoreClient.make_client(
    multiprocess_mode=multiprocess_mode,
    asyncio_mode=False,
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=self.log_stats,
)

self.logger_manager: StatLoggerManager | None = None
```

renderer 与输入处理器共享配置；输出处理器使用对应 tokenizer；客户端根据执行模式建立核心边界。`LLM` 随后复用这里创建的部分组件引用。

### 3.2 add_request 调用输入处理

普通 `EngineInput` 路径中的关键调用是：

```python
# 文件：vllm/v1/engine/llm_engine.py
# 类：LLMEngine
# 函数：add_request
# 源码范围：L250–L261（连续节选）
request = self.input_processor.process_inputs(
    request_id,
    prompt,
    params,
    supported_tasks=self.get_supported_tasks(),
    arrival_time=arrival_time,
    lora_request=lora_request,
    tokenization_kwargs=tokenization_kwargs,
    trace_headers=trace_headers,
    priority=priority,
)
prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)
```

`InputProcessor` 构造执行请求，同时 `extract_prompt_components` 提取可用原文，供输出状态保存。直接传入 `EngineCoreRequest` 的兼容分支仍存在，但这不是本章主线。

### 3.3 先注册输出状态，再提交

```python
# 文件：vllm/v1/engine/llm_engine.py
# 类：LLMEngine
# 函数：add_request
# 源码范围：L263–L277（连续节选）
self.input_processor.assign_request_id(request)

req_id = request.request_id

# Use cloned params that may have been updated in process_inputs()
params = request.params

n = params.n if isinstance(params, SamplingParams) else 1

if n == 1:
    # Make a new RequestState and queue.
    self.output_processor.add_request(request, prompt_text, None, 0)
    # Add the request to EngineCore.
    self.engine_core.add_request(request)
    return req_id
```

这段顺序很关键：

1. 调用 processor 的静态方法，保留外部 ID 并生成内部 ID。
2. 使用请求中已经克隆和补全的参数，而非继续使用原始参数对象。
3. 为请求建立输出状态，记录输入、解码器和输出模式等上下文。
4. 将请求交给客户端，进入核心执行。

核心返回的是增量信息，输出侧需要先有状态才能解释这些信息。即使后台执行很快，注册与提交之间的顺序也保持一致。

`add_request` 返回内部 ID，供入口提交过程记录和取消；最终用户结果会恢复外部 ID。`n>1` 的分支则通过 `ParentRequest` 拆出子请求，逐个注册并提交，输出时再合并回一条输入请求的候选列表。

### 3.4 step 组织输出处理与取消反馈

```python
# 文件：vllm/v1/engine/llm_engine.py
# 类：LLMEngine
# 函数：step
# 源码范围：L296–L334（连续节选）
def step(self) -> list[RequestOutput | PoolingRequestOutput]:
    if self.should_execute_dummy_batch:
        self.should_execute_dummy_batch = False
        self.engine_core.execute_dummy_batch()
        return []

    # 1) Get EngineCoreOutput from the EngineCore.
    with record_function_or_nullcontext("llm_engine step: get_output"):
        outputs = self.engine_core.get_output()

    # 2) Process EngineCoreOutputs.
    with record_function_or_nullcontext("llm_engine step: process_outputs"):
        iteration_stats = IterationStats() if self.log_stats else None
        processed_outputs = self.output_processor.process_outputs(
            outputs.outputs,
            engine_core_timestamp=outputs.timestamp,
            iteration_stats=iteration_stats,
        )
        self.output_processor.update_scheduler_stats(outputs.scheduler_stats)

    # 3) Abort any reqs that finished due to stop strings.
    with record_function_or_nullcontext("llm_engine step: abort_requests"):
        self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

    # 4) Record stats
    with record_function_or_nullcontext("llm_engine step: record_stats"):
        if (
            self.logger_manager is not None
            and outputs.scheduler_stats is not None
            and len(outputs.outputs) > 0
        ):
            self.logger_manager.record(
                scheduler_stats=outputs.scheduler_stats,
                iteration_stats=iteration_stats,
                mm_cache_stats=self.renderer.stat_mm_cache(),
            )
            self.do_log_stats_with_interval()

    return processed_outputs.request_outputs
```

普通文本、无特殊数据并行空批次的主线可以读成：

```text
get_output：取得 EngineCoreOutputs
    → process_outputs：用请求状态处理每条增量
    → abort_requests：取消输出侧已经判停的请求
    → 返回 processed_outputs.request_outputs
```

一次 `LLMEngine.step` 是入口侧的一轮交互。它可能包含多个请求的结果，也可能没有用户可见结果，不与“某个请求生成一个 Token”严格一一对应。

`has_unfinished_requests` 主要读取输出处理器的未完成状态，并结合数据并行相关执行状态判断是否继续。离线入口因此无需直接查看 Scheduler 的队列。

### 3.5 客户端接口契约

入口把 EngineCoreClient 当作提交与结果接口，不读取其内部队列或调用核心步骤。

| 调用 | 输入 | 入口取得什么 |
| --- | --- | --- |
| add_request | EngineCoreRequest | 提交完成；后续结果通过 get_output 获取 |
| get_output | 无 | 一批 EngineCoreOutputs，可能没有用户可见结果 |
| abort_requests | 内部请求 ID 列表 | 向核心发出取消指令 |

客户端选择、同进程调用和后台消息循环的实现唯一归属 02。本篇到这三个接口为止。

### 3.6 取消也需要跨越输入输出两侧

```python
# 文件：vllm/v1/engine/llm_engine.py
# 类：LLMEngine
# 函数：abort_request
# 源码范围：L212–L216（连续节选）
def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
    """Remove request_ids from EngineCore and Detokenizer."""

    request_ids = self.output_processor.abort_requests(request_ids, internal)
    self.engine_core.abort_requests(request_ids)
```

用户使用外部 ID 发起取消时，输出处理器先找到相应内部请求，清理入口状态并返回内部 ID，再由客户端通知核心。提交过程中回收已添加请求时，可以直接指定 `internal=True`。

入口输出状态清理与核心资源清理是协作的两个动作；核心如何移出请求、释放 KV 引用，继续由核心和缓存管理章节解释。

## 4. 手工推演：从提交到完成

```text
提交外部请求 "0"
    → 构造 EngineCoreRequest
    → 生成内部 ID，保留 external_req_id="0"
    → 建立 request_states[内部 ID]
    → 客户端提交核心

第一个 step：取得增量 Token
    → 输出处理器累积，FINAL_ONLY 尚未产生用户结果

后续 step：核心结束，或输出侧检测到停止字符串
    → 构造 RequestOutput，恢复 request_id="0"
    → 清理输出状态；必要时反馈核心取消
    → 离线循环收集最终结果
```

## 5. 设计原因与自测

LLMEngine 将请求构造、输出状态与客户端组合成入口侧闭环。客户端隔离核心执行方式，输出处理器隔离增量结果的组织方式，使离线编排逻辑能复用这些能力。

读完后应能解释：为什么状态注册放在核心提交之前？`self.engine_core` 是什么对象？get_output 的返回值为什么还不是用户结果？停止字符串为什么需要反馈取消？

[章节目录](../README.md) · [上一篇：InputProcessor 与请求构造](04-InputProcessor-and-EngineCoreRequest.md) · [下一篇：输出处理与文本还原](06-OutputProcessor-and-Detokenization.md)
