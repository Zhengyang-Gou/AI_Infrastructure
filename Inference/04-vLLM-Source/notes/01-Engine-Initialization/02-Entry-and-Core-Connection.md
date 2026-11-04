# 02 · 建立入口与核心之间的连接

这一篇接着引擎工厂往下走：LLMEngine 建立输入输出组件，创建核心客户端，再通过客户端进入 EngineCore。

## 1. 先认识两个名称相近的对象

| 对象 | 负责什么 | 后续会在哪里用到 |
| --- | --- | --- |
| LLMEngine | 将入口输入、核心客户端和输出处理连接起来 | 提交请求、取回核心输出并继续处理 |
| EngineCore | 组织调度、缓存管理和模型执行 | 核心一轮轮执行时使用 |
| EngineCoreClient | 让 LLMEngine 能联系 EngineCore | 提交、取输出、取消、查询任务等 |

`LLMEngine.engine_core` 保存 **InprocClient**，客户端的 `engine_core` 属性再保存真正的 EngineCore。沿这两个属性就能从入口找到核心。

## 2. 创建入口侧的三个组件

源码定位：[v1/engine/llm_engine.py](../../vllm/vllm/v1/engine/llm_engine.py) · `LLMEngine.__init__`。

```python
# 文件：vllm/v1/engine/llm_engine.py
# 类：LLMEngine
# 函数：__init__
# 源码范围：L91–L111（连续节选）
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
```

这段代码先建立三个长期复用的组件，再建立核心客户端：

| 组件 | 初始化时准备什么 | 请求到来后做什么 |
| --- | --- | --- |
| Renderer | 根据模型配置建立输入处理方式，取得 tokenizer | 将文本等输入整理为统一输入表示 |
| InputProcessor | 保存运行配置和 renderer 的引用 | 检查请求参数与长度，构造 EngineCoreRequest |
| OutputProcessor | 保存 tokenizer、输出间隔及统计相关设置 | 为各条请求保存输出状态，解码并组织用户结果 |

tokenizer 负责文本和 token ID 之间的转换。Renderer 处理输入形式，InputProcessor 检查请求，OutputProcessor 处理返回结果；三者会在第二章和第八章遇到真实请求时展开。

OutputProcessor 在引擎初始化时建立，逐请求的 RequestState 则在提交请求时注册。

## 3. 核心客户端怎样选择

源码定位：[v1/engine/core_client.py](../../vllm/vllm/v1/engine/core_client.py) · `EngineCoreClient.make_client`。

`make_client` 根据调用方式创建客户端，返回的 InprocClient 随后建立核心：

```python
# 文件：vllm/v1/engine/core_client.py
# 类：EngineCoreClient
# 函数：make_client
# 源码范围：L104–L112（连续节选）
if multiprocess_mode and asyncio_mode:
    return EngineCoreClient.make_async_mp_client(
        vllm_config, executor_class, log_stats
    )

if multiprocess_mode and not asyncio_mode:
    return SyncMPClient(vllm_config, executor_class, log_stats)

return InprocClient(vllm_config, executor_class, log_stats)
```

`asyncio_mode` 描述入口怎样收发核心消息；`async_scheduling` 描述核心怎样安排尚在执行中的工作。这两个名称都带 async，但控制的是不同事情。

## 4. 一行构造继续进入整个执行体系

InprocClient 的初始化很短：

```python
# 文件：vllm/v1/engine/core_client.py
# 类：InprocClient
# 函数：__init__
# 源码范围：L316–L317（连续节选）
def __init__(self, *args, **kwargs):
    self.engine_core = EngineCore(*args, **kwargs)
```

`EngineCore(*args, **kwargs)` 是一次真正的构造调用。只有核心内部的模型、缓存和调度器等初始化都完成，这一行才会返回。

调用在这里一层层进入：

```text
LLM.__init__ 尚未结束
└─ LLMEngine.from_engine_args 尚未结束
   └─ LLMEngine.__init__ 尚未结束
      └─ EngineCoreClient.make_client
         └─ InprocClient.__init__
            └─ EngineCore.__init__
               ├─ Executor / Worker / 模型加载（第 03、04 篇）
               ├─ KV 准备与预热（第 05、06 篇）
               └─ Scheduler 与核心收尾（第 07 篇）
```

所以，后面几篇是在进入更深层的构造函数。全部完成后，代码会沿这个结构逐层返回，最后才回到用户的 `LLM(...)` 调用之后。

## 5. 对象之间怎样连接

创建完成后，各对象通过属性引用连接起来：

```text
用户进程
LLM
└─ llm_engine：LLMEngine
   ├─ renderer
   ├─ input_processor
   ├─ output_processor
   └─ engine_core：InprocClient
      └─ engine_core：EngineCore
         ├─ model_executor：UniProcExecutor
         │  └─ driver_worker：WorkerWrapperBase
         │     └─ worker：GPU Worker
         │        └─ model_runner：GPUModelRunner
         │           └─ model：加载权重后的模型
         └─ scheduler：Scheduler
            └─ kv_cache_manager：管理块的使用情况
```

这里的层级表示“谁保存了谁的引用”。下一篇沿 WorkerWrapperBase 继续找到实际 Worker。

## 6. 建立这些组件后得到什么

入口已经有输入和输出处理能力，并且正在建立可联系的核心。后面一批请求会共用这些组件，而每条请求各自拥有独立状态。

例如以后一次 generate 提交 A、B、C 三条输入，会重复调用同一个 InputProcessor，向同一个核心提交，再由同一个 OutputProcessor 根据不同请求 ID 分别处理输出。组件复用和请求状态独立可以同时成立。

构造函数中的日志与缓存收尾，会在第 07 篇回到 LLMEngine 时接着说明。

阅读检查：

- LLMEngine.engine_core 的真实类型是什么？
- 入口侧已经有 OutputProcessor，为什么还不能说请求的输出状态已经建立？
- 调用 InprocClient 构造函数时，哪几个外层构造函数仍在等待返回？
- 怎样沿属性引用从 LLMEngine 找到实际模型？

[章节目录](README.md) · [上一篇：从 LLM 构造到运行配置](01-LLM-and-Engine-Configuration.md) · [下一篇：启动 Executor、Worker 与执行环境](03-Executor-Worker-and-Device.md)
