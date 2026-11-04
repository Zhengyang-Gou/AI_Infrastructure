# 01 · LLM 与初始化

本篇从用户创建 `LLM` 开始，跟踪配置如何交给引擎，以及 `generate` 如何进入离线推理流程。

**阅读场景：** 本地源码、decoder-only 文本生成、`n=1`。先掌握可复用引擎的创建与公共接口，再沿后续笔记深入每次请求的处理。

## 1. 职责与边界

`LLM` 是用户持有的离线推理入口。它接收模型和运行配置，创建 `LLMEngine`，保存组件引用，提供 `generate`、`chat`、`enqueue` 等公共接口。

```text
LLM(...)：准备可重复使用的引擎
    → EngineArgs → LLMEngine.from_engine_args
    → 保存配置、renderer、计数器等入口状态

LLM.generate(...)：提交这一次生成工作
    → 检查任务，补齐默认采样参数
    → OfflineInferenceMixin._run_completion
```

初始化配置描述模型与运行资源；请求参数描述本次希望怎样生成。模型前向、调度和缓存分配继续由引擎下游负责。

## 2. 源码定位与关键数据

下文 `# 文件` 路径均相对于 `04-vLLM-Source/vllm/`，代码块是本地源码的连续节选；函数所在文件、类和名称在块首标明。

| 文件 | 类 | 重点函数 |
| --- | --- | --- |
| [entrypoints/llm.py](../../../vllm/vllm/entrypoints/llm.py) | `LLM` | `__init__`、`from_engine_args`、`get_default_sampling_params`、`generate`、`chat`、`enqueue`、`wait_for_completion` |
| [v1/engine/llm_engine.py](../../../vllm/vllm/v1/engine/llm_engine.py) | `LLMEngine` | `from_engine_args` |

| 数据 | 生命周期与用途 |
| --- | --- |
| `EngineArgs` | 构造入口时汇集用户配置，交给引擎工厂 |
| `VllmConfig` | 引擎工厂创建的统一运行配置，供下游组件读取 |
| `self.llm_engine` | 跨多次生成调用复用的引擎对象 |
| `self.renderer`、`self.input_processor` | 引用引擎所创建的组件，保持配置和处理方式一致 |
| `self.request_counter` | 为离线请求分配递增的外部请求 ID |
| `self.default_sampling_params` | 惰性缓存默认生成配置的差异字典 |
| `SamplingParams` | 每次生成使用的温度、长度、停止条件等参数 |

## 3. 实现分析

### 3.1 初始化：从用户配置到引擎与组件引用

`__init__` 先归一化配置，再构造 `EngineArgs`。例如，字典配置转换成对应配置类，整数编译模式转换成 `CompilationConfig`；还会处理默认统计开关、自定义 worker 类的序列化以及数据并行入口的约束。

这些准备完成后，入口创建引擎并保存运行状态：

```python
# 文件：vllm/entrypoints/llm.py
# 类：LLM
# 函数：__init__
# 源码范围：L339–L356（连续节选）
self.llm_engine = LLMEngine.from_engine_args(
    engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
)
self.model_config = self.llm_engine.model_config
self.engine_class = type(self.llm_engine)

self.request_counter = Counter()
self.default_sampling_params: dict[str, Any] | None = None

supported_tasks = self.llm_engine.get_supported_tasks()
self.supported_tasks = supported_tasks

self.runner_type = self.model_config.runner_type
self.renderer = self.llm_engine.renderer
self.chat_template = load_chat_template(chat_template)
self.input_processor = self.llm_engine.input_processor

self.renderer.warmup(ChatParams(chat_template=self.chat_template))
```

这段代码建立了三种状态：

1. **可执行的引擎**：通过工厂进入配置生成、核心和执行资源初始化。
2. **入口自己的状态**：请求计数器和默认参数缓存。
3. **共享组件引用**：renderer 与输入处理器来自同一个 `LLMEngine`。

`renderer.warmup` 是输入处理侧的预热。完整的模型加载与 GPU 预热继续发生在引擎下游，不能将两者视为同一件事。构造函数后面还初始化其他 mixin 和辅助状态，以上是生成主线所需的节选。

### 3.2 两个同名工厂承担不同工作

用户入口的工厂将已有参数展开，再调用 `LLM.__init__`：

```python
# 文件：vllm/entrypoints/llm.py
# 类：LLM
# 函数：from_engine_args
# 源码范围：L378–L381（连续节选）
@classmethod
def from_engine_args(cls, engine_args: EngineArgs) -> "LLM":
    """Create an LLM instance from EngineArgs."""
    return cls(**vars(engine_args))
```

引擎工厂则将参数转为统一配置，选择执行器，再构造 `LLMEngine`：

```python
# 文件：vllm/v1/engine/llm_engine.py
# 类：LLMEngine
# 函数：from_engine_args
# 源码范围：L160–L186（连续节选）
@classmethod
def from_engine_args(
    cls,
    engine_args: EngineArgs,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    stat_loggers: list[StatLoggerFactory] | None = None,
    enable_multiprocessing: bool = False,
) -> "LLMEngine":
    """Creates an LLM engine from the engine arguments."""

    # Create the engine configs.
    vllm_config = engine_args.create_engine_config(usage_context)
    executor_class = Executor.get_class(vllm_config)

    if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
        logger.debug("Enabling multiprocessing for LLMEngine.")
        enable_multiprocessing = True

    # Create the LLMEngine.
    return cls(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=not engine_args.disable_log_stats,
        usage_context=usage_context,
        stat_loggers=stat_loggers,
        multiprocess_mode=enable_multiprocessing,
    )
```

因此，`LLM.from_engine_args` 仍然走用户入口的初始化；`LLMEngine.from_engine_args` 才决定如何创建引擎。运行模式受这里的环境开关与参数共同影响，单张 GPU 并不意味着入口和核心必然在同一进程。

### 3.3 默认参数：缓存字典，每次创建参数对象

```python
# 文件：vllm/entrypoints/llm.py
# 类：LLM
# 函数：get_default_sampling_params
# 源码范围：L407–L412（连续节选）
def get_default_sampling_params(self) -> SamplingParams:
    if self.default_sampling_params is None:
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
    if self.default_sampling_params:
        return SamplingParams.from_optional(**self.default_sampling_params)
    return SamplingParams()
```

第一次调用时，从模型配置读取默认生成配置的差异字典；后续复用这个字典，但每次返回新的 `SamplingParams`。默认值来源还受 `generation_config` 等配置影响。

需要区分三步：`LLM` 在未传参数时创建默认对象；mixin 在离线提交时将输出模式设为 `FINAL_ONLY`；`InputProcessor` 再克隆和补全执行侧参数。后两步分别见第 02、04 篇。

### 3.4 generate：检查任务，委托流程编排

下面省略函数签名和 docstring，保留实际函数体：

```python
# 文件：vllm/entrypoints/llm.py
# 类：LLM
# 函数：generate
# 源码范围：L457–L477（连续节选）
runner_type = self.model_config.runner_type
if runner_type != "generate":
    raise ValueError(
        "LLM.generate() is only supported for generative models. "
        "Try passing `--runner generate` to use the model as a "
        "generative model."
    )

if sampling_params is None:
    sampling_params = self.get_default_sampling_params()

return self._run_completion(
    prompts=prompts,
    params=sampling_params,
    output_type=RequestOutput,
    use_tqdm=use_tqdm,
    lora_request=lora_request,
    tokenization_kwargs=tokenization_kwargs,
    priority=priority,
    mm_processor_kwargs=mm_processor_kwargs,
)
```

它没有在这里逐 Token 执行模型，而是将工作交给继承的 `_run_completion`。`LLM` 同时使用继承和组合：mixin 方法运行在同一个 `LLM` 实例上，`self.llm_engine` 则引用另一个组件对象。

这次调用最终返回 `list[RequestOutput]`。外层列表对应输入请求，`RequestOutput.outputs` 中的候选结果才是 `CompletionOutput`；`n>1` 时，一条输入可以有多个候选。

### 3.5 chat、enqueue 与等待接口

| 接口 | 本地实现的委托路径 | 用户侧含义 |
| --- | --- | --- |
| `generate` | `_run_completion` | 准备普通 prompts，提交并等待结果 |
| `chat` | `_run_chat` | 先处理消息和聊天模板，再提交并等待结果 |
| `enqueue` | `_add_completion_requests` | 提交普通 prompts，返回请求 ID |
| `wait_for_completion` | `_run_engine` | 收集当前引擎内所有未完成请求 |

本地 `chat` 函数体直接走 `_run_chat`，虽然它的 docstring 还描述了调用 `generate` 的过程，阅读应以实际函数体为准。

等待接口的实现如下：

```python
# 文件：vllm/entrypoints/llm.py
# 类：LLM
# 函数：wait_for_completion
# 源码范围：L539–L561（连续节选）
def wait_for_completion(
    self,
    output_type: type[Any] | tuple[type[Any], ...] | None = None,
    *,
    use_tqdm: bool | Callable[..., tqdm] = True,
) -> list[Any]:
    """Wait for all enqueued requests to complete and return results.

    This method processes all requests currently in the engine queue
    and returns their outputs. Use after enqueue() to get results.

    Args:
        output_type: The expected output type(s). If not provided, accepts
            both RequestOutput and PoolingRequestOutput.
        use_tqdm: If True, shows a tqdm progress bar.

    Returns:
        A list of output objects for all completed requests.
    """
    if output_type is None:
        output_type = (RequestOutput, PoolingRequestOutput)

    return self._run_engine(output_type, use_tqdm=use_tqdm)
```

它没有按请求 ID 选择等待范围；多次 `enqueue` 后，当前未完成请求会一起被收集。多进程模式下，核心可能在提交后就开始执行，因此 `enqueue` 分开的是提交与等待两个入口动作。

## 4. 手工推演：复用一个引擎处理两批请求

```text
创建 LLM
    → 配置与引擎初始化一次，计数器准备好

第一次 generate([A, B])
    → 为 A、B 分配外部 ID
    → 提交、收集完成结果，返回与输入对应的列表

第二次 generate([C])
    → 复用已有引擎和组件
    → 继续使用请求计数器，创建 C 的请求状态
```

每次请求使用新的输入与输出状态，入口持有的引擎和组件则跨调用复用。权重和请求状态的生命周期由此分开。

## 5. 设计原因与自测

统一配置工厂让入口适配不同执行方式；mixin 复用离线流程；组件引用让接口与具体处理职责保持清晰。同步 `generate` 描述的是用户等待结果的方式，内部仍通过多轮增量执行完成生成。

读完后应能解释：为什么初始化不放在每次 `generate` 内？两个 `from_engine_args` 有何区别？默认参数为什么缓存字典而每次创建对象？

[章节目录](../README.md) · [整体框架](../Overview.md) · [下一篇：离线推理流程编排](02-Offline-Inference-Flow.md)
