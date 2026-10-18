# LLM

## 1. 模块职责

`LLM` 是 vLLM 的离线推理用户入口：接收模型配置，创建并持有推理引擎，对外提供文本生成、对话生成和分步提交等接口。

源码入口：[vllm/entrypoints/llm.py](../../../vllm/vllm/entrypoints/llm.py)。本文以本地源码的生成任务为主线，整体协作流程见 [Overview](../Overview.md)。

`LLM` 将用户调用连接到引擎执行体系。请求如何批量组织、输入如何转换、结果如何逐轮收集，分别由 mixin 和引擎组件承担；调度、KV Cache 管理与模型前向计算继续下沉到 Engine Core 及其下游。

```python
class LLM(BeamSearchOfflineMixin, PoolingOfflineMixin, OfflineInferenceMixin):
    ...
```

这里同时使用了继承与组合：mixin 为 `LLM` 提供方法，`self.llm_engine` 等属性则引用它所调用的组件。`self._run_completion(...)` 是同一个 `LLM` 实例上的继承方法调用。

## 2. 核心数据与接口

### 2.1 初始化配置与运行状态

| 数据或属性 | 作用 |
| --- | --- |
| `EngineArgs` | 汇集模型、分词器、并行方式、显存与编译等初始化参数，交给引擎工厂创建配置和引擎。 |
| `self.llm_engine` | 持有 `LLMEngine`，承接请求提交和结果处理。 |
| `self.model_config` | 引擎的模型配置，供任务类型判断和默认生成参数读取等逻辑使用。 |
| `self.renderer` | 引用引擎创建的 renderer，供继承的预处理方法使用。 |
| `self.input_processor` | 引用引擎的输入处理器；生成主线中由 `LLMEngine.add_request` 调用它。 |
| `self.request_counter` | 为离线请求提供递增计数，供 mixin 分配请求 ID。 |
| `self.default_sampling_params` | 惰性缓存默认采样参数的差异字典，初始为 `None`。 |
| `self.supported_tasks` / `self.runner_type` | 记录引擎支持的任务和模型运行类型。 |
| `self.chat_template` | 保存加载后的对话模板，参与 renderer 预热。 |

`EngineArgs` 是创建引擎时的配置载体；`model_config`、`renderer` 等则是引擎创建后被入口持有或引用的运行状态。构造 `LLM` 会进入引擎初始化流程，包括下游模型与运行资源的准备。

### 2.2 生成接口的输入与输出

`generate` 的核心接口可简写为：

```text
generate(
    prompts: PromptType | Sequence[PromptType],
    sampling_params: SamplingParams | Sequence[SamplingParams] | None,
    ...
) → list[RequestOutput]
```

| 数据 | 用户侧含义 |
| --- | --- |
| `PromptType` | 单条输入，可表达文本、包含 token IDs 的输入及其他受支持的输入形式。 |
| `SamplingParams` | 控制生成行为，例如温度、采样范围、生成长度和停止条件。 |
| `RequestOutput` | 一条输入请求的结果，包含请求标识、输入信息、候选输出和完成状态。 |
| `CompletionOutput` | `RequestOutput.outputs` 中的一条候选，包含生成文本、token IDs 和结束原因等。 |

单个 `SamplingParams` 会用于所有输入；参数序列则与输入逐条对应，长度必须匹配。结果列表按输入顺序返回；每个 `RequestOutput` 内可以包含多条候选，因此外层列表与候选列表具有不同含义。

`use_tqdm` 控制进度展示；`lora_request`、`priority`、`tokenization_kwargs` 和 `mm_processor_kwargs` 等补充请求或输入处理配置，其中优先级需要相应的调度策略支持。

## 3. 重要函数

| 函数 | 核心作用 | 主要下游 |
| --- | --- | --- |
| `__init__` | 归一化配置、创建引擎、建立组件引用并预热 renderer。 | `EngineArgs`、`LLMEngine.from_engine_args`。 |
| `from_engine_args` | 将已有 `EngineArgs` 展开后交给 `LLM` 构造函数。 | `cls(**vars(engine_args))`。 |
| `get_default_sampling_params` | 根据模型生成配置创建默认 `SamplingParams`。 | `model_config.get_diff_sampling_param`。 |
| `generate` | 校验生成任务、补齐默认参数，并运行离线生成流程。 | `_run_completion`。 |
| `chat` | 接收对话消息和模板相关选项，运行对话生成流程。 | `_run_chat`。 |
| `enqueue` / `enqueue_chat` | 完成输入准备与请求提交，返回请求 ID 列表。 | `_add_completion_requests` / `_add_chat_requests`。 |
| `wait_for_completion` | 收集当前引擎中所有未完成请求的最终结果。 | `_run_engine`。 |
| `get_tokenizer` | 取得引擎使用的 tokenizer。 | `llm_engine.get_tokenizer`。 |

此外，`LLM` 还提供缓存重置、性能分析、休眠唤醒和权重更新等管理接口。阅读生成主线时，可先掌握上表中的函数，再按需要展开这些能力。

## 4. 核心实现分析

### 4.1 初始化：将用户配置连接到可用引擎

初始化主线如下：

```text
LLM(...)
  → 归一化配置与检查运行方式
  → EngineArgs(...)
  → LLMEngine.from_engine_args(...)
      → 创建 VllmConfig、选择 Executor 并构造 LLMEngine
  → 保存模型配置与组件引用
  → 初始化请求计数器、默认参数缓存等入口状态
  → renderer.warmup(...)
```

构造函数允许部分配置使用字典或配置对象，也允许 `compilation_config` 使用整数模式。它先将这些形式转换为下游需要的配置对象，再汇集到 `EngineArgs`。例如，内部 `_make_config` 处理 `None`、字典和已有实例，编译配置的整数形式则单独转换为 `CompilationConfig`。

入口还处理少量运行准备工作：默认关闭统计日志；必要时序列化以类形式传入的 `worker_cls`；检查当前单进程入口是否允许指定的数据并行配置。这些工作发生在引擎创建之前，用于使配置适配实际运行方式。

引擎创建及关键引用绑定的源码主干为：

```python
self.llm_engine = LLMEngine.from_engine_args(
    engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
)
self.model_config = self.llm_engine.model_config
# 中间的其他状态初始化省略
self.renderer = self.llm_engine.renderer
self.chat_template = load_chat_template(chat_template)
self.input_processor = self.llm_engine.input_processor

self.renderer.warmup(ChatParams(chat_template=self.chat_template))
```

`Renderer`、`InputProcessor` 和 `OutputProcessor` 在 `LLMEngine` 中组装；入口复用其中的组件引用。这样，预处理流程与引擎侧能够基于一致的模型配置和 renderer 协作。

需要区分两个同名工厂：`LLM.from_engine_args` 将参数展开后构造用户入口；`LLMEngine.from_engine_args` 才负责将参数转换为引擎配置并创建引擎。前者仍然会进入 `LLM.__init__`。

### 4.2 默认参数：缓存配置字典，每次创建参数对象

源码中的实现很短：

```python
def get_default_sampling_params(self) -> SamplingParams:
    if self.default_sampling_params is None:
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
    if self.default_sampling_params:
        return SamplingParams.from_optional(**self.default_sampling_params)
    return SamplingParams()
```

第一次调用时，从模型配置中读取默认生成参数的差异字典；后续调用复用该字典。每次返回时仍构造一个新的 `SamplingParams` 对象，缓存的对象类型是字典。

默认值来源受 `generation_config` 及其覆盖配置影响，可能来自模型的生成配置，也可能使用 vLLM 默认值。`generate` 仅在 `sampling_params is None` 时调用这一方法；用户显式传入参数时，入口将其继续交给下游处理。

### 4.3 `generate`：检查任务后委托离线流程

`generate` 的主体可归纳为三步：

1. 检查 `self.model_config.runner_type` 是否为 `"generate"`，不符合时抛出 `ValueError`。
2. 若未提供 `sampling_params`，调用 `get_default_sampling_params`。
3. 将 prompts、参数和附加选项交给 `_run_completion`，指定 `output_type=RequestOutput`。

后续调用关系为：

```text
LLM.generate
  → OfflineInferenceMixin._run_completion
      ├─ _add_completion_requests：预处理并提交请求
      └─ _run_engine：持续收集结果并恢复输入顺序
```

批量参数对齐、请求 ID 分配、`FINAL_ONLY` 输出模式以及收集循环均由继承方法处理。输入内容和生成参数的进一步校验则由输入处理层承担。`generate` 主要决定调用哪个流程，以及该流程返回什么类型的结果。

`generate` 返回前等待当前引擎中的未完成请求处理结束。输出按请求 ID 排序的逻辑位于 `_run_engine`，因此内部完成顺序可以与输入顺序不同。

### 4.4 `chat`：使用独立的对话预处理路径

本地源码中，`chat` 在任务检查和默认参数处理后，直接调用 `_run_chat`：

```text
generate → _run_completion → _add_completion_requests ─┐
                                                    ├─ 请求提交 → _run_engine
chat     → _run_chat       → _add_chat_requests ──────┘
```

两条路径的差别集中在输入准备：`generate` 接收 prompts，`chat` 接收消息列表，并传递模板、是否添加生成提示等选项。对话路径由 mixin 和 renderer 处理消息及模板，随后复用请求提交与结果收集能力。

阅读时应以当前实现为准：虽然 `chat` 的 docstring 描述了调用 `generate` 的过程，这份源码的实际函数体委托给 `_run_chat`。

### 4.5 `enqueue` 与 `wait_for_completion`：拆开提交和收集

`generate` 将提交与等待合在一次调用中；这两个接口允许用户分开组织它们：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="/path/to/local/model")  # 替换为实际模型路径
params = SamplingParams(temperature=0.8, max_tokens=64)

request_ids = llm.enqueue(["解释 KV Cache 的作用。"], params)
outputs = llm.wait_for_completion(use_tqdm=False)
print(outputs[0].outputs[0].text)
```

`enqueue` 调用 `_add_completion_requests` 后返回请求 ID，不进入 `_run_engine` 等待结果。`wait_for_completion` 调用 `_run_engine`，默认接受 `RequestOutput` 或 `PoolingRequestOutput`，也允许显式指定输出类型。

`wait_for_completion` 没有接收请求 ID 的参数，收集范围是当前引擎的所有未完成请求；若此前调用过多次 `enqueue`，这些请求会一起被处理和收集。

这里分开的是入口侧的提交与结果收集。后台 Engine Core 可能在请求提交后就开始调度，因此 `enqueue` 返回并不意味着核心仍未开始执行；是否需要控制调度时机，应结合引擎运行方式理解。

## 5. 设计思想

**用统一入口组织配置与调用。** 用户通过 `LLM` 创建可重复使用的引擎实例，再调用生成接口。初始化配置与每次请求的采样参数具有不同生命周期，分别通过构造函数和生成方法进入系统。

**用 mixin 复用流程，用组件承接具体职责。** `LLM` 继承离线推理编排方法，同时调用 renderer 和引擎组件。这让文本生成、对话生成等入口复用请求管理与收集能力，并保留各自的输入处理路径。

**在边界逐层完成校验。** `LLM` 判断接口是否适用于当前任务，mixin 对齐批量输入和参数，输入处理层再检查请求内容与生成约束。校验位置对应各层掌握的信息。

**同步调用封装增量执行。** 用户得到完整的 `list[RequestOutput]`，内部仍持续接收和处理增量输出。`enqueue` 与 `wait_for_completion` 进一步开放提交和收集的组合方式。

**保持结果与输入的对应关系。** 每条输入使用请求 ID 贯穿提交、执行和输出处理，最后恢复结果顺序。对用户而言，一批请求可以并行推进，同时保留明确的结果归属。

## 6. 总结

理解 `LLM` 可以抓住两条主线：初始化时，将用户配置转换为 `EngineArgs`，创建引擎并建立组件引用；生成时，检查任务、取得参数，再委托 mixin 完成请求提交和结果收集。

继续阅读 [OfflineInferenceMixin](02-OfflineInferenceMixin.md)，可展开 `_run_completion`、请求提交和 `_run_engine` 的具体实现；输入与输出转换分别见 [Input Processing](03-Input-Processing.md) 和 [Output Processing](04-Output-Processing.md)。
