# Inference Entry Overview

## 1. 整体职责

Inference Entry 连接用户接口与推理引擎：接收 prompt 和生成参数，将输入转换为引擎可接收的请求，再收集引擎输出并返回用户可使用的结果。

本文以已经完成 `LLM` 初始化后的离线文本生成调用 `LLM.generate(...)` 为主线，默认每个 prompt 生成一条结果（`n=1`）。调度、KV Cache 管理和模型执行由 Engine Core 及其下游模块负责，本章关注请求如何进入这些模块，以及结果如何返回。

各模块在这条主线中的分工如下：

| 模块 | 在协作中的职责 |
| --- | --- |
| `LLM` | 提供 `generate` 等用户接口，持有引擎与输入处理所需的组件。 |
| `OfflineInferenceMixin` | 组织输入预处理、批量请求提交和结果收集循环。 |
| `Renderer` | 将不同形式的用户输入转换为统一的 `EngineInput`。 |
| `LLMEngine` | 衔接输入处理、输出状态管理与 Engine Core 客户端。 |
| `InputProcessor` | 校验输入和生成参数，构造 `EngineCoreRequest`。 |
| `EngineCoreClient` | 向 Engine Core 提交请求，并取得引擎输出。 |
| `OutputProcessor` | 维护请求的输出状态，将引擎增量输出组织成 `RequestOutput`。 |

`OfflineInferenceMixin` 是 `LLM` 继承的能力，调用链中的这些方法运行在同一个 `LLM` 实例上；`Renderer`、`LLMEngine` 等则是它所使用的组件。

## 2. 一次推理的完整调用链

```text
LLM.generate(prompts, sampling_params)
└─ OfflineInferenceMixin._run_completion(...)
   │
   ├─ _add_completion_requests(...)                 请求准备与提交
   │  ├─ 统一 prompts 与参数的批量形式
   │  └─ _render_and_add_requests(...)              逐条消费输入
   │     ├─ _preprocess_cmpl_one(...)
   │     │  └─ _preprocess_cmpl(...)
   │     │     └─ Renderer.render_cmpl(...) → EngineInput
   │     └─ _add_request(...)
   │        └─ LLMEngine.add_request(...)
   │           ├─ InputProcessor.process_inputs(...) → EngineCoreRequest
   │           ├─ OutputProcessor.add_request(...)   注册输出状态
   │           └─ EngineCoreClient.add_request(...) → Engine Core
   │
   └─ _run_engine(...)                             结果收集循环
      ├─ while LLMEngine.has_unfinished_requests():
      │  └─ LLMEngine.step()
      │     ├─ EngineCoreClient.get_output() ← Engine Core
      │     ├─ OutputProcessor.process_outputs(...) → RequestOutput
      │     └─ 向 Engine Core 取消已由输出侧判停的请求（如有）
      └─ 收集 finished 的结果，按原请求顺序返回 list[RequestOutput]
```

这条链分为两个阶段：先完成本次调用的请求提交，再进入 `_run_engine` 收集结果。提交阶段通过生成器逐条预处理、逐条提交，无需等所有 prompt 都转换成 `EngineInput` 后才开始提交。

`LLMEngine.step()` 是入口侧的一次结果处理迭代，不等同于单个请求的一次模型前向计算。`EngineCoreClient` 屏蔽执行方式的差异：同进程模式可以在取输出时推进核心执行，多进程模式则从后台 Engine Core 获取输出。

## 3. 关键阶段分析

### 3.1 输入接收

用户通过 `LLM.generate` 传入单个 prompt 或一批 prompts，以及 `SamplingParams`。`LLM` 检查当前模型是否支持生成任务；如果没有显式提供采样参数，就取得默认参数，然后交给 `_run_completion`。

`OfflineInferenceMixin` 将单条输入转换为批量形式，并对齐每条输入对应的采样参数、LoRA 配置和优先级。这样，后续流程可以按统一方式逐条处理请求，无需反复区分单条调用与批量调用。

此时的数据仍然表达用户意图：**要基于什么输入生成，以及希望怎样生成。**

### 3.2 输入预处理

`OfflineInferenceMixin` 通过 `_preprocess_cmpl_one` 调用 `_preprocess_cmpl`，解析用户输入，再交给 `Renderer.render_cmpl` 完成渲染、分词及相应的模型输入处理，得到 `EngineInput`。

这里的变化是把面向用户的输入形式转换为引擎侧可以继续处理的输入表示。普通文本需要变为 token IDs；已经提供 token IDs 的输入可以复用这些信息；多模态输入还需要相应的预处理和位置等信息。

因此，`EngineInput` 表达的是**模型要消费的输入内容**。它还需要与请求标识、生成参数等结合，才能成为一次完整的引擎请求。

本地主线中的分词主要由 `Renderer` 负责，后续 `InputProcessor` 侧重请求校验与封装。详见 [Input Processing](Core-Modules/03-Input-Processing.md)。

### 3.3 请求提交

`OfflineInferenceMixin._add_request` 分配请求 ID，并将离线生成的输出模式设为 `FINAL_ONLY`，随后调用 `LLMEngine.add_request`。

`LLMEngine` 在这里协调三个动作：

1. **构造请求。** `InputProcessor.process_inputs` 接收 `EngineInput` 和生成参数，校验任务、输入长度与参数等，补全必要的生成配置，构造 `EngineCoreRequest`。
2. **注册输出状态。** `OutputProcessor.add_request` 记录请求上下文，为后续的 token 累积、文本解码与结束处理做好准备。
3. **提交核心执行。** `EngineCoreClient.add_request` 将请求交给 Engine Core，进入后续调度和执行流程。

`EngineInput → EngineCoreRequest` 的意义是从“输入内容”变为“可提交执行的请求”：引擎不仅需要知道输入是什么，还需要知道如何识别、生成和管理这次请求。

输出状态必须在提交时建立，因为之后返回的通常是携带请求 ID 的增量输出，需要依靠这些状态才能还原为完整结果。如果设置 `n>1`，`LLMEngine` 还会拆分子请求，并通过父请求状态组织多条候选结果。

### 3.4 结果收集

请求提交后，`OfflineInferenceMixin._run_engine` 在仍有未完成请求时反复调用 `LLMEngine.step()`。每轮通过 `EngineCoreClient.get_output` 取得核心输出，再交给 `OutputProcessor.process_outputs`。

Engine Core 返回的是面向执行的增量信息，例如本轮新增的 token IDs 和结束状态。`OutputProcessor` 根据请求 ID 找到提交时注册的状态，累积生成内容、进行文本解码、处理停止字符串，并构造面向用户的 `RequestOutput`。

结束判定由两侧协作完成：Engine Core 可以报告请求结束；输出侧也可能在解码后发现停止字符串。后一种情况下，`LLMEngine` 会把取消请求传回 Engine Core，使执行侧同步停止并清理请求。

`FINAL_ONLY` 表示只在请求完成时产出最终结果，内部仍需要持续处理增量 token。`_run_engine` 收集 `finished=True` 的结果，等待所有请求完成，再根据入口分配的请求 ID 恢复输入顺序，返回 `list[RequestOutput]`。因此，内部完成顺序可以与用户输入顺序不同。

## 4. 核心数据的变化

从用户视角看，主线可以概括为：

```text
Prompt → EngineInput → EngineCoreRequest → RequestOutput
```

若把核心执行与输出处理之间的边界展开，则是：

```text
Prompt
  ── Renderer ──→ EngineInput
  ── InputProcessor（结合请求 ID、生成参数等）──→ EngineCoreRequest
  ── Engine Core 调度与执行 ──→ EngineCoreOutput（多轮增量）
  ── OutputProcessor（结合请求状态）──→ RequestOutput
```

| 数据变化 | 为什么需要变化 | 负责模块 |
| --- | --- | --- |
| `Prompt → EngineInput` | 将文本、token 输入等形式转换为一致的模型输入表示，隔离输入格式差异。 | `OfflineInferenceMixin` 调用 `Renderer`。 |
| `EngineInput → EngineCoreRequest` | 将输入与经过校验的生成参数、请求标识等组合为可提交的执行请求。 | `LLMEngine` 调用 `InputProcessor`。 |
| `EngineCoreRequest → EngineCoreOutput` | 引擎逐步执行请求，产生增量 token 和状态，而非一次性生成最终用户对象。 | Engine Core 及其下游模块。 |
| `EngineCoreOutput → RequestOutput` | 将增量执行结果与历史状态合并，转换为用户可读取的文本和完成信息。 | `OutputProcessor`。 |

`EngineCoreOutputs` 是客户端取得的一批核心输出的容器，其中包含各请求的 `EngineCoreOutput`。用户最终取得的列表则由 `_run_engine` 收集完成后的 `RequestOutput` 组成。

## 5. 关键设计思想

**接口与流程编排分开。** `LLM` 提供清晰的用户接口，`OfflineInferenceMixin` 复用请求准备、提交和收集逻辑。理解入口时，可以先抓住公共接口，再沿编排方法追踪组件协作。

**输入表示与执行请求分开。** `Renderer` 解决“不同输入如何转换成模型输入”，`InputProcessor` 解决“这些输入与参数能否构成合法请求”。两次转换分别对应输入格式与执行语义两个边界。

**计算执行与输出组织分开。** Engine Core 负责调度和执行，`OutputProcessor` 负责将增量 token 组织成用户结果。后者依赖持续维护的请求状态，也承担需要文本解码后才能完成的停止判断。

**同步接口封装增量执行。** 一次 `generate` 调用等待整批结果返回，内部却持续执行、持续处理增量输出。同步描述的是用户如何取得结果，并不表示所有请求必须按输入顺序逐个执行。

**通过客户端隔离执行边界。** `LLMEngine` 通过 `EngineCoreClient` 提交请求、获取输出，让入口流程能够适配不同的核心运行方式，而无需直接组织调度器或模型执行器。

## 6. 总结

一次离线推理由 `LLM` 接收调用，`OfflineInferenceMixin` 串起处理流程；输入依次经过 `Renderer` 和 `InputProcessor`，成为交给 Engine Core 的请求；核心产生的增量结果再由 `OutputProcessor` 还原为用户输出，最终由入口收集并返回。

继续阅读时，可沿 [LLM](Core-Modules/01-LLM.md) → [OfflineInferenceMixin](Core-Modules/02-OfflineInferenceMixin.md) → [Input Processing](Core-Modules/03-Input-Processing.md) → [Output Processing](Core-Modules/04-Output-Processing.md) 展开各模块细节。
