# 00 · vLLM V1 整体架构与请求流转

先跟踪一次请求从进入系统到返回结果的过程，再按 01–08 深入各个模块。本篇对应当前仓库中的 V1 实现，以单 GPU、纯文本、自回归生成作为主线。

## 1. 先看完整流程

```text
离线：LLM.generate / 在线：API Server
    ↓
输入处理：文本转 token IDs、校验参数
    ↓
LLMEngine / AsyncLLM：登记请求和输出状态
    ↓
EngineCoreClient：提交请求、接收结果
    ↓
EngineCore：驱动推理循环
    ↓
Scheduler：选择请求与本轮 token 数 ←→ KV Cache Manager：分配、复用 KV 块
    ↓
SchedulerOutput：本轮执行计划
    ↓
Executor → GPU Worker → Model Runner
    ↓
准备 token IDs、positions、block tables、Attention 元数据
    ↓
模型前向：Embedding → Decoder 层 → 输出隐藏状态
    │                    ↕
    │              层内 Attention：写入本轮 KV、读取上下文 KV
    ↓
LM Head：计算 logits
    ↓
Sampler：产生候选输出 token
    ↓
Scheduler.update_from_output：更新进度和结束状态
    ├─ 请求完成？
    │   ├─ 未完成 → 回到 Scheduler，进入下一轮
    │   └─ 完成 → 释放请求的 KV 块引用
    │
    └─ EngineCoreOutputs → EngineCoreClient
            ↓
        OutputProcessor：反分词、处理停止字符串、组织输出
            ├─ 离线收集最终结果 / 在线按配置返回流式或完整结果
            └─ 停止字符串触发结束
                    ↓
                向 EngineCore 提交终止请求
                    ↓
                释放请求的 KV 块引用
```

这张图是模块职责图。请求可能跨进程传递；模块之间的箭头不一定对应直接的 Python 函数调用。循环中的部分步骤可以异步执行或重叠，第一遍先理解它们的数据依赖。

**核心循环：调度 → 准备输入 → 模型计算与采样 → 更新状态 → 下一轮。**

## 2. 每一层负责什么

| 层级 | 主要职责 | 阅读入口（路径相对源码仓库） |
|---|---|---|
| 推理入口 | 接收 prompt 和生成参数，发起请求；在线入口还处理 HTTP 协议 | `vllm/entrypoints/llm.py`、`vllm/entrypoints/openai/` |
| 输入处理 | 将输入转换成引擎可用的 token 和请求数据，校验参数；文本渲染和分词也涉及 Renderer | `vllm/v1/engine/input_processor.py`、`vllm/renderers/` |
| LLMEngine / AsyncLLM | 管理前端请求和输出状态，连接 EngineCore；分别服务同步与异步使用方式 | `vllm/v1/engine/llm_engine.py`、`vllm/v1/engine/async_llm.py` |
| EngineCoreClient | 提交请求、接收核心引擎输出，封装进程通信或进程内调用 | `vllm/v1/engine/core_client.py` |
| EngineCore | 组织调度、执行、采样和状态更新 | `vllm/v1/engine/core.py` |
| Scheduler | 管理等待和运行中的请求，决定本轮各请求计算多少 token，处理执行结果 | `vllm/v1/core/sched/scheduler.py` |
| KV Cache Manager | 管理 KV 块元数据与分配、前缀命中、引用释放 | `vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/block_pool.py` |
| Executor | 将执行任务派发给一个或多个 Worker，汇集执行结果 | `vllm/v1/executor/abstract.py`、`vllm/v1/executor/uniproc_executor.py` |
| GPU Worker | 初始化 GPU、加载模型，调用 Model Runner；多 GPU 时还涉及通信 | `vllm/v1/worker/gpu_worker.py` |
| Model Runner | 维护执行所需状态，准备批量张量和 Attention 元数据，运行模型及采样 | `vllm/v1/worker/gpu_model_runner.py`、`vllm/v1/worker/gpu/model_runner.py` |
| 模型与计算层 | 实现模型前向、Attention、线性层、归一化等计算 | `vllm/model_executor/models/`、`vllm/model_executor/layers/` |
| Attention 后端与 kernel | 根据 KV 布局和后端实现执行 Attention 等底层计算 | `vllm/v1/attention/`、`csrc/` |
| 输出处理 | 将输出 token 转成文本，处理停止字符串，组织用户可见结果 | `vllm/v1/engine/output_processor.py`、`vllm/v1/engine/detokenizer.py` |

当前 Worker 中存在两种 Model Runner 实现，通过 `use_v2_model_runner` 选择。这个 V2 指 Model Runner 的实现版本，所在引擎仍然是 V1。阅读时先查看 Worker 实际选用了哪一种。

## 3. 一条请求如何走完生命周期

### 3.1 请求进入：文本变成引擎请求

以离线调用为例：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="<本地模型路径>")
outputs = llm.generate(
    ["请介绍一下 KV Cache。"],
    SamplingParams(temperature=0.8, max_tokens=32),
)
```

`LLM` 初始化阶段已经涉及模型加载、KV Cache 配置和预热。`generate()` 提交请求时，系统处理 prompt 和参数，得到 token IDs，登记请求 ID，并建立对应的输出状态。

在当前实现中，`LLMEngine.add_request()` 会调用输入处理器，再分别向 `OutputProcessor` 和 `EngineCoreClient` 登记请求。核心侧把请求交给 Scheduler，等待获得计算资源。

第一遍跟踪三个信息即可：**请求 ID、输入 token IDs、生成参数**。

### 3.2 调度：决定这一轮计算什么

Scheduler 同时维护多条请求。每次调度要结合 token 预算、请求进度和 KV 空间，决定哪些请求可以执行，以及每条请求本轮计算多少 token。

- **Prefill**：计算 prompt 的 token，建立各层 KV；较长 prompt 可以拆成多个轮次。
- **Decode**：使用已有上下文 KV，计算新输入 token，并产生下一个输出 token。
- **Continuous Batching**：每轮重新组织执行批次，允许请求加入、完成和离开。

V1 的调度实现围绕 token 进度和预算组织逻辑。学习时可用 prefill/decode 区分计算语义，但不要预设源码必须维护两套独立调度循环。

Scheduler 调用 KV Cache Manager，检查可复用的前缀块，并为本轮计算分配需要的 KV 块。最终生成 `SchedulerOutput`，其中包含各请求的调度 token 数、请求数据更新和缓存块信息等。

**SchedulerOutput 是本轮的执行计划；它还需要 Model Runner 转换成 GPU 张量。**

### 3.3 KV Cache：连接调度与 Attention

KV Cache 在不同层有不同的表示：

| 所在层 | 关心的信息 |
|---|---|
| Scheduler / KV Cache Manager | 请求需要多少块、哪些块可以复用、哪些块可以释放 |
| Model Runner | token 对应的写入位置、请求的 block table、Attention 元数据 |
| Attention / GPU | 各层实际存储的 K/V 张量，以及本轮计算如何读写它们 |

因此，理解分页缓存要把 **CPU 侧块管理 → 执行侧位置映射 → GPU 侧 KV 读写** 串起来。KV Cache Manager 管理块的分配和复用，具体 KV 数值由模型执行中的 Attention 路径写入和读取。

### 3.4 执行：计划变成 GPU 计算

```text
EngineCore
    → Executor.execute_model(SchedulerOutput)
    → GPU Worker.execute_model(...)
    → Model Runner.execute_model(...)
```

Executor 负责派发，Worker 提供设备执行环境，Model Runner 将计划转成模型需要的输入：token IDs、positions、block tables，以及 Attention 元数据等。

随后模型执行 Embedding 和各 Decoder 层。Attention 在 Decoder 层内部运行：计算本轮 Q/K/V，将新 KV 写入缓存，并读取上下文 KV 完成注意力计算。经过后续层得到隐藏状态，再通过 LM Head 计算 logits。

Sampler 根据 logits 和采样参数产生输出 token。在当前 `EngineCore.step()` 中，模型执行可能直接返回结果，也可能先返回 `None`，随后由 `sample_tokens()` 完成采样。读代码时需要把执行和采样两条调用一起看。

### 3.5 更新与返回：闭合推理循环

`Scheduler.update_from_output()` 接收执行结果，更新请求进度、生成 token 和结束状态。未完成的请求继续参与后续调度；完成的请求释放 KV 块引用。

开启前缀缓存时，释放请求引用后，一些块仍可保留缓存信息供后续请求复用，直到需要被回收。请求结束不等于立即清空对应的所有 GPU KV 数据。

核心侧的结果经 `EngineCoreClient` 返回前端，由 `OutputProcessor` 反分词并组织输出。结束判断分布在核心侧和输出侧：例如 EOS、长度限制涉及核心侧状态，停止字符串则需要输出侧处理文本，并在必要时通知核心终止请求。

- 离线 `LLM.generate()` 持续推进并收集完成的输出，最后返回结果列表。
- 在线 API 通过异步路径运行，可以按请求配置流式返回生成结果。

## 4. 用一个小例子理解 prefill 与 decode

假设 prompt 分词后是 `[p0, p1, p2, p3]`，请求最多生成三个 token；忽略前缀复用、分块 prefill 和推测解码，并假设期间没有提前结束：

| 轮次 | 本轮模型输入 | KV 的变化 | 本轮采样结果 |
|---|---|---|---|
| 1：Prefill | `p0, p1, p2, p3` | 写入四个 prompt token 的 KV | `t0` |
| 2：Decode | `t0` | 写入 `t0` 的 KV，并读取已有上下文 KV | `t1` |
| 3：Decode | `t1` | 写入 `t1` 的 KV，并读取已有上下文 KV | `t2`，达到生成长度上限 |

**采样出的 token 在作为下一轮输入被计算时，才产生自己的 KV。** 最后一轮采样出的 `t2` 已经可以返回给用户，请求结束后无需再为它执行一次模型前向。

若 prompt 被分块，尚未完成的 prefill 轮次可能只推进输入计算进度，不产生可返回的新输出 token。因此不能把“一轮调度”理解为“每条请求一定生成一个 token”。

## 5. 第一遍阅读抓住两个循环

### 前端：持续接收和处理核心输出

[entrypoints/offline_utils.py](../../vllm/vllm/entrypoints/offline_utils.py) 的 `_run_engine()` 在有未完成请求时不断调用 `LLMEngine.step()`。当前 [LLMEngine.step()](../../vllm/vllm/v1/engine/llm_engine.py) 的主要工作是获取核心输出、处理输出，并通知核心终止由输出侧判定结束的请求。

### 核心：调度、执行、更新

[EngineCore.step()](../../vllm/vllm/v1/engine/core.py) 的主线可以概括为：

```python
# 阅读用伪代码；省略异步、异常处理和其他任务类型。
plan = scheduler.schedule(...)
future = model_executor.execute_model(plan, non_block=True)
result = future.result()
if result is None:
    result = model_executor.sample_tokens(...)
outputs = scheduler.update_from_output(plan, result)
```

前端的 `LLMEngine.step()` 和核心的 `EngineCore.step()` 职责不同。多进程模式下，核心拥有自己的循环；前端获取输出不代表它在同一进程内直接完成一次 GPU 执行。

先读普通 `step()` 建立理解，再看 `step_with_batch_queue()` 和异步调度如何让执行重叠。

## 6. 与后续八个阶段的关系

| 阶段 | 回到总流程中需要回答的问题 |
|---|---|
| [01 · Inference Entry](../01-Inference-Entry/) | prompt 和参数怎样变成引擎请求？ |
| [02 · Engine Core](../02-Engine-Core/) | 前端、Client 和核心怎样协作，谁驱动循环？ |
| [03 · Scheduler](../03-Scheduler/) | 本轮选哪些请求、分配多少 token，执行后如何更新？ |
| [04 · KV Cache](../04-KV-Cache/) | 块如何分配、复用、映射和释放？ |
| [05 · Worker and Model Runner](../05-Worker-and-Model-Runner/) | Executor 如何派发任务，执行计划如何转成 GPU 张量？ |
| [06 · Model Execution](../06-Model-Execution/) | token 怎样经过模型层变成 logits？ |
| [07 · Attention and Kernels](../07-Attention-and-Kernels/) | Attention 怎样读取分页 KV，后端和 kernel 如何执行？ |
| [08 · Distributed and Optimization](../08-Distributed-and-Optimization/) | 计算如何分布到多 GPU，通信和执行优化如何影响流程？ |

读完本篇后，尝试不看图讲清一条请求的完整路径，并区分 `SchedulerOutput`、GPU KV 张量和用户收到的 `RequestOutput`。接下来进入 01，沿同一条请求逐步补齐源码细节。
