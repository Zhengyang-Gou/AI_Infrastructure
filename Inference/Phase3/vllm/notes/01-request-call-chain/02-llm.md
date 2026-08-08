# LLM

## 1. 文件定位

- 文件路径：`vllm/entrypoints/llm.py`
- 所属层次：离线推理入口层
- 核心职责：把用户侧 API 接入 vLLM 的底层引擎与通用离线推理流程。
- 在调用链中的位置：位于用户代码与 `OfflineInferenceMixin` / `LLMEngine` 之间。

当前阶段只围绕下面这条主线学习：

```text
basic.py
  ↓
LLM(...)
  ↓
LLM.generate(...)
  ↓
OfflineInferenceMixin._run_completion(...)
  ↓
offline_utils.py
```

`LLM` 是一个用户侧门面（facade）。它负责接收参数、创建底层 `LLMEngine` 并提供同步生成 API，但不亲自实现 Scheduler、KV Cache、模型前向或 Sampling。

---

## 2. 本阶段阅读范围

### 重点阅读

1. `class LLM(...)` 的继承关系，尤其是 `OfflineInferenceMixin`。
2. `LLM.__init__()` 中的 `EngineArgs → LLMEngine.from_engine_args()`。
3. `get_default_sampling_params()` 的职责。
4. `LLM.generate()` 的主逻辑。
5. 可选阅读 `enqueue()` / `wait_for_completion()`，用于理解“提交请求”和“驱动执行”的区别。

### 暂时跳过

- `EngineArgs` 的大量具体字段；
- tensor parallel / data parallel；
- quantization；
- compilation / CUDA Graph；
- LoRA 与 multimodal 细节；
- `chat()`、`collective_rpc()`、prefix cache、profiling 等其他接口；
- Scheduler、KV Cache、ModelRunner、Sampler 的内部实现。

当前原则：

> 只沿着 `basic.py → LLM.generate() → _run_completion()` 这条 Request Call Chain 前进。

---

## 3. `LLM` 的继承关系

当前最需要关注的是 `OfflineInferenceMixin`。

可以先建立下面的结构：

```text
LLM
│
├── 对外 API
│   └── generate()
│
└── OfflineInferenceMixin
    ├── _run_completion()
    ├── _add_completion_requests()
    └── _run_engine()
```

因此，在 `LLM.generate()` 中看到：

```python
return self._run_completion(...)
```

时，不要继续在 `llm.py` 里寻找 `_run_completion()` 的实现。

它来自 `OfflineInferenceMixin`，定义在：

```text
vllm/entrypoints/offline_utils.py
```

这是读完 `llm.py` 后最重要的源码跳转点。

---

## 4. `LLM.__init__()`：只抓初始化主线

`LLM.__init__()` 参数很多，当前阶段不需要逐项研究。

### 4.1 构造 `EngineArgs`

大量用户参数最终会被集中封装到 `EngineArgs`：

```python
engine_args = EngineArgs(
    model=model,
    tokenizer=tokenizer,
    tensor_parallel_size=tensor_parallel_size,
    dtype=dtype,
    quantization=quantization,
    gpu_memory_utilization=gpu_memory_utilization,
    ...
)
```

当前只需要形成：

```text
LLM(...) 的用户参数
        ↓
    EngineArgs
```

`EngineArgs` 可以暂时理解为：

> 用户配置进入底层引擎初始化之前的统一参数载体。

不要继续追每一个字段如何变成 `VllmConfig`。

### 4.2 创建 `LLMEngine`

当前最重要的一段是：

```python
self.llm_engine = LLMEngine.from_engine_args(
    engine_args=engine_args,
    usage_context=UsageContext.LLM_CLASS,
)
```

因此初始化主线可以压缩成：

```text
LLM(model=...)
      ↓
LLM.__init__()
      ↓
EngineArgs
      ↓
LLMEngine.from_engine_args(...)
      ↓
self.llm_engine
```

这里是当前阶段的阅读边界：

```text
用户参数整理
    ↓
EngineArgs
    ↓
------------------------
LLMEngine.from_engine_args()
    ↓
底层初始化细节
```

`EngineArgs → VllmConfig → Platform → Engine` 的完整初始化链留到 configuration / initialization 专题。

### 4.3 记住 `self.llm_engine`

引擎创建后，`LLM` 会保存一些常用对象，例如：

```python
self.llm_engine
self.model_config
self.runner_type
self.renderer
self.input_processor
```

当前最重要的是：

```python
self.llm_engine
```

后续 `LLM` 的很多能力最终都要落到底层 `LLMEngine`。

---

## 5. `get_default_sampling_params()`：知道职责即可

`generate()` 允许用户不显式传入 `SamplingParams`：

```python
if sampling_params is None:
    sampling_params = self.get_default_sampling_params()
```

当前只记：

```text
用户传入 SamplingParams
        ↓
     直接使用

用户没有传入
        ↓
get_default_sampling_params()
        ↓
得到默认采样参数
```

generation config 的合并规则以及真正的 Sampling 实现都暂时不深入。

---

## 6. `LLM.generate()`：本文件最重要的代码

可以把 `generate()` 的主逻辑抽象成：

```python
def generate(
    self,
    prompts,
    sampling_params=None,
    ...,
) -> list[RequestOutput]:

    runner_type = self.model_config.runner_type

    if runner_type != "generate":
        raise ValueError(...)

    if sampling_params is None:
        sampling_params = self.get_default_sampling_params()

    return self._run_completion(
        prompts=prompts,
        params=sampling_params,
        output_type=RequestOutput,
        ...,
    )
```

它主要完成三件事。

### 6.1 检查 runner 类型

```python
runner_type = self.model_config.runner_type

if runner_type != "generate":
    raise ValueError(...)
```

作用只是确认当前模型运行模式支持生成任务。

```text
generate runner
    ↓
继续

非 generate runner
    ↓
拒绝调用 generate()
```

这里不是推理核心逻辑。

### 6.2 补齐默认 `SamplingParams`

```python
if sampling_params is None:
    sampling_params = self.get_default_sampling_params()
```

作用是保证后面的 completion 流程总能得到有效的生成参数。

### 6.3 转交给 `_run_completion()`

真正关键的是：

```python
return self._run_completion(
    prompts=prompts,
    params=sampling_params,
    output_type=RequestOutput,
    ...,
)
```

因此 `generate()` 可以理解成：

```text
LLM.generate()
     │
     ├── API 校验
     ├── 默认参数准备
     │
     └── _run_completion()
             ↓
       真正的离线 completion 流程
```

所以：

> `LLM.generate()` 是同步生成 API 的门面，不是模型执行循环本身。

它不会直接负责：

```text
Scheduler.schedule()
KV Cache 管理
model.forward()
GPU kernel
真正的 token sampling
```

---

## 7. `_run_completion()`：当前最重要的跳转点

`_run_completion()` 来自 `OfflineInferenceMixin`。

当前先理解它把两件事情组合起来：

```text
                 _run_completion()
                 /              \
                /                \
_add_completion_requests()      _run_engine()
        │                            │
        ↓                            ↓
     提交请求                    驱动执行
```

因此：

```text
用户调用 generate()
        ↓
提交全部请求
        ↓
不断驱动 Engine
        ↓
直到所有请求完成
        ↓
返回 list[RequestOutput]
```

这也解释了为什么 `basic.py` 中：

```python
outputs = llm.generate(prompts, sampling_params)
```

函数返回后可以直接遍历完整结果。

---

## 8. 可选阅读：`enqueue()` / `wait_for_completion()`

如果当前源码版本提供这两个接口，可以快速看一下。

它们非常适合帮助理解：

```text
提交请求 ≠ 驱动执行
```

可以抽象成：

```text
enqueue()
    ↓
_add_completion_requests()
    ↓
提交请求
```

```text
wait_for_completion()
    ↓
_run_engine()
    ↓
驱动执行直到完成
```

而普通的：

```python
LLM.generate(...)
```

通过 `_run_completion()` 把两者组合成一个同步接口。

这里看懂概念即可，不需要继续深入实现。

---

## 9. 与 `basic.py` 串起来

`basic.py` 中最核心的是：

```python
llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(prompts, sampling_params)
```

现在可以展开成：

```text
basic.py
│
├── LLM(model=...)
│       ↓
│   LLM.__init__()
│       ↓
│   EngineArgs
│       ↓
│   LLMEngine.from_engine_args()
│       ↓
│   self.llm_engine
│
└── LLM.generate(...)
        ↓
    检查 runner_type
        ↓
    准备 SamplingParams
        ↓
    _run_completion(...)
        ↓
    offline_utils.py
```

这就是当前阶段读 `llm.py` 的完整主线。

---

## 10. `LLM`、`LLMEngine` 与 `OfflineInferenceMixin` 的边界

### `LLM`

面向用户：

```text
LLM(...)
LLM.generate(...)
```

主要职责：

- 提供易用 API；
- 整理初始化参数；
- 创建并持有 `LLMEngine`；
- 将生成请求交给通用离线推理流程。

### `OfflineInferenceMixin`

面向通用离线执行流程：

```text
_run_completion()
_add_completion_requests()
_run_engine()
```

当前先理解为：

> 负责组织“提交请求 + 同步驱动引擎”这一整套离线推理流程。

### `LLMEngine`

面向更底层的请求与输出管理。

当前阶段先记住：

```text
LLM
    ↓
OfflineInferenceMixin
    ↓
LLMEngine
```

不要在 `llm.py` 阶段提前进入 `LLMEngine` 内部细节。

---

## 11. 当前阶段检查清单

学完 `llm.py` 后，应该能够不看源码回答：

- [ ] `LLM` 在离线推理调用链中处于什么位置？
- [ ] 为什么说 `LLM` 是 facade，而不是核心执行器？
- [ ] `LLM` 为什么能够调用 `_run_completion()`？
- [ ] `LLM.__init__()` 当前最值得看的两步是什么？
- [ ] `EngineArgs` 当前应该如何理解？
- [ ] `self.llm_engine` 在哪里创建？
- [ ] `generate()` 主要完成哪三件事？
- [ ] `generate()` 为什么不会直接执行 `model.forward()`？
- [ ] `_run_completion()` 定义在哪个文件？
- [ ] `_add_completion_requests()` 和 `_run_engine()` 分别负责什么？
- [ ] 为什么说“提交请求”和“驱动执行”是两条不同路径？
- [ ] 从 `llm.py` 下一步应该跳转到哪个文件？

---

## 12. 一句话总结

`LLM` 是 vLLM 离线推理的用户侧门面：构造阶段通过 `EngineArgs` 创建底层 `LLMEngine`，生成阶段由 `generate()` 完成入口校验和默认参数准备，然后把真正的同步离线推理流程交给 `OfflineInferenceMixin._run_completion()`。

---

## 13. 下一步

下一篇阅读：

```text
vllm/entrypoints/offline_utils.py
```

重点追踪两条路径：

```text
_run_completion()
        ↓
_add_completion_requests()
        ↓
LLMEngine.add_request()
```

以及：

```text
_run_engine()
        ↓
LLMEngine.step()
```

到这里再进入 `LLMEngine`，不要提前钻入 Scheduler、KV Cache 或模型执行细节。
