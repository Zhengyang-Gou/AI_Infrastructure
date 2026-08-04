# LLM

## 1. 文件定位

- 文件路径：`vllm/entrypoints/llm.py`
- 所属层次：离线推理入口层
- 核心职责：把易用的用户参数转换成引擎配置，并向用户提供同步生成接口。
- 在调用链中的位置：位于用户代码和 `LLMEngine` 之间。

`LLM` 面向离线批量推理。它屏蔽引擎初始化、请求批处理和执行循环等内部细节，不负责亲自执行调度或模型前向计算。

## 2. 核心类与函数

| 类 / 函数 | 作用 | 调用者 | 调用对象 |
| --- | --- | --- | --- |
| `LLM` | 离线推理的公开 API | 用户代码 | `LLMEngine` |
| `LLM.__init__()` | 接收模型与运行参数，构造底层引擎 | 用户代码 | `EngineArgs`、`LLMEngine` |
| `LLM.from_engine_args()` | 根据已有 `EngineArgs` 创建 `LLM` | 初始化逻辑 | `LLMEngine.from_engine_args()` |
| `LLM.generate()` | 校验生成任务并启动 completion 流程 | 用户代码 | `_run_completion()` |
| `OfflineInferenceMixin` | 提供请求批量提交和同步执行循环 | `LLM` | `LLMEngine` |

## 3. 主执行流程

```text
LLM.__init__()
→ 整理用户配置
→ 构造 EngineArgs
→ 创建 LLMEngine
→ 保存 model_config、tokenizer 等公共对象
```

```text
LLM.generate()
→ 检查 runner_type
→ 准备默认 SamplingParams
→ _run_completion()
→ 返回 RequestOutput 列表
```

`_run_completion()` 来自 `OfflineInferenceMixin`，定义在 `vllm/entrypoints/offline_utils.py`。因此，`llm.py` 负责公开 API，而通用离线执行流程被放在 Mixin 中复用。

## 4. 输入与输出

### 输入

- 初始化输入：模型路径、tokenizer、并行配置、dtype、量化方式、显存配置等。
- 生成输入：`PromptType`、`SamplingParams`、LoRA、优先级和多模态处理参数等。

### 输出

`generate()` 返回 `list[RequestOutput]`，结果顺序与输入 prompt 顺序一致。

### 状态变化

- 初始化阶段创建并保存 `LLMEngine`。
- `generate()` 将请求加入底层引擎，并同步驱动引擎直至所有请求结束。
- 真正的请求状态由 `LLMEngine` 和核心引擎维护。

## 5. 关键代码解析

### `LLM.__init__()`

### `LLM.from_engine_args()`

### `LLM.generate()`

## 6. 与其他文件的关系

- 上游：离线推理示例或用户业务代码。
- 下游：`vllm/entrypoints/offline_utils.py` 和 `vllm/v1/engine/llm_engine.py`。
- 传递的数据：prompt、采样参数以及引擎配置。
- 返回的数据：面向用户的 `RequestOutput`。

## 7. 当前结论

`LLM` 是离线推理的门面对象：构造阶段负责准备底层引擎，`generate()` 负责校验输入并把执行工作交给通用离线推理流程。
