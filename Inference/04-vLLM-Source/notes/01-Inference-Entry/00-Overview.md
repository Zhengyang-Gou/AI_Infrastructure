# 01 · 推理入口

本章以本地 vLLM V1 的离线 `LLM.generate()` 为主线，先看纯文本、每个 prompt 生成一个结果的情况。

**入口层负责把用户输入变成引擎请求，再把引擎输出变成用户结果。** 请求何时执行由 Scheduler 决定，模型计算在 Worker 和 Model Runner 中完成。

## 1. 输入和输出是什么

| 数据 | 含义 |
|---|---|
| `prompts` | 输入文本，也可以是已有的 token IDs |
| `SamplingParams` | 生成规则，例如 `temperature`、`max_tokens`、停止条件 |
| `RequestOutput` | 一条请求的结果，包含生成文本、token IDs 和完成状态等信息 |

`generate()` 返回 `list[RequestOutput]`，结果顺序与输入 prompt 顺序一致。

## 2. 一次 generate 做了什么

```text
用户调用 LLM.generate(prompts, sampling_params)
    ↓
整理输入：统一 prompt 和参数格式，文本分词为 token IDs
    ↓
构建请求：校验输入与参数，生成 EngineCoreRequest
    ↓
登记请求：保存输出处理状态，通过 EngineCoreClient 提交给核心
    ↓
核心执行：调度 → 模型计算 → 采样 → 更新状态
    ↓
处理输出：把输出 token 转成文本，整理为 RequestOutput
    ↓
等待请求完成，按输入顺序返回结果
```

这是处理流程，箭头可能跨进程，不代表全部步骤都在 `generate()` 所在进程中执行。

## 3. 主要模块各做什么

| 模块 | 职责 |
|---|---|
| `LLM.generate()` | 接收输入，提交请求，等待结果 |
| `OfflineInferenceMixin` / Renderer | 整理 prompt 与参数、分词、批量提交请求 |
| `InputProcessor` | 校验输入和参数，构建 `EngineCoreRequest` |
| `LLMEngine` | 连接输入处理、核心请求提交和输出处理 |
| `EngineCoreClient` | 向核心提交请求、接收输出；封装进程内调用或进程通信 |
| `OutputProcessor` | 反分词、处理文本停止字符串、组织输出结果 |

## 4. 先记住三个区别

- **提交不等于执行。** `add_request()` 登记请求，核心通过后续调度决定何时计算。
- **前端 step 与核心 step 职责不同。** `LLMEngine.step()` 获取并处理输出；`EngineCore.step()` 组织调度、执行和状态更新。
- **停止条件可能在不同位置判定。** 长度上限和停止 token 可在核心侧处理；文本停止字符串由输出侧处理，触发后需通知核心终止请求。

模型加载主要发生在 `LLM(...)` 初始化阶段。同一个实例多次调用 `generate()` 会复用已加载模型。

## 5. 源码阅读顺序

路径相对本地 `vllm/` 源码目录。

1. [entrypoints/llm.py](../../vllm/vllm/entrypoints/llm.py)：从 `LLM.generate()` 开始。
2. [entrypoints/offline_utils.py](../../vllm/vllm/entrypoints/offline_utils.py)：看 `_run_completion()` 如何提交请求，再由 `_run_engine()` 收集结果。
3. [v1/engine/llm_engine.py](../../vllm/vllm/v1/engine/llm_engine.py)：看 `add_request()` 和 `step()`。
4. [v1/engine/input_processor.py](../../vllm/vllm/v1/engine/input_processor.py)：看 `process_inputs()` 如何构建请求。
5. [v1/engine/output_processor.py](../../vllm/vllm/v1/engine/output_processor.py)：看 `process_outputs()` 如何形成用户结果。

读完应能说明：**prompt 在哪里变成 token IDs，请求如何提交给核心，输出 token 如何变成最终文本。**

接下来：[LLM.generate](01-LLM-Generate.md) → [请求处理](02-Request-Processing.md)。
