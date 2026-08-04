# Basic Offline Inference

## 1. 文件定位

- 文件路径：`examples/basic/offline_inference/basic.py`
- 所属层次：用户示例层
- 核心职责：展示如何创建 `LLM`、设置 `SamplingParams`、调用 `generate()` 并读取生成结果。
- 在调用链中的位置：整个离线推理请求的最外层入口。

> 当前仓库中没有检出 `examples` 目录，本笔记按照 `Roadmap.md` 指定的目标文件建立。恢复示例目录后，再对照实际代码补充关键代码。

## 2. 核心对象

| 对象 | 作用 | 来源 | 传递给 |
| --- | --- | --- | --- |
| `prompts` | 保存一个或多个用户输入 | 用户代码 | `LLM.generate()` |
| `SamplingParams` | 描述温度、最大 token 数等生成参数 | 用户代码 | `LLM.generate()` |
| `LLM` | 提供离线推理 API 并持有底层引擎 | `vllm` | 用户代码 |
| `RequestOutput` | 保存一次请求的完整生成结果 | `LLM.generate()` | 用户代码 |

## 3. 主执行流程

```text
准备 prompts
→ 创建 SamplingParams
→ 初始化 LLM
→ 调用 LLM.generate()
→ 遍历 RequestOutput
→ 读取 prompt 和生成文本
```

这个示例只负责演示公开 API。请求如何被处理、调度和执行，分别由 `LLM`、`LLMEngine` 与 `EngineCore` 完成。

## 4. 输入与输出

### 输入

- 模型名称或本地模型路径。
- 一个 prompt 或由多个 prompt 组成的序列。
- 控制生成行为的 `SamplingParams`。

### 输出

`LLM.generate()` 返回 `RequestOutput` 列表。每个元素对应一个输入请求，并包含生成文本、token ID 和结束原因等信息。

### 状态变化

示例代码本身不维护请求状态。`LLM.generate()` 是同步接口，会持续运行，直到本批请求全部完成后再返回。

## 5. 关键代码解析

### `LLM(...)`

### `SamplingParams(...)`

### `LLM.generate(...)`

## 6. 与其他文件的关系

- 上游：用户或业务代码。
- 下游：`vllm/entrypoints/llm.py`。
- 传递的数据：模型配置、prompt、采样参数。
- 返回的数据：`RequestOutput`。

## 7. 当前结论

`basic.py` 是用户视角的最小离线推理入口。它说明 vLLM 对外暴露什么 API，但不包含请求调度和模型执行的实现。
