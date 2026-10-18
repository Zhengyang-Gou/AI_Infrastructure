# Inference Entry 学习导航

## 模块职责

接收离线推理调用，将用户输入转换为引擎请求，驱动推理并返回生成结果。

## 调用链

```text
LLM.generate → OfflineInferenceMixin._run_completion
  ├─ _add_completion_requests → Renderer → LLMEngine.add_request → InputProcessor → EngineCoreClient
  └─ _run_engine → LLMEngine.step → OutputProcessor → RequestOutput
```

## 学习清单

源码路径相对于 `../../vllm/`，建议按以下顺序阅读：

- [vllm/entrypoints/llm.py](../../vllm/vllm/entrypoints/llm.py)：`LLM` 初始化与 `generate` 入口。
- [vllm/entrypoints/offline_utils.py](../../vllm/vllm/entrypoints/offline_utils.py)：`OfflineInferenceMixin` 的请求提交与执行循环。
- [vllm/renderers/base.py](../../vllm/vllm/renderers/base.py)：输入渲染与分词入口。
- [vllm/v1/engine/llm_engine.py](../../vllm/vllm/v1/engine/llm_engine.py)：`add_request` 与 `step` 的衔接。
- [vllm/v1/engine/input_processor.py](../../vllm/vllm/v1/engine/input_processor.py)：请求校验与 `EngineCoreRequest` 构造。
- [vllm/v1/engine/output_processor.py](../../vllm/vllm/v1/engine/output_processor.py)：输出处理、文本解码与结束判定。
- [vllm/outputs.py](../../vllm/vllm/outputs.py)：`RequestOutput` 与 `CompletionOutput`。

## 笔记索引

- [Overview](Overview.md)：本章整体流程与模块关系。
- [01 · LLM](Core-Modules/01-LLM.md)：对外接口与初始化。
- [02 · OfflineInferenceMixin](Core-Modules/02-OfflineInferenceMixin.md)：离线推理流程编排。
- [03 · Input Processing](Core-Modules/03-Input-Processing.md)：从用户输入到引擎请求。
- [04 · Output Processing](Core-Modules/04-Output-Processing.md)：从引擎输出到用户结果。
