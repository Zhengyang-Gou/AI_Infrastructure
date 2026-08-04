# Input Rendering

## 1. 文件定位

- 路径：`vllm/renderers/`、`vllm/v1/engine/input_processor.py`。
- 职责：应用 chat template、tokenize、多模态预处理，并生成合法的 `EngineCoreRequest`。

## 2. 主执行流程

```text
messages / prompt
→ OnlineRenderer 或 BaseRenderer
→ chat template
→ tokenize / multimodal processor
→ EngineInput
→ InputProcessor.process_inputs()
→ 参数与长度校验
→ EngineCoreRequest
```

Renderer 负责面向模型的输入语义，InputProcessor 负责面向 Engine Core 的统一请求结构与约束。

## 3. 输入与输出

- 输入：文本、token IDs、messages、多模态数据和 tokenization kwargs。
- 输出：tokenized `EngineInput` 与 `EngineCoreRequest`。
- 状态：更新多模态 processor cache，并为请求分配内部 request ID。

## 4. 关键代码解析

### `renderer_from_config()`

### `BaseRenderer.render_cmpl()`

### `BaseRenderer.render_chat()`

### `BaseRenderer.process_for_engine()`

### `OnlineRenderer.render_chat()`

### `InputProcessor.process_inputs()`

### `InputProcessor._validate_prompt_len()`

## 5. 当前结论

输入渲染层把多种公开协议统一成模型 token 与特征，再由 InputProcessor 收敛为核心引擎唯一接受的请求格式。
