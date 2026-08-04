# Logits, Logprobs and Structured Output

## 1. 文件定位

- 主要路径：`vllm/v1/sample/logits_processor/`、`vllm/v1/engine/logprobs.py`、`vllm/v1/structured_output/`。
- 所属层次：采样约束与概率输出层。
- 核心职责：在采样前修改 logits、生成 grammar bitmask，并把模型概率转换成用户输出格式。

## 2. 核心对象

| 对象 | 作用 |
| --- | --- |
| `LogitsProcessor` | 根据请求参数或状态对 logits 做变换 |
| `StructuredOutputRequest` | 保存 choice、regex、JSON Schema 或 grammar 约束 |
| `StructuredOutputManager` | 异步编译 grammar，并为本轮 batch 生成 bitmask |
| `LogprobsProcessor` | 累积 prompt/sample logprobs 并修正解码文本边界 |

## 3. 主执行流程

```text
raw logits
→ penalties / bias / bad words
→ custom LogitsProcessor
→ StructuredOutput grammar bitmask
→ temperature / top-k / top-p
→ sample token
```

```text
模型返回 token IDs 与 top logprobs
→ LogprobsProcessor.update_from_output()
→ 累积 prompt 和 sample logprobs
→ Detokenizer 修正文本片段
→ CompletionOutput.logprobs
```

Grammar 通常异步编译。Scheduler 只有在 grammar ready 后才调度请求，并把 bitmask 传给采样器。

## 4. 输入与输出

### 输入

- raw logits、SamplingParams 和请求历史 token。
- choice、regex、JSON Schema 或 grammar 描述。
- 模型返回的 sampled token 及 top logprobs。

### 输出

- 经过约束的 logits 和可采样 token 集合。
- prompt logprobs、sample logprobs 及 decoded token 信息。
- 满足结构约束的生成 token。

### 状态变化

- Stateful logits processor 随生成 token 更新内部状态。
- Grammar 在每次接受 token 后推进自动机状态。
- LogprobsProcessor 跨多个 engine step 累积概率结果。

## 5. 关键代码解析

### `LogitsProcessor.validate_params()`

### `LogitsProcessor.apply()`

### `LogitsProcessor.update_state()`

### `StructuredOutputRequest.from_sampling_params()`

### `StructuredOutputManager.grammar_init()`

### `StructuredOutputManager.grammar_bitmask()`

### `StructuredOutputManager.should_advance()`

### `LogprobsProcessor.from_new_request()`

### `LogprobsProcessor.update_from_output()`

## 6. 与其他文件的关系

- Scheduler：等待 grammar 就绪并获取每轮 bitmask。
- Sampler：按 processor 和 bitmask 修改 logits。
- Detokenizer：为 logprobs 中的 token 生成稳定文本表示。
- 推测解码：rejection sampler 也必须应用相同采样约束。

## 7. 当前结论

这一层决定哪些 token 可以被采样，以及概率如何返回给用户。
