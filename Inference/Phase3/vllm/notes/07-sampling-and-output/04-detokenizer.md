# Incremental Detokenizer

## 1. 学习目标

理解新增 token IDs 如何在不重复解码全部序列的情况下增量转换为文本，并理解 stop token、stop string、流式 DELTA 输出和停止字符串缓冲之间的关系。

## 2. 文件定位

- 文件路径：`vllm/v1/engine/detokenizer.py`
- 所属层次：Frontend Engine 文本输出层
- 核心职责：选择快慢增量解码实现、累积生成 token、检查 stop string，并按累计或 DELTA 模式返回安全的输出文本。
- 在调用链中的位置：由 `OutputProcessor.process_outputs()` 调用，结果最终进入 `CompletionOutput.text`。

## 3. 核心类与组件

| 类 / 组件 | 作用 |
| --- | --- |
| `IncrementalDetokenizer` | 公共接口与“跳过 detokenization”时的空实现 |
| `BaseIncrementalDetokenizer` | 实现 token 累积、stop string 检查和输出切片的通用逻辑 |
| `FastIncrementalDetokenizer` | 使用 `tokenizers.decoders.DecodeStream` 进行原生增量解码 |
| `SlowIncrementalDetokenizer` | 使用 Python 侧 `detokenize_incrementally()` 与 offset 状态增量解码 |
| `check_stop_strings()` | 在新生成的文本范围内查找最早完成的停止字符串并计算截断位置 |

## 4. 主执行流程

### 实现选择

```text
IncrementalDetokenizer.from_new_request(tokenizer, request)
→ tokenizer 为 None：使用空实现，仅记录 token IDs
→ tokenizers 版本满足要求且为 TokenizersBackend：FastIncrementalDetokenizer
→ 其他情况：SlowIncrementalDetokenizer
```

### 增量更新

```text
BaseIncrementalDetokenizer.update(new_token_ids, stop_terminated)
→ 根据 include_stop_str_in_output 决定是否跳过最后一个 stop token 的解码
→ 逐个保存 token ID
→ decode_next(token_id)
→ 追加到 output_text
→ 达到 min_tokens 后检查 stop strings
→ 命中时按配置保留或裁掉 stop string
→ 返回匹配字符串或 None
```

### 生成用户文本

```text
get_next_output_text(finished, delta)
→ 未结束时保留 stop_buffer_length 个尾部字符
→ cumulative 模式返回当前可见累计文本
→ delta 模式按 _last_output_text_offset 返回新增片段
→ 请求结束时释放尾部缓冲
```

尾部缓冲用于避免流式输出提前泄露某个 stop string 的前缀。缓冲长度取最长 stop string 的长度减一，且仅在 stop string 不应包含在输出中时启用。

## 5. 输入与输出

### 输入

- `EngineCoreRequest`：提供 prompt token IDs 与 detokenization / stop 相关采样参数。
- `new_token_ids`：当前 `EngineCoreOutput` 新产生的 token IDs。
- `stop_terminated`：核心是否因 token 级 STOP 原因结束。
- `finished` 与 `delta`：控制文本是否释放尾部缓冲以及返回累计或增量文本。

### 输出

- `update()` 返回命中的 stop string；没有命中时返回 `None`。
- `get_next_output_text()` 返回当前应暴露给用户的文本。
- `output_token_ids` 提供已生成 token IDs；慢速实现会排除为解码上下文保存的 prompt IDs。

### 状态变化

- 累积 `token_ids` 与 `output_text`。
- 快速实现推进 `DecodeStream` 状态。
- 慢速实现更新 tokens、`prefix_offset` 与 `read_offset`。
- DELTA 模式更新 `_last_output_text_offset`。

## 6. 关键代码解析

### `IncrementalDetokenizer.from_new_request()`

### `BaseIncrementalDetokenizer.update()`

### `BaseIncrementalDetokenizer.decode_next()`

### `BaseIncrementalDetokenizer.get_next_output_text()`

### `FastIncrementalDetokenizer.decode_next()`

### `FastIncrementalDetokenizer._protected_step()`

### `SlowIncrementalDetokenizer.decode_next()`

### `check_stop_strings()`

## 7. 与其他文件的关系

- 上游：`OutputProcessor` 把 `EngineCoreOutput.new_token_ids` 传给 `update()`。
- 请求参数：`SamplingParams` 决定 stop strings、min tokens、special token 和输出包含规则。
- 快速实现：依赖 Hugging Face `tokenizers` 的 `DecodeStream`。
- 慢速实现：依赖 `vllm/tokenizers/detokenizer_utils.py`。
- 下游：`RequestState._new_completion_output()` 调用 `get_next_output_text()`，并写入 `CompletionOutput.text`。
- 核心协作：字符串停止条件在前端命中时，`OutputProcessor` 可能要求 Engine Core 中止请求。

## 8. 当前结论

Detokenizer 不只是 token 到字符串的转换器。它还维护增量解码状态、保护流式输出不泄露 stop string 前缀，并补充 Scheduler 无法在 token ID 层完成的字符串停止判断。
