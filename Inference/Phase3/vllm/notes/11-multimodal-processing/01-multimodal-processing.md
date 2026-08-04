# Multimodal Processing

## 1. 文件定位

- 路径：`vllm/multimodal/processing/processor.py`、`inputs.py`、`context.py`、`vllm/multimodal/parse.py`。
- 职责：把文本和不同模态媒体转换为模型可消费的 token IDs、关键字张量与占位范围。

## 2. 数据转换

```text
Prompt + MultiModalDataDict
→ 解析每个 modality/item
→ 调用模型对应 HF processor
→ 得到 pixel_values / feature tensors 等 mm_kwargs
→ 对 prompt 执行 insertion 或 replacement
→ tokenization
→ 定位各 item 的 placeholder range
→ 校验数量、长度与字段绑定
```

模型实现通过 `_get_mm_fields_config()` 描述 HF 输出字段怎样按媒体 item 切分，通过 `_get_prompt_updates()` 描述文本或 token 占位符怎样更新。

## 3. 输入与输出

- 输入：文本 prompt 或 token IDs，以及 image、video、audio 等媒体 item。
- 输出：处理后的 prompt token IDs、`MultiModalKwargs`、媒体 hashes、placeholder ranges。
- Placeholder range 标记语言模型序列中将被 encoder embeddings 替换或合并的位置。
- Processor 必须校验媒体 item 数、占位符数和特征长度一致。

## 4. 关键代码解析

### `BaseMultiModalProcessor.__call__()`

### `BaseMultiModalProcessor.apply()`

### `BaseMultiModalProcessor._apply_hf_processor()`

### `BaseMultiModalProcessor._apply_prompt_updates()`

### `BaseMultiModalProcessor._validate_mm_kwargs()`

### `BaseMultiModalProcessor._validate_mm_placeholders()`

### `find_mm_placeholders()`

## 5. 与其他文件的关系

- 上游：Renderer 或 InputProcessor 提供用户 prompt 和媒体对象。
- 模型注册表创建对应具体 Processor。
- 下游：Engine request 携带 mm features、hashes 和 placeholder ranges。
- Worker 的 EncoderRunner 消费处理后的特征。

## 6. 当前结论

Processor 不负责运行视觉或音频 encoder；它负责建立媒体与文本序列之间可验证的结构化映射。
