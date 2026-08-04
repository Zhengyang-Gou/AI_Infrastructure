# Qwen2-VL

## 1. 文件定位

- 路径：`vllm/model_executor/models/qwen2_vl.py`。
- 职责：以一个完整视觉语言模型展示 Processor 注册、视觉 Transformer、embedding 合并和语言模型 forward。

## 2. 输入处理

- `Qwen2VLProcessingInfo` 计算图像或视频在不同尺寸、帧数下产生的视觉 token 数。
- `Qwen2VLMultiModalProcessor` 定义图像/视频字段和 prompt placeholder 更新规则。
- 图片和视频包含 pixel values 与 grid THW，用于恢复时空 patch 布局。
- mRoPE positions 需要同时表达文本和视觉时空位置。

## 3. 模型执行

```text
pixel_values + grid_thw
→ Vision Patch Embed
→ Vision Transformer blocks
→ Patch Merger
→ multimodal embeddings
→ 写入 prompt placeholder positions
→ Qwen2 language model
→ hidden states
→ logits
```

视觉 token 数必须与 placeholder range 长度匹配，否则 embedding merge 无法保持序列位置正确。

## 4. 关键代码解析

### `Qwen2VLMultiModalProcessor._get_prompt_updates()`

### `Qwen2VLMultiModalProcessor._get_mm_fields_config()`

### `Qwen2VLProcessingInfo.get_num_image_tokens()`

### `Qwen2VLForConditionalGeneration.get_mrope_input_positions()`

### `Qwen2VisionTransformer.forward()`

### `Qwen2VLForConditionalGeneration.embed_multimodal()`

### `Qwen2VLForConditionalGeneration.forward()`

### `Qwen2VLForConditionalGeneration.compute_logits()`

## 5. 阅读建议

- 先用单张图片追 `pixel_values → grid_thw → visual embeddings → placeholder`。
- 再比较视频输入，观察 frame 和 temporal patch 如何改变 token 数。
- 最后阅读 encoder CUDA Graph、分块与 profiling 分支，避免一开始被优化路径打断主线。

## 6. 当前结论

Qwen2-VL 将多模态通用框架具体化：模型文件同时声明“输入怎样处理”和“特征怎样计算”，通用 Engine 负责缓存、调度与批处理。
