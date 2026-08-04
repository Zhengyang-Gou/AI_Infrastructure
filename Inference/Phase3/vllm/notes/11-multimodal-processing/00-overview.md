# Multimodal Processing

## 学习目标

本阶段理解图像、视频、音频等数据如何被解析和预处理，如何在文本 prompt 中占据准确的 token 范围，以及 encoder embeddings 如何进入语言模型。

## 阅读顺序

| 顺序 | 笔记 | 主要内容 |
| --- | --- | --- |
| 1 | `01-multimodal-processing.md` | Processor、prompt updates 和占位范围 |
| 2 | `02-registry-and-cache.md` | 模型注册、媒体哈希与两侧缓存 |
| 3 | `03-encoder-execution.md` | 调度预算、encoder forward 与 embedding 合并 |
| 4 | `04-qwen2-vl.md` | 用 Qwen2-VL 串起完整模型实现 |

## 主调用链

```text
文本 + image/video/audio
→ MultiModalDataParser
→ BaseMultiModalProcessor
→ Hugging Face processor
→ input_ids + mm_kwargs + mm_placeholders
→ Scheduler 分配 encoder budget
→ Worker 执行 multimodal encoder
→ 按 placeholder range 合并 embeddings
→ Language Model forward
```

## 完成标准

- 能区分原始媒体、processor 输出、placeholder tokens 和 encoder embeddings。
- 能说明一个媒体 item 与 prompt token range 如何一一对应。
- 能解释 processor cache 与 Worker receiver cache 的边界。
- 能追踪 encoder cache 的分配、命中和释放。
- 能以一个具体 VLM 说明模型注册、视觉 encoder 和语言模型的连接方式。

## 当前结论

多模态链路的核心约束是“媒体 item、处理后特征、占位 token 范围”三者始终对齐；任何缓存、切块或批处理都不能破坏这一映射。
