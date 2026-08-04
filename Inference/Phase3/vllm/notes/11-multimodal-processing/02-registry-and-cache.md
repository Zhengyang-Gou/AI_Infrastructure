# Registry and Cache

## 1. 文件定位

- 路径：`vllm/multimodal/registry.py`、`cache.py`、`hasher.py`。
- 职责：为模型绑定 processing components，并用稳定 hash 在 frontend、Engine 和 Worker 之间复用处理结果。

## 2. 注册机制

- 多模态模型通过 decorator 注册 Processor、ProcessingInfo 和 DummyInputsBuilder factory。
- Registry 根据 `ModelConfig` 找到实际模型类并构造对应处理上下文。
- ProcessingInfo 提供支持的模态、每 item 最大 token 数和 HF processor 配置。
- DummyInputsBuilder 用于显存 profiling 和最大输入预算估算。

## 3. 缓存边界

```text
Frontend Processor Cache
→ 以媒体 hash 复用昂贵的 HF preprocessing
→ 发送新特征或缓存引用
→ Engine / Worker Receiver Cache
→ 恢复 Worker 执行 encoder 所需特征
```

- Cache key 必须覆盖媒体内容和影响预处理结果的配置。
- Sender 与 Receiver 的命中状态必须同步，不能让接收端引用已淘汰条目。
- 共享内存实现降低大张量跨进程复制，但引入对象生命周期与清理问题。

## 4. 关键代码解析

### `MultiModalRegistry.register_processor()`

### `MultiModalRegistry.create_processor()`

### `MultiModalRegistry.processor_cache_from_config()`

### `MultiModalRegistry.engine_receiver_cache_from_config()`

### `BaseMultiModalCache.get_and_update()`

### `BaseMultiModalCache.clear_cache()`

### `BaseMultiModalProcessorCache.is_cached()`

### `BaseMultiModalProcessorCache.make_stats()`

## 5. 与调度缓存的区别

- Processor cache 保存预处理后的媒体输入或传输对象。
- Encoder cache 保存模型 encoder 已计算出的 embeddings。
- KV cache 保存语言模型 attention 的 key/value blocks。
- 三者键、容量单位和释放时机不同，不能混为一个缓存层。

## 6. 当前结论

多模态缓存的价值主要来自避免重复 CPU preprocessing、跨进程复制和 encoder forward；分析命中时必须明确命中的是哪一层结果。
