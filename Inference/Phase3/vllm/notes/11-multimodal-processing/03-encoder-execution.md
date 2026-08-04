# Encoder Execution

## 1. 文件定位

- 调度侧：`vllm/v1/core/encoder_cache_manager.py`。
- Worker 侧：`vllm/v1/worker/gpu/mm/encoder_runner.py`、`model_states/default.py`。
- 职责：控制多模态 encoder 的 token 预算、缓存占用和 GPU 执行，并将 embeddings 放入语言模型输入。

## 2. 调度流程

```text
Scheduler 检查媒体 item 的 encoder cache
→ 命中：复用已有 embeddings
→ 未命中：检查本轮 encoder compute budget 与 cache space
→ 分配 cache entry 并调度 encoder input
→ Worker 运行 encoder
→ embeddings 按 request/item 缓存
→ placeholder positions 从缓存 gather
```

Encoder budget 防止单轮媒体计算占满执行预算；cache manager 则防止 embeddings 长期占用超出配置容量。

## 3. Worker 数据路径

- `prepare_mm_inputs()` 将请求级媒体 kwargs 搬到设备并组织 batch。
- `execute_mm_encoder()` 调用模型暴露的多模态 embedding 接口。
- 结果按 item 拆分并写入 Worker encoder cache。
- `gather_mm_embeddings()` 根据当前请求 placeholder 需要取回 embeddings。
- `get_inputs_embeds()` 将文本 token embeddings 与多模态 embeddings 合并。

## 4. 关键代码解析

### `compute_mm_encoder_budget()`

### `EncoderCacheManager.check_and_update_cache()`

### `EncoderCacheManager.can_allocate()`

### `EncoderCacheManager.allocate()`

### `EncoderCacheManager.free()`

### `EncoderRunner.prepare_mm_inputs()`

### `EncoderRunner.execute_mm_encoder()`

### `EncoderRunner.gather_mm_embeddings()`

### `EncoderRunner.get_inputs_embeds()`

## 5. 与 KV Cache 的关系

- Encoder embeddings 可跨 prompt chunk 复用，避免每个 chunk 重跑视觉 encoder。
- Embeddings 合入语言模型后，相关序列位置仍会产生普通 attention KV。
- Encoder cache 的媒体 item 生命周期与 KV block 生命周期独立。

## 6. 当前结论

多模态调度同时管理两种资源：encoder forward 的计算预算和 encoder embeddings 的缓存容量；二者都满足后才能调度新媒体 item。
