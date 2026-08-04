# Speculative Execution Flow

## 1. 文件定位

- 路径：`vllm/v1/worker/gpu/model_runner.py`、`vllm/v1/spec_decode/metadata.py`、`vllm/v1/worker/gpu/spec_decode/utils.py`。
- 职责：把 draft tokens 合入下一轮 Target 输入，并在验证后更新请求状态、产生新候选。

## 2. 跨轮闭环

```text
第 N 轮 sample_tokens()
→ 基于 Target hidden states / history 调用 speculator.propose()
→ DraftTokensHandler 暂存候选
→ Executor.take_draft_token_ids() 交给 Scheduler

第 N+1 轮 SchedulerOutput
→ Model Runner 将候选加入 input IDs 和 positions
→ Target execute_model() 批量验证
→ sample() 进入 RejectionSampler
→ postprocess_sampled() 按接受/拒绝数更新状态
→ 再次 propose()
```

候选生成与本轮输出的异步 D2H copy 可以重叠，这是推测解码集成到执行流水线后的额外并行机会。

## 3. 关键状态

- `SpecDecodeMetadata` 描述待验证请求和候选布局。
- `num_sampled` 表示最终从该轮得到的 token 数。
- `num_rejected` 用于修正乐观推进的 computed token 和 KV 状态。
- `draft_tokens` 只属于下一轮候选，不能提前进入用户可见输出。
- PP 模式需要把多 token 采样结果广播给非末级 rank，保持各 stage 状态一致。

## 4. 关键代码解析

### `SpecDecodeMetadata.__post_init__()`

### `SpecDecodeMetadata.make_dummy()`

### `GPUModelRunner.execute_model()`

### `GPUModelRunner.sample()`

### `GPUModelRunner.sample_tokens()`

### `GPUModelRunner.postprocess_sampled()`

### `GPUModelRunner.take_draft_token_ids()`

### `DraftTokensHandler.set_draft_tokens()`

### `DraftTokensHandler.get_draft_tokens()`

## 5. 性能观察

- 接受长度：平均每次 Target 验证实际输出多少 token。
- 草稿开销：propose 的模型、kernel 和同步时间。
- 验证膨胀：一次 Target forward 计算了多少最终被拒绝的位置。
- 批处理影响：不同候选长度导致的 padding、形状变化和 CUDA Graph 覆盖。

## 6. 当前结论

推测解码不是采样器内的局部优化，而是跨两个调度轮次的状态机；读代码时必须把“本轮输出”和“下轮候选”分开追踪。
