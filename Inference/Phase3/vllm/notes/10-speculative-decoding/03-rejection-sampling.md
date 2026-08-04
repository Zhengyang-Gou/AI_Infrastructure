# Rejection Sampling

## 1. 文件定位

- 路径：`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`、`rejection_sampler_utils.py`。
- 兼容路径：`vllm/v1/sample/rejection_sampler.py`。
- 职责：用 Target logits 验证 draft tokens，返回每个请求最终接受的 token 序列及拒绝数量。

## 2. 验证语义

- Greedy：draft token 与 Target argmax 连续一致时接受；首个不一致位置输出 Target token 并停止接受后续候选。
- Random：按 Target 与 Draft 概率之比决定是否接受；拒绝时从校正后的残差分布采样，使最终分布仍等价于 Target。
- Bonus token：若所有候选均被接受，可以再使用 Target 对下一个位置的预测。

接受一定是前缀语义：某位置被拒绝后，建立在它之上的后续 draft tokens 全部失效。

## 3. 张量布局

```text
不同请求的 draft 长度
→ 展平为待验证 token positions
→ 对 Target logits 应用 penalty / temperature / constraint
→ 分块执行验证以控制临时显存
→ 生成 flattened sampled tokens
→ 统计每请求 num_sampled / num_rejected
```

Logprobs 必须对应最终保留 token，而不是简单返回所有 draft 位置的 Target logits。

## 4. 关键代码解析

### `RejectionSampler.__call__()`

### `RejectionSampler._verify()`

### `RejectionSampler._verify_in_chunks()`

### `rejection_sample()`

### `RejectionSampler.forward()`

### `RejectionSampler.apply_logits_processors()`

### `RejectionSampler.apply_penalties()`

### `apply_sampling_constraints()`

### `sample_recovered_tokens()`

## 5. 正确性检查

- 比较关闭和开启 speculative decoding 时固定 seed 的采样语义。
- 分别覆盖 greedy、temperature、top-k/top-p 和 logits processor。
- 覆盖请求具有不同 draft 长度、零候选和全部接受的批次。
- 检查拒绝后的 token history、penalty state、logprobs 与 KV 位置。

## 6. 当前结论

拒绝采样是推测解码的正确性边界：proposer 可以近似 Target，但最终保留 token 的分布与状态必须由验证器恢复为 Target 语义。
