# InputBatch

## 1. 文件定位

- 文件路径：`vllm/v1/worker/gpu/input_batch.py`
- 所属层次：GPU Model Runner 的输入数据与批处理 kernel 层
- 核心职责：定义模型执行批次的数据结构，并提供准备 token、position、sequence length 和执行后状态更新的 Triton kernel。
- 在调用链中的位置：由 `GPUModelRunner.prepare_inputs()` 调用，输出随后进入 Attention 准备、模型 forward 和采样。

本文件把多个请求压缩成连续 Tensor，并用 `idx_mapping`、`query_start_loc` 等索引保留批次与请求状态之间的对应关系。

## 2. 核心类与数据

| 类 / 数据 | 作用 | 典型形状 |
| --- | --- | --- |
| `InputBuffers` | 预分配、跨 step 复用的 GPU Tensor | 按最大请求数和最大 token 数分配 |
| `InputBatch` | 描述当前 step 的完整批次 | dataclass |
| `idx_mapping` | `batch_idx → req_state_idx` | `[num_reqs]` |
| `query_start_loc` | 每个请求 token 在扁平批次中的前缀和边界 | `[num_reqs + 1]` |
| `input_ids` | 本轮送入模型的 token IDs | `[num_tokens_after_padding]` |
| `positions` | 每个 input token 的序列位置 | `[num_tokens_after_padding]` |
| `seq_lens` | 本轮计算后每个请求的序列长度 | `[num_reqs]` |
| `logits_indices` | 从 hidden states 中选取需要计算 logits 的位置 | `[total_num_logits]` |
| `expanded_idx_mapping` | 推测解码下每个 logit 对应的请求槽位 | `[total_num_logits]` |

## 3. 批次布局

多个请求的本轮 token 被连接成一个连续序列：

```text
request A tokens | request B tokens | request C tokens
0                qsl[1]              qsl[2]              qsl[3]
```

`query_start_loc[i]` 和 `query_start_loc[i + 1]` 给出第 `i` 个 batch 请求的 token 区间。`idx_mapping[i]` 再找到该请求在持久化 `RequestState` 中的槽位。

`num_tokens` 表示真实 token 数，`num_tokens_after_padding` 表示 CUDA Graph 或并行执行要求填充后的 token 数；类似地，`num_reqs_after_padding` 可能大于真实请求数。

## 4. 输入准备流程

### Prefill token

```text
prepare_prefill_inputs()
→ 读取每个请求的 prefill_len 和 num_computed_tokens
→ 从 all_token_ids 取出本轮尚未计算的 prompt token
→ 写入连续 input_ids
→ 保存推测解码需要的后续 prefill token
```

### Position 与 sequence length

```text
prepare_pos_seq_lens()
→ position = num_computed_tokens + 请求内偏移
→ seq_len = num_computed_tokens + query_len
→ 为 full CUDA graph 清零未使用的 seq_lens
```

### Decode 与 draft token

```text
combine_sampled_and_draft_tokens()
→ 将上轮 sampled token 写入 decode 输入
→ 将 draft tokens 写入后续位置
→ 计算每个请求需要采样的 logits_indices
```

## 5. 执行后状态更新

```text
get_num_sampled_and_rejected()
→ 结合 prefill 状态计算 sampled / rejected 数量

post_update()
→ 追加 sampled token 到 all_token_ids
→ 更新 last_sampled_tokens 与 total_len
→ 更新 penalty bin counts
→ 按 rejected 数量修正 num_computed_tokens
```

对于不采样的 pooling 或非最后 PP rank 路径，`post_update_num_computed_tokens()` 只按本轮 query length 增加已计算 token 数。

## 6. 输入与输出

### 输入

- 持久化的请求状态 Tensor：完整 token 序列、prefill 长度、已计算 token 数和 draft token。
- Scheduler 决定的请求顺序和每个请求本轮 token 数。
- `InputBuffers` 中预分配的目标 Tensor。

### 输出

- `InputBatch`：模型 forward、Attention 和 Sampler 共享的本轮批次描述。
- `input_ids`、`positions`、`seq_lens` 和 `logits_indices` 等 GPU Tensor。
- 执行后更新的请求 token、计数和长度状态。

### 状态变化

- 输入阶段在预分配 buffer 中写入本轮 token 和 position。
- 采样后把新 token 写回持久化 `RequestState` Tensor。
- 推测解码时根据拒绝 token 数修正已计算进度。

## 7. 关键代码解析

### `InputBuffers.__init__()`

### `InputBatch.make_dummy()`

### `prepare_prefill_inputs()`

### `prepare_pos_seq_lens()`

### `combine_sampled_and_draft_tokens()`

### `get_num_sampled_and_rejected()`

### `post_update()`

### `post_update_num_computed_tokens()`

### `expand_idx_mapping()`

## 8. 与其他文件的关系

- 上游：`GPUModelRunner.prepare_inputs()` 组织 Scheduler 数据并调用本文件函数。
- 请求状态：与 `vllm/v1/worker/gpu/states.py` 中的 `RequestState` 配合。
- Cache 映射：`InputBatch` 中的索引和 positions 传给 `block_table.py`。
- 下游：具体模型 forward、Attention metadata、Sampler 和状态回写逻辑。

## 9. 当前结论

`input_batch.py` 负责把“每个请求本轮执行多少 token”转换为紧凑、可复用且适合 GPU kernel 的批次表示，并在采样后高效更新持久化请求状态。
