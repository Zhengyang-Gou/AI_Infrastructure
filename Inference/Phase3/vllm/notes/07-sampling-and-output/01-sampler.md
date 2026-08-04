# GPU Sampler

## 1. 学习目标

理解 `Sampler` 如何把一个 batch 中不同请求的 SamplingParams 转换成 GPU 上的批量 logits 处理与采样操作，并产生供 Model Runner 回传的 `SamplerOutput`。

## 2. 文件定位

- 文件路径：`vllm/v1/worker/gpu/sample/sampler.py`
- 所属层次：GPU Model Runner 的采样层
- 核心职责：维护逐请求采样状态，处理 logits，选择采样实现，并生成 token IDs 与 logprobs。
- 在调用链中的位置：位于模型的 `compute_logits()` 与 `ModelRunnerOutput` 之间。

本文件接收的已经是 logits。hidden states 到 logits 的转换由 GPU Model Runner 调用模型的 `compute_logits()` 完成。

## 3. 核心类与组件

| 类 / 组件 | 作用 |
| --- | --- |
| `Sampler` | 协调整个 batch 的 logits 处理、采样和 logprobs 计算 |
| `SamplingStates` | 保存 temperature、min-p、top-k、top-p、seed 等逐请求状态 |
| `PenaltiesState` | 应用 presence、frequency、repetition 等 penalty |
| `LogitBiasState` | 应用 logit bias、allowed token 与 min tokens 等约束 |
| `BadWordsState` | 屏蔽当前请求不允许生成的 bad words |
| `LogprobTokenIdsState` | 保存额外需要返回 logprob 的 token IDs |
| `InputBatch` | 提供请求索引映射、位置、输入 token 与 logits 位置 |
| `SamplerOutput` | 保存采样 token、logprobs 和 NaN / speculative 统计 |

各状态组件按 request index 保存参数，使同一 GPU batch 中的请求能够使用不同 SamplingParams。

## 4. 主执行流程

### 请求加入采样器

```text
Sampler.add_request(req_idx, prompt_len, sampling_params)
→ SamplingStates.add_request()
→ PenaltiesState.add_request()
→ LogitBiasState.add_request()
→ BadWordsState.add_request()
→ LogprobTokenIdsState.add_request()
→ apply_staged_writes() 将暂存状态同步到执行缓冲区
```

### 一轮采样

```text
Sampler.__call__(logits, input_batch)
→ 取得 batch 的请求索引和位置映射
→ 判断是否需要返回 logprobs
→ Sampler.sample()
→ Sampler.apply_sampling_params()
→ logit bias
→ penalties
→ bad words mask
→ temperature
→ min-p
→ top-k / top-p
→ FlashInfer sampler 或 Gumbel sampler
→ 可选 compute_topk_scores()
→ SamplerOutput
```

`sample()` 会先处理除 top-k/top-p 外的采样参数，再根据请求状态决定是否使用 FlashInfer 采样路径；否则显式应用 top-k/top-p 后进入 Gumbel 采样路径。greedy、显式 seed 或特定 logprobs 模式会影响路径选择。

## 5. 输入与输出

### 输入

- `logits`：形状对应本轮待采样位置的词表分数 Tensor。
- `input_batch`：包含请求到 logits 行的映射、位置、input IDs 与 batch 元数据。
- 预先注册的逐请求 `SamplingParams` 状态。

### 输出

`SamplerOutput` 包含：

- `sampled_token_ids`：二维 GPU Tensor，每个请求对应本轮生成 token。
- `logprobs_tensors`：按请求要求计算的 logprobs，未请求时为 `None`。
- `num_nans`：启用相关统计时记录 logits 中的 NaN 数。
- `num_sampled` 与 `num_rejected`：描述实际采样与 speculative rejection 数量。

### 状态变化

- `add_request()` 将新请求参数写入多个逐请求状态对象。
- `apply_staged_writes()` 把暂存修改应用到采样状态缓冲区。
- 采样本身返回新 Tensor，不直接修改 Scheduler 中的 `Request`。

## 6. 关键代码解析

### `Sampler.__init__()`

### `Sampler.add_request()`

### `Sampler.apply_staged_writes()`

### `Sampler.__call__()`

### `Sampler.apply_sampling_params()`

### `Sampler._requires_logits_processing()`

### `Sampler.sample()`

## 7. 与其他文件的关系

- 上游：`vllm/v1/worker/gpu/model_runner.py` 选择 hidden states、计算 logits 并调用 Sampler。
- 参数载体：`vllm/sampling_params.py` 中的 `SamplingParams`。
- batch 数据：`vllm/v1/worker/gpu/input_batch.py`。
- 低层采样：`vllm/v1/sample/ops/topk_topp_sampler.py` 与 `vllm/v1/worker/gpu/sample/gumbel.py`。
- 输出结构：`vllm/v1/worker/gpu/sample/output.py` 中的 `SamplerOutput`。
- 下游：Model Runner 将结果整理为 `ModelRunnerOutput`，再由 Scheduler 回写请求状态。

## 8. 当前结论

`Sampler` 是批量采样协调器。它把逐请求 SamplingParams 组织为 GPU 状态，按固定顺序处理 logits，并选择适合当前 batch 的采样实现，最终返回 token IDs 与可选 logprobs。
