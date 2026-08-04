# GPUModelRunner

## 1. 文件定位

- 文件路径：`vllm/v1/worker/gpu/model_runner.py`
- 所属层次：GPU 模型执行核心层
- 核心职责：维护 GPU 侧请求状态，把 `SchedulerOutput` 转换为模型输入和 Attention 元数据，运行模型 forward，并协调 logits、采样与状态回写。
- 在调用链中的位置：位于 GPU Worker 与具体 PyTorch 模型、Attention backend、Sampler 之间。

该文件是 V2 Model Runner 主线。它只保留不同模型共同需要的执行逻辑，模型专属行为由 Model State、具体模型文件和其他辅助组件承担。

## 2. 核心类与组件

| 类 / 组件 | 作用 | 主要数据 |
| --- | --- | --- |
| `GPUModelRunner` | 协调整轮 GPU 执行 | 配置、模型、请求状态和 Cache |
| `RequestState` | 保存所有活跃请求的 GPU 侧状态 | token、长度、计算进度和 draft token |
| `InputBuffers` | 预分配可复用的输入 Tensor | input IDs、positions、seq lens |
| `InputBatch` | 描述当前执行批次 | 请求映射、token 布局、采样索引 |
| `BlockTables` | 管理请求到 KV block 的映射 | block IDs 与 slot mappings |
| Model State | 适配不同模型类型的公共执行接口 | Attention 元数据和模型附加输入 |
| `Sampler` | 对 logits 执行普通采样 | sampled token IDs |
| `RejectionSampler` | 推测解码时验证 draft token | 接受与拒绝数量 |
| `ExecuteModelState` | 在 forward 与 sampling 两步之间暂存状态 | batch、hidden states 和 Attention 元数据 |

## 3. 初始化与模型加载

```text
GPUModelRunner.__init__()
→ 保存各类 VllmConfig
→ 初始化并行、KV、LoRA、多模态和推测解码状态
→ 创建 RequestState
→ 创建 InputBuffers
→ 准备采样与执行状态占位

GPUModelRunner.load_model()
→ get_model_loader()
→ ModelLoader.load_model()
→ 初始化 Model State
→ 创建 Sampler / RejectionSampler
→ 创建 PromptLogprobsWorker
→ 创建 PoolingRunner（如需要）
```

模型加载使用 `DeviceMemoryProfiler` 记录权重显存。真正的具体模型类选择与权重加载由 Model Loader 和 Model Registry 完成，本阶段只关注它们的调用边界。

## 4. 请求状态同步

每次真实执行开始时，Model Runner 先让 GPU 侧状态与 `SchedulerOutput` 对齐：

```text
finish_requests()
→ 删除 finished / preempted 请求状态

add_requests()
→ 添加 scheduled_new_reqs
→ 写入 token、模型状态、block table、LoRA 和采样状态

update_requests()
→ 更新 cached request 的 num_computed_tokens
→ 追加新 block IDs
→ 处理 Cache block 清零与 copy-on-write
```

`req_id_to_index` 把外部字符串请求 ID 映射到预分配状态数组的整数槽位。`InputBatch.idx_mapping` 再把本轮 batch 行号映射到这些请求槽位。

## 5. 输入与 Attention 准备

```text
prepare_inputs(scheduler_output, batch_desc)
→ 排序本轮 req_ids
→ 构造 idx_mapping
→ 计算 num_scheduled_tokens
→ 计算 query_start_loc
→ 写入 prefill / sampled / draft token
→ 生成 positions 与 seq_lens
→ 生成 logits_indices
→ 返回 InputBatch
```

```text
prepare_attn(input_batch)
→ BlockTables.gather_block_tables()
→ BlockTables.compute_slot_mappings()
→ 返回 block_tables 与 slot_mappings
```

随后 Model State 根据 `InputBatch`、block tables、slot mappings 和 Cache 配置构造具体 Attention backend 所需的元数据。

## 6. 模型前向主流程

```text
GPUModelRunner.execute_model()
→ 同步请求状态
→ 选择 eager / piecewise graph / full CUDA graph
→ prepare_inputs()
→ prepare_attn()
→ ModelState.prepare_attn()
→ 准备 input_ids、positions、inputs_embeds
→ set_forward_context(...)
→ self.model(**model_inputs)
→ 保存 hidden_states 到 ExecuteModelState
```

普通 eager 路径直接调用 `self.model(**model_inputs)`；piecewise 和 full graph 路径通过 `ModelCudaGraphManager` 执行。无论采用哪种路径，最后一个 Pipeline Parallel rank 都得到 hidden states，其他 rank 返回 `IntermediateTensors`。

## 7. 采样与状态回写

```text
GPUModelRunner.sample_tokens(grammar_output)
→ 取出 ExecuteModelState
→ sample(hidden_states, input_batch, grammar_output)
→ 选取 logits_indices 对应 hidden states
→ self.model.compute_logits()
→ 应用 grammar bitmask（可选）
→ Sampler / RejectionSampler
→ 构造异步输出
→ postprocess_sampled()
→ 更新 RequestState
→ 返回 AsyncOutput / ModelRunnerOutput
```

前向与采样拆分后，`ExecuteModelState` 是两者之间的桥梁。采样后，`postprocess_sampled()` 更新已计算 token 数、最后采样 token、完整 token 序列和 penalty 计数等 GPU 状态。

## 8. 输入与输出

### 输入

- `VllmConfig` 和当前 `torch.device`。
- Scheduler 产生的 `SchedulerOutput`。
- 结构化输出采样使用的 `GrammarOutput`。
- Pipeline Parallel 场景中的 `IntermediateTensors`。

### 输出

- `execute_model()`：最后 rank 通常返回 `None` 并保存执行状态；非最后 rank 返回 `IntermediateTensors`。
- `sample_tokens()`：返回 `AsyncOutput`、`ModelRunnerOutput` 或空值。
- `get_kv_cache_spec()`：返回模型各层的 Cache 需求。

### 状态变化

- 活跃请求在 `RequestState` 中新增、更新或移除。
- block table 根据 Scheduler 分配结果追加物理 block ID。
- 模型前向结果暂存在 `ExecuteModelState`。
- 采样结果写回请求 token、长度和推测解码状态。

## 9. 关键代码解析

### `GPUModelRunner.__init__()`

### `GPUModelRunner.load_model()`

### `GPUModelRunner.initialize_kv_cache()`

### `GPUModelRunner.finish_requests()`

### `GPUModelRunner.add_requests()`

### `GPUModelRunner.update_requests()`

### `GPUModelRunner.prepare_inputs()`

### `GPUModelRunner.prepare_attn()`

### `GPUModelRunner.execute_model()`

### `GPUModelRunner.sample()`

### `GPUModelRunner.sample_tokens()`

### `GPUModelRunner.postprocess_sampled()`

### `sort_batch_req_ids()`

## 10. 与其他文件的关系

- 上游：`vllm/v1/worker/gpu_worker.py`。
- 输入准备：`vllm/v1/worker/gpu/input_batch.py`。
- Cache 地址准备：`vllm/v1/worker/gpu/block_table.py`。
- 模型加载：`vllm/model_executor/model_loader` 与 Model Registry。
- 模型执行：`vllm/model_executor/models/*.py`。
- Attention：Model State、Attention backend 和 forward context。
- 采样：`vllm/v1/worker/gpu/sample/sampler.py` 等组件。

## 11. 当前结论

`GPUModelRunner` 是 Scheduler 与真实 GPU 计算之间的核心转换层：它同步请求状态、准备 Tensor 与 KV 地址、执行模型，并将 hidden states 转换为可返回 Scheduler 的采样结果。
