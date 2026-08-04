# GPU Execution

## 学习目标

这一阶段关注 `SchedulerOutput` 离开 `EngineCore` 后，如何经过单进程 Executor、GPU Worker 和 V2 GPU Model Runner，最终完成输入张量准备、模型前向与采样。

完成本阶段后，应该能够说明：

1. `UniProcExecutor` 如何创建并调用唯一的 Worker。
2. GPU device、分布式环境、模型和 KV Cache 分别在哪里初始化。
3. `SchedulerOutput` 如何被转换为 `InputBatch`、block table 和 slot mapping。
4. Model Runner 如何维护新增、缓存、抢占和完成请求的 GPU 侧状态。
5. 模型前向在哪里真正发生，hidden states 如何交给采样逻辑。
6. 单卡主线中每一层对象的职责边界是什么。

本阶段只追踪单进程和 V2 GPU Model Runner 主线，不展开多进程 Executor、具体模型层、Attention kernel 和采样算法内部实现。

## 阅读顺序

| 顺序 | 文件 | 主要关注点 |
| --- | --- | --- |
| 1 | `vllm/v1/executor/uniproc_executor.py` | Executor 如何初始化 Worker，并转发执行与采样调用 |
| 2 | `vllm/v1/worker/gpu_worker.py` | GPU、模型、显存和 KV Cache 的初始化，以及 Worker 调用边界 |
| 3 | `vllm/v1/worker/gpu/model_runner.py` | 请求状态同步、批次准备、模型 forward 和采样主流程 |
| 4 | `vllm/v1/worker/gpu/input_batch.py` | Scheduler 数据如何变成模型使用的 Tensor 批次 |
| 5 | `vllm/v1/worker/gpu/block_table.py` | 逻辑 block table 如何整理为 Attention 使用的表和 slot mapping |

## 整体结构

```text
EngineCore
   ↓ SchedulerOutput
UniProcExecutor
   ↓ collective_rpc()
Worker
   ↓ execute_model()
GPUModelRunner
   ├─ 同步请求状态
   ├─ 构造 InputBatch
   ├─ 准备 block tables / slot mappings
   ├─ 调用模型 forward
   └─ 计算 logits 并采样
```

## 初始化调用链

```text
UniProcExecutor._init_executor()
→ WorkerWrapperBase.init_worker()
→ Worker.init_device()
→ 创建 GPUModelRunner
→ Worker.load_model()
→ GPUModelRunner.load_model()
→ ModelLoader.load_model()
```

模型加载后，Worker 会通过 profile run 估算可用于 KV Cache 的显存，然后根据 `KVCacheConfig` 让 Model Runner 创建 KV Cache、`BlockTables` 和 Attention 相关状态。

## 单轮执行调用链

```text
EngineCore
→ UniProcExecutor.execute_model(SchedulerOutput)
→ Worker.execute_model(SchedulerOutput)
→ GPUModelRunner.execute_model(SchedulerOutput)
→ finish_requests() / add_requests() / update_requests()
→ prepare_inputs()
→ prepare_attn()
→ set_forward_context(...)
→ self.model(...)
→ hidden_states
```

当前 V2 路径把采样从模型前向中分离：

```text
EngineCore
→ UniProcExecutor.sample_tokens(GrammarOutput)
→ Worker.sample_tokens(GrammarOutput)
→ GPUModelRunner.sample_tokens(GrammarOutput)
→ GPUModelRunner.sample(...)
→ model.compute_logits(...)
→ Sampler / RejectionSampler
→ ModelRunnerOutput
```

## 关键数据变化

| 数据 | 产生位置 | 作用 |
| --- | --- | --- |
| `SchedulerOutput` | Scheduler | 描述本轮请求、token、block 和完成状态 |
| `RequestState` | `GPUModelRunner` | 保存 GPU 侧请求 token 与执行进度 |
| `InputBatch` | `GPUModelRunner.prepare_inputs()` | 聚合模型 forward 所需的批次张量与索引 |
| `block_tables` | `BlockTables.gather_block_tables()` | 为本轮 batch 整理每个请求的物理 block ID |
| `slot_mappings` | `BlockTables.compute_slot_mappings()` | 将 token 位置映射到 KV Cache 物理 slot |
| `AttentionMetadata` | Model State | 组织具体 Attention backend 所需元数据 |
| `hidden_states` | 模型 forward | 最后一层隐藏状态，供 logits 与采样使用 |
| `ModelRunnerOutput` | `GPUModelRunner.sample_tokens()` | 将采样 token、logprobs 和 KV connector 输出返回 Scheduler |

## 文件职责边界

| 文件 | 一句话职责 |
| --- | --- |
| `uniproc_executor.py` | 在单进程模式下持有一个 Worker，并统一转发 Executor RPC |
| `gpu_worker.py` | 管理 GPU 运行环境和生命周期，并把执行工作委托给 Model Runner |
| `gpu/model_runner.py` | 把调度结果转换为模型输入，执行 forward，并协调采样和状态回写 |
| `gpu/input_batch.py` | 定义批次数据并提供批量生成 token、position 和状态更新的 GPU kernel |
| `gpu/block_table.py` | 保存请求 block table，并生成 Attention 使用的 block table 与 slot mapping |

## 当前阶段的检查清单

- [ ] 能画出 `EngineCore → Executor → Worker → GPUModelRunner` 调用链。
- [ ] 能说明 Worker 与 Model Runner 的职责差异。
- [ ] 能找到 GPU device、模型和 KV Cache 的初始化位置。
- [ ] 能说明 `SchedulerOutput` 如何变为 `InputBatch`。
- [ ] 能解释 `idx_mapping`、`query_start_loc`、`block_tables` 和 `slot_mappings` 的用途。
- [ ] 能找到模型 forward 的实际调用位置。
- [ ] 能说明 `execute_model()` 与 `sample_tokens()` 为什么是两步。

## 一句话总结

GPU 执行阶段把 Scheduler 描述的逻辑工作，经 Executor 和 Worker 转发为 GPUModelRunner 中的真实 Tensor 批次、KV Cache 地址、模型前向与采样输出。

## 补充专题

- `06-compilation-and-cuda-graphs.md`：补齐 `torch.compile`、图切分、编译缓存与 CUDA Graph 分发。
