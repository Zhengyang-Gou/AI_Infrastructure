# GPU Worker

## 1. 文件定位

- 文件路径：`vllm/v1/worker/gpu_worker.py`
- 所属层次：GPU 设备与模型运行生命周期管理层
- 核心职责：初始化 CUDA 和分布式环境，创建 Model Runner，加载模型，估算 KV Cache 显存，并转发执行与采样操作。
- 在调用链中的位置：位于 Executor 与 `GPUModelRunner` 之间。

Worker 负责“让 GPU 运行环境可用”，Model Runner 负责“把一轮调度真正变成模型计算”。这两个职责在本文件边界处明确分开。

## 2. 核心类与组件

| 类 / 组件 | 作用 | 生命周期阶段 |
| --- | --- | --- |
| `Worker` | GPU Worker 主类 | 初始化、加载、执行、关闭 |
| `GPUModelRunner` | 管理模型和 GPU 侧请求状态 | `init_device()` 中创建 |
| `MemorySnapshot` | 记录设备初始显存状态 | 设备初始化 |
| `memory_profiling()` | 通过 profile run 估算非 KV Cache 显存 | KV Cache 容量确定 |
| `KVCacheConfig` | 描述最终 KV Cache 分配方案 | Cache 初始化 |
| `AsyncIntermediateTensors` | 为 Pipeline Parallel 中间张量延迟等待通信 | PP 执行路径 |
| `WorkerSentinel` | 故障容错场景下处理 Worker 状态 | 可选功能 |

## 3. 初始化流程

```text
Worker.__init__()
→ 保存 VllmConfig 与 rank 信息
→ 配置 profiler、fault tolerance、sleep mode 等状态

Worker.init_device()
→ 计算实际 local rank
→ 选择并设置 CUDA device
→ 校验 dtype
→ 初始化分布式环境和模型并行组
→ 设置随机种子
→ 清理缓存并记录 MemorySnapshot
→ 创建 GPUModelRunner

Worker.load_model()
→ 进入 weights 内存池上下文
→ GPUModelRunner.load_model()
```

当前配置的 `use_v2_model_runner` 为真时，Worker 从 `vllm/v1/worker/gpu/model_runner.py` 创建新版 `GPUModelRunner`；否则会选择旧版 `gpu_model_runner.py`。

## 4. 显存与 KV Cache 流程

```text
Worker.determine_available_memory()
→ GPUModelRunner.profile_run()
→ 记录模型前向峰值与非 KV Cache 显存
→ 估算 CUDA Graph 显存
→ requested_memory - non_kv_cache_memory
→ 可用于 KV Cache 的字节数
```

如果用户显式设置 `kv_cache_memory_bytes`，Worker 仍会执行 profile run 以触发模型编译，但直接采用指定容量，而不按 `gpu_memory_utilization` 自动推导。

```text
Worker.initialize_from_config(kv_cache_config)
→ 更新 cache_config.num_gpu_blocks
→ 初始化 KV transfer connector
→ GPUModelRunner.initialize_kv_cache()
→ 必要时初始化 KV block zeroing 元数据
```

## 5. 主执行流程

### 模型前向

```text
Executor.execute_model()
→ Worker.execute_model(scheduler_output)
→ 等待前一轮非阻塞 PP send
→ 必要时接收上一 PP rank 的 IntermediateTensors
→ GPUModelRunner.execute_model()
→ 返回 ModelRunnerOutput、IntermediateTensors 或 None
```

Pipeline Parallel 的非最后 rank 会发送 `IntermediateTensors` 给下一 rank；普通单卡路径则直接由 Model Runner 保存 hidden states，等待后续采样调用。

### 采样

```text
Executor.sample_tokens()
→ Worker.sample_tokens(grammar_output)
→ GPUModelRunner.sample_tokens(grammar_output)
→ ModelRunnerOutput / AsyncModelRunnerOutput
```

## 6. 输入与输出

### 输入

- 初始化输入：`VllmConfig`、`rank`、`local_rank` 和分布式初始化地址。
- 执行输入：Scheduler 产生的 `SchedulerOutput`。
- 采样输入：结构化输出使用的 `GrammarOutput`。
- Cache 输入：Engine Core 计算出的 `KVCacheConfig`。

### 输出

- `determine_available_memory()` 返回可分配给 KV Cache 的字节数。
- `get_kv_cache_spec()` 返回模型层的 KV Cache 需求。
- `execute_model()` 返回模型运行结果、PP 中间张量或空值。
- `sample_tokens()` 返回同步或异步的 `ModelRunnerOutput`。

### 状态变化

- CUDA device、分布式通信组和随机种子完成初始化。
- `model_runner` 持有模型权重、请求状态和 KV Cache Tensor。
- 初始化阶段记录显存快照，并据此确定 KV Cache 容量。
- PP 模式下保存尚未完成的异步 send handle。

## 7. 关键代码解析

### `Worker.__init__()`

### `Worker.init_device()`

### `Worker.load_model()`

### `Worker.determine_available_memory()`

### `Worker.get_kv_cache_spec()`

### `Worker.initialize_from_config()`

### `Worker.compile_or_warm_up_model()`

### `Worker.execute_model()`

### `Worker.sample_tokens()`

### `Worker.shutdown()`

### `init_worker_distributed_environment()`

## 8. 与其他文件的关系

- 上游：`vllm/v1/executor/uniproc_executor.py` 或其他 Executor 实现。
- 下游：V2 路径为 `vllm/v1/worker/gpu/model_runner.py`。
- 初始化依赖：分布式环境、平台实现、显存分析和 workspace 管理器。
- 向下传递：`SchedulerOutput`、`GrammarOutput` 和 `KVCacheConfig`。
- 向上返回：可用 KV Cache 容量与 `ModelRunnerOutput`。

## 9. 当前结论

`Worker` 是 GPU 生命周期管理者：它完成设备、通信、模型和 Cache 的初始化，并在运行阶段把 Executor 调用交给 Model Runner；具体 Tensor 批次和模型 forward 不在 Worker 中实现。
