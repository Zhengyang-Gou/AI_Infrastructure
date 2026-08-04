# UniProcExecutor

## 1. 文件定位

- 文件路径：`vllm/v1/executor/uniproc_executor.py`
- 所属层次：核心引擎与 Worker 之间的单进程执行层
- 核心职责：在当前进程中初始化唯一的 driver Worker，并实现 Executor 接口所需的调用转发。
- 在调用链中的位置：位于 `EngineCore` 与 `Worker` 之间。

`UniProcExecutor` 不准备模型输入，也不直接调用 CUDA kernel。它把统一的 Executor 操作转换成对 `WorkerWrapperBase` 的本地方法调用。

## 2. 核心类与组件

| 类 / 组件 | 作用 | 关注点 |
| --- | --- | --- |
| `UniProcExecutor` | 单进程 Executor 实现 | 初始化 Worker、转发模型执行和采样 |
| `WorkerWrapperBase` | 包装实际 Worker 对象 | 提供统一的 Worker 初始化与方法调用入口 |
| `AsyncOutputFuture` | 把异步 Model Runner 输出适配成 `Future` | 延迟取得 `AsyncModelRunnerOutput` 的真实结果 |
| `ExecutorWithExternalLauncher` | 支持 torchrun 等外部 launcher | 每个 Executor 仍只创建一个 Worker，但 rank 由环境变量决定 |
| `run_method()` | 按方法名或 callable 调用 Worker | `collective_rpc()` 的本地调用实现 |

## 3. 初始化流程

```text
UniProcExecutor._init_executor()
→ 创建 WorkerWrapperBase(rpc_rank=0)
→ _distributed_args()
→ 组装 Worker 初始化参数
→ WorkerWrapperBase.init_worker()
→ WorkerWrapperBase.init_device()
→ WorkerWrapperBase.load_model()
→ 根据 backend 更新 block size
```

普通单进程路径的全局 rank 固定为 `0`。`local_rank` 优先取设备字符串中的显式索引，否则使用 `0`；分布式初始化地址则使用本机 IP 和空闲端口构造。

## 4. 主执行流程

### 模型前向

```text
EngineCore
→ UniProcExecutor.execute_model(scheduler_output)
→ collective_rpc("execute_model", single_value=True)
→ Worker.execute_model(scheduler_output)
→ ModelRunnerOutput / Future / None
```

### 采样

```text
EngineCore
→ UniProcExecutor.sample_tokens(grammar_output)
→ collective_rpc("sample_tokens", single_value=True)
→ Worker.sample_tokens(grammar_output)
→ ModelRunnerOutput / Future / None
```

`collective_rpc()` 保留了其他 Executor 的统一接口形式。单进程只有一个 Worker，因此 `single_value=False` 时将结果包装为单元素列表，`single_value=True` 时直接返回该 Worker 的结果。

## 5. 输入与输出

### 输入

- `SchedulerOutput`：本轮需要执行的请求、token、KV block 更新和完成请求信息。
- `GrammarOutput`：结构化输出采样所需的 grammar 约束。
- 字符串方法名或 callable，以及对应的 `args`、`kwargs`。

### 输出

- 同步模式返回单个值或单元素列表。
- 非阻塞模式返回 `Future`。
- 若 Worker 返回 `AsyncModelRunnerOutput`，同步路径会立即取出真实输出，非阻塞路径由 `AsyncOutputFuture` 延迟取出。

### 状态变化

- 初始化后，`driver_worker` 持有实际 Worker。
- 模型权重和设备资源由 Worker 初始化并持有。
- `shutdown()` 将资源释放操作继续转交给 Worker。

## 6. 关键代码解析

### `AsyncOutputFuture.result()`

### `UniProcExecutor._init_executor()`

### `UniProcExecutor._distributed_args()`

### `UniProcExecutor.collective_rpc()`

### `UniProcExecutor.execute_model()`

### `UniProcExecutor.sample_tokens()`

### `UniProcExecutor.take_draft_token_ids()`

### `UniProcExecutor.shutdown()`

### `ExecutorWithExternalLauncher._distributed_args()`

## 7. 与其他文件的关系

- 上游：`vllm/v1/engine/core.py` 和 Executor 抽象接口。
- 下游：`vllm/v1/worker/worker_base.py` 创建的具体 Worker，GPU 路径最终进入 `gpu_worker.py`。
- 向下传递：`SchedulerOutput`、`GrammarOutput` 和控制操作。
- 向上返回：`ModelRunnerOutput`、异步 `Future` 或管理操作结果。

## 8. 当前结论

`UniProcExecutor` 是单进程调用适配器：它负责创建一个 Worker，并把 Engine Core 的统一 Executor 调用直接转发给该 Worker，本身不参与批次准备和模型计算。
