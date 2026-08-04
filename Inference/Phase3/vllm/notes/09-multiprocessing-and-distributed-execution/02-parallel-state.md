# Parallel State

## 1. 文件定位

- 路径：`vllm/distributed/parallel_state.py`、`communication_op.py`。
- 职责：初始化分布式环境，创建不同维度的进程组，并为模型层提供统一 collective 接口。

## 2. 并行维度

| 维度 | 切分对象 | 典型通信 |
| --- | --- | --- |
| TP | 层内权重、激活或 attention heads | all-reduce、all-gather、reduce-scatter |
| PP | 连续模型层 | send、recv |
| DP | 请求批次或 Engine 副本 | 调度协调、状态聚合 |
| EP | MoE experts | dispatch、combine、all-to-all 类操作 |

同一 rank 可以同时位于多个组中。分析通信前应先确定 tensor 当前沿哪个并行维度被切分。

## 3. 初始化与调用链

```text
init_distributed_environment()
→ 初始化 world process group
→ initialize_model_parallel()
→ 构造 TP / PP / DP / EP GroupCoordinator
→ 模型层调用 communication_op
→ 定位对应 GroupCoordinator
→ 后端执行 collective
```

`GroupCoordinator` 屏蔽 NCCL、PyTorch process group 和自定义 communicator 的差异，并记录组内 rank 关系。

## 4. 关键代码解析

### `init_distributed_environment()`

### `initialize_model_parallel()`

### `get_tp_group()`

### `get_pp_group()`

### `get_dp_group()`

### `get_ep_group()`

### `GroupCoordinator.all_reduce()`

### `GroupCoordinator.all_gather()`

### `GroupCoordinator.reduce_scatter()`

### `GroupCoordinator.broadcast_tensor_dict()`

### `tensor_model_parallel_all_reduce()`

### `tensor_model_parallel_all_gather()`

## 5. 与模型代码的关系

- Column/Row Parallel Linear 使用 TP collective 拼接或归约结果。
- Pipeline stage 使用前后 rank 传递中间激活。
- MoE 层借助 EP 组将 token 派发给不同 expert。
- Executor 决定进程与 rank，parallel state 决定 rank 间逻辑关系。

## 6. 当前结论

并行状态是分布式模型代码的坐标系；任何 collective 都应同时记录输入张量形状、切分维度、组内 rank 和通信后的形状。
