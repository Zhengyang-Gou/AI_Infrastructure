# Distributed Coordination

## 1. 文件定位

- 路径：`vllm/v1/engine/coordinator.py`、`vllm/v1/executor/ray_executor.py`、`ray_executor_v2.py`、`ray_utils.py`。
- 职责：协调多个数据并行 Engine、管理跨节点 Worker 资源，并决定请求送往哪个 Engine。

## 2. 数据并行协调

```text
Frontend / AsyncLLM
→ DP Coordinator 输入 socket
→ 收集各 Engine 的队列与负载状态
→ 选择目标 Engine
→ 转发请求
→ Engine 发布状态和输出
```

Coordinator 管的是 Engine 副本之间的请求分配，不替代模型内部 TP/PP collective。两者可以叠加，例如每个 DP Engine 内部再使用 TP。

## 3. 多节点执行

- Ray placement group 预留跨节点 GPU 资源并固定 actor 布局。
- Ray Executor 创建远端 Worker actor，保持 rank、节点和设备映射。
- 模型调用仍遵守 Executor 接口，因此 Engine Core 不必理解具体 actor 通信。
- 节点故障、actor 异常和资源不足最终需要回传到 Engine Core。

## 4. 关键代码解析

### `DPCoordinator.__init__()`

### `DPCoordinator.get_engine_socket_addresses()`

### `DPCoordinator.shutdown()`

### `DPCoordinatorProc.run_coordinator()`

### `DPCoordinatorProc.process_input_socket()`

### `initialize_ray_cluster()`

### `RayDistributedExecutor._init_executor()`

### `RayDistributedExecutor.execute_model()`

### `RayDistributedExecutor.collective_rpc()`

## 5. 与其他文件的关系

- 上游：在线 Engine Client 和 API Server。
- 横向：Coordinator 处理 DP 副本，parallel state 处理副本内部并行组。
- 下游：Multiproc 或 Ray Worker 最终都进入相同 Worker/Model Runner 接口。

## 6. 当前结论

数据并行扩展包含两层路由：外层把请求分给 Engine，内层 Executor 把一次模型步广播给组成该 Engine 的 Worker。
