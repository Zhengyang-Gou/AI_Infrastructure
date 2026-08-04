# Multiprocess Executor

## 1. 文件定位

- 路径：`vllm/v1/executor/multiproc_executor.py`。
- 职责：在本机创建和管理 Worker 进程，通过共享消息队列广播调用并收集结果。

## 2. 主执行流程

```text
MultiprocExecutor 初始化
→ 创建 request / response MessageQueue
→ 为各 rank 启动 WorkerProc
→ Worker 加载设备、模型与 ModelRunner
→ execute_model() 广播任务
→ Worker busy loop 执行方法
→ driver rank 返回 ModelRunnerOutput
```

`collective_rpc()` 是统一远程调用入口；模型执行和采样在其上增加了固定的方法名、返回 rank 和异步结果处理。

## 3. 进程生命周期

- 启动：确定 world size、rank 和设备映射，再等待各 Worker ready。
- 运行：请求队列负责广播，响应队列按调用收集结果。
- 监控：death pipe 和 Worker monitor 将子进程异常传回 Executor。
- 退出：先通知 Worker 停止，再回收进程、队列和监控线程。

## 4. 关键代码解析

### `MultiprocExecutor._init_executor()`

### `MultiprocExecutor.execute_model()`

### `MultiprocExecutor.collective_rpc()`

### `MultiprocExecutor.shutdown()`

### `MultiprocExecutor.check_health()`

### `WorkerProc.make_worker_process()`

### `WorkerProc.worker_main()`

### `WorkerProc.worker_busy_loop()`

## 5. 与其他文件的关系

- 上游：Engine Core 根据 `distributed_executor_backend` 选择 Executor。
- 下游：每个进程持有 Worker 和 Model Runner。
- 并列实现：Ray Executor 面向多节点和 Ray placement group。

## 6. 当前结论

`MultiprocExecutor` 是本地多 GPU 的控制平面：它不实现模型计算，而是保证同一调用可靠地到达所有相关 rank，并把输出和故障带回 Engine Core。
