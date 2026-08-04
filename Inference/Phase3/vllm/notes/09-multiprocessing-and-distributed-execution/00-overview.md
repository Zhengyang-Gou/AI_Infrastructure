# Multiprocessing and Distributed Execution

## 学习目标

本阶段从单进程 Engine Core 扩展到多进程、多 GPU 和多节点执行，理解 Executor 如何管理 Worker、并行组如何组织 collective，以及 KV Connector 如何支持分离式部署。

## 阅读顺序

| 顺序 | 笔记 | 主要内容 |
| --- | --- | --- |
| 1 | `01-multiprocess-executor.md` | 本地 Worker 进程、消息队列和故障处理 |
| 2 | `02-parallel-state.md` | TP、PP、DP、EP 并行组与 collective |
| 3 | `03-distributed-coordination.md` | 数据并行协调、负载分发和跨节点资源 |
| 4 | `04-kv-transfer-and-disaggregated-serving.md` | KV 传输、Prefill/Decode 分离和 Connector 生命周期 |

## 主执行拓扑

```text
API Server
→ EngineCoreProc
→ MultiprocExecutor / RayDistributedExecutor
→ rank 0..N Worker
→ GPUModelRunner

各 rank 同时属于若干并行组：
TP / PP / DP / EP

分离式部署：
Prefill Worker → KV Connector → Decode Worker
```

## 完成标准

- 能说明 Engine Core、Executor、Worker 进程之间的控制与数据通道。
- 能区分 TP、PP、DP、EP 的切分维度和通信操作。
- 能解释多进程启动、健康检查、异常传播和退出顺序。
- 能追踪 KV 从 Prefill 实例传给 Decode 实例的接口边界。

## 当前结论

分布式执行层把一次模型调用拆成进程管理、rank 拓扑、collective 通信和可选 KV 传输四类问题；读代码时应先画进程与并行组，再追单个函数。
