# 10 · Distributed Execution

**多个设备和 rank 怎样分工、通信并保持执行一致？**

状态：边界页；正文待写。

## 1. 职责边界

| 项目 | 契约 |
| --- | --- |
| 输入 | 模型与并行配置、设备资源及单 Worker 的执行接口。 |
| 输出 | Worker 拓扑、参数分片、通信与跨 rank 执行协调。 |
| 本章负责 | TP/PP/DP/EP；进程组和 Worker 拓扑；张量与权重切分；集合通信；跨 rank 请求、批次与流水协调。 |

本章遇到其他模块只使用接口结果，不展开其实现。

| 内容 | 唯一归属 |
| --- | --- |
| 单核心客户端消息与后台循环 | 02 Engine Core |
| 单 Worker 通用权重读取与装载 | 05 Model and Weight Loading |
| 单 Worker 基础执行 | 06 Model Execution |
| 通用 batch queue 与优化算法 | 09 Runtime Optimizations |
| HTTP 与连接生命周期 | 11 Online Serving |

## 2. 正文范围

1. 并行配置与进程拓扑。
2. TP 与权重切分。
3. PP 与流水协调。
4. DP、EP 与跨 rank 通信。

[返回模块索引](../README.md) · [源码纵览](../00-Overview.md)
