# 11 · Online Serving

**在线请求怎样接入，怎样异步交付结果并处理连接生命周期？**

状态：边界页；正文待写。

## 1. 职责边界

| 项目 | 契约 |
| --- | --- |
| 输入 | HTTP 请求、连接状态及引擎返回的用户结果。 |
| 输出 | HTTP/SSE 响应、异步结果交付与必要的取消请求。 |
| 本章负责 | API 路由与请求适配；AsyncLLM 调用；异步结果队列和消费者；流式交付；断连、超时、背压与取消发起。 |

本章遇到其他模块只使用接口结果，不展开其实现。

| 内容 | 唯一归属 |
| --- | --- |
| 共用输入处理、输出状态与解码算法 | 01 Inference Entry |
| 核心客户端通信与执行循环 | 02 Engine Core |
| 接纳、预算与请求退场 | 03 Scheduler |
| AsyncScheduler、GPU 重叠与推测解码 | 09 Runtime Optimizations |
| 多 rank 拓扑与执行协调 | 10 Distributed Execution |

## 2. 正文范围

1. HTTP 接入与参数适配。
2. AsyncLLM 与异步结果交付。
3. SSE 流式响应。
4. 断连、超时、背压与取消。

[返回模块索引](../README.md) · [源码纵览](../00-Overview.md)
