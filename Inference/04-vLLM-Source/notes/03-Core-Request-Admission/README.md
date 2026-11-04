# 03 · 请求进入核心，等待执行

从核心接收请求到进入第一次调度。

## 1. 本章在推理引擎中的位置

```mermaid
flowchart TD
    S1["01 创建引擎与准备环境<br/>LLM(...)"]
    S2["02 提交一批离线请求<br/>LLM.generate(prompts)"]
    S3["03 核心接收并入队 ← 本章<br/>EngineCore → Scheduler"]
    S4["04 决定本轮执行内容<br/>选择请求与安排 KV"]
    S5["05 准备模型输入<br/>Executor → Worker → Model Runner"]
    S6["06 模型前向与采样<br/>模型 → logits → token"]
    S7["07 回写请求状态<br/>继续执行或释放资源"]
    S8["08 解码与收集结果<br/>OutputProcessor → RequestOutput"]
    R["返回本次 generate 的结果"]
    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    S7 -->|未完成请求继续| S4
    S7 -->|增量 token 与结束信息| S8
    S8 -->|停止字符串触发取消| S7
    S8 -->|收齐全部结果| R
    classDef current fill:#dbeafe,stroke:#2563eb,stroke-width:3px,color:#0f172a,font-weight:bold;
    class S3 current;
```

## 2. 本章调用路线与用途

```mermaid
flowchart TD
    A["InprocClient.add_request<br/>接收已构造的核心请求"]
    B["EngineCore.preprocess_add_request<br/>准备核心使用的请求对象"]
    C["Request.from_engine_core_request<br/>建立可持续更新的 Request"]
    D["EngineCore.add_request<br/>检查后交给调度器"]
    E["Scheduler.add_request<br/>按 ID 登记请求"]
    F["Scheduler._enqueue_waiting_request<br/>将请求放入等待队列"]
    G["LLMEngine.step<br/>开始一次结果处理"]
    H["InprocClient.get_output<br/>调用核心执行方法并取输出"]
    I["EngineCore.step<br/>进入本轮调度与执行"]
    A --> B --> C --> D --> E --> F
    F -->|请求等待执行| G --> H --> I
```

## 3. 章节范围

- 核心客户端与请求交接。
- EngineCoreRequest 转为核心 Request。
- 请求表、等待队列与初始状态。
- 核心执行循环的驱动与调度入口。

本章正文待迁移与整理。

[总目录](../README.md) · [固定配置](../00-Reading-Setup.md) · [全局执行路线](../Overview.md) · [上一章：提交一批离线请求](../02-Offline-Request-Submission/README.md) · [下一章：决定本轮执行哪些请求](../04-Step-Scheduling/README.md)
