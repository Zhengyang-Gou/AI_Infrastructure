# 04 · 决定本轮执行哪些请求

从未完成请求与资源状态到本轮 SchedulerOutput。

## 1. 本章在推理引擎中的位置

```mermaid
flowchart TD
    S1["01 创建引擎与准备环境<br/>LLM(...)"]
    S2["02 提交一批离线请求<br/>LLM.generate(prompts)"]
    S3["03 核心接收并入队<br/>EngineCore → Scheduler"]
    S4["04 决定本轮执行内容 ← 本章<br/>选择请求与安排 KV"]
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
    class S4 current;
```

## 2. 本章调用路线与用途

```mermaid
flowchart TD
    A["Scheduler.schedule<br/>准备本轮 token 预算"]
    B["遍历 running<br/>计算运行请求的本轮需求"]
    C["KVCacheManager.allocate_slots<br/>为运行请求申请所需 KV 块"]
    D{"分配成功？"}
    E["登记本轮计算量<br/>记录 token 数，扣减预算"]
    P["Scheduler._preempt_request<br/>释放引用，将请求放回等待队列"]
    Q{"当前请求已被抢占？"}
    X{"还有运行请求和预算？"}
    W{"无本轮抢占，且有等待请求、预算和名额？"}
    F["KVCacheManager.get_computed_blocks<br/>查找等待请求可复用的前缀"]
    G["KVCacheManager.allocate_slots<br/>检查接纳条件并申请 KV 块"]
    H{"可以接纳？"}
    I["接纳等待请求<br/>移入 running，登记计算量并扣减预算"]
    J["构造 SchedulerOutput<br/>打包本轮请求、token 数和缓存信息"]
    K["Scheduler._update_after_schedule<br/>推进计算进度与在途计数"]
    A --> X
    X -->|是| B --> C --> D
    D -->|是| E --> X
    D -->|否| P --> Q
    Q -->|否，重试分配| C
    Q -->|是| J
    X -->|否| W
    W -->|是| F --> G --> H
    H -->|是| I --> W
    H -->|否| J
    W -->|否| J
    J --> K
```

## 3. 章节范围

- token 进度、预算与运行请求选择。
- 等待请求接纳与连续批处理。
- Chunked Prefill、前缀缓存查询与 KV 块分配。
- 缓存压力、抢占与恢复。
- SchedulerOutput 构造与调度侧进度推进。

本章正文待迁移与整理。

[总目录](../README.md) · [固定配置](../00-Reading-Setup.md) · [全局执行路线](../Overview.md) · [上一章：请求进入核心，等待执行](../03-Core-Request-Admission/README.md) · [下一章：把执行计划变成模型输入](../05-Model-Input-Preparation/README.md)
