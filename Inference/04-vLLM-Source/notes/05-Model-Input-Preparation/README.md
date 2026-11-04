# 05 · 把执行计划变成模型输入

从 SchedulerOutput 到准备好本轮 GPU 计算输入。

## 1. 本章在推理引擎中的位置

```mermaid
flowchart TD
    S1["01 创建引擎与准备环境<br/>LLM(...)"]
    S2["02 提交一批离线请求<br/>LLM.generate(prompts)"]
    S3["03 核心接收并入队<br/>EngineCore → Scheduler"]
    S4["04 决定本轮执行内容<br/>选择请求与安排 KV"]
    S5["05 准备模型输入 ← 本章<br/>Executor → Worker → Model Runner"]
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
    class S5 current;
```

## 2. 本章调用路线与用途

```mermaid
flowchart TD
    A["UniProcExecutor.execute_model<br/>接收 SchedulerOutput"]
    B["UniProcExecutor.collective_rpc<br/>调用 Worker 的执行方法"]
    C["Worker.execute_model<br/>将本轮计划交给 Model Runner"]
    D["GPUModelRunner.execute_model<br/>组织本轮输入准备"]
    E["GPUModelRunner._update_states<br/>同步请求、缓存块与 batch 状态"]
    F["GPUModelRunner._prepare_inputs<br/>准备 token、位置与计算长度"]
    G["GPUModelRunner._get_slot_mappings<br/>确定本轮 token 的 KV 写入位置"]
    H["GPUModelRunner._build_attention_metadata<br/>整理 Attention 所需长度与块信息"]
    I["GPUModelRunner._preprocess<br/>取得模型前向需要的输入张量"]
    J["交给前向执行<br/>输入张量与 Attention 信息就绪"]
    A --> B --> C --> D --> E --> F --> G --> H --> I --> J
```

## 3. 章节范围

- Executor、Worker 与 Model Runner 的执行交接。
- 执行侧请求状态与 batch 组织。
- token 展平、位置与序列长度。
- block table、slot mapping 与 CPU / GPU 数据准备。

本章正文待迁移与整理。

[总目录](../README.md) · [固定配置](../00-Reading-Setup.md) · [全局执行路线](../Overview.md) · [上一章：决定本轮执行哪些请求](../04-Step-Scheduling/README.md) · [下一章：完成本轮前向与采样](../06-Forward-and-Sampling/README.md)
