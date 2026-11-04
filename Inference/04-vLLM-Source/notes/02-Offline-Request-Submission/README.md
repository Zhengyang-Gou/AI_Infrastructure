# 02 · 提交一批离线请求

从 LLM.generate(prompts) 到逐条提交核心请求。

## 1. 本章在推理引擎中的位置

```mermaid
flowchart TD
    S1["01 创建引擎与准备环境<br/>LLM(...)"]
    S2["02 提交一批离线请求 ← 本章<br/>LLM.generate(prompts)"]
    S3["03 核心接收并入队<br/>EngineCore → Scheduler"]
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
    class S2 current;
```

## 2. 本章调用路线与用途

```mermaid
flowchart TD
    A["LLM.generate<br/>接收一批输入与生成参数"]
    B["OfflineInferenceMixin._run_completion<br/>组织提交与结果收集"]
    C["_add_completion_requests<br/>对齐输入和逐请求参数"]
    D["_render_and_add_requests<br/>逐条取输入并提交"]
    E["_preprocess_cmpl_one → _preprocess_cmpl<br/>整理一条输入与分词参数"]
    F["BaseRenderer.render_cmpl<br/>分词并建立 EngineInput"]
    G["_add_request<br/>分配外部 ID，设置 FINAL_ONLY"]
    H["LLMEngine.add_request<br/>组织请求构造与登记"]
    I["InputProcessor.process_inputs<br/>检查输入，构造 EngineCoreRequest"]
    J["InputProcessor.assign_request_id<br/>建立核心使用的内部 ID"]
    K["OutputProcessor.add_request<br/>登记这条请求的输出状态"]
    L["InprocClient.add_request<br/>将请求交给核心"]
    M["_run_engine<br/>开始循环取得结果"]
    A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K --> L
    L -->|还有输入| D
    L -->|整批已提交| M
```

## 3. 章节范围

- 批量输入与生成参数对齐。
- Renderer、分词与统一输入表示。
- InputProcessor 校验与 EngineCoreRequest 构造。
- 请求 ID、输出状态注册与提交。

本章正文待迁移与整理。

[总目录](../README.md) · [固定配置](../00-Reading-Setup.md) · [全局执行路线](../Overview.md) · [上一章：创建引擎与准备执行环境](../01-Engine-Initialization/README.md) · [下一章：请求进入核心，等待执行](../03-Core-Request-Admission/README.md)
