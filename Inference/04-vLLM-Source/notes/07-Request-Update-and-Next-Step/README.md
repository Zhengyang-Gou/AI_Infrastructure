# 07 · 回写执行结果，推进下一轮

从模型执行结果到下一轮请求状态或核心收尾。

## 1. 本章在推理引擎中的位置

```mermaid
flowchart TD
    S1["01 创建引擎与准备环境<br/>LLM(...)"]
    S2["02 提交一批离线请求<br/>LLM.generate(prompts)"]
    S3["03 核心接收并入队<br/>EngineCore → Scheduler"]
    S4["04 决定本轮执行内容<br/>选择请求与安排 KV"]
    S5["05 准备模型输入<br/>Executor → Worker → Model Runner"]
    S6["06 模型前向与采样<br/>模型 → logits → token"]
    S7["07 回写请求状态 ← 本章<br/>继续执行或释放资源"]
    S8["08 解码与收集结果<br/>OutputProcessor → RequestOutput"]
    R["返回本次 generate 的结果"]
    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    S7 -->|未完成请求继续| S4
    S7 -->|增量 token 与结束信息| S8
    S8 -->|停止字符串触发取消| S7
    S8 -->|收齐全部结果| R
    classDef current fill:#dbeafe,stroke:#2563eb,stroke-width:3px,color:#0f172a,font-weight:bold;
    class S7 current;
```

## 2. 本章调用路线与用途

```mermaid
flowchart TD
    A["Scheduler.update_from_output<br/>接收执行计划与模型结果"]
    B["按 request_id 找到 Request<br/>扣减在途计数，取出对应 token"]
    C["Scheduler._update_request_with_output<br/>处理本轮新增 token"]
    D["Request.append_output_token_ids<br/>更新输出与完整 token 历史"]
    E["check_stop<br/>检查停止 token 与长度等条件"]
    F{"请求已结束？"}
    G["Scheduler._free_request<br/>登记完成并启动资源清理"]
    H["Scheduler._free_blocks<br/>释放块引用，移除请求记录"]
    I["保留 Request<br/>等待后续轮次继续执行"]
    J["构造 EngineCoreOutput<br/>整理新增 token 和结束信息"]
    K["整理队列与 EngineCoreOutputs<br/>移出结束请求，按客户端返回增量"]
    L["Scheduler.schedule<br/>为未完成请求形成下一轮计划"]
    A --> B --> C --> D --> E --> F
    F -->|是| G --> H --> J
    F -->|否| I --> J
    J --> K
    K -->|仍有未完成请求| L
    K -->|返回入口| M["进入第 08 章<br/>解码并收集用户结果"]
```

## 3. 章节范围

- 按请求 ID 关联结果与追加有效 token。
- 计算进度、在途状态与停止条件。
- 完成请求退出与 KV 资源引用释放。
- 未完成请求进入下一轮调度。
- EngineCoreOutput 与核心增量返回。

本章正文待迁移与整理。

[总目录](../README.md) · [固定配置](../00-Reading-Setup.md) · [全局执行路线](../Overview.md) · [上一章：完成本轮前向与采样](../06-Forward-and-Sampling/README.md) · [下一章：将增量还原成文本，返回离线结果](../08-Output-and-Result-Collection/README.md)
