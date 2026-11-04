# 08 · 将增量还原成文本，返回离线结果

从核心增量输出到本次 generate 的最终结果。

## 1. 本章在推理引擎中的位置

```mermaid
flowchart TD
    S1["01 创建引擎与准备环境<br/>LLM(...)"]
    S2["02 提交一批离线请求<br/>LLM.generate(prompts)"]
    S3["03 核心接收并入队<br/>EngineCore → Scheduler"]
    S4["04 决定本轮执行内容<br/>选择请求与安排 KV"]
    S5["05 准备模型输入<br/>Executor → Worker → Model Runner"]
    S6["06 模型前向与采样<br/>模型 → logits → token"]
    S7["07 回写请求状态<br/>继续执行或释放资源"]
    S8["08 解码与收集结果 ← 本章<br/>OutputProcessor → RequestOutput"]
    R["返回本次 generate 的结果"]
    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    S7 -->|未完成请求继续| S4
    S7 -->|增量 token 与结束信息| S8
    S8 -->|停止字符串触发取消| S7
    S8 -->|收齐全部结果| R
    classDef current fill:#dbeafe,stroke:#2563eb,stroke-width:3px,color:#0f172a,font-weight:bold;
    class S8 current;
```

## 2. 本章调用路线与用途

```mermaid
flowchart TD
    A["OfflineInferenceMixin._run_engine<br/>循环取得请求结果"]
    B["LLMEngine.step<br/>组织取输出、解码与取消反馈"]
    C["InprocClient.get_output<br/>推进核心，取得 EngineCoreOutputs"]
    D["OutputProcessor.process_outputs<br/>按内部 ID 找到 RequestState"]
    E["IncrementalDetokenizer.update<br/>增量解码并检查停止字符串"]
    F["RequestState.make_request_output<br/>按输出方式构造 RequestOutput"]
    G["OutputProcessor._finish_request<br/>清理已经完成的入口状态"]
    H["InprocClient.abort_requests<br/>将停止字符串触发的取消交给核心"]
    I["_run_engine 收集 finished 结果<br/>保存已完成的 RequestOutput"]
    J{"还有未完成请求？"}
    K["_run_engine 排序并返回<br/>按外部 ID 恢复输入顺序"]
    A --> B --> C --> D --> E --> F
    F -->|请求完成| G --> H
    F -->|继续等待输出| H
    H --> I --> J
    J -->|是| B
    J -->|否| K
```

## 3. 章节范围

- 客户端取输出与 OutputProcessor 增量解码。
- 停止字符串、取消反馈与入口状态清理。
- FINAL_ONLY 与 RequestOutput 构造。
- 完成结果收集、排序与 generate 返回。

本章正文待迁移与整理。

[总目录](../README.md) · [固定配置](../00-Reading-Setup.md) · [全局执行路线](../Overview.md) · [上一章：回写执行结果，推进下一轮](../07-Request-Update-and-Next-Step/README.md)
