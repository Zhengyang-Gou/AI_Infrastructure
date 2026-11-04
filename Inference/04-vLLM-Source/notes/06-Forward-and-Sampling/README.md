# 06 · 完成本轮前向与采样

从本轮模型输入到 ModelRunnerOutput。

## 1. 本章在推理引擎中的位置

```mermaid
flowchart TD
    S1["01 创建引擎与准备环境<br/>LLM(...)"]
    S2["02 提交一批离线请求<br/>LLM.generate(prompts)"]
    S3["03 核心接收并入队<br/>EngineCore → Scheduler"]
    S4["04 决定本轮执行内容<br/>选择请求与安排 KV"]
    S5["05 准备模型输入<br/>Executor → Worker → Model Runner"]
    S6["06 模型前向与采样 ← 本章<br/>模型 → logits → token"]
    S7["07 回写请求状态<br/>继续执行或释放资源"]
    S8["08 解码与收集结果<br/>OutputProcessor → RequestOutput"]
    R["返回本次 generate 的结果"]
    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    S7 -->|未完成请求继续| S4
    S7 -->|增量 token 与结束信息| S8
    S8 -->|停止字符串触发取消| S7
    S8 -->|收齐全部结果| R
    classDef current fill:#dbeafe,stroke:#2563eb,stroke-width:3px,color:#0f172a,font-weight:bold;
    class S6 current;
```

## 2. 本章调用路线与用途

```mermaid
flowchart TD
    A["GPUModelRunner.execute_model<br/>组织本轮模型前向"]
    B["set_forward_context<br/>提供 Attention 元数据与 KV 位置"]
    C["GPUModelRunner._model_forward<br/>调用实际模型"]
    D["LlamaForCausalLM.forward<br/>进入 Llama 主体计算"]
    E["LlamaModel.forward<br/>依次执行各层 Attention 与 MLP"]
    F["LlamaForCausalLM.compute_logits<br/>为候选 token 计算分数"]
    G["保存 execute_model_state 并返回<br/>留存 logits 与本轮执行状态"]
    H["UniProcExecutor.sample_tokens → Worker.sample_tokens<br/>由核心继续发起采样"]
    I["GPUModelRunner.sample_tokens<br/>读取本轮 logits 与采样状态"]
    J["GPUModelRunner._sample<br/>根据生成参数采样 token"]
    K["GPUModelRunner._bookkeeping_sync<br/>整理有效 token 与请求对应关系"]
    L["构造 ModelRunnerOutput<br/>将本轮采样结果返回核心"]
    A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K --> L
```

## 3. 章节范围

- 模型 forward 与必要的算子调用。
- Attention Backend 与 KV 读写。
- logits、采样与正式输出 token。
- Prefill 分块的输出行为与 ModelRunnerOutput。

本章正文待迁移与整理。

[总目录](../README.md) · [固定配置](../00-Reading-Setup.md) · [全局执行路线](../Overview.md) · [上一章：把执行计划变成模型输入](../05-Model-Input-Preparation/README.md) · [下一章：回写执行结果，推进下一轮](../07-Request-Update-and-Next-Step/README.md)
