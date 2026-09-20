# LLM 推理基础

阶段目标：**能解释一次生成请求如何执行**,已有笔记按以下顺序阅读。

1. [Transformer 架构](01-Transformer-Architecture.md)
2. [Attention Mechanism](02-Attention-Mechanism.md)
3. [Prefill 与 Decode](03-Prefill-and-Decode.md)
4. [KV Cache 原理](04-KV-Cache-Principles.md)
5. [Sampling Strategies](05-Sampling-Strategies.md)
6. [LLM 推理流程与性能指标](06-LLM-Inference-Metrics.md)

## 阶段验收

- [ ] 解释从输入 Prompt 到首 Token，再到逐 Token 生成的流程。
- [ ] 说明 Prefill、Decode、KV Cache 和 MHA/GQA/MQA 的关系。
- [ ] 解释采样方式，以及 TTFT、TPOT、吞吐等指标。

下一步：[02 · Mini Inference Engine](../02-Mini-Inference-Engine/)。

[返回总路线](../Roadmap.md)
