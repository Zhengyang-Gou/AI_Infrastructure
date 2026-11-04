# 08 · Sampling

**logits 怎样变成输出 token 与概率信息？**

状态：边界页；正文待写。

## 1. 职责边界

| 项目 | 契约 |
| --- | --- |
| 输入 | logits、采样参数和历史 token 等采样上下文。 |
| 输出 | 采样 token、所需 logprobs 与采样侧结果。 |
| 本章负责 | logits 变换、惩罚与约束；temperature、top-k、top-p；greedy 与随机采样；logprobs。 |

本章遇到其他模块只使用接口结果，不展开其实现。

| 内容 | 唯一归属 |
| --- | --- |
| 追加正式输出、EOS/停止 token/长度判停与请求退场 | 03 Scheduler |
| 停止字符串、增量解码与 RequestOutput | 01 Inference Entry |
| 模型前向与 Sampler 的调用编排 | 06 Model Execution |
| 推测 token 的验证与接受/拒绝 | 09 Runtime Optimizations |

## 2. 正文范围

1. Sampler 输入与输出。
2. logits 变换与约束。
3. greedy、top-k、top-p 与随机采样。
4. logprobs 与结果组织。

[返回模块索引](../README.md) · [源码纵览](../00-Overview.md)
