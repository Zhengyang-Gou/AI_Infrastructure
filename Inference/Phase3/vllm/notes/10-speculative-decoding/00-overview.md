# Speculative Decoding

## 学习目标

本阶段理解推测解码如何用低成本 proposer 先生成多个候选 token，再由目标模型一次验证，从而减少目标模型串行 decode 步数，同时保持目标采样分布。

## 阅读顺序

| 顺序 | 笔记 | 主要内容 |
| --- | --- | --- |
| 1 | `01-proposers.md` | N-gram、Suffix 等无草稿模型候选器 |
| 2 | `02-draft-models.md` | Draft Model、EAGLE、MTP 等模型候选器 |
| 3 | `03-rejection-sampling.md` | 验证、接受、拒绝和恢复采样 |
| 4 | `04-speculative-execution-flow.md` | Scheduler、Model Runner 与下一轮状态闭环 |

## 核心循环

```text
已接受 token 序列
→ Proposer 生成 K 个 draft tokens
→ 下一轮 Scheduler 把 draft tokens 加入验证输入
→ Target Model 一次计算 K+1 个位置的 logits
→ Rejection Sampler 接受前缀并处理首个拒绝位置
→ 更新已计算 token、KV 与输出
→ 基于新状态再次 propose
```

## 完成标准

- 能区分基于匹配的 proposer 与基于小模型/额外 head 的 proposer。
- 能解释目标模型为什么可一次验证多个候选位置。
- 能说明 greedy 与随机采样下的接受规则。
- 能追踪拒绝后 token、computed position 和 KV 状态如何回退或覆盖。
- 能用接受长度、draft 成本和目标验证成本判断是否真正加速。

## 当前结论

推测解码优化的是目标模型调用的串行次数，不保证每轮计算更少；只有候选足够便宜且接受率足够高时，端到端吞吐或延迟才会改善。
