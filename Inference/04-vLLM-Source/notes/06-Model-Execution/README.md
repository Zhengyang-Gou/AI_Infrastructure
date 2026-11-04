# 06 · Model Execution

**执行计划怎样变成输入张量、一次模型执行与执行结果？**

状态：边界页；正文待写。

## 1. 职责边界

| 项目 | 契约 |
| --- | --- |
| 输入 | SchedulerOutput，以及已加载的模型、KV 张量和执行侧请求状态。 |
| 输出 | ModelRunnerOutput：请求索引映射、采样 token 和所需概率信息等。 |
| 本章负责 | Executor → Worker → Model Runner 的执行交接；执行侧请求状态的建立、更新与删除；批次拼装；input_ids、positions、block table 与 slot_mapping；模型前向、Attention 与 Sampler 的调用位置。 |

本章遇到其他模块只使用接口结果，不展开其实现。

| 内容 | 唯一归属 |
| --- | --- |
| 请求选择、token 预算与判停 | 03 Scheduler |
| 物理块的分配、引用与回收算法 | 04 KV Cache Management |
| 模型、权重与 KV 张量初始化 | 05 Model and Weight Loading |
| 后端元数据的具体格式、KV 布局与 kernel | 07 Attention Backend |
| logits 处理和采样算法 | 08 Sampling |
| 图执行、推测路径和批次重叠 | 09 Runtime Optimizations |

## 2. 正文范围

1. 执行器与 Worker 的调用接口。
2. 执行侧请求状态与批次维护。
3. 输入张量、block table 与 slot_mapping。
4. 模型前向与 Attention 接口。
5. 采样调用与 ModelRunnerOutput。

[返回模块索引](../README.md) · [源码纵览](../00-Overview.md)
