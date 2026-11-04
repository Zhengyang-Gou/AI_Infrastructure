# 07 · Attention Backend

**Attention 后端怎样选择，怎样解释缓存布局并执行注意力？**

状态：边界页；正文待写。

## 1. 职责边界

| 项目 | 契约 |
| --- | --- |
| 输入 | Q/K/V、已分配的 KV 存储、块表与位置等原始执行信息。 |
| 输出 | Attention 输出，以及按指定物理布局写入的 K/V。 |
| 本章负责 | 后端选择和兼容性；Attention 元数据构造与格式；逻辑块到后端 KV 物理布局的对应；缓存读写；Attention 算子与 kernel 调用。 |

本章遇到其他模块只使用接口结果，不展开其实现。

| 内容 | 唯一归属 |
| --- | --- |
| 谁持有哪个块、前缀复用与淘汰 | 04 KV Cache Management |
| 整批输入拼装和通用 block table、slot_mapping 构造 | 06 Model Execution |
| logits 与 token 采样 | 08 Sampling |
| 图捕获与重放策略 | 09 Runtime Optimizations |

## 2. 正文范围

1. 后端选择与接口。
2. Attention 元数据。
3. KV 布局与块大小适配。
4. KV 写入与 Attention kernel 调用。

[返回模块索引](../README.md) · [源码纵览](../00-Overview.md)
