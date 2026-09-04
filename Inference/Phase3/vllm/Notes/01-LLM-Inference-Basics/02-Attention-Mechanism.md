# Attention Mechanism

> 状态：已完成初稿 ｜ 更新：2026-09-04

## 1. Attention 在做什么

Attention 的本质是：**根据当前 Token 与其他 Token 的相关性，对它们的信息进行加权汇总。**

输入 `X ∈ [B, S, H]` 经过三个线性投影：

```text
Q = XWq   # 当前 Token 想查询什么
K = XWk   # 每个 Token 能用什么特征被匹配
V = XWv   # 每个 Token 实际提供的信息
```

可以把它理解为：用 Q 和 K 计算匹配分数，再按分数对 V 求加权和。

## 2. Scaled Dot-Product Attention

核心公式：

```text
scores = QKᵀ / √d
weights = softmax(scores + mask)
output = weights × V
```

其中 `d` 是单个 Attention Head 的维度。

为什么除以 `√d`？当维度增大时，点积的数值范围也会变大，Softmax 容易进入饱和区域。缩放能让分数保持在较稳定的范围内。

为什么需要 Softmax？它把一行分数转换为和为 1 的权重，使模型能按相关程度组合多个 Token 的信息。

实际实现会使用数值稳定的 Softmax：

```text
softmax(xᵢ) = exp(xᵢ - max(x)) / Σ exp(xⱼ - max(x))
```

减去最大值不会改变结果，但能避免指数运算溢出。

## 3. Causal Mask

Decoder-only 模型不能看到未来 Token，因此位置 `i` 只能关注 `0...i`：

```text
Token 0: ✓ ✗ ✗ ✗
Token 1: ✓ ✓ ✗ ✗
Token 2: ✓ ✓ ✓ ✗
Token 3: ✓ ✓ ✓ ✓
```

实现时，未来位置的 score 会被加上 `-∞`，经过 Softmax 后权重变为 0。Mask 只限制信息流向，不减少普通 Attention 的理论计算复杂度。

## 4. Multi-Head Attention

模型会把隐藏维度拆成多个 Head：

```text
Q: [B, S, H] → [B, num_heads, S, head_dim]
H = num_heads × head_dim
```

不同 Head 可以学习不同关系，例如局部依赖、长距离依赖或语义关联。各 Head 的输出拼接后，再经过输出投影 `Wo`。

推理中常重点关注 Query Head 与 KV Head 的数量：

- **MHA**：每个 Query Head 都有独立的 K、V，表达能力强但 KV Cache 大。
- **GQA**：一组 Query Head 共享一个 KV Head，是当前主流的质量与性能折中。
- **MQA**：所有 Query Head 共享一组 K、V，KV Cache 最小。

## 5. Prefill 与 Decode 中的区别

### Prefill

Prompt 中所有 Token 同时计算，Attention Score 的形状近似为：

```text
[B, num_heads, S, S]
```

标准 Attention 的计算复杂度随序列长度呈 `O(S²)` 增长，中间 Score/Weight 矩阵也可能占用大量显存。FlashAttention 通过分块计算和在线 Softmax，避免把完整的 `S × S` 矩阵写回显存，主要优化数据搬运，而不是近似 Attention。

### Decode

每一步只有一个新 Query，但需要读取长度为 `L` 的历史 K、V：

```text
Q: [B, heads, 1, d]
K/V Cache: [B, kv_heads, L, d]
```

单步 Attention 计算量约随上下文长度 `L` 线性增长。此时并行度较低，又要反复读取 KV Cache，因此通常更容易受到显存带宽限制。

KV Cache 保存历史 Token 已经计算好的 K、V，使 Decode 不必每一步重新处理完整上下文；代价是显存占用随序列长度增长。

## 6. 常见误区

- Attention 权重高表示当前计算中的相关性强，不一定等同于可解释的“重要性”。
- FlashAttention 是精确 Attention 的 IO 优化算法，不是稀疏或近似算法。
- KV Cache 只避免重复计算历史 K、V，并没有消除当前 Query 对历史 KV 的读取。
- Causal Mask 保证自回归约束，但标准实现仍可能计算被遮挡区域。

## 7. 面试快速回答

**Q、K、V 为什么来自不同投影？** 这样模型可以分别学习“查询特征”“匹配特征”和“输出内容”，比直接对原始表示做相似度和加权更灵活。

**为什么长上下文很贵？** Prefill 的标准 Attention 计算随长度平方增长；Decode 的单步 KV 读取随上下文线性增长，同时 KV Cache 也线性占用显存。

**GQA 为什么能加速推理？** 它减少 KV Head 数，从而降低 KV Cache 容量和读取带宽，同时保留较多 Query Head 的表达能力。

## 总结

Attention 可以压缩为三步：**QKᵀ 计算相关性、Softmax 生成权重、权重乘 V 聚合信息。** 对推理系统而言，Prefill 的核心问题是长序列下的二次计算和显存访问，Decode 的核心问题则是持续读取不断增长的 KV Cache。
