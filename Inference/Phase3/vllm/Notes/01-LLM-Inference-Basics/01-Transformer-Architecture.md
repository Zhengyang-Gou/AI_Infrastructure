# Transformer 架构

> 状态：已完成初稿 ｜ 更新：2026-09-04

## 1. 从推理视角看 Transformer

大模型推理通常使用 **Decoder-only Transformer**。它接收已有 Token，一层层提取上下文信息，最后预测下一个 Token；新 Token 再作为输入重复这一过程。

```text
Token IDs
   ↓
Token Embedding
   ↓
N × Transformer Block
   ↓
Final Norm
   ↓
LM Head → Logits → Sampling → Next Token
```

位置通常不再通过独立的位置向量相加，而是在 Attention 中用 **RoPE** 旋转 Query 和 Key，使注意力感知相对位置。

## 2. 一个 Transformer Block

以常见的 Llama 类架构为例：

```text
x ──→ RMSNorm → Self-Attention → + ──→ RMSNorm → MLP → + ──→ output
│                                  ↑  │                       ↑
└──────────────────────────────────┘  └───────────────────────┘
```

- **RMSNorm**：稳定数值分布。现代 LLM 多采用 Pre-Norm，即先归一化再进入子层。
- **Self-Attention**：让当前 Token 聚合历史 Token 的信息。
- **MLP**：对每个 Token 独立进行非线性变换，常见实现是 SwiGLU。
- **Residual Connection**：保留原始信息并改善深层网络训练。

MLP 通常占参数量和计算量的大头；Attention 的 KV Cache 则是推理阶段显存管理的重点。

## 3. Self-Attention 的核心计算

输入 `X ∈ [B, S, H]`，通过线性投影得到 Q、K、V：

```text
Q = XWq,  K = XWk,  V = XWv
Attention(Q, K, V) = softmax(QKᵀ / √d + causal_mask)V
```

- `B`：批大小；`S`：序列长度；`H`：隐藏维度；`d`：单个 Head 的维度。
- **Causal Mask** 保证位置 `i` 只能看到 `0...i`，不能看到未来 Token。
- 多个 Attention Head 并行学习不同关系，拼接后再经过输出投影。

常见 KV Head 设计：

| 类型 | KV Head 数量 | 特点 |
| --- | ---: | --- |
| MHA | 与 Query Head 相同 | 表达能力强，KV Cache 最大 |
| GQA | 少于 Query Head | 质量与显存/速度折中，现代 LLM 常用 |
| MQA | 只有一组 | KV Cache 最小，但可能影响模型能力 |

## 4. 为什么架构会影响推理性能

推理分为两个特征不同的阶段：

- **Prefill**：并行处理整个 Prompt，大矩阵乘较多，通常更容易充分利用 GPU 算力。
- **Decode**：每次只生成一个 Token，需要反复读取模型权重和历史 KV Cache，通常更受显存带宽限制。

KV Cache 的元素量近似为：

```text
2 × layers × tokens × kv_heads × head_dim
```

其中 `2` 代表 Key 和 Value，再乘数据类型的字节数即可估算显存占用。因此，层数、上下文长度、KV Head 数和精度都会直接影响可服务的并发量。

## 5. 需要记住的架构差异

- **Norm**：LayerNorm 或 RMSNorm，Pre-Norm 或 Post-Norm。
- **Position Encoding**：绝对位置编码、RoPE、ALiBi。
- **Attention**：MHA、GQA、MQA，以及 Sliding Window 等稀疏模式。
- **MLP**：GELU FFN 或 SwiGLU。
- **模型类型**：Encoder-only 适合理解，Encoder-Decoder 适合条件生成，Decoder-only 是当前 LLM 生成主流。

## 6. 面试快速回答

**Transformer Block 由什么组成？** 归一化、Self-Attention、MLP 和两条残差连接；Decoder-only 模型还通过 Causal Mask 保证自回归约束。

**为什么推理需要 KV Cache？** 历史 Token 的 K、V 在后续 Decode 中不会变化，缓存后可避免每一步重复计算整个上下文。

**为什么 GQA 有利于推理？** 多个 Query Head 共享较少的 KV Head，在保持多头 Query 表达能力的同时，减少 KV Cache 容量和读取带宽。

## 总结

从 AI Infra 角度，不仅要知道 Transformer “有什么层”，还要抓住三条主线：**权重决定基础显存与带宽开销，Attention 产生并读取 KV Cache，Prefill 与 Decode 具有不同的性能瓶颈。**
