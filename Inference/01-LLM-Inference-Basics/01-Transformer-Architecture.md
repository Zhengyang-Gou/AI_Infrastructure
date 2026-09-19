# Transformer 架构

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

## 4. 一次前向推理的张量维度变化

以 **Llama 类 Decoder-only + GQA + SwiGLU** 为例。一次前向计算得到下一个 Token 的概率分布；生成完整回答需要先执行一次 Prefill，再反复执行 Decode。下面使用逻辑张量形状，实际推理引擎可能采用 Token 打包、融合算子或分页 KV Cache 等物理布局。

### 4.1 维度约定

| 符号 | 含义 | 本例取值（示意配置） |
| --- | --- | ---: |
| `B` | Batch Size | 1 |
| `S` | Prompt 的 Token 数 | 4 |
| `H` | Hidden Size | 4096 |
| `nq` | Query Head 数 | 32 |
| `nkv` | KV Head 数 | 8 |
| `d` | 每个 Head 的维度，`H = nq × d` | 128 |
| `I` | MLP 中间维度 | 14336 |
| `Vocab` | 词表大小 | 32000 |
| `L` | Transformer Block 层数 | 32 |

这里每 `nq / nkv = 4` 个 Query Head 共享一组 K、V。以下线性层权重按 `Y = XW` 的数学约定书写。

### 4.2 Prefill：输入 4 个 Token，预测第 5 个 Token

| 步骤 | 计算 / 说明 | 输出形状 | 本例形状 |
| --- | --- | --- | --- |
| 输入 | Token IDs | `[B, S]` | `[1, 4]` |
| Embedding | 查表，Embedding 权重为 `[Vocab, H]` | `[B, S, H]` | `[1, 4, 4096]` |
| Attention 前 RMSNorm | 沿隐藏维度归一化，形状不变 | `[B, S, H]` | `[1, 4, 4096]` |
| Q 投影 | `XWq`，`Wq: [H, nq × d]` | `[B, S, nq × d]` | `[1, 4, 4096]` |
| K、V 投影 | `XWk`、`XWv`，权重各为 `[H, nkv × d]` | 各为 `[B, S, nkv × d]` | 各为 `[1, 4, 1024]` |
| 拆分 Head 并转置 | Q | `[B, nq, S, d]` | `[1, 32, 4, 128]` |
| 拆分 Head 并转置 | K、V | 各为 `[B, nkv, S, d]` | 各为 `[1, 8, 4, 128]` |
| RoPE | 旋转 Q、K，形状不变；缓存旋转后的 K 和 V | Q、K 形状同上 | 同上 |
| Attention Scores | 各 Query Head 使用对应 KV Head，计算 `QKᵀ / √d` | `[B, nq, S, S]` | `[1, 32, 4, 4]` |
| Mask + Softmax | 加因果掩码，沿最后一维归一化 | `[B, nq, S, S]` | `[1, 32, 4, 4]` |
| 加权求和 | Attention 概率乘对应的 V | `[B, nq, S, d]` | `[1, 32, 4, 128]` |
| 合并 Head | 转置回 `[B, S, nq, d]`，再拼接 | `[B, S, H]` | `[1, 4, 4096]` |
| 输出投影 + 残差 | `Wo: [H, H]`，投影后与 Block 输入相加 | `[B, S, H]` | `[1, 4, 4096]` |
| MLP 前 RMSNorm | 形状不变 | `[B, S, H]` | `[1, 4, 4096]` |
| Gate / Up 投影 | 两个权重各为 `[H, I]` | 各为 `[B, S, I]` | 各为 `[1, 4, 14336]` |
| SwiGLU | `SiLU(XWgate) ⊙ (XWup)`，`⊙` 为逐元素乘法 | `[B, S, I]` | `[1, 4, 14336]` |
| Down 投影 + 残差 | `Wdown: [I, H]`，投影后与 MLP 子层输入相加 | `[B, S, H]` | `[1, 4, 4096]` |
| 重复共 `L` 个 Block | 每层输入、输出形状相同，参数和 KV Cache 各自独立 | `[B, S, H]` | `[1, 4, 4096]` |
| Final RMSNorm | 最后一层归一化 | `[B, S, H]` | `[1, 4, 4096]` |
| 取最后一个位置 | 普通下一 Token 生成只需最后位置的隐藏状态 | `[B, H]` | `[1, 4096]` |
| LM Head | 乘 `[H, Vocab]` 权重，得到 Logits | `[B, Vocab]` | `[1, 32000]` |
| Sampling / Argmax | 每个请求选出一个 Token ID | `[B, 1]` | `[1, 1]` |

如果对所有位置执行 LM Head，Logits 是 `[B, S, Vocab]`；这里只预测下一个 Token，因此可以先取最后位置再做投影，节省计算。上例假设没有 Padding；有 Padding 时应取每个请求最后一个有效位置。

**注意**：GQA 在逻辑上让多个 Query Head 共享 KV Head，无须把缓存中的 K、V 真正复制到 `nq` 个 Head。表中的 Attention Scores 是数学上的形状，FlashAttention 等融合实现通常不会在显存中完整物化这个矩阵。

### 4.3 Decode：输入第 5 个 Token，预测第 6 个 Token

Prefill 结束后，每层 K、V Cache 的形状各为 `[1, 8, 4, 128]`。Prefill 采样得到的第 5 个 Token 尚未经过模型，下一次前向计算才会为它产生 K、V。

设调用开始时已缓存 `T` 个 Token，本次只输入一个新 Token，则 **Query 长度为 1，KV 长度为 `T + 1`**：

| 步骤 | 通用形状 | 本例：`T = 4` |
| --- | --- | --- |
| 新 Token IDs | `[B, 1]` | `[1, 1]` |
| Embedding / Norm | `[B, 1, H]` | `[1, 1, 4096]` |
| 新 Q（拆分 Head 后） | `[B, nq, 1, d]` | `[1, 32, 1, 128]` |
| 新 K、V（拆分 Head 后） | 各为 `[B, nkv, 1, d]` | 各为 `[1, 8, 1, 128]` |
| RoPE 后追加 KV Cache | 各为 `[B, nkv, T + 1, d]` | 各为 `[1, 8, 5, 128]` |
| Q 与全部缓存 K 计算 Scores | `[B, nq, 1, T + 1]` | `[1, 32, 1, 5]` |
| Softmax 后与全部缓存 V 加权求和 | `[B, nq, 1, d]` | `[1, 32, 1, 128]` |
| 合并 Head / 输出投影 / 残差 | `[B, 1, H]` | `[1, 1, 4096]` |
| MLP 中间张量 | `[B, 1, I]` | `[1, 1, 14336]` |
| MLP 输出 / 残差 | `[B, 1, H]` | `[1, 1, 4096]` |
| 经过全部 Block 和 Final Norm，取唯一位置 | `[B, H]` | `[1, 4096]` |
| LM Head → 采样 | `[B, Vocab] → [B, 1]` | `[1, 32000] → [1, 1]` |

此时 RoPE 使用新 Token 的实际位置（本例从 0 开始编号，位置为 4），不能因为本次输入长度为 1 就重新从位置 0 开始。每层只计算新 Token 的 Q、K、V，但 Attention 会读取该层全部可见的历史 K、V。

记住三条维度变化：**Block 外部始终保持 `[B, 当前输入长度, H]`；MLP 内部把 `H` 扩到 `I` 再缩回；Attention 的最后两维是 `[Query 长度, KV 长度]`，Decode 时为 `[1, T + 1]`。**

## 5. 为什么架构会影响推理性能

推理分为两个特征不同的阶段：

- **Prefill**：并行处理整个 Prompt，大矩阵乘较多，通常更容易充分利用 GPU 算力。
- **Decode**：每次只生成一个 Token，需要反复读取模型权重和历史 KV Cache，通常更受显存带宽限制。

KV Cache 的元素量近似为：

```text
2 × layers × tokens × kv_heads × head_dim
```

其中 `2` 代表 Key 和 Value，再乘数据类型的字节数即可估算显存占用。因此，层数、上下文长度、KV Head 数和精度都会直接影响可服务的并发量。

## 6. 需要记住的架构差异

- **Norm**：LayerNorm 或 RMSNorm，Pre-Norm 或 Post-Norm。
- **Position Encoding**：绝对位置编码、RoPE、ALiBi。
- **Attention**：MHA、GQA、MQA，以及 Sliding Window 等稀疏模式。
- **MLP**：GELU FFN 或 SwiGLU。
- **模型类型**：Encoder-only 适合理解，Encoder-Decoder 适合条件生成，Decoder-only 是当前 LLM 生成主流。

## 7. 面试快速回答

**Transformer Block 由什么组成？** 归一化、Self-Attention、MLP 和两条残差连接；Decoder-only 模型还通过 Causal Mask 保证自回归约束。

**为什么推理需要 KV Cache？** 历史 Token 的 K、V 在后续 Decode 中不会变化，缓存后可避免每一步重复计算整个上下文。

**为什么 GQA 有利于推理？** 多个 Query Head 共享较少的 KV Head，在保持多头 Query 表达能力的同时，减少 KV Cache 容量和读取带宽。

## 总结

从 AI Infra 角度，不仅要知道 Transformer “有什么层”，还要抓住三条主线：**权重决定基础显存与带宽开销，Attention 产生并读取 KV Cache，Prefill 与 Decode 具有不同的性能瓶颈。**
