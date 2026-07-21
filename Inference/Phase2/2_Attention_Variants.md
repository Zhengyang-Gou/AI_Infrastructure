# Attention Variants
MHA：每个 Query 头都有自己独立的 Key、Value

MQA：所有 Query 头共享同一套 Key、Value

GQA：若干 Query 头组成一组，每组共享一套 Key、Value

MLA：不直接缓存完整 Key、Value，而是缓存一个低维潜变量，各个头再用不同投影读取它

它们的演化，本质上是在解决同一个问题：

如何在尽量保持多头注意力能力的同时，减少自回归生成时庞大的 KV Cache 和显存带宽消耗

## Attention
![alt text](image.png)

## MHA
### 为什么需要多头
如果只有一个头，模型只能从一种角度去理解句子

多头机制允许模型在同一时间，从多个不同的“子空间”（维度视角）去捕捉多样的语义关系

### MHA 计算流程
假设输入是一段序列，每个 Token 的特征维度是 $d_{model}$

把它分成 $h$ 个头，那么每个头分到的维度就是 $d_k = d_{model} / h$

1. 线性映射：输入矩阵 $X$ 通过三组不同的线性层，生成 $Q, K, V$：
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

2. 切分成多头
把 $Q, K, V$ 从形状 [Batch_size, Seq_len, embed_dim] 拆成 [Batch_size, num_heads, Seq_len, head_dim]

    让这 $h$ 个头并行计算

3. 缩放点积注意力
对于每一个头，计算公式为：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

    $Q K^T$：计算每个字和其它所有字之间的相似度分数

    $\sqrt{d_k}$：缩放因子，防止向量维度太高时，点积结果太大，导致 Softmax 梯度消失

    $\text{softmax}$：将分数归一化为概率

    $\times V$：用这些概率对具体内容进行加权求和，得到融合了上下文的新特征

4. 拼接与最终线性变换
    把所有头的输出拼回原来的大小，再过最后一层线性层 $W_O$ 整合信息

## MQA
### 为什么需要 MQA
在 MHA 中，如果我们要分 $h$ 个头，每个头都有自己独立的一套 $Q、K、V$

在模型推理时，因为是自回归的一个字一个字往外蹦，为了避免重复计算前面的字，把之前计算好的 $K$ 和 $V$ 缓存起来，这叫 KV Cache

MHA 的问题：随着序列越来越长，KV Cache 会吃掉海量的显存，很多时候大模型推不动，不是因为显卡算力不够，而是显存被 KV Cache 撑爆了

MQA 的解法：能不能让所有的头，共享同一组 K 和 V

### MHA vs MQA 的区别
MHA：有 8 个 Query 头，同时对应 8 个 Key 头和 8 个 Value 头

MQA：有 8 个 Query 头，但只有 1 个 Key 头和 1 个 Value 头。所有的 Query 头都去和这同一个 Key 算相似度，然后去乘以同一个 Value

假设 embed_dim = 512，num_heads = 8，则每个头的 head_dim = 64

| 机制  | Q 的形状           | K 的形状           | V 的形状           |
| --- | --------------- | --------------- | --------------- |
| MHA | `[B, 8, L, 64]` | `[B, 8, L, 64]` | `[B, 8, L, 64]` |
| MQA | `[B, 8, L, 64]` | `[B, 1, L, 64]` | `[B, 1, L, 64]` |

## GQA
GQA 是目前绝大多数现代开源大模型的标准配置

### GQA 的核心思想
将 Query 头部进行分组，每一组 Query 共享一个单独的 Key 头和 Value 头

### 维度变化
假设 embed_dim = 512，num_heads = 8，组数 num_groups = 2

8 个 Q 头被分成 2 组，每组 4 个 Q 头，同时，K 和 V 此时也拥有 2 个头

| 机制  | Q 的总头数 | K/V 的总头数 | 每个 K/V 头被几个 Q 共享 |
| --- | -----: | -------: | ---------------- |
| MHA |      8 |        8 | 1（不共享）           |
| MQA |      8 |        1 | 8（全员共享）          |
| GQA |      8 |        2 | 4（组内共享）          |

当 num_groups = num_heads 时，GQA 就退化成了 MHA

当 num_groups = 1 时，GQA 就变成了 MQA

## MLA