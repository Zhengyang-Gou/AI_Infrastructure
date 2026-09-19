# Prefill 与 Decode

## 1. 一次生成包含两个阶段

Decoder-only LLM 的推理可以简化为：

```text
Prompt Tokens
     ↓
Prefill：并行处理整个 Prompt，建立 KV Cache
     ↓
生成第一个 Token
     ↓
Decode：每轮生成一个 Token，并追加 KV Cache
     ↓
遇到 EOS、停止词或长度上限
```

两者执行的是同一套 Transformer 权重，区别主要在于输入形状、并行度和内存访问模式。

## 2. Prefill 阶段

Prefill 一次处理全部输入 Token。假设 Prompt 长度为 `S`：

```text
input:  [B, S]
hidden: [B, S, H]
Q/K/V: 处理 S 个位置
output: 取最后位置的 logits 生成首个 Token
```

主要工作：

- 对所有 Prompt Token 执行完整前向计算。
- 在每一层生成并保存 K、V，形成初始 KV Cache。
- 计算最后一个位置的 logits，用于采样第一个输出 Token。

Prefill 的矩阵乘规模较大、并行度高，通常更容易利用 GPU 算力。标准 Attention 对序列长度的计算复杂度为 `O(S²)`，因此长 Prompt 会显著增加首 Token 延迟。

Prefill 主要影响 **TTFT（Time To First Token）**。

## 3. Decode 阶段

Decode 每轮只处理最新的一个 Token：

```text
new token:  [B, 1]
new Q/K/V:  只计算当前 Token
attention:  当前 Q 读取全部历史 K/V
cache:      将当前 K/V 追加到 KV Cache
```

历史 Token 的 K、V 已经缓存，不需要重新计算。但每生成一个 Token，仍需：

1. 读取模型权重完成一轮前向计算。
2. 读取不断增长的历史 KV Cache。
3. 采样下一个 Token，并进入下一轮。

Decode 单轮并行度较低，矩阵通常较“瘦”，又需要频繁读取权重和 KV Cache，因此常见瓶颈是显存带宽而非峰值算力。

Decode 主要影响 **TPOT（Time Per Output Token）** 或 **ITL（Inter-Token Latency）**。

## 4. 核心差异

| 对比项 | Prefill | Decode |
| --- | --- | --- |
| 每次处理 Token 数 | 整个 Prompt | 每轮一个新 Token |
| 并行度 | 高 | 低 |
| Attention 范围 | Prompt 内所有合法位置 | 当前 Query 对全部历史 KV |
| KV Cache | 创建并批量写入 | 每轮读取并追加 |
| 常见瓶颈 | 计算量、长序列 Attention | 权重与 KV Cache 带宽 |
| 核心指标 | TTFT | TPOT / ITL |

若输出 `N` 个 Token，粗略估算端到端延迟：

```text
latency ≈ TTFT + (N - 1) × TPOT
```

实际还包括排队、调度、采样、网络传输等开销。

## 5. 为什么需要 KV Cache

如果没有 KV Cache，Decode 每一步都要重新计算完整上下文，生成越长，重复计算越严重。

使用 KV Cache 后：

- 历史 K、V 只计算一次。
- 当前 Token 只生成一组新的 K、V。
- Attention 仍然要读取历史 KV，因此时间和显存开销仍会随上下文长度增长。

KV Cache 是用显存换计算，也是 LLM Serving 并发容量的重要限制。

## 6. Serving 中的相互影响

长 Prefill 可能长时间占用 GPU，使正在 Decode 的请求出现 Token 卡顿；如果只优先 Decode，新请求的 TTFT 又会恶化。因此调度器需要在两者之间分配 Token Budget。

常见优化：

- **Prefill**：FlashAttention、Prefix Caching、Chunked Prefill。
- **Decode**：Continuous Batching、CUDA Graph、GQA/MQA、KV Cache 量化、Speculative Decoding。
- **系统层**：将长 Prefill 切块，或使用 Prefill/Decode 分离部署，降低阶段间干扰。

## 7. 面试快速回答

**Prefill 和 Decode 为什么性能特征不同？** Prefill 同时处理大量 Token，矩阵乘大、并行度高；Decode 每轮只有一个新 Token，需要反复读取权重和历史 KV Cache，通常更受显存带宽限制。

**有 KV Cache 后，Decode 为什么仍会变慢？** 它虽然不再重新计算历史 K、V，但当前 Query 仍需读取越来越长的历史 KV；上下文越长，单步读取量越大。

**Chunked Prefill 解决什么问题？** 它把长 Prompt 切成多个调度块，使 Prefill 能与 Decode 交错执行，减少长 Prefill 对在线 Token 输出的阻塞，但会增加调度复杂度。

## 总结

Prefill 决定“多久看到第一个 Token”，Decode 决定“后续 Token 出得是否流畅”。优化 LLM Serving 时必须分别分析 TTFT 与 TPOT，并处理两个阶段对 GPU 计算和显存带宽的竞争。
