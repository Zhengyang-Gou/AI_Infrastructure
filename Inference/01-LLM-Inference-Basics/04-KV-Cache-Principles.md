# KV Cache 原理

## 1. 为什么需要 KV Cache

自回归生成每次只增加一个 Token。对于已经生成的历史 Token，它们在每层 Attention 中的 K、V 不会改变。

如果不缓存，生成新 Token 时必须重新计算整个上下文：

```text
第 1 步：计算 token 1...S
第 2 步：重新计算 token 1...S+1
第 3 步：重新计算 token 1...S+2
```

KV Cache 将历史 K、V 保存下来，使每轮 Decode 只计算新 Token：

```text
Q_new, K_new, V_new = project(new_token)
output = Attention(Q_new, [K_cache; K_new], [V_cache; V_new])
```

因此，KV Cache 的本质是：**用额外显存换取更少的重复计算。**

## 2. 缓存的具体内容

每个 Transformer Layer 都有独立的 KV Cache，通常保存经过投影和位置编码后的 K，以及投影后的 V：

```text
K cache: [num_kv_heads, sequence_length, head_dim]
V cache: [num_kv_heads, sequence_length, head_dim]
```

为什么不缓存 Q？历史 Query 只在对应 Token 生成时使用一次；未来 Token 只需要用新的 Q 查询历史 K、V。

Prefill 会一次性写入所有 Prompt Token 的 K、V；Decode 每轮读取历史 Cache，并追加当前 Token 的 K、V。

## 3. 显存占用如何计算

单条序列的 KV Cache 大小近似为：

```text
2 × layers × tokens × kv_heads × head_dim × bytes_per_element
```

- `2`：Key 和 Value。
- `tokens`：Prompt 与已生成 Token 的总长度。
- `kv_heads`：MHA 中通常等于 Query Head 数，GQA/MQA 中更少。
- FP16/BF16 每个元素通常为 2 Bytes。

示例：32 层、32 个 KV Head、Head Dimension 为 128、FP16：

```text
每 Token = 2 × 32 × 32 × 128 × 2 Bytes = 512 KiB
2048 Tokens ≈ 1 GiB
```

若使用 8 个 KV Head 的 GQA，同样长度只需约 `256 MiB`。这也是 GQA 能显著提高推理并发量的重要原因。

在线 Serving 中，各请求长度不同，总占用应按所有活跃请求的 Token 数累计。

## 4. 生命周期

```text
请求到达
  ↓
Prefill：分配空间并写入 Prompt KV
  ↓
Decode：读取历史 KV，追加新 Token KV
  ↓
请求完成或取消
  ↓
释放 KV Cache
```

管理系统还可能支持：

- **Prefix Caching**：多个请求共享相同前缀的 KV Block。
- **Copy-on-Write**：共享前缀发生分叉时，只复制需要修改的部分。
- **Preemption**：显存不足时暂停或移出部分请求。
- **Sliding Window**：模型只关注固定窗口，旧 KV 可被丢弃。

## 5. 连续内存与分页管理

简单实现会为每条请求预留一段连续空间，但输出长度通常无法提前确定：

- 按最大长度预留会浪费显存。
- 按当前长度分配又难以原地扩展。
- 不同长度的请求反复进入和退出会产生碎片。

Paged KV Cache 将 Cache 划分为固定大小的物理 Block，并用 Block Table 将逻辑 Token 位置映射到物理位置。请求增长时只需申请新 Block，不要求物理连续。这是 vLLM PagedAttention 的关键基础。

## 6. KV Cache 的性能代价

KV Cache 避免了历史 Token 的重复前向计算，但没有让 Attention 变成常数开销：

- 当前 Query 仍要读取全部有效历史 K、V。
- 上下文越长，每轮 Decode 的读取量越大。
- Decode 常受显存带宽限制，Cache 布局和访问连续性会影响效率。
- Cache 占用越大，GPU 能同时容纳的请求越少。

常见优化方向包括 GQA/MQA、低精度 KV Cache、PagedAttention、Prefix Caching 和只保留局部窗口。

## 7. 面试快速回答

**KV Cache 为什么只缓存 K、V？** 新 Token 的 Query 只需要与历史 K 做匹配，再对历史 V 加权；历史 Query 不会再次使用。

**KV Cache 是否降低了 Attention 的复杂度？** 它消除了历史 Token 的重复投影和前向计算，但单步 Decode 仍需访问长度为 `L` 的历史 KV，Attention 开销仍随 `L` 线性增长。

**PagedAttention 解决什么问题？** 它用固定大小的 Block 管理动态增长的 KV Cache，减少预留浪费和内存碎片，并支持更灵活的共享与回收。

## 总结

KV Cache 是 LLM 高效自回归推理的基础：它让 Decode 每轮只计算一个新 Token，但代价是持续增长的显存占用和读取带宽。对 Serving 系统而言，KV Cache 管理效率直接决定最大并发量、Decode 延迟和显存利用率。

---

[学习目录](README.md) · [上一篇：Prefill 与 Decode](03-Prefill-and-Decode.md) · [下一篇：Sampling Strategies](05-Sampling-Strategies.md)
