# 07 · Attention 与 KV Cache 读写

把第 03 篇的页表、第 05 篇的 Context 与实际 Attention 计算连接起来。

对应源码：[attention.py](../source/nano-vllm/nanovllm/layers/attention.py)。

## attention.py

```python
@triton.jit
def store_kvcache_kernel(...):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    ...
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```
自定义 Triton Kernel 将本轮产生的 K/V 写入 Paged KV Cache：

- 每个 Program 负责一个 Token
- `slot_mapping[idx]` 给出该 Token 的扁平物理槽位
- 一个 Token 的所有 KV Head 和 Head Dim 被当作长度 `D` 的连续区域复制
- `slot=-1` 用于 CUDA Graph Padding，表示该位置不应写缓存

```python
def forward(self, q, k, v):
    context = get_context()
    if k_cache.numel() and v_cache.numel():
        store_kvcache(
            k, v, k_cache, v_cache, context.slot_mapping
        )
```
Attention 每一层首先把新 K/V 写入该层自己的缓存。模型预热发生在 KV Cache 分配前，此时 Cache 是空张量，因此跳过写入。

```python
if context.is_prefill:
    if context.block_tables is not None:
        k, v = k_cache, v_cache
    o = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=context.cu_seqlens_q,
        cu_seqlens_k=context.cu_seqlens_k,
        max_seqlen_q=context.max_seqlen_q,
        max_seqlen_k=context.max_seqlen_k,
        softmax_scale=self.scale,
        causal=True,
        block_table=context.block_tables,
    )
```
Prefill 使用变长 FlashAttention：

- 普通 Prefill：直接使用本轮连续的 Q/K/V
- 命中前缀缓存：K/V 改为整个分页缓存，并通过 `block_table` 找到当前序列的历史块
- `cu_seqlens` 描述扁平 Batch 中每条序列的边界
- `causal=True` 保证 Token 只能看到自己及之前的位置

```python
else:
    o = flash_attn_with_kvcache(
        q.unsqueeze(1), k_cache, v_cache,
        cache_seqlens=context.context_lens,
        block_table=context.block_tables,
        softmax_scale=self.scale,
        causal=True,
    )
```
Decode 时每条序列只有一个 Query，使用专门的 KV Cache Attention：

- `context_lens` 限制每条序列的有效缓存长度
- `block_table` 将逻辑位置映射到非连续的物理块
- 不需要复制、拼接每条序列的历史 K/V

因此 PagedAttention 的核心并不是改变注意力公式，而是让注意力 Kernel 能通过页表直接读取离散物理块。

---

[学习目录](README.md) · [上一篇：Qwen3 模型与基础算子](06-Qwen3-and-Basic-Layers.md) · [下一篇：张量并行与权重加载](08-Tensor-Parallel-and-Weight-Loading.md)
