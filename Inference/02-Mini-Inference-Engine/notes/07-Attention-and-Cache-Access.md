# 07 · Attention 与 KV Cache 读写

把第 03 篇的页表、第 05 篇的 Context 与实际 Attention 计算连接起来。

对应源码：[attention.py](../source/nano-vllm/nanovllm/layers/attention.py)。

## attention.py

```python
@triton.jit
def store_kvcache_kernel(...):
    # 每个 Triton Program 负责一个 Token。
    idx = tl.program_id(0)
    # 读取当前 Token 对应的扁平物理槽位。
    slot = tl.load(slot_mapping_ptr + idx)
    # 读取当前 Token 对应的扁平物理槽位。
    # CUDA Graph 补齐位置不写缓存。
    if slot == -1:
        return
    ...
    # 写入该 Token 的全部 KV Head 和 Head 维度。
    tl.store(k_cache_ptr + cache_offsets, key)
    # V 使用与 K 对应的物理槽位。
    tl.store(v_cache_ptr + cache_offsets, value)
```

**功能描述：** 将本轮计算出的 K/V 散写到分页缓存中的指定槽位，使后续注意力计算可以通过页表读取历史数据。

```python
def forward(self, q, k, v):
    # 读取执行器为本批次准备的元数据。
    context = get_context()
    # 缓存非空才执行写入。
    if k_cache.numel() and v_cache.numel():
        # 按 slot_mapping 将新 K/V 写入本层缓存。
        store_kvcache(
            k, v, k_cache, v_cache, context.slot_mapping
        )
```

**功能描述：** 在每层 Attention 前向中更新该层 KV Cache。预热时缓存尚未分配，因此跳过写入，直接使用本轮 K/V 计算。

```python
if context.is_prefill:
    # 存在页表时改用分页 K/V，以包含已缓存前缀。
    if context.block_tables is not None:
        k, v = k_cache, v_cache
    # 用变长内核处理扁平拼接的 Query。
    o = flash_attn_varlen_func(
        q, k, v,
        # Query 和 Key 分别给出序列边界。
        cu_seqlens_q=context.cu_seqlens_q,
        cu_seqlens_k=context.cu_seqlens_k,
        # 提供本批次最大长度用于内核执行。
        max_seqlen_q=context.max_seqlen_q,
        max_seqlen_k=context.max_seqlen_k,
        softmax_scale=self.scale,
        # 每个位置只能关注自身及之前的 Token。
        causal=True,
        # 将逻辑块映射到物理缓存块。
        block_table=context.block_tables,
    )
```

**功能描述：** 计算变长批次的 Prefill 注意力。普通输入使用本轮 K/V，存在缓存前缀时通过页表读取完整上下文，同时保持因果可见范围。

```python
else:
    # 使用适合逐 Token 解码的缓存注意力接口。
    o = flash_attn_with_kvcache(
        # 补出长度为 1 的 Query 序列维。
        q.unsqueeze(1), k_cache, v_cache,
        # 每条序列仅访问有效上下文。
        cache_seqlens=context.context_lens,
        # 从逻辑位置定位物理缓存块。
        block_table=context.block_tables,
        softmax_scale=self.scale,
        # 维持自回归因果约束。
        causal=True,
    )
```

**功能描述：** 计算 Decode 的单 Query 注意力，通过页表直接访问离散的历史 KV 块，避免为每条序列复制或拼接完整缓存。

---

[学习目录](README.md) · [上一篇：Qwen3 模型与基础算子](06-Qwen3-and-Basic-Layers.md) · [下一篇：张量并行与权重加载](08-Tensor-Parallel-and-Weight-Loading.md)
