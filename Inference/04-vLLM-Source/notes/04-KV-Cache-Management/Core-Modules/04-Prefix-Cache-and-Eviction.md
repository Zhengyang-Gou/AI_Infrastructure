# 04 · 前缀缓存与淘汰

## 1. 职责与边界

本篇负责前缀的身份、查找、登记与失效。输入是请求 token 历史、隔离元数据和块池索引；输出是可复用块与命中长度。取得引用使用块池 touch 接口，不重讲引用数实现。

## 2. 源码定位与索引

| 文件 | 重点 |
| --- | --- |
| [kv_cache_utils.py](../../../vllm/vllm/v1/core/kv_cache_utils.py) | hash_block_tokens、get_request_block_hasher、resolve_kv_cache_block_sizes |
| [kv_cache_manager.py](../../../vllm/vllm/v1/core/kv_cache_manager.py) | get_computed_blocks、prefix_cache_lookup_enabled |
| [single_type_kv_cache_manager.py](../../../vllm/vllm/v1/core/single_type_kv_cache_manager.py) | FullAttentionManager.find_longest_cache_hit、cache_blocks |
| [block_pool.py](../../../vllm/vllm/v1/core/block_pool.py) | get_cached_block、cache_full_blocks、_maybe_evict_cached_block、evict_blocks |

主线中 hash_block_size、scheduler_block_size 与缓存 block_size 一致。前缀身份还带 group_id，避免不同缓存组的相同 token hash 被混用。

## 3. 实现分析

### 3.1 当前块的身份包含历史前缀

```python
# 文件：vllm/v1/core/kv_cache_utils.py
# 类：无（模块级函数）
# 函数：hash_block_tokens
# 源码范围：L617–L623（连续节选）
if not parent_block_hash:
    parent_block_hash = NONE_HASH

curr_block_token_ids_tuple = tuple(curr_block_token_ids)
return BlockHash(
    hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
)
```

hash 输入包含父块 hash、当前 token 序列和 extra_keys。同样的当前 token，在不同历史前缀下不能直接复用 KV。extra_keys 可包含 cache_salt、LoRA、多模态或 embedding 相关身份；本章只把它们作为缓存隔离输入，不解释内容预处理。

get_request_block_hasher 只为新增完整 hash 单元生成链式 hash。生成 hash 表明 token 内容已知，不表明对应 K/V 已经计算。

### 3.2 命中必须是可连续复用的前缀

get_computed_blocks 返回三元组：`(KVCacheBlocks, num_new_computed_tokens, shared_prefix_boundary)`。单组主线最后一项为 0；其他分组使用它记录共享前缀边界。

入口先限制最多命中的位置：

```python
# 文件：vllm/v1/core/kv_cache_manager.py
# 类：KVCacheManager
# 函数：get_computed_blocks
# 源码范围：L259–L264（连续节选）
max_cache_hit_length = request.num_tokens - 1
computed_blocks, num_new_computed_tokens, num_uncached = (
    self.coordinator.find_longest_cache_hit(
        request.block_hashes, max_cache_hit_length
    )
)
```

需要留下最后一个已知 token 的前向以得到 logits。完整块对齐可能让重算范围大于一个 token；不能将“所有 Prompt 的 KV 都存在”理解为本次完全无需前向。

FullAttention 的完整块查询从头开始，一旦有一块未命中便停止：

```python
# 文件：vllm/v1/core/single_type_kv_cache_manager.py
# 类：FullAttentionManager
# 函数：find_longest_cache_hit
# 源码范围：L731–L739（连续节选）
# Phase 1: longest run of cached full blocks from the start. A missing
# block implies every later block misses too (chained hashes).
for block_hash in itertools.islice(full_block_hashes, max_length // block_size):
    cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
    if not cached_block:
        break
    for computed, cached in zip(computed_blocks, cached_block):
        computed.append(cached)
hit_length = len(computed_blocks[0]) * block_size
```

这里只查索引、收集候选，不增加新请求的引用。接入与容量兑现由 allocate_slots 完成。

### 3.3 登记和复用不等于复制 GPU 数据

cache_full_blocks 根据请求 hash 将完整块登记到 hash+group_id 索引；同一内容可以有不止一个物理块，索引取一个可用候选。请求正式计算范围推进时，只补登记尚未缓存的部分。

本地 allocate_slots 会在执行下发前登记可缓存的范围。此时必须区分“可按 hash 找到”和“写入已结算”：登记本身不执行 GPU 完成检查，安全复用还依赖调用方与执行路径的顺序约束。本篇只解释缓存元数据，不凭 hash 登记推断 GPU 完成。

复用完整命中块时使用同一物理 ID 并取得引用，普通完整块前缀不要求复制 K/V。

### 3.4 淘汰移除身份，归还只减少引用

```python
# 文件：vllm/v1/core/block_pool.py
# 类：BlockPool
# 函数：_maybe_evict_cached_block
# 源码范围：L694–L700（连续节选）
evicted_hashes = self._remove_cached_block_hashes(block)
if not evicted_hashes:
    # The block doesn't have hash, eviction is not needed
    return False

self._emit_block_removed_events(evicted_hashes)
return True
```

当空闲缓存块被借作新内容时，旧身份必须移除。evict_blocks 也可让仍有引用的块失去前缀身份，但不会仅因此把该块归还到空闲队列。

reset_prefix_cache 需要没有普通请求占用；源码用“只剩 null block 在用”作为检查条件，不是对正在使用的 KV 强行全部释放。

### 3.5 本地细粒度分支的范围

本地还支持多组缓存下更细的 hash 单元，以及 FullAttentionManager 的部分尾块查找和 CoW。部分命中后，继续写入需要私有块并安排复制，不能直接改共享尾块。

但单组 UnitaryKVCacheCoordinator 要求 hash_block_size 等于块大小，基础主线不进入该细粒度分支。阅读 docstring 中“只缓存完整块”时，也要检查具体配置和函数体，不能将它推广到所有本地实现。

## 4. 手工推演：完整命中仍需留出前向

给定 block_size=16，Prompt 有 32 个 token，已有对应的两个完整缓存块：

```text
最多复用位置 = 32−1 = 31
主线按完整块向下对齐：floor(31/16) = 1 个块
命中 16 个 token；剩余 16 个 token 需要重算
```

如果另一个请求在不同的历史前缀后出现相同的第二块 token，其父 hash 不同，也不会与本例第二块共用身份。

## 5. 设计原因与自测

- 为什么只用当前块 token 作为键不够？
- 查询命中时是否已经取得请求引用？
- 为什么 Prompt 恰好落在块边界时仍可能重算一整块？
- hash 被移除后，持有该块的请求是否必然结束？
- 单组完整块规则能否直接套到细粒度混合缓存？

[章节目录](../README.md) · [本章概览](../Overview.md)
