# 03 · 请求块与增量分配

## 1. 职责与边界

输入是外部给定的新计算量、前缀命中长度与块引用；输出是新增块集合或 None。本篇负责一次分配怎样兑现，不决定计算量、是否抢占或抢占谁。

## 2. 源码定位与请求账本

| 文件 | 重点 |
| --- | --- |
| [kv_cache_manager.py](../../../vllm/vllm/v1/core/kv_cache_manager.py) | allocate_slots、get_blocks、get_block_ids、KVCacheBlocks |
| [kv_cache_coordinator.py](../../../vllm/vllm/v1/core/kv_cache_coordinator.py) | coordinator 选择、按组预测和分配 |
| [single_type_kv_cache_manager.py](../../../vllm/vllm/v1/core/single_type_kv_cache_manager.py) | req_to_blocks、get_num_blocks_to_allocate、add_local_computed_blocks、allocate_new_blocks |

| 状态/参数 | 本篇中的含义 |
| --- | --- |
| req_to_blocks[request_id] | 一个分组内，请求持有的有序块引用 |
| num_new_tokens | 本次外部要求计算、需要存储的新 token 数 |
| num_new_computed_tokens | 本次新找到的本地前缀命中长度 |
| new_computed_blocks | 上述命中对应的块，尚需接入请求并取得引用 |
| num_tokens_need_slot | 旧进度、命中、新计算量及额外预留合成的存储范围 |

本章单组场景使用 UnitaryKVCacheCoordinator 和 FullAttentionManager。多组时 coordinator 汇总各组需求，不把它们当成同一个块表。

## 3. 实现分析

### 3.1 先算覆盖范围，再检查能否兑现

allocate_slots 先合成已知的本地/外部计算进度，再确定本次需要覆盖到的位置：

```python
# 文件：vllm/v1/core/kv_cache_manager.py
# 类：KVCacheManager
# 函数：allocate_slots
# 源码范围：L490–L493（连续节选）
num_tokens_main_model = total_computed_tokens + num_new_tokens
num_tokens_need_slot = min(
    num_tokens_main_model + num_lookahead_tokens, self.max_model_len
)
```

固定主线中 external 和 lookahead 均为 0。请求需要的逻辑块数为 ceil(覆盖位置数/block_size)，但本次只补足已有块之外的部分。

full_sequence_must_fit 是额外容量门槛：检查当前已知完整序列是否能容纳，不等于立刻分配未来全部输出。reserved_blocks 和 watermark 会减少可用余量；调度器何时传这些参数不在本篇展开。

### 3.2 容量检查还要计算空闲的命中块

```python
# 文件：vllm/v1/core/single_type_kv_cache_manager.py
# 类：SingleTypeKVCacheManager
# 函数：get_num_blocks_to_allocate
# 源码范围：L215–L230（连续节选）
# Among the `new_computed_blocks`, the first `num_skipped_blocks` worth
# of blocks are skipped; `num_req_blocks` of those may already be in
# `req_to_blocks`, so only skip the remainder from `new_computed_blocks`.
num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

# If a computed block is an eviction candidate (in the free queue and
# ref_cnt == 0), it will be removed from the free queue when touched by
# the allocated request, so we must count it in the free-capacity check.
num_evictable_blocks = self._get_num_evictable_blocks(
    new_computed_blocks[num_skipped_new_computed_blocks:]
)
if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
    # Reserve the extra block that allocate_new_blocks pulls for the
    # partial-hit CoW redirect.
    num_new_blocks += 1
return num_new_blocks + num_evictable_blocks
```

命中块若 ref_cnt=0，接入请求时会从空闲队列移出，也要计入可用容量消耗；若 ref_cnt>0，已经不在空闲队列，不再额外扣一次。细粒度部分命中可能需要一个 CoW 私有块，该扩展不参与基础算例。

检查失败返回 None。不能将整个 allocate_slots 理解为完全无副作用的试探：源码可能在最终容量检查前移除窗口外已安全可弃的块。基础 FullAttention 场景没有这种窗口回收。

### 3.3 接入命中块，再追加新块

容量通过后，命中块通过 add_local_computed_blocks 接入：取得引用，扩展 req_to_blocks，并记录已缓存前缀范围。hash 匹配本身由 Core 04 负责，此处只消费匹配结果。

追加新块的主分支是：

```python
# 文件：vllm/v1/core/single_type_kv_cache_manager.py
# 类：SingleTypeKVCacheManager
# 函数：allocate_new_blocks
# 源码范围：L359–L369（连续节选）
req_blocks = self.req_to_blocks[request_id]
num_required_blocks = cdiv(num_tokens, self.block_size)
num_new_blocks = num_required_blocks - len(req_blocks)
if num_new_blocks <= 0:
    return cow_blocks
else:
    new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
    req_blocks.extend(new_blocks)
    if self._record_new_block_ids:
        self.new_block_ids.extend(b.block_id for b in new_blocks)
    return cow_blocks + new_blocks
```

列表中已有的引用继续保留；新增块由池借出后追加。没有新增需求时可返回空集合，不能把它与 None 的分配失败混淆。

allocate_slots 返回的是**新增分配块**。若需要请求完整块表，应使用 get_blocks/get_block_ids；KVCacheBlocks 按组包装结果，执行侧随后消费块 ID。

### 3.4 分配与缓存登记的交接

启用前缀缓存时，allocate_slots 还调用 cache_blocks，为可提交的 token 范围登记 hash；范围不超过正式 token 历史。登记不等于 GPU 已完成写入，调用者必须维持执行与复用的安全顺序。hash 身份规则由 Core 04 负责。

## 4. 最小分配算例

给定 block_size=16，请求已经持有 2 个块，本次要求存储范围覆盖到第 40 个位置，无命中、无预留：

```text
需要块数 = ceil(40/16) = 3
已有块数 = 2
新增块数 = 1
原有 [P, Q] → [P, Q, R]
返回新增集合 [R]；完整块集合为 [P, Q, R]
```

P/Q/R 是池返回的物理 ID 的代称；这一步不构造 slot_mapping。

## 5. 设计原因与自测

- 空集合与 None 分别表示什么？
- 为什么命中一个空闲块也会消耗可用块数？
- 为什么每轮只申请增量，而不是重新申请整条序列？
- full_sequence_must_fit 是否预留所有未来输出？
- 分配失败是否必然意味着块池完全没有空闲块？

[章节目录](../README.md) · [本章概览](../Overview.md)
