# 02 · 块池与引用计数

## 1. 职责与边界

本篇只解释单个块的状态和块池操作。调用者已经决定需要多少块；本篇不计算请求需求，不做前缀匹配，也不选择抢占对象。

## 2. 源码定位与持久状态

| 文件 | 重点 |
| --- | --- |
| [kv_cache_utils.py](../../../vllm/vllm/v1/core/kv_cache_utils.py) | KVCacheBlock、FreeKVCacheBlockQueue |
| [block_pool.py](../../../vllm/vllm/v1/core/block_pool.py) | BlockPool.__init__、get_new_blocks、touch、free_blocks |

| 状态 | 含义 |
| --- | --- |
| block_id | 物理存储槽的整数标识 |
| ref_cnt | 当前持有该块的引用数，不只由 hash 是否存在决定 |
| block_hash | 前缀查找身份；存在不表示当前有请求占用 |
| prev/next_free_block | 双向空闲链表链接，支持已知块的常数时间移除 |
| is_null | 占位块标记，需排除普通借还与缓存 |

KVCacheBlock 是 CPU 元数据，不包含实际 K/V 张量。所有块对象在池初始化时一次创建，之后重复借还。

## 3. 实现分析

### 3.1 null block 不参加普通借还

```python
# 文件：vllm/v1/core/block_pool.py
# 类：BlockPool
# 函数：__init__
# 源码范围：L187–L191（连续节选）
# To represent a placeholder block with block_id=0.
# The ref_cnt of null_block is not maintained, needs special care to
# avoid freeing it.
self.null_block = self.free_block_queue.popleft()
self.null_block.is_null = True
```

null block 从空闲队列取出，保留为占位。其 ref_cnt 不按普通块维护；不能仅凭 ref_cnt=0 将它当成可借块。

### 3.2 借出新的存储槽

```python
# 文件：vllm/v1/core/block_pool.py
# 类：BlockPool
# 函数：get_new_blocks
# 源码范围：L658–L668（连续节选）
if num_blocks > self.get_num_free_blocks():
    raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

# In order to only iterate the list once, we duplicated code a bit
if self.enable_caching:
    for block in ret:
        self._maybe_evict_cached_block(block)
        assert block.ref_cnt == 0
        block.ref_cnt += 1
```

调用先检查空闲数，再从队首取块。启用缓存时，借作新内容会移除该块的旧 hash 身份，然后将引用数从 0 增为 1。get_new_blocks 并不按请求 hash 查找缓存。

### 3.3 命中已有块时取得引用

```python
# 文件：vllm/v1/core/block_pool.py
# 类：BlockPool
# 函数：touch
# 源码范围：L710–L717（连续节选）
for block in blocks:
    # ref_cnt=0 means this block is in the free list (i.e. eviction
    # candidate), so remove it.
    if block.ref_cnt == 0 and not block.is_null:
        self.free_block_queue.remove(block)
    block.ref_cnt += 1
    if self.metrics_collector:
        self.metrics_collector.on_block_accessed(block)
```

块若在空闲队列中，touch 先将其移除；若本来已经被其他请求引用，则直接增加引用数。两种情况都不分配新的 GPU 存储。

### 3.4 归还引用和队列优先级

```python
# 文件：vllm/v1/core/block_pool.py
# 类：BlockPool
# 函数：free_blocks
# 源码范围：L728–L742（连续节选）
blocks_with_hash = []
blocks_without_hash = []
for block in ordered_blocks:
    block.ref_cnt -= 1
    if block.ref_cnt == 0 and not block.is_null:
        # When caching is disabled we always append for better
        # GPU cache locality from reusing recently used blocks
        if block.block_hash is None and self.enable_caching:
            blocks_without_hash.append(block)
        else:
            blocks_with_hash.append(block)

# Blocks without hash get evicted first - prepend them last to the tail
self.free_block_queue.prepend_n(blocks_without_hash)
self.free_block_queue.append_n(blocks_with_hash)
```

只有 ref_cnt 降到 0 的非 null 块重新进入空闲队列。启用缓存时，无 hash 的块被放到前部，优先借出；保留 hash 的块追加到后部。不能把本地实现概括为“所有释放块都简单追加队尾”。

是否有 hash 和是否空闲是两个维度：

| ref_cnt | hash | 状态 |
| ---: | --- | --- |
| >0 | 无 | 请求正在使用，尚无前缀身份 |
| >0 | 有 | 请求正在使用，也能参与前缀共享 |
| 0 | 有 | 可淘汰候选；数据仍可能命中 |
| 0 | 无 | 普通空闲块，优先用于新内容 |

此表排除 null block。hash 的建立与移除规则由 Core 04 负责。

## 4. 手工推演：共享块引用

给定普通块 P 已被请求 X 使用且已登记 hash：

```text
X 持有：ref_cnt=1
Y 命中并接入 P：touch，1→2
X 归还：2→1，不进空闲队列
Y 归还：1→0，进入空闲队列，hash 可保留
Z 再次命中：touch，0→1，并移出空闲队列
```

这段只检查块状态，不推演请求调度轮次。

## 5. 设计原因与自测

引用数负责“还能不能回收”，hash 负责“能不能按前缀找到”，空闲队列负责“下次优先借谁”。

- 空闲块为什么可能仍带 hash？
- 一个被其他请求引用的命中块，是否会消耗新的空闲块？
- free_blocks 是否会清零 GPU 张量？
- 为什么 null block 需要单独处理？

[章节目录](../README.md) · [本章概览](../Overview.md)
