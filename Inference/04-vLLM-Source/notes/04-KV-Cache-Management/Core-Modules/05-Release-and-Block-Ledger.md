# 05 · 释放与块账本

## 1. 职责与边界

本篇输入是外部发来的块释放指令，负责请求块账本退出与引用交还。谁决定请求结束、是否抢占由 03 负责；本篇不重讲判停和调度队列。

## 2. 源码定位与安全契约

| 文件 | 重点 |
| --- | --- |
| [kv_cache_manager.py](../../../vllm/vllm/v1/core/kv_cache_manager.py) | free、pop_blocks_for_free、remove_skipped_blocks |
| [kv_cache_coordinator.py](../../../vllm/vllm/v1/core/kv_cache_coordinator.py) | 将释放指令交给各组，汇总待归还块 |
| [single_type_kv_cache_manager.py](../../../vllm/vllm/v1/core/single_type_kv_cache_manager.py) | pop_blocks_for_free、free |

调用者必须确认块不会再被在途工作读写，或先把仍持有引用的块交给延迟归还路径。请求从账本消失和块可供新内容使用不是同一时刻。

## 3. 实现分析

### 3.1 立即归还：移出账本，再逆序交还

```python
# 文件：vllm/v1/core/single_type_kv_cache_manager.py
# 类：SingleTypeKVCacheManager
# 函数：pop_blocks_for_free
# 源码范围：L513–L517（连续节选）
# Default to [] in case a request is freed (aborted) before alloc.
req_blocks = self.req_to_blocks.pop(request_id, [])
self.num_cached_block.pop(request_id, None)
self._partial_hit_reqs.pop(request_id, None)
return req_blocks
```

请求的块列表、已缓存块进度及部分命中记录一起移出。此方法返回原来的块对象，但没有减少它们的引用数。

free 再把列表逆序交给池：

```python
# 文件：vllm/v1/core/single_type_kv_cache_manager.py
# 类：SingleTypeKVCacheManager
# 函数：free
# 源码范围：L519–L527（连续节选）
def free(self, request_id: str) -> None:
    """
    Free the blocks for the request.

    Args:
        request_id: The request ID.
    """
    # Free blocks in reverse order so that the tail blocks are freed first.
    self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))
```

逆序让较后面的内容先进入对应的归还优先次序。最终队列还会按有无 hash 分流，具体规则只在 Core 02 展开。

### 3.2 延迟归还：账本已经退出，引用仍需保留

KVCacheManager.pop_blocks_for_free 汇总各组块以及必要的额外 pin，交给调用者延后传入 BlockPool.free_blocks。它本身不代表块已经可借，也不能省略最后的归还。

这种拆分使请求可以停止被安排，同时防止未完成的 GPU 写入覆盖后来借用相同块 ID 的新内容。在途队列如何驱动与何时宣布完成不在本章展开。

### 3.3 窗口回收使用安全处理位置

窗口类缓存还可能在请求未结束时移除不再需要的块。allocate_slots 传入回收接口的起点按下式扣除在途量：

```python
# 文件：vllm/v1/core/kv_cache_manager.py
# 类：KVCacheManager
# 函数：allocate_slots
# 源码范围：L504–L508（连续节选）
self.coordinator.remove_skipped_blocks(
    request.request_id,
    max(0, total_computed_tokens - request.num_in_flight_tokens),
    num_prompt_tokens=request.num_prompt_tokens,
)
```

因此不能仅凭乐观推进的 num_computed_tokens 就回收窗口外存储。基础 FullAttention 不做滑窗丢弃；具体滑窗和 Mamba 规则不套到案例 B。

## 4. 案例 B：唯一的完整块账本

条件：一条 600-token Prompt，block_size=16；单组 FullAttention；无前缀命中、无额外预留和细粒度部分块缓存；容量足够。外部已给定三轮计算位置为 **256 / 256 / 88**。本篇只对块，不解释这些计算量为何被选中。

| 轮次 | 本次新覆盖位置数 | 累计覆盖位置数 | 请求持有块数 | 本次新增块 | 块内有效范围 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 256 | 256 | 16 | 16 | 16 个满块 |
| 2 | 256 | 512 | 32 | 16 | 32 个满块 |
| 3 | 88 | 600 | 38 | 6 | 37 个满块，尾块 8 个有效位置 |

```text
总块数 = ceil(累计覆盖位置数 / 16)
本次新增 = 总块数 − 请求已有块数
第三轮：ceil(600/16)−32 = 38−32 = 6
```

每个普通块只有这一个请求引用时，持有期间 ref_cnt=1。用 P0…P37 表示分配到的物理块：它们的真实 ID 不必连续，也不是默认的 0…37；0 是池保留的 null block。

这张表只统计 Prompt KV。采样得到的下一个 token 尚未因此自动写入 KV；后续 Decode 不在本算例展开。

假设上述写入均已结算，此后调用者发出释放指令：

| 对象 | 释放后的状态 |
| --- | --- |
| req_to_blocks[该请求] | 被删除 |
| P0…P36 | 引用归零，可保留完整块 hash，作为空闲缓存候选 |
| P37 | 引用归零，基础场景无完整块 hash，进入优先复用的空闲部分 |
| GPU KV 大张量 | 继续存在，物理槽可重新借用；不因一次请求结束而释放整块 GPU 分配 |

若其中某些块还被其他请求引用，相应 ref_cnt 不会归零，也不会加入空闲队列。这里没有重复构造 slot_mapping 或用户结果。

## 5. 设计原因与自测

- pop_blocks_for_free 为什么不能等同于 free_blocks？
- 请求已结束，为什么仍可能有块暂时不能复用？
- 案例 B 为什么是 38 个块，而不是 600 个块或 38×层数？
- 第三轮新增 6 块，为什么只有 5 个新增满块？
- 释放后的 37 个 hash 是否仍然都可立即命中，还取决于什么执行与淘汰条件？

[章节目录](../README.md) · [本章概览](../Overview.md)
