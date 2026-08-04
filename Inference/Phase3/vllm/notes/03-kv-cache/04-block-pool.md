# Block Pool

## 1. 文件定位

- 文件路径：`vllm/v1/core/block_pool.py`
- 所属层次：KV Cache block 资源与 Prefix Cache 索引层
- 核心职责：统一管理全部 `KVCacheBlock`，执行 block 分配、引用计数、释放、哈希查找和缓存驱逐。
- 在调用链中的位置：被所有 `SingleTypeKVCacheManager` 共享，是 Scheduler 侧 KV block 元数据的最底层资源池。

`BlockPool` 管理的是 block 元数据和 `block_id`，不直接保存 GPU 上的 K/V tensor。Worker 通过同一个 `block_id` 找到实际的 KV Cache page。

## 2. 核心类与组件

| 类 / 组件 | 作用 |
| --- | --- |
| `BlockHashToBlockMap` | 保存 `(block_hash, group_id) → KVCacheBlock` 的 Prefix Cache 索引 |
| `BlockPool` | 管理 block 总量、分配、释放、缓存和驱逐 |
| `KVCacheBlock` | 保存 `block_id`、`ref_cnt`、hash 和空闲链表指针 |
| `FreeKVCacheBlockQueue` | 用双向链表维护空闲及可驱逐 blocks 的顺序 |
| `null_block` | 表示无需实际 KV 存储的 block table 占位项 |

> `KVCacheBlock` 与 `FreeKVCacheBlockQueue` 实际定义在 `vllm/v1/core/kv_cache_utils.py`，本文件导入并使用它们。

### `KVCacheBlock` 关键字段

| 字段 | 作用 |
| --- | --- |
| `block_id` | Block Pool 内唯一编号，并对应 Worker 侧 KV page |
| `ref_cnt` | 当前持有该 block 的请求或临时操作数量 |
| `block_hash` | Prefix Cache 使用的 `(hash, group_id)` 键 |
| `block_hash_num_tokens` | 该 hash 表示的前缀 token 长度 |
| `prev_free_block` / `next_free_block` | 空闲双向链表指针 |
| `is_null` | 标记不会被正常分配和缓存的占位 block |

## 3. 主执行流程

### 初始化

```text
BlockPool.__init__()
→ 创建 num_gpu_blocks 个 KVCacheBlock
→ 全部放入 FreeKVCacheBlockQueue
→ 创建 BlockHashToBlockMap
→ 取出一个 block 作为 null_block
→ null_block 不再作为普通空闲 block 分配
```

### 分配新 blocks

```text
get_new_blocks(num_blocks)
→ 从 free_block_queue 头部取出 blocks
→ 如果 block 带旧 hash，则先从 Prefix Cache 驱逐
→ 确认 ref_cnt == 0
→ ref_cnt 增加为 1
→ 返回给 SingleTypeKVCacheManager
```

空闲队列中既可以有从未缓存的普通空闲 blocks，也可以有 `ref_cnt == 0`、仍带 hash 的 Prefix Cache 驱逐候选。

### 写入 Prefix Cache

```text
cache_full_blocks()
→ 根据 Request.block_hashes 取得对应前缀 hash
→ 组合 kv_cache_group_id
→ 将 hash 写入 KVCacheBlock
→ 插入 BlockHashToBlockMap
→ block 保持当前引用，不改变请求 block table
```

同一个 hash 可以对应多个物理 blocks。实现不会为相同内容强制去重，以保持已分配 block table 的 block IDs 只追加、不被替换。

### 查询并复用缓存

```text
get_cached_block(block_hash, group_ids)
→ 为每个 group 组合 hash key
→ BlockHashToBlockMap.get_one_block()
→ 任一 group 缺失则整体返回 None
→ 命中 blocks 交给 manager
→ BlockPool.touch()
→ 若 ref_cnt == 0，先从 free queue 移除
→ ref_cnt += 1
```

### 释放 blocks

```text
free_blocks(ordered_blocks)
→ 每个 block.ref_cnt -= 1
→ ref_cnt > 0：仍被其他请求共享，不回收
→ ref_cnt == 0 且无 hash：放到空闲队列前部，优先复用
→ ref_cnt == 0 且有 hash：放到空闲队列尾部，作为缓存保留
```

这使 Prefix Cache 具备近似 LRU 的驱逐顺序：无缓存价值的 block 优先分配，带 hash 的缓存 block 尽量延后驱逐。

### 重置 Prefix Cache

```text
reset_prefix_cache()
→ 确认除 null_block 外没有正在使用的 blocks
→ 清空哈希索引
→ 重置所有 block hash 元数据
→ 保留 block 对象和 free queue
```

## 4. 输入与输出

### 输入

- 初始化输入：GPU block 总数、Prefix Cache 开关、hash block size 和事件配置。
- 分配输入：需要的 block 数。
- 查找输入：`block_hash` 和一个或多个 KV Cache group IDs。
- 缓存输入：`Request`、请求 block table、完整 block 范围和 group ID。
- 释放输入：按驱逐优先级排列的 blocks。

### 输出

- 新分配的 `list[KVCacheBlock]`。
- Prefix Cache 命中时，各 group 对应的 `list[KVCacheBlock]`；任一 group miss 时为 `None`。
- 空闲 block 数、KV Cache 使用比例和可选的 KV Cache 事件。

### 状态变化

- 分配会从空闲队列移除 blocks、驱逐旧 hash 并增加引用计数。
- `touch()` 会保护缓存命中 blocks，增加其引用计数。
- 缓存操作会把 hash 写入 block，并更新 hash 索引。
- 释放会减少引用计数，并在归零时按是否带 hash 决定空闲队列位置。
- 驱逐只清除 Prefix Cache hash；如果 `ref_cnt > 0`，block 仍由请求持有。

## 5. 关键代码解析

### `BlockHashToBlockMap.get_one_block()`

### `BlockHashToBlockMap.insert()`

### `BlockHashToBlockMap.pop()`

### `BlockPool.__init__()`

### `BlockPool.get_cached_block()`

### `BlockPool.cache_full_blocks()`

### `BlockPool.cache_partial_block()`

### `BlockPool.get_new_blocks()`

### `BlockPool._maybe_evict_cached_block()`

### `BlockPool.touch()`

### `BlockPool.free_blocks()`

### `BlockPool.evict_blocks()`

### `BlockPool.reset_prefix_cache()`

### `BlockPool.get_num_free_blocks()`

### `BlockPool.get_usage()`

## 6. 与其他文件的关系

- 上游：`vllm/v1/core/single_type_kv_cache_manager.py`。
- 间接上游：`KVCacheCoordinator` 和 `KVCacheManager`。
- block 与队列定义：`vllm/v1/core/kv_cache_utils.py`。
- 请求 hash 来源：`vllm/v1/request.py` 和 `kv_cache_utils.py` 的 block hash 生成逻辑。
- 配置来源：`KVCacheConfig.num_blocks` 最终决定 Block Pool 容量。
- 下游连接：`block_id` 经 `SchedulerOutput` 传到 Worker，映射到实际 GPU KV Cache page。

## 7. 当前结论

`BlockPool` 把有限 KV Cache 容量组织成一个带引用计数和哈希索引的共享资源池：正在使用的 blocks 被保护，空闲的缓存 blocks 可继续命中，同时也能在容量不足时按顺序驱逐复用。
