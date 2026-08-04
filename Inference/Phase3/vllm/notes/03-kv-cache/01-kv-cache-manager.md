# KV Cache Manager

## 1. 文件定位

- 文件路径：`vllm/v1/core/kv_cache_manager.py`
- 所属层次：Scheduler 与 KV Cache 内部实现之间的统一接口层
- 核心职责：查询 Prefix Cache、计算和分配 slots、返回 block IDs、缓存完整 blocks，并在请求结束时释放 blocks。
- 在调用链中的位置：由 `Scheduler` 直接持有，向下委托 `KVCacheCoordinator` 和 `BlockPool`。

`KVCacheManager` 隐藏了单组与混合 KV Cache 的内部差异。Scheduler 只需要面向请求和 token 数工作，不需要直接操作每一种 attention 类型的 block table。

## 2. 核心类与组件

| 类 / 组件 | 作用 |
| --- | --- |
| `KVCacheBlocks` | 封装按 KV Cache group 组织的 blocks，并向 Scheduler 提供 block IDs |
| `KVCacheManager` | Scheduler 使用的 KV Cache 门面对象 |
| `coordinator` | 将统一操作协调到一个或多个 single-type managers |
| `block_pool` | 保存所有 blocks、空闲队列和 Prefix Cache 哈希索引 |
| `empty_kv_cache_blocks` | 预创建的不可变空结果，减少频繁分配与 GC |

### `KVCacheBlocks` 的组织方式

```text
blocks[group_index][token_block_index]
→ KVCacheBlock
→ block_id
```

外层按 KV Cache group 组织，是因为不同 group 将来可以拥有不同 block size；内层则按该请求的 token 位置顺序排列。

## 3. 主执行流程

### 初始化

```text
KVCacheManager.__init__()
→ get_kv_cache_coordinator()
→ 创建 NoPrefix / Unitary / Hybrid Coordinator
→ 从 Coordinator 取得共享 BlockPool
→ 保存 KVCacheConfig 与 group 数量
→ 根据 watermark 计算接纳请求时应保留的 blocks
```

### 查询 Prefix Cache

```text
get_computed_blocks(request)
→ 检查是否启用 Prefix Cache、请求是否跳过读取
→ max_cache_hit_length = request.num_tokens - 1
→ coordinator.find_longest_cache_hit()
→ 得到每组命中 blocks 和命中 token 数
→ 包装为 KVCacheBlocks
→ 返回 blocks、num_new_computed_tokens、shared_prefix_boundary
```

### 分配 slots

`allocate_slots()` 将 token 布局分成已计算、本地新命中、外部命中、本轮新 token 和 speculative lookahead 几部分，主要经过三个阶段：

```text
1. 清理已经超出 attention 可见范围的 blocks
   并计算所需新增 block 数

2. 接入 Prefix Cache / 外部 KV 命中的 blocks
   为仍需要物理 slots 的外部 token 分配 blocks

3. 为本轮新 token 和 lookahead token 分配 blocks
   并把可提交的完整 blocks 写入 Prefix Cache
```

容量判断同时考虑：

- 当前 Block Pool 可用 blocks。
- async KV load 为其他 in-flight 请求保留的 `reserved_blocks`。
- waiting / preempted 请求接纳时使用的 watermark。
- 可选的“整个序列必须容纳” admission gate。

若所需容量超过可用容量，方法返回 `None`，不会执行部分分配。

### 释放请求

```text
Scheduler 抢占或完成请求
→ KVCacheManager.free(request)
→ 释放 partial-tail pins
→ coordinator.free(request_id)
→ 各 single-type manager 清理 req_to_blocks
→ BlockPool.free_blocks()
```

## 4. 输入与输出

### 输入

- 初始化输入：`KVCacheConfig`、最大模型长度、调度与哈希 block size、Prefix Cache 开关和并行配置。
- 查找输入：`Request.block_hashes`、请求 token 数和 cache 读取策略。
- 分配输入：请求、本轮 token 数、Prefix Cache 命中 blocks、外部命中 token 数和 lookahead token 数。
- 释放输入：需要结束或被抢占的 `Request`。

### 输出

- `get_computed_blocks()` 返回 `KVCacheBlocks`、本地命中 token 数和共享前缀边界。
- `allocate_slots()` 成功时返回本轮新分配的 `KVCacheBlocks`，容量不足时返回 `None`。
- `get_block_ids()` 返回按 group 组织的 `tuple[list[int], ...]`。
- `usage` 返回当前 KV Cache 使用比例。
- 事件与统计接口返回 Prefix Cache 命中和 block 生命周期信息。

### 状态变化

- Prefix Cache 命中 blocks 被加入请求的 block table，并增加引用。
- 新 blocks 从 Block Pool 取出并加入请求的 block table。
- 新完整 blocks 可以获得 hash 并进入 Prefix Cache 索引。
- attention 已不再访问的 blocks 可被替换为 null block 并提前释放。
- 请求释放时，各 group 的 block table 被清除，引用计数减少。

## 5. 关键代码解析

### `KVCacheBlocks.get_block_ids()`

### `KVCacheBlocks.get_unhashed_block_ids_all_groups()`

### `KVCacheManager.__init__()`

### `KVCacheManager.get_computed_blocks()`

### `KVCacheManager.get_computed_blocks_for_connector()`

### `KVCacheManager.allocate_slots()`

### `KVCacheManager.free()`

### `KVCacheManager.pop_blocks_for_free()`

### `KVCacheManager.get_num_common_prefix_blocks()`

### `KVCacheManager.get_blocks()`

### `KVCacheManager.get_block_ids()`

### `KVCacheManager.cache_blocks()`

### `KVCacheManager.new_step_starts()`

## 6. 与其他文件的关系

- 上游：`vllm/v1/core/sched/scheduler.py`。
- 协调层：`vllm/v1/core/kv_cache_coordinator.py`。
- 类型策略：`vllm/v1/core/single_type_kv_cache_manager.py`。
- block 资源：`vllm/v1/core/block_pool.py`。
- 请求哈希：`vllm/v1/request.py` 与 `vllm/v1/core/kv_cache_utils.py`。
- 配置来源：`vllm/v1/kv_cache_interface.py` 中的 `KVCacheConfig` 和各类 `KVCacheSpec`。
- 下游：返回的 block IDs 经 `SchedulerOutput` 传到 Worker / Model Runner。

## 7. 当前结论

`KVCacheManager` 是 Scheduler 侧 KV Cache 的门面：它把“请求还要计算多少 token”转换为 Prefix Cache 查找、跨 group 容量检查、block table 扩展和最终资源释放。
