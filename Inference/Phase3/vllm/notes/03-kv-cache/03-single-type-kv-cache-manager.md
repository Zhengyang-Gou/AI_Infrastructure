# Single Type KV Cache Manager

## 1. 文件定位

- 文件路径：`vllm/v1/core/single_type_kv_cache_manager.py`
- 所属层次：单一 KV Cache spec 的策略实现层
- 核心职责：维护某个 KV Cache group 的请求 block table，并实现该 attention 类型的分配、Prefix Cache 命中、缓存和跳过 block 规则。
- 在调用链中的位置：由 `KVCacheCoordinator` 创建并调用，向下使用共享 `BlockPool`。

这个文件把“所有 KV Cache 都共有的引用和分配机制”与“不同 attention 类型的可见范围和可复用规则”组合在一起。

## 2. 核心类与组件

| 类 | 对应场景 | 核心差异 |
| --- | --- | --- |
| `SingleTypeKVCacheManager` | 抽象基类 | 提供 block table、分配、缓存和释放的公共实现 |
| `FullAttentionManager` | Full Attention / MLA | 从序列开头连续查找缓存，通常保留全部历史 blocks |
| `RSWAManager` | Reference Sliding Window Attention | 保留 prompt 前缀与当前 decode window，释放中间 gap |
| `SlidingWindowManager` | Sliding Window Attention | 只要求窗口覆盖范围内的连续缓存 blocks |
| `ChunkedLocalAttentionManager` | Chunked Local Attention | 按 local attention chunk 计算可跳过 blocks |
| `MambaManager` | Mamba / 线性状态缓存 | 缓存和查找状态 block，而非普通全历史 K/V |
| `CrossAttentionManager` | Encoder-Decoder Cross Attention | 为请求分配静态 blocks，不参与跨请求 Prefix Cache |
| `SinkFullAttentionManager` | 带 attention sink 的 Full Attention | 额外保留固定 sink blocks |

### 核心状态

| 字段 | 作用 |
| --- | --- |
| `req_to_blocks` | `request_id → list[KVCacheBlock]`，即该 group 的请求 block table |
| `num_cached_block` | 记录 running 请求已有多少 blocks 已写入 Prefix Cache |
| `block_size` | 此 group 一个实际 block 对应的 token 数 |
| `scheduler_block_size` | 多 group 共同采用的调度对齐粒度 |
| `kv_cache_group_id` | 当前 manager 所属 group |
| `_null_block` | 在跳过位置占位，维持 token 位置到 block table 下标的关系 |
| `_partial_hit_reqs` | 记录落在 block 内部的细粒度 Prefix Cache 命中 |

## 3. 主执行流程

### 计算新增 block 数

```text
get_num_blocks_to_allocate()
→ ceil(num_tokens / block_size)
→ 减去请求已经持有的 blocks
→ 考虑 Prefix Cache 命中 blocks
→ 减去 attention 已跳过的 blocks
→ 将 ref_cnt = 0 的命中 block 计入容量占用
→ partial hit 时预留一个 CoW block
→ 返回本 group 需要新增的 block 数
```

### 接入 Prefix Cache 命中

```text
add_local_computed_blocks()
→ 根据 attention 类型计算已跳过 blocks
→ BlockPool.touch(命中 blocks)
→ 跳过位置用 null block 填充
→ 命中 blocks 加入 req_to_blocks
→ 更新 num_cached_block
```

`touch()` 会增加引用计数；若命中 block 原本 `ref_cnt == 0` 且位于空闲队列，它会先被移出空闲队列，防止随后被驱逐。

### 分配新 blocks

```text
allocate_new_blocks()
→ 计算请求 block table 的目标长度
→ partial hit 时先创建私有 CoW block
→ BlockPool.get_new_blocks()
→ 追加到 req_to_blocks[request_id]
→ 返回本轮新增 blocks
```

### 缓存完整 blocks

```text
cache_blocks()
→ 根据 num_tokens 计算新的完整 block 范围
→ reachable_block_mask() 决定哪些 blocks 具有可复用价值
→ BlockPool.cache_full_blocks()
→ 为 block 绑定 block_hash + group_id
→ 更新 num_cached_block
```

基类默认缓存所有非 null 的完整 blocks。Sliding Window、Mamba 等子类可以通过 mask 只缓存其命中算法未来会查询的 blocks。

### 提前移除不可见 blocks

```text
remove_skipped_blocks()
→ get_num_skipped_tokens()
→ 找到 attention 已不会读取的 block 范围
→ _remove_blocks_in_range()
→ 对应位置替换为 null block
→ 原 block 返回 BlockPool
```

### 释放请求

```text
free(request_id)
→ pop_blocks_for_free()
→ 删除 req_to_blocks 与 num_cached_block 状态
→ 逆序调用 BlockPool.free_blocks()
```

逆序释放使尾部 blocks 在驱逐顺序中更早被复用，优先保留更靠近公共前缀的缓存 blocks。

## 4. 输入与输出

### 输入

- 当前 group 的 `KVCacheSpec`、共享 Block Pool、group ID 和 block size。
- 请求 ID、目标 token 数、Prefix Cache 命中 blocks 和当前计算进度。
- `Request.block_hashes`、最大命中长度和跨 group 对齐粒度。

### 输出

- 此 group 需要新增的 block 数。
- 新分配或 Prefix Cache 命中的 `list[KVCacheBlock]`。
- Prefix Cache 的精确命中 token 长度。
- 当前请求的 block table 和公共前缀 block 数。

### 状态变化

- `req_to_blocks` 随请求接纳、扩展、滑动窗口回收和完成而变化。
- Prefix Cache 命中时增加 block 引用。
- 完整 blocks 获得 hash 后进入 Block Pool 的缓存索引。
- 被跳过的位置保留 null block，以维持 block table 的逻辑位置。
- 请求释放时 manager 级别的跟踪状态被删除，实际 block 引用交给 Block Pool 减少。

## 5. 关键代码解析

### `SingleTypeKVCacheManager.__init__()`

### `SingleTypeKVCacheManager.get_num_blocks_to_allocate()`

### `SingleTypeKVCacheManager.add_local_computed_blocks()`

### `SingleTypeKVCacheManager.allocate_external_computed_blocks()`

### `SingleTypeKVCacheManager.allocate_new_blocks()`

### `SingleTypeKVCacheManager.cache_blocks()`

### `SingleTypeKVCacheManager.pop_blocks_for_free()`

### `SingleTypeKVCacheManager.free()`

### `SingleTypeKVCacheManager.remove_skipped_blocks()`

### `FullAttentionManager.find_longest_cache_hit()`

### `SlidingWindowManager.find_longest_cache_hit()`

### `SlidingWindowManager.get_num_skipped_tokens()`

### `MambaManager.find_longest_cache_hit()`

### `MambaManager.allocate_new_blocks()`

### `get_manager_for_kv_cache_spec()`

### `register_all_kvcache_specs()`

## 6. 与其他文件的关系

- 上游：`vllm/v1/core/kv_cache_coordinator.py`。
- 共享资源：`vllm/v1/core/block_pool.py`。
- block、hash 与空闲队列类型：`vllm/v1/core/kv_cache_utils.py`。
- 策略配置：`vllm/v1/kv_cache_interface.py` 中的各种 `KVCacheSpec`。
- Registry：`vllm/v1/kv_cache_spec_registry.py` 将 spec 类型映射到 manager 类型。
- 请求数据：`vllm/v1/request.py` 提供 token、block hashes 和 Prefix Cache 边界。

## 7. 当前结论

`SingleTypeKVCacheManager` 是 attention 语义落到 block table 的位置：公共基类管理引用、分配与释放，各子类则决定哪些历史 blocks 必须保留、哪些 blocks 可以命中以及哪些位置能够提前回收。
