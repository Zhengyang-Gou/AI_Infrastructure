# KV Cache

## 学习目标

这一阶段关注 Scheduler 侧的 KV Cache 元数据管理：请求如何查询 Prefix Cache、获得 block table、增加 block 引用、申请新 block，以及在抢占或完成时释放 block。

完成本阶段后，应该能够说明：

1. `request_id`、block table、`block_id` 和物理 KV Cache page 之间的关系。
2. `KVCacheManager.allocate_slots()` 如何判断请求需要多少 blocks。
3. `KVCacheCoordinator` 为什么要协调多个 KV Cache groups。
4. 不同 attention 类型为什么需要不同的 `SingleTypeKVCacheManager`。
5. Prefix Cache 如何用 token 前缀的 block hash 查找可复用 block。
6. `ref_cnt` 如何保证被多个请求共享的 block 不会被提前回收。
7. Block Pool 如何同时承担空闲块管理和可驱逐 Prefix Cache 管理。
8. 请求被抢占或完成后，blocks 如何回到 Block Pool。

本阶段研究的是 Scheduler 侧的 block 元数据和分配策略。GPU 上 KV tensor 的创建、布局和读写留到 GPU 执行与 Attention 阶段。

## 阅读顺序

| 顺序 | 文件 | 主要关注点 |
| --- | --- | --- |
| 1 | `vllm/v1/core/kv_cache_manager.py` | Scheduler 使用的统一分配、查询和释放接口 |
| 2 | `vllm/v1/core/kv_cache_coordinator.py` | 多个 KV Cache groups 如何被统一协调 |
| 3 | `vllm/v1/core/single_type_kv_cache_manager.py` | 不同 attention 类型的 block table 与命中策略 |
| 4 | `vllm/v1/core/block_pool.py` | block 分配、引用计数、哈希索引和驱逐顺序 |

## 整体结构

```text
Scheduler
   ↓ get_computed_blocks() / allocate_slots() / free()
KVCacheManager
   ↓ 统一 Scheduler 接口
KVCacheCoordinator
   ↓ 协调每个 KV cache group
SingleTypeKVCacheManager × N
   ↓ 维护 request_id → blocks
BlockPool
   ↓ 维护全部 KVCacheBlock、空闲队列和哈希索引
KVCacheBlock
   ↓ block_id 对应 Worker 侧实际 KV Cache page
GPU KV Cache Tensor
```

## 核心映射关系

```text
request_id
→ SingleTypeKVCacheManager.req_to_blocks[request_id]
→ 按 token 位置排列的 KVCacheBlock 列表
→ block_id
→ Worker 侧 KV Cache page
```

```text
token prefix
→ Request.block_hashes
→ block_hash + kv_cache_group_id
→ BlockPool.cached_block_hash_to_block
→ 可复用 KVCacheBlock
```

| 字段 / 映射 | 作用 |
| --- | --- |
| `req_to_blocks` | 保存每个请求在某一 KV Cache group 中的 block table |
| `block_id` | 标识 Block Pool 中的 block，并传给 Worker 定位 KV page |
| `block_hash` | Prefix Cache 的内容寻址键，包含前缀链信息 |
| `kv_cache_group_id` | 区分使用同一 token 前缀但属于不同 KV Cache group 的 block |
| `ref_cnt` | 记录 block 当前被多少请求或临时操作持有 |
| `free_block_queue` | 保存空闲 block 和 `ref_cnt == 0` 的可驱逐缓存 block |

## Prefix Cache 查找流程

```text
Request 初始化或追加 token
→ 生成完整 hash block 的 Request.block_hashes
→ Scheduler 调用 KVCacheManager.get_computed_blocks()
→ Coordinator.find_longest_cache_hit()
→ SingleType manager 按 attention 类型检查可复用前缀
→ BlockPool.get_cached_block()
→ 返回各 KV cache group 的命中 blocks
→ 增加引用并加入请求 block table
→ 减少本轮实际 Prefill token 数
```

即使整个 prompt 都命中，当前实现也会限制最大命中长度为 `request.num_tokens - 1`，以便重算最后一个 token 并获得 logits。

## slots 分配流程

```text
Scheduler 计算 num_new_tokens
→ KVCacheManager.allocate_slots()
→ 计算本地 / 外部已计算 token 和 lookahead token
→ 移除 attention 已不会访问的 blocks
→ Coordinator.get_num_blocks_to_allocate()
→ 与 BlockPool 可用 blocks、watermark、预留 blocks 比较
→ 空间不足：返回 None
→ 接入 Prefix Cache 命中 blocks
→ Coordinator.allocate_new_blocks()
→ 为请求扩展 block table
→ 对已完成的完整 blocks 建立 Prefix Cache 索引
→ 返回本轮新 blocks 的 block IDs
```

`allocate_slots()` 返回 `None` 表示当前资源不足，Scheduler 会停止接纳 waiting 请求，或通过抢占 running 请求释放容量后重试。

## block 生命周期

```text
BlockPool 中的空闲 block（ref_cnt = 0）
→ get_new_blocks()
→ ref_cnt = 1，加入请求 block table
→ block 计算完整后写入 Prefix Cache 哈希索引
→ 其他同前缀请求命中并 touch()
→ ref_cnt 增加
→ 请求结束或被抢占
→ free_blocks()
→ ref_cnt 减少
→ ref_cnt = 0 时回到 free_block_queue
→ 有 hash：保留为可驱逐 Prefix Cache block
→ 无 hash：优先成为下一次分配对象
```

“释放请求引用”不一定等于“立刻删除缓存内容”。带 hash 且 `ref_cnt == 0` 的 block 仍能留在 Prefix Cache 中，但已经进入空闲队列，可以被后续分配驱逐和复用。

## 多 KV Cache group

模型可能包含 Full Attention、Sliding Window、Mamba、Cross Attention 等不同 KV Cache spec。它们的 block size、保留范围和 Prefix Cache 命中规则并不相同，因此当前结构分为三层：

- `KVCacheManager`：向 Scheduler 暴露统一接口。
- `KVCacheCoordinator`：跨 group 计算总需求，并协调一致的缓存命中长度。
- `SingleTypeKVCacheManager`：实现某一种 KV Cache spec 的具体策略。

多个 group 最终共享一个 `BlockPool`，因此 block 数量检查必须把各 group 的需求相加。

## 跨文件调用表

| 调用方 | 被调用方 | 作用 |
| --- | --- | --- |
| `Scheduler.schedule()` | `KVCacheManager.get_computed_blocks()` | 查询 waiting 请求的本地 Prefix Cache 命中 |
| `Scheduler.schedule()` | `KVCacheManager.allocate_slots()` | 为本轮 token 分配 KV slots |
| `KVCacheManager` | `KVCacheCoordinator` | 将统一操作分发到一个或多个 groups |
| `KVCacheCoordinator` | `SingleTypeKVCacheManager` | 执行 attention 类型相关的 block 计算与命中策略 |
| `SingleTypeKVCacheManager` | `BlockPool.get_new_blocks()` | 取得新的 blocks |
| `SingleTypeKVCacheManager` | `BlockPool.touch()` | 持有 Prefix Cache 命中的 blocks |
| `SingleTypeKVCacheManager` | `BlockPool.cache_full_blocks()` | 为完整 blocks 建立哈希索引 |
| `Scheduler` | `KVCacheManager.free()` | 抢占或完成时释放请求 blocks |
| `BlockPool` | Worker block table | 通过 `block_id` 连接调度元数据与实际 KV page |

## 当前阶段检查清单

- [ ] 能画出 `Request → KVCacheManager → Coordinator → SingleType Manager → BlockPool`。
- [ ] 能说明 block table 保存在哪里，以及为什么按 KV Cache group 组织。
- [ ] 能区分 `block_id`、`block_hash` 和 `ref_cnt`。
- [ ] 能说明 Prefix Cache 命中为什么会减少 Prefill 计算量。
- [ ] 能按三个阶段描述 `allocate_slots()`。
- [ ] 能说明为什么 `allocate_slots()` 可能返回 `None`。
- [ ] 能解释共享 block 的引用计数如何增加和减少。
- [ ] 能解释释放后的带 hash block 为什么仍可能留在 Prefix Cache。
- [ ] 能说明 null block 在 Sliding Window 等稀疏 block table 中的作用。

## 一句话总结

KV Cache 管理层把请求的 token 位置映射为可共享、可引用计数和可驱逐的 blocks，并用 `block_id` 把 Scheduler 的 block table 连接到 Worker 上的实际 KV Cache page。

## 补充专题

- `05-advanced-kv-cache.md`：扩展到 KV Cache Spec、Hybrid Cache、Encoder Cache 和 KV Offload。
