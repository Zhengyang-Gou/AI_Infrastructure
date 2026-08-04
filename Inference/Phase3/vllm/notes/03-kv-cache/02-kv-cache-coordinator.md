# KV Cache Coordinator

## 1. 文件定位

- 文件路径：`vllm/v1/core/kv_cache_coordinator.py`
- 所属层次：KV Cache 跨 group 协调层
- 核心职责：为每个 KV Cache group 创建类型专用 manager，汇总 block 需求，并协调多种 attention 类型的一致 Prefix Cache 命中长度。
- 在调用链中的位置：位于 `KVCacheManager` 与多个 `SingleTypeKVCacheManager` 之间。

一个模型可能只有一种 KV Cache spec，也可能同时包含 Full Attention、Sliding Window 或 Mamba 等不同 spec。Coordinator 负责把这些 group 组合成对上层一致的行为。

## 2. 核心类与组件

| 类 / 组件 | 适用场景 | 作用 |
| --- | --- | --- |
| `KVCacheCoordinator` | 所有配置 | 定义跨 group 的公共操作 |
| `KVCacheCoordinatorNoPrefixCache` | Prefix Cache 关闭 | 支持任意 group 数，不执行缓存命中 |
| `UnitaryKVCacheCoordinator` | 只有一个 KV Cache group | 直接使用单个 manager 查询缓存 |
| `HybridKVCacheCoordinator` | 存在多个 KV Cache groups | 协调不同 spec 的 block 与命中长度 |
| `SpecGroup` | Hybrid 模式 | 将相同 spec 的 groups 合并执行缓存查找 |
| `single_type_managers` | 每个 Coordinator | 每个 KV Cache group 对应一个类型专用 manager |
| `block_pool` | 所有 managers 共享 | 统一管理有限的 block 资源 |

## 3. 主执行流程

### 选择 Coordinator

```text
get_kv_cache_coordinator()
├─ Prefix Cache 关闭       → KVCacheCoordinatorNoPrefixCache
├─ 只有一个 KV Cache group → UnitaryKVCacheCoordinator
└─ 多个 KV Cache groups    → HybridKVCacheCoordinator
```

### 初始化 groups

```text
KVCacheCoordinator.__init__()
→ 创建共享 BlockPool
→ 遍历 KVCacheConfig.kv_cache_groups
→ get_manager_for_kv_cache_spec()
→ 每个 group 创建一个 SingleTypeKVCacheManager
→ 保存 scheduler_block_size 和缓存保留策略
```

`scheduler_block_size` 是跨 group 的调度粒度，必须是 hash block size 和每个 group 实际 block size 的整数倍。

### 计算和执行分配

```text
get_num_blocks_to_allocate()
→ 遍历 single_type_managers
→ 每个 manager 计算自己的新增 block 数
→ 对所有 group 求和

allocate_new_computed_blocks()
→ 第一阶段：先 touch 所有 group 的本地缓存命中 blocks
→ 第二阶段：再为所有 group 分配外部 KV 命中所需 blocks

allocate_new_blocks()
→ 每个 manager 扩展自己的 request block table
→ 返回 tuple[list[KVCacheBlock], ...]
```

本地命中采用“两阶段”处理，是为了防止某个 group 分配外部 blocks 时驱逐另一个 group 尚未增加引用的本地命中 block。

### Unitary Prefix Cache 命中

```text
UnitaryKVCacheCoordinator.find_longest_cache_hit()
→ 单个 manager.find_longest_cache_hit()
→ 返回该 group 的 blocks 和命中 token 数
```

### Hybrid Prefix Cache 命中

```text
HybridKVCacheCoordinator.verify_and_split_kv_cache_groups()
→ 按相同 KVCacheSpec 组合 SpecGroup
→ Full Attention group 优先

HybridKVCacheCoordinator.find_longest_cache_hit()
→ 以 max_cache_hit_length 为候选长度
→ 每种 attention spec 检查该长度是否可命中
→ 任一类型缩短候选长度时重新协调
→ 直到所有类型接受同一个边界
→ 按最终长度裁剪每个 group 的命中 blocks
```

Full Attention 需要从开头连续命中，而 Sliding Window 或 Mamba 可以只保留特定窗口或状态 block。Hybrid Coordinator 必须找到所有 groups 都能正确恢复的共同 token 边界。

### 释放

```text
KVCacheCoordinator.free(request_id)
→ 遍历 single_type_managers
→ manager.free(request_id)
→ 各 group 清理 request block table
→ blocks 返回共享 BlockPool
```

## 4. 输入与输出

### 输入

- `KVCacheConfig` 及其中的 KV Cache groups 和 specs。
- 请求 ID、目标 token 数、本地 Prefix Cache 命中 blocks 和外部命中 token 数。
- 请求的 block hash 链与最大可命中长度。

### 输出

- 所有 groups 合计所需的新增 block 数。
- 按 group 组织的新分配 blocks。
- 按 group 组织的 Prefix Cache 命中 blocks、统一命中 token 数和未被稀疏组缓存的公共前缀长度。
- 每个 group 的公共前缀 block 数。

### 状态变化

- 初始化时创建共享 Block Pool 和每个 group 的 manager。
- Prefix Cache 命中 blocks 被各 manager 持有并写入请求 block table。
- 新 blocks 在各 group 中分配，但都消耗同一个 Block Pool 的容量。
- 缓存与释放操作被广播到所有 single-type managers。

## 5. 关键代码解析

### `KVCacheCoordinator.__init__()`

### `KVCacheCoordinator.get_num_blocks_to_allocate()`

### `KVCacheCoordinator.allocate_new_computed_blocks()`

### `KVCacheCoordinator.allocate_new_blocks()`

### `KVCacheCoordinator.cache_blocks()`

### `KVCacheCoordinator.free()`

### `KVCacheCoordinator.get_blocks()`

### `KVCacheCoordinatorNoPrefixCache.find_longest_cache_hit()`

### `UnitaryKVCacheCoordinator.find_longest_cache_hit()`

### `HybridKVCacheCoordinator.verify_and_split_kv_cache_groups()`

### `HybridKVCacheCoordinator.cache_blocks()`

### `HybridKVCacheCoordinator.find_longest_cache_hit()`

### `get_kv_cache_coordinator()`

## 6. 与其他文件的关系

- 上游：`vllm/v1/core/kv_cache_manager.py`。
- 类型实现：`vllm/v1/core/single_type_kv_cache_manager.py`。
- 共享资源：`vllm/v1/core/block_pool.py`。
- 配置定义：`vllm/v1/kv_cache_interface.py`。
- block 与 hash 类型：`vllm/v1/core/kv_cache_utils.py`。
- 请求输入：`vllm/v1/request.py`。

## 7. 当前结论

`KVCacheCoordinator` 解决的是“一个请求跨多个 KV Cache groups 必须作为整体分配和命中”的问题：它汇总容量需求，并让不同 attention 类型最终对同一个可复用前缀边界达成一致。
