# Advanced KV Cache

## 1. 文件定位

- 主要路径：`vllm/v1/kv_cache_interface.py`、`kv_cache_spec_registry.py`、`vllm/v1/core/encoder_cache_manager.py`、`vllm/v1/kv_offload/`。
- 所属层次：缓存规格、异构缓存与多级存储层。
- 核心职责：支持 Full Attention、Sliding Window、Mamba、Encoder Cache 和 CPU/外部介质 offload。

## 2. 核心对象

| 对象 | 作用 |
| --- | --- |
| `KVCacheSpec` | 描述某类层的缓存形状、page 大小和最大内存需求 |
| `KVCacheSpecRegistry` | 将缓存规格类型映射到对应 manager |
| `KVCacheGroupSpec` | 把布局兼容的层组成缓存组 |
| `KVCacheConfig` | 描述实际分配的缓存 tensors 和 group 配置 |
| `EncoderCacheManager` | 管理多模态或 Encoder-Decoder 输入的 Encoder 输出缓存 |
| `OffloadingManager` | 在 GPU 缓存和更低层级存储之间查找、加载和保存 blocks |

## 3. 主执行流程

```text
模型各 Attention / Mamba 层
→ 生成 KVCacheSpec
→ KVCacheSpecRegistry 选择 manager
→ 按 page size 和层类型形成 KVCacheGroupSpec
→ 分配 KVCacheTensor
→ Scheduler 按 group 协调 block 分配
```

```text
GPU Cache miss
→ OffloadingManager.lookup()
→ prepare_load()
→ 从 CPU 或外部介质恢复 block
→ complete_load()
→ GPU Cache 可用
```

Hybrid Cache 让不同层类型使用各自的缓存规格和命中规则，再由 coordinator 取共同可用的前缀。

## 4. 输入与输出

### 输入

- 模型每一层的缓存类型、head 数、dtype 和 block size。
- Scheduler 的请求状态、token 位置和缓存命中结果。
- Offload tier、locality 和存储策略。

### 输出

- 分组后的 `KVCacheConfig` 和实际缓存 tensors。
- Encoder 输入的缓存命中与释放结果。
- Offload load/store 计划及完成事件。

### 状态变化

- 不同缓存组独立维护 blocks 和 Prefix Cache。
- Encoder Cache 按请求与多模态 input ID 计费和释放。
- Offload manager 在层级存储之间迁移 block 数据。

## 5. 关键代码解析

### `KVCacheSpec.page_size_bytes()`

### `KVCacheSpec.max_memory_usage_bytes()`

### `KVCacheSpecRegistry.register()`

### `KVCacheSpecRegistry.get_manager_class()`

### `EncoderCacheManager.check_and_update_cache()`

### `EncoderCacheManager.allocate()`

### `EncoderCacheManager.free()`

### `OffloadingManager.lookup()`

### `OffloadingManager.prepare_load()`

### `OffloadingManager.prepare_store()`

## 6. 与其他文件的关系

- 基础分配：已有的 `KVCacheManager`、Coordinator、SingleType Manager 和 BlockPool。
- 多模态：Encoder Cache 与多模态 processor、encoder forward 相连。
- 分布式：远程 KV Connector 与 Prefill/Decode 分离放在阶段九学习。
- Worker：Model Runner 根据 `KVCacheConfig` 创建物理缓存并绑定 Attention 层。

## 7. 当前结论

高级 KV Cache 把单一 Attention block 管理扩展为多缓存规格、多层类型和多存储层级的统一系统。
