# KV Cache Management Overview

**本章管理缓存的规格与块所有权。** 它接收已经测得的字节预算和外部给定的 token 需求，交出缓存配置、块引用或失败结果。

## 1. 两类工作

```text
容量规划（启动时）
每层 KVCacheSpec + 可用缓存字节数
    → 按兼容规格分组
    → 计算池中块数与各层存储需求
    → KVCacheConfig

块管理（请求运行时）
查询可复用前缀 → 接入命中块 → 按需申请新块
    → 维护请求持有的块与前缀索引
    → 收到释放指令后交还引用
```

容量规划不创建 GPU 张量；运行期间的块借还也不反复调用 GPU 大张量分配。

## 2. 对象怎样分工

```text
KVCacheManager：面向请求的查询、分配、释放接口
└─ KVCacheCoordinator：协调多个缓存分组
   ├─ SingleTypeKVCacheManager：维护每组 req_to_blocks
   │  └─ FullAttentionManager：本章基础场景使用的类型规则
   └─ BlockPool：共享块元数据、空闲队列和 hash 索引
```

| 对象 | 描述什么 |
| --- | --- |
| KVCacheSpec | 一层缓存的 block_size、dtype、KV heads 等规格 |
| KVCacheGroupSpec | 共用一套请求块表的层集合 |
| KVCacheConfig | num_blocks、各层存储配置和分组 |
| KVCacheBlock | block_id、ref_cnt、hash 与空闲队列链接；不保存 K/V 张量 |
| req_to_blocks | 一条请求在一个缓存分组中持有的有序块引用 |
| KVCacheBlocks | 按组包装的块引用集合，可取得对应的块 ID |

单个 FullAttention 分组下，一个逻辑块位置在所有组内层上各有对应存储；不是把层数再乘进请求块表长度。

## 3. 对外接口到哪里为止

| 接口 | 输入 | 输出/效果 |
| --- | --- | --- |
| get_computed_blocks | 请求 token 历史与 hash | 分组块引用、命中 token 数、shared_prefix_boundary |
| allocate_slots | 请求、本次新计算量、命中块与命中长度等 | 新分配块；不足时返回 None |
| get_blocks / get_block_ids | request_id | 请求目前持有的完整块集合/ID |
| free | 请求，且调用者已确认可安全归还 | 删除请求块账本并交还引用 |
| pop_blocks_for_free | 请求 | 移出块账本，交出仍持有引用的块，供调用者延后归还 |

查询命中尚未取得新的请求引用；allocate_slots 中接入命中块时才固定这些引用。它返回的新增块也不是该请求的完整块表。

## 4. 必须分开的几件事

- **请求释放与缓存淘汰**：前者减少引用；后者移除前缀查找记录。引用归零后，数据可能留在空闲块中继续命中。
- **块可复用与 GPU 写入完成**：hash 登记及调度进度都不能单独证明写入完成，调用者必须遵守在途工作的交还约束。
- **逻辑位置与物理 ID**：请求中的第几个块由列表位置表示，物理 ID 由分配结果给出。
- **单组规则与混合缓存**：主线可按完整块对齐计算；多组、细粒度 hash、滑窗与 Mamba 会改变匹配和回收规则，不能套用主线公式。

## 5. 阅读路线

依次进入 [01 容量](Core-Modules/01-Spec-Groups-and-Capacity.md)、[02 块池](Core-Modules/02-Block-Pool-and-References.md)、[03 分配](Core-Modules/03-Allocation-and-Request-Blocks.md)、[04 前缀](Core-Modules/04-Prefix-Cache-and-Eviction.md)、[05 释放](Core-Modules/05-Release-and-Block-Ledger.md)。完整数字推演集中在 Core 05。

[章节目录](README.md)
