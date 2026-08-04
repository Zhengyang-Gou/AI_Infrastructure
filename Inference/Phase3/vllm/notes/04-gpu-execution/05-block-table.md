# BlockTables

## 1. 文件定位

- 文件路径：`vllm/v1/worker/gpu/block_table.py`
- 所属层次：GPU 侧 KV Cache 地址映射层
- 核心职责：保存每个请求的物理 block ID，并为当前 batch 生成 Attention forward 所需的 block tables 和 slot mappings。
- 在调用链中的位置：Scheduler 分配 block 之后、Attention metadata 构造之前。

Scheduler 管理逻辑 block 的分配与生命周期；本文件负责把分配结果整理为 GPU kernel 可以直接使用的物理地址索引。

## 2. 核心类与组件

| 类 / 组件 | 作用 | 关注点 |
| --- | --- | --- |
| `BlockTables` | 管理一个或多个 KV Cache group 的请求 block table | 请求槽位到物理 block IDs |
| `StagedWriteTensor` | 先在 CPU/UVA 侧暂存小规模更新，再批量应用 | 避免频繁零散写 GPU |
| `FusedStagedWriter` | 多 Cache group 时融合写入 | 一次应用多组 staged writes |
| `UvaBackedTensor` | 保存每个请求当前 block 数 | CPU 可更新、GPU kernel 可读取 |
| `_gather_block_tables_kernel` | 为当前 batch 收集并重排请求行 | `req_state_idx → batch_idx` |
| `_compute_slot_mappings_kernel` | 计算 token 到 KV Cache slot 的映射 | position、block ID、block offset |

## 3. 内部数据布局

每个 KV Cache group 都有一个二维 block table：

```text
block_tables[group][req_state_idx][logical_block_index]
    → physical_block_id
```

`num_blocks[group][req_state_idx]` 记录该请求当前有效 block 数。`input_block_tables` 是为本轮模型 forward 准备的持久 Tensor，其行顺序按当前 batch 重排，并可保留稳定地址供 CUDA Graph 捕获与重放。

当 Cache block size 大于 kernel block size 时，一个 Cache block ID 会展开成多个连续 kernel block ID，展开比例保存在 `blocks_per_kv_block` 中。

## 4. Block 更新流程

```text
SchedulerOutput 中的新 block IDs
→ GPUModelRunner.add_requests() / update_requests()
→ BlockTables.append_block_ids()
→ StagedWriteTensor.stage_write()
→ BlockTables.apply_staged_writes()
→ GPU block table 更新完成
```

新请求使用 `overwrite=True` 从头写入 block table；已在运行的请求使用 `overwrite=False` 从现有 `num_blocks` 之后追加。

## 5. 当前批次的 block table

```text
InputBatch.idx_mapping
→ gather_block_tables()
→ _gather_block_tables_kernel
→ 按 batch 顺序复制请求 block table 行
→ 清零 CUDA Graph 填充行
```

输入的 `idx_mapping` 把 batch 行映射到请求状态槽位，kernel 据此从持久化 block table 中收集正确的行，并写入 forward 使用的 `input_block_tables`。

## 6. Slot mapping 计算

每个 token 的物理 slot 可概括为：

```text
logical_block_index = position // block_size
block_offset         = position % block_size
physical_block_id    = block_table[request][logical_block_index]
slot_id              = physical_block_id * block_size + block_offset
```

```text
compute_slot_mappings()
→ 读取 idx_mapping、query_start_loc 和 positions
→ 查找每个 position 对应的 physical block ID
→ 计算每个 KV Cache group 的 slot ID
→ 将填充区域写成 PAD_SLOT_ID
```

启用 Decode Context Parallelism 时，kernel 还会判断当前位置是否属于本 rank，并把非本地位置映射为 `PAD_SLOT_ID`。

## 7. 输入与输出

### 输入

- Scheduler 为新请求或缓存请求分配的 `new_block_ids`。
- 当前 batch 的 `idx_mapping`、`query_start_loc` 和 `positions`。
- 每个 Cache group 的 block size、kernel block size 和最大 block 数。

### 输出

- 当前 batch 使用的 `tuple[torch.Tensor, ...]` block tables。
- 形状为 `[num_kv_cache_groups, num_tokens_padded]` 的 slot mappings。
- dummy run 使用的稳定地址 block table 和 padding slot mappings。

### 状态变化

- 新 block ID 被写入请求对应的持久化 block table 行。
- `num_blocks` 随追加或覆盖操作更新。
- 每轮执行前，`input_block_tables` 和 `slot_mappings` 被刷新为当前 batch 内容。

## 8. 关键代码解析

### `BlockTables.__init__()`

### `BlockTables.init_block_table_layout_tensors()`

### `BlockTables.append_block_ids()`

### `BlockTables.apply_staged_writes()`

### `BlockTables.gather_block_tables()`

### `BlockTables.compute_slot_mappings()`

### `BlockTables.get_dummy_block_tables()`

### `BlockTables.get_dummy_slot_mappings()`

### `_gather_block_tables_kernel()`

### `_compute_slot_mappings_kernel()`

## 9. 与其他文件的关系

- 上游：Scheduler 和 KV Cache Manager 决定 block 的分配，`GPUModelRunner` 接收并写入这些结果。
- 批次索引：使用 `input_batch.py` 中 `InputBatch` 的映射和位置 Tensor。
- 下游：Model State 和 Attention backend 使用 block tables、slot mappings 构造执行元数据。
- Cache 实体：最终索引指向 `GPUModelRunner.initialize_kv_cache()` 创建的 KV Cache Tensor。

## 10. 当前结论

`block_table.py` 是逻辑 Cache 分配与 Attention 物理寻址之间的桥梁：它保存请求持有的 block ID，并把本轮 token position 转换成可直接访问 KV Cache 的 slot ID。
