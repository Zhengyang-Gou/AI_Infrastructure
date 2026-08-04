# KV Transfer and Disaggregated Serving

## 1. 文件定位

- 路径：`vllm/distributed/kv_transfer/kv_connector/`、`vllm/v1/worker/kv_connector_model_runner_mixin.py`。
- 职责：定义 KV Connector 生命周期，将调度侧块状态与 Worker 侧 KV 张量传输连接起来。

## 2. 分离式调用链

```text
Prefill Engine 计算 prompt KV
→ Scheduler 构造 connector metadata
→ Worker 保存各层 KV
→ Connector 传输或发布 KV
→ Decode Engine 查询可复用 token
→ 分配目标 KV blocks
→ Worker 按层加载 KV
→ Decode 从已计算位置继续
```

调度器只管理 token、block 和请求状态；真正的张量搬运由 Worker 侧 Connector 完成，两侧通过 metadata 对齐。

## 3. 生命周期与一致性

- Factory 根据配置创建具体 Connector 实现。
- Scheduler 侧查询远端命中、构造传输 metadata，并处理请求完成状态。
- Worker 侧在 forward 前启动加载，在每层计算附近等待或保存 KV。
- Connector 必须处理异步完成、传输失败、抢占、释放和 shutdown。
- Prefill 与 Decode 必须在模型、KV layout、block size 和并行映射上兼容。

## 4. 关键代码解析

### `KVConnectorFactory.create_connector()`

### `KVConnectorBase_V1.get_num_new_matched_tokens()`

### `KVConnectorBase_V1.update_state_after_alloc()`

### `KVConnectorBase_V1.build_connector_meta()`

### `KVConnectorBase_V1.start_load_kv()`

### `KVConnectorBase_V1.wait_for_layer_load()`

### `KVConnectorBase_V1.save_kv_layer()`

### `KVConnectorBase_V1.request_finished()`

### `KVConnectorModelRunnerMixin.maybe_get_kv_connector_output()`

### `KVConnectorModelRunnerMixin.finalize_kv_connector()`

## 5. 与其他阶段的关系

- KV Cache 阶段提供 block、prefix cache 和分配基础。
- Scheduler 决定哪些 token 本地已有、哪些可从远端加载。
- GPU Model Runner 在 layer forward 边界调用 Connector。
- 在线服务层负责将 Prefill 与 Decode 实例组织成实际部署。

## 6. 当前结论

分离式服务的核心不是复制一个完整请求，而是让两套独立调度状态通过稳定 metadata 精确指向同一段 KV 内容。
