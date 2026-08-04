# Attention Layer

## 1. 学习目标

理解模型产生的 Q、K、V 如何通过统一 `Attention` 层连接到 KV Cache、运行时 Attention Metadata 和具体后端实现，并明确统一层与 kernel 实现之间的边界。

## 2. 文件定位

- Roadmap 路径：`vllm/attention/layer.py`
- 当前实际路径：`vllm/model_executor/layers/attention/attention.py`
- 所属层次：模型执行层的 Attention 抽象入口
- 核心职责：选择 Attention 后端、创建后端实现、描述本层 KV Cache 规格，并在 forward 时调度 KV Cache 更新与 Attention 计算。
- 在调用链中的位置：位于 `LlamaAttention` 等具体模型层与 `AttentionImpl` 之间。

## 3. 核心类与组件

| 类 / 组件 | 作用 |
| --- | --- |
| `Attention` | 面向模型代码的统一 Attention 模块，持有后端类、实现对象和本层 KV Cache 引用 |
| `AttentionBackend` | 描述后端能力并提供具体实现类 |
| `AttentionImpl` | 执行后端专属 Attention 逻辑 |
| `ForwardContext` | 提供当前 forward 的逐层 metadata、KV Cache 绑定和 slot mapping |
| `KVCacheSpec` | 描述本层所需 KV Cache 的 block 大小、head 数、head size、dtype 等信息 |
| `AttentionBackendEnum` | 保存当前后端的规范名称并连接注册表 |

`Attention.__init__()` 会调用 `get_attn_backend()`，再通过 `get_impl_cls()` 实例化后端实现。初始化时先放置空的 `self.kv_cache` 占位 Tensor，后续由 Model Runner 的 KV Cache 初始化流程绑定实际存储。

## 4. 主执行流程

### 初始化流程

```text
模型层构造 Attention(...)
→ 读取 VllmConfig 与 CacheConfig
→ get_attn_backend(...)
→ 得到具体 AttentionBackend 类
→ backend.get_impl_cls()
→ 创建 self.impl
→ 将当前层注册到 static_forward_context
→ 建立 KV Cache 占位引用与量化 scale
```

### Forward 流程

```text
Attention.forward(query, key, value)
→ 必要时计算或应用 KV Cache 量化 scale
→ reshape Q、K、V 为 head 维度
→ 必要时单独调用 unified_kv_cache_update()
→ unified_attention_with_output()
→ get_attention_context(layer_name)
→ 取得 attn_metadata、Attention 层和 kv_cache
→ self.impl.forward(...)
→ 返回 output
```

若后端的 `forward_includes_kv_cache_update` 为 `False`，且本层不共享其他层的 KV Cache，统一层会先显式执行 KV Cache 更新；否则 K、V 写入通常由后端 forward 路径负责。

## 5. 输入与输出

### 输入

- `query`：当前 token 或 token chunk 的 Query Tensor。
- `key`：当前 token 或 token chunk 的 Key Tensor。
- `value`：当前 token 或 token chunk 的 Value Tensor。
- `output_shape`：部分后端或模型可显式指定输出形状。
- `output_dtype`：可选的输出数据类型，默认使用 Query 的 dtype。
- 运行时隐式输入：`ForwardContext` 中的 Attention Metadata、KV Cache 与 slot mapping。

### 输出

- `Attention.forward()` 返回形状恢复为二维的 Attention 输出 Tensor。
- KV Cache 作为副作用写入当前 K、V，具体写入位置由 slot mapping 决定。

### 状态变化

- 初始化时保存 `self.attn_backend`、`self.impl`、`self.backend` 与层名。
- `self.kv_cache` 从初始化占位 Tensor 变为 Model Runner 绑定的实际缓存 Tensor。
- 动态 KV Cache 量化启用时，首轮计算后更新 scale 并关闭重复计算。

## 6. 关键代码解析

### `Attention.__init__()`

### `Attention.forward()`

### `Attention.get_attn_backend()`

### `Attention.get_kv_cache_spec()`

### `get_attention_context()`

### `unified_kv_cache_update()`

### `unified_attention_with_output()`

## 7. 与其他文件的关系

- 上游：`vllm/model_executor/models/llama.py` 等模型实现，负责生成 Q、K、V。
- 后端选择：`vllm/v1/attention/selector.py`，结合配置与平台能力返回后端类。
- 后端注册：`vllm/v1/attention/backends/registry.py`，解析后端名称和类路径。
- 下游：`vllm/v1/attention/backends/<实际使用后端>.py` 中的实现类。
- 运行时数据来源：Model Runner 构造 Attention Metadata、block table 和 slot mapping，并绑定 KV Cache。
- KV Cache 配置：`get_kv_cache_spec()` 将本层 head、dtype、sliding window 等信息转换为缓存规格。

## 8. 当前结论

`Attention` 是模型代码与后端 kernel 之间的稳定边界。模型只需传入 Q、K、V；统一层负责取得本轮缓存和 metadata，并把具体计算委托给运行时选定的 `AttentionImpl`。
