# Attention Backend

## 1. 学习目标

在确认实际运行配置采用的后端后，只阅读该后端文件，理解它如何定义 KV Cache 布局、构造 Attention Metadata、写入当前 K、V 并调用具体 CUDA 或 Triton kernel。

## 2. 文件定位

- Roadmap 路径：`vllm/v1/attention/backends/<实际使用后端>.py`
- 实际路径：必须根据本机运行配置和 selector 的最终结果选择，不能在通用笔记中固定。
- 所属层次：Attention 后端实现与 kernel 适配层
- 核心职责：把统一 Attention 接口的数据转换成特定 kernel 需要的布局与参数，并执行后端专属 forward。
- 在调用链中的位置：位于 `Attention` 统一层与底层 CUDA、Triton 或外部 Attention 库之间。

Roadmap 给出的候选文件在当前仓库均存在：

```text
vllm/v1/attention/backends/flash_attn.py
vllm/v1/attention/backends/flashinfer.py
vllm/v1/attention/backends/triton_attn.py
```

这三个文件不是按固定顺序全部阅读的清单。应先确认 `vllm_config.attention_config.backend`、平台能力与 `get_attn_backend()` 的最终选择，再将本笔记中的接口名称映射到该文件的具体类名。

## 3. 核心类与组件

| 抽象角色 | 作用 | 在所选后端中应寻找的实现 |
| --- | --- | --- |
| `AttentionBackend` | 声明后端名称、实现类、builder、KV Cache 形状和支持能力 | 名称通常以 `Backend` 结尾的类 |
| `AttentionMetadata` | 保存当前 batch 的序列边界、block table 和执行参数 | 后端专属 metadata 数据类 |
| `AttentionMetadataBuilder` | 从 Model Runner 的公共输入构造后端 metadata | 名称通常以 `MetadataBuilder` 结尾的类 |
| `AttentionImpl` | 执行 KV Cache 更新和 Attention forward | 名称通常以 `Impl` 结尾的类 |
| Attention kernel | 完成 prefill、decode 或统一变长 Attention 计算 | 后端导入或定义的 CUDA / Triton 调用 |

具体类名因后端而异。例如当前仓库中的 FlashAttention、FlashInfer 和 Triton 后端各自拥有不同的 Backend、MetadataBuilder、Metadata 与 Impl 类，不能将其中一个后端的类名套用到另一个后端。

## 4. 主执行流程

### Metadata 构造

```text
SchedulerOutput
→ Model Runner 整理 request index、query length、sequence length、block table
→ 所选后端的 AttentionMetadataBuilder.build()
→ 后端专属 AttentionMetadata
→ 写入 ForwardContext
```

### Attention 执行

```text
Attention.forward(query, key, value)
→ get_attention_context()
→ 取得所选后端 metadata、KV Cache、slot mapping
→ 所选 AttentionImpl.forward()
→ 写入当前 K、V（或由独立更新函数完成）
→ 根据 metadata 选择并调用后端 kernel
→ 将结果写入 output Tensor
```

Prefill 与 Decode 可能使用不同 kernel，也可能由一个统一 kernel 根据 metadata 处理。应以实际后端的 `forward()` 和 builder 逻辑为准。

## 5. 输入与输出

### 输入

- `query`、`key`、`value`：模型层本轮产生的 Q、K、V Tensor。
- `kv_cache`：当前 Attention 层绑定的物理缓存 Tensor。
- `attn_metadata`：后端专属运行时 metadata。
- `slot_mapping`：当前 K、V 的写入位置；部分后端在独立更新路径中使用。
- 后端初始化参数：head 数、head size、scale、sliding window、KV Cache dtype 与 Attention 类型等。

### 输出

- 写入调用方预分配的 Attention `output` Tensor。
- 以副作用将当前 K、V 写入 KV Cache。
- metadata builder 产生供本轮或 CUDA Graph 重放使用的后端 metadata。

### 状态变化

- KV Cache 中对应物理 slot 被写入当前 K、V。
- 后端 wrapper 或 workspace 可能随 batch 形状更新；具体状态以所选实现为准。

下列函数由 `vllm/v1/attention/backend.py` 定义。阅读实际后端时，应定位相应的覆盖实现，而不是假设某个固定的具体类名。

## 6. 关键代码解析

### `AttentionBackend.get_name()`

### `AttentionBackend.get_impl_cls()`

### `AttentionBackend.get_builder_cls()`

### `AttentionBackend.get_kv_cache_shape()`

### `AttentionMetadataBuilder.build()`

### `AttentionImpl.forward()`

## 7. 与其他文件的关系

- 上游统一层：`vllm/model_executor/layers/attention/attention.py`。
- 后端选择：`vllm/v1/attention/selector.py` 与当前 platform 实现。
- 后端注册：`vllm/v1/attention/backends/registry.py`。
- 接口定义：`vllm/v1/attention/backend.py`。
- metadata 输入：`vllm/v1/worker/gpu/attn_utils.py` 与 Model Runner 的 Attention 准备流程。
- 缓存输入：Model Runner 根据 Scheduler 提供的 block IDs 维护 block table，并把物理 KV Cache 绑定到各层。
- 下游：所选后端依赖的 CUDA 扩展、Triton kernel 或第三方 Attention 库。

## 8. 当前结论

具体后端是 Paged KV Cache 与实际 Attention kernel 的适配器。学习时最重要的是沿所选后端的 metadata builder 和 implementation forward，确认 block table、slot mapping、KV Cache 与 kernel 参数如何衔接；无需同时展开其他后端。
