# Attention

## 1. 学习目标

本阶段沿着模型层产生的 Q、K、V，追踪它们如何进入统一 Attention 层、绑定运行时 KV Cache 与 Attention Metadata，再交给实际后端执行 kernel，最后返回 Attention 输出。

完成本阶段后，应能够说明以下链路：

```text
模型 Attention 层产生 Q、K、V
→ Attention.forward()
→ 从 ForwardContext 取得 KV Cache、slot mapping 和 Attention Metadata
→ 选定 AttentionBackend 对应的实现类
→ 写入当前 K、V 并调用具体 Attention kernel
→ 返回 Attention output
```

## 2. 阅读文件与顺序

| 顺序 | 笔记 | 当前仓库源码 | 阅读目的 |
| --- | --- | --- | --- |
| 1 | `01-attention-layer.md` | `vllm/model_executor/layers/attention/attention.py` | 理解模型代码与后端实现之间的统一入口 |
| 2 | `02-backend-registry.md` | `vllm/v1/attention/backends/registry.py` | 理解后端名称、类路径和自定义覆盖的注册机制 |
| 3 | `03-attention-backend.md` | `vllm/v1/attention/backends/<实际使用后端>.py` | 只选择当前运行配置最终采用的一个后端，理解 metadata、KV Cache 与 kernel 的连接 |

Roadmap 中的 `vllm/attention/layer.py` 在当前仓库不存在。对应的 `Attention` 类现位于 `vllm/model_executor/layers/attention/attention.py`，因此第一篇笔记按当前源码路径记录。

## 3. 核心对象

| 对象 | 所在层次 | 作用 |
| --- | --- | --- |
| `Attention` | 模型执行层 | 接收 Q、K、V，选择后端实现并提供统一 forward 入口 |
| `AttentionBackend` | 后端接口层 | 声明后端名称、实现类、metadata builder 和 KV Cache 布局等能力 |
| `AttentionImpl` | 后端实现层 | 使用 KV Cache 与 Attention Metadata 执行具体 Attention 计算 |
| `AttentionMetadataBuilder` | Model Runner 与后端之间 | 将本轮 batch、序列长度和 block table 等信息构造成后端 metadata |
| `ForwardContext` | 一次模型 forward 的运行时上下文 | 按层保存 Attention Metadata、KV Cache 绑定和 slot mapping |
| `AttentionBackendEnum` | 注册层 | 将后端名称映射到可延迟导入的后端类路径，并支持运行时覆盖 |

## 4. 主执行流程

### 后端确定与初始化

```text
Attention.__init__()
→ get_attn_backend()
→ 根据 AttentionConfig、平台、dtype、head size、KV Cache dtype 等条件选择后端
→ backend.get_impl_cls()
→ 创建具体 AttentionImpl
→ 将 Attention 层注册到 static_forward_context
```

`registry.py` 负责“名称到类路径”的注册与解析；真正结合平台和配置选择后端的逻辑位于 `vllm/v1/attention/selector.py` 及平台实现中。两者职责不能混为一谈。

### 一轮 Attention 执行

```text
query / key / value
→ Attention.forward()
→ reshape 为 head 维度
→ unified_kv_cache_update()（仅部分后端需要独立执行）
→ unified_attention_with_output()
→ get_attention_context()
→ 取得当前层的 metadata、KV Cache 和 slot mapping
→ AttentionImpl.forward()
→ CUDA / Triton 等 kernel
→ output
```

是否由 `AttentionImpl.forward()` 内部同时写入 KV Cache，取决于后端的 `forward_includes_kv_cache_update`。不能假设所有后端都使用完全相同的写入路径。

## 5. 关键数据如何连接

| 数据 | 来源 | 去向 | 作用 |
| --- | --- | --- | --- |
| `query`、`key`、`value` | 具体模型的 Attention 层 | `Attention.forward()` | 当前批次要执行 Attention 的 Q、K、V |
| `kv_cache` | Model Runner 初始化并绑定到各 Attention 层 | 具体后端实现 | 保存历史 K、V，并接收当前 K、V |
| `block_table` | Scheduler 的 block 分配结果，经 Model Runner 整理 | 后端 metadata | 将请求的逻辑 token 位置映射到 KV Cache 物理 block |
| `slot_mapping` | Model Runner 根据调度结果构造 | KV Cache 更新路径 | 指定当前 K、V 写入的物理 slot |
| `attn_metadata` | 后端对应的 metadata builder | `AttentionImpl.forward()` | 描述序列边界、上下文长度、block table 等运行时信息 |
| `output` | 具体 Attention kernel | 模型 Decoder Layer | 作为 Attention 子层的输出继续参与投影与残差计算 |

## 6. 后端阅读边界

第三篇笔记不固定为 FlashAttention、FlashInfer 或 Triton。应先确认当前运行环境最终选择的后端，再只阅读对应文件，例如：

```text
vllm/v1/attention/backends/flash_attn.py
vllm/v1/attention/backends/flashinfer.py
vllm/v1/attention/backends/triton_attn.py
```

不同 GPU 平台、软件依赖、模型特性、数据类型、KV Cache 类型和显式配置可能得到不同结果。因此，本阶段只建立通用阅读框架，不虚构一个固定后端。

## 7. 完成标准

- 能说明 `Attention` 抽象层为什么不直接写死某一种 kernel。
- 能区分后端选择、后端注册、metadata 构造和 kernel 执行四个职责。
- 能说明 `block_table` 与 `slot_mapping` 分别参与 KV Cache 的读取和写入。
- 能指出 KV Cache 与 Attention Metadata 如何通过 `ForwardContext` 进入当前层的 forward。
- 能沿实际运行后端追踪到具体实现类的 `forward()`。

## 8. 当前结论

vLLM 的 Attention 主线不是“模型层直接调用某个固定 kernel”，而是由统一 `Attention` 层接收 Q、K、V，通过运行配置选择后端，并在 forward 时从上下文取出当前层的 KV Cache、slot mapping 与 metadata，再把计算委托给具体后端实现。
