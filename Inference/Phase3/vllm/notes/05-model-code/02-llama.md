# Llama Model

## 1. 文件定位

- 文件路径：`vllm/model_executor/models/llama.py`
- 所属层次：具体模型实现层
- 核心职责：定义与 Hugging Face Llama 权重兼容的推理专用模型结构、forward、logits 计算和权重加载。
- 在调用链中的位置：由 Model Loader 实例化，由 `GPUModelRunner` 在 forward 和 sampling 阶段调用。

该实现没有训练路径，重点围绕高效推理、Tensor Parallel、Pipeline Parallel、KV Cache Attention 和融合权重组织。

## 2. 核心类与组件

| 类 / 组件 | 作用 | 关键子层 |
| --- | --- | --- |
| `LlamaForCausalLM` | Causal LM 最外层模型 | `LlamaModel`、`ParallelLMHead`、`LogitsProcessor` |
| `LlamaModel` | embedding、decoder layers 与 final norm | `VocabParallelEmbedding`、`LlamaDecoderLayer` |
| `LlamaDecoderLayer` | 单层 Transformer block | RMSNorm、Attention、MLP |
| `LlamaAttention` | QKV 投影、RoPE、Attention 和输出投影 | `QKVParallelLinear`、`Attention`、`RowParallelLinear` |
| `LlamaMLP` | SwiGLU 风格前馈网络 | `MergedColumnParallelLinear`、`SiluAndMul`、`RowParallelLinear` |
| `WeightsMapper` | 将 HF 参数名映射到融合参数 | QKV 与 gate/up mapping |

## 3. 模型初始化结构

```text
LlamaForCausalLM
├─ LlamaModel
│  ├─ VocabParallelEmbedding
│  ├─ N × LlamaDecoderLayer
│  │  ├─ RMSNorm
│  │  ├─ LlamaAttention
│  │  ├─ RMSNorm
│  │  └─ LlamaMLP
│  └─ RMSNorm
├─ ParallelLMHead
└─ LogitsProcessor
```

Pipeline Parallel 只在当前 rank 创建属于自己的 layer 范围。第一 rank 负责 embedding，最后 rank 负责 final norm、LM Head 和 logits；不属于当前 rank 的边界层使用 `PPMissingLayer` 占位。

## 4. Attention 流程

```text
LlamaAttention.forward(positions, hidden_states)
→ qkv_proj(hidden_states)
→ 按 q_size / kv_size 切分 q、k、v
→ rotary_emb(positions, q, k)
→ Attention(q, k, v)
→ o_proj(attn_output)
→ output
```

`QKVParallelLinear` 按 head 维度切分输出。查询头总数必须能被 TP size 整除；KV 头多于 TP rank 时切分，KV 头少于 TP rank 时在多个 rank 间复制，以支持 GQA 和 MQA。

具体 K、V 如何写入 KV Cache，以及 Prefill / Decode kernel 如何执行，由 `Attention` 和所选 backend 完成。

## 5. Decoder Layer 与 MLP 流程

```text
输入 hidden_states
→ input_layernorm（同时处理 residual）
→ self_attn
→ post_attention_layernorm（融合 residual）
→ mlp
→ 返回 hidden_states, residual
```

```text
LlamaMLP.forward(x)
→ gate_up_proj(x)
→ SiluAndMul()
→ down_proj(x)
→ output
```

`MergedColumnParallelLinear` 将 HF 的 `gate_proj` 与 `up_proj` 融合为一次投影；激活后，`RowParallelLinear` 将每个 rank 的局部结果 all-reduce 回完整 hidden size。

## 6. 整体 Forward 与 Logits

```text
LlamaForCausalLM.forward()
→ LlamaModel.forward()
→ 第一 PP rank 执行 embedding
→ 依次执行本 rank 的 decoder layers
→ 非最后 PP rank 返回 IntermediateTensors
→ 最后 PP rank 执行 final RMSNorm
→ 返回 hidden_states
```

```text
LlamaForCausalLM.compute_logits(hidden_states)
→ LogitsProcessor(lm_head, hidden_states)
→ logits
```

模型 forward 不在每个 token 位置都立即计算 logits。Model Runner 会先通过 `logits_indices` 选出需要采样的位置，再调用 `compute_logits()`。

## 7. 权重加载

`LlamaModel.hf_to_vllm_mapper` 定义以下映射：

```text
.q_proj    → .qkv_proj, shard "q"
.k_proj    → .qkv_proj, shard "k"
.v_proj    → .qkv_proj, shard "v"
.gate_proj → .gate_up_proj, shard 0
.up_proj   → .gate_up_proj, shard 1
```

`AutoWeightsLoader` 遍历 checkpoint 参数，应用名称映射后调用目标参数绑定的 `weight_loader`。若 input embedding 与 LM Head 共享权重，最外层 loader 会跳过重复的 `lm_head` 权重。

## 8. 输入与输出

### 输入

- `input_ids` 或已计算的 `inputs_embeds`。
- 与 token 一一对应的 `positions`。
- 非第一 PP rank 接收的 `IntermediateTensors`。
- 初始化时的 `VllmConfig` 与 Hugging Face `LlamaConfig`。

### 输出

- 最后 PP rank 返回 hidden states。
- 非最后 PP rank 返回包含 hidden states 和 residual 的 `IntermediateTensors`。
- `compute_logits()` 返回 vocabulary logits。
- `load_weights()` 返回已成功加载的参数名集合。

### 状态变化

- 初始化阶段根据 TP / PP 配置创建本 rank 所需参数分片。
- 权重加载阶段把 HF 独立权重写入 vLLM 融合参数的对应 shard。
- forward 期间 Attention 通过 forward context 使用本轮 KV Cache 元数据。

## 9. 关键代码解析

### `LlamaMLP.__init__()`

### `LlamaMLP.forward()`

### `LlamaAttention.__init__()`

### `LlamaAttention.forward()`

### `LlamaAttention._init_rotary_emb()`

### `LlamaDecoderLayer.__init__()`

### `LlamaDecoderLayer.forward()`

### `LlamaModel.__init__()`

### `LlamaModel.forward()`

### `LlamaModel.load_weights()`

### `LlamaForCausalLM.__init__()`

### `LlamaForCausalLM.forward()`

### `LlamaForCausalLM.compute_logits()`

### `LlamaForCausalLM.load_weights()`

## 10. 与其他文件的关系

- 上游选择：`registry.py` 和 Model Loader。
- 执行入口：`vllm/v1/worker/gpu/model_runner.py`。
- 并行 Linear：`vllm/model_executor/layers/linear.py`。
- embedding 与 LM Head：`vllm/model_executor/layers/vocab_parallel_embedding.py`。
- Attention：模型层创建通用 `Attention`，具体 backend 在 forward context 中工作。
- 权重加载：依赖 `AutoWeightsLoader`、`WeightsMapper` 和各层的参数 loader。

## 11. 当前结论

`llama.py` 将标准 Llama 结构重组为适合 vLLM 推理的 PP/TP 模型：融合 QKV 与 gate/up 投影，保留 residual 与 KV Cache Attention 边界，并提供兼容 Hugging Face checkpoint 的权重映射。
