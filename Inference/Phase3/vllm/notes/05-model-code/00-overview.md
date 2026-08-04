# Model Code

## 学习目标

这一阶段从 GPU Model Runner 调用 `self.model(...)` 的位置继续向下，理解 vLLM 如何选择具体模型类，以及 Llama 模型如何使用张量并行 Linear、词表并行 Embedding 和 LM Head 完成推理。

完成本阶段后，应该能够说明：

1. Hugging Face config 中的 `architectures` 如何映射到 vLLM 模型类。
2. vLLM 为什么采用 lazy registration，以及何时真正导入模型模块。
3. `LlamaForCausalLM → LlamaModel → LlamaDecoderLayer` 的 forward 链路。
4. Q、K、V 和 MLP gate/up 权重如何融合为并行 Linear。
5. Column Parallel 与 Row Parallel 分别切分哪个维度、何时发生通信。
6. token embedding 和 LM Head 如何按 vocabulary 维度切分。
7. Hugging Face 权重名如何映射到 vLLM 的融合参数名。

本阶段先理解非量化 Llama 主线。量化方法、具体 Attention backend、RoPE、RMSNorm 和 logits processor 内部细节留在相关专题中展开。

## 阅读顺序

| 顺序 | 文件 | 主要关注点 |
| --- | --- | --- |
| 1 | `vllm/model_executor/models/registry.py` | architecture 如何解析为具体 vLLM 模型类 |
| 2 | `vllm/model_executor/models/llama.py` | Llama 模型结构、forward 和权重映射 |
| 3 | `vllm/model_executor/layers/linear.py` | QKV、MLP 和输出投影使用的张量并行 Linear |
| 4 | `vllm/model_executor/layers/vocab_parallel_embedding.py` | embedding 与 LM Head 的 vocabulary parallel 实现 |

## 模型选择调用链

```text
Hugging Face config.architectures
→ ModelRegistry.inspect_model_cls()
→ 确认模型能力与接口
→ ModelRegistry.resolve_model_cls()
→ lazy import 模型模块
→ ModelLoader 实例化具体模型类
→ 加载权重
```

在当前源码中，`_ModelRegistry` 是类名，`ModelRegistry` 是其全局实例。内置 Llama 映射为：

```text
"LlamaForCausalLM"
→ vllm.model_executor.models.llama
→ LlamaForCausalLM
```

## Llama Forward 调用链

```text
GPUModelRunner.execute_model()
→ LlamaForCausalLM.forward()
→ LlamaModel.forward()
→ VocabParallelEmbedding.forward()
→ LlamaDecoderLayer.forward()
   ├─ RMSNorm
   ├─ LlamaAttention.forward()
   │  ├─ QKVParallelLinear
   │  ├─ RoPE
   │  ├─ Attention
   │  └─ RowParallelLinear
   └─ LlamaMLP.forward()
      ├─ MergedColumnParallelLinear
      ├─ SiluAndMul
      └─ RowParallelLinear
→ final RMSNorm
→ hidden_states
```

采样阶段再使用：

```text
hidden_states
→ LlamaForCausalLM.compute_logits()
→ LogitsProcessor
→ ParallelLMHead weight
→ logits
```

## 张量并行关系

| 层 | 切分方式 | forward 后通信 |
| --- | --- | --- |
| `QKVParallelLinear` | 按 attention head / 输出维度切分 | 默认不 gather |
| `MergedColumnParallelLinear` | 分别切分融合后的 gate 和 up 输出 | 默认不 gather |
| `RowParallelLinear` | 按输入维度切分 | 默认 all-reduce 输出 |
| `VocabParallelEmbedding` | 按 vocabulary 行切分 | mask 后 all-reduce embedding |
| `ParallelLMHead` | 按 vocabulary 行切分 | 由 logits 处理与采样路径使用分片权重 |

Column Parallel 后的局部输出可以直接作为 Row Parallel 的局部输入，因此 Attention 和 MLP 内部不必在两层之间先 gather 完整 Tensor。

## 权重名称转换

Llama 使用 `WeightsMapper` 将 Hugging Face 的独立权重映射到 vLLM 的融合权重：

```text
q_proj / k_proj / v_proj
→ qkv_proj 的 q / k / v shard

gate_proj / up_proj
→ gate_up_proj 的 shard 0 / shard 1
```

具体 Linear 的 `weight_loader()` 再根据 tensor parallel rank、融合 shard ID 和参数维度选择当前 rank 应加载的切片。

## 文件职责边界

| 文件 | 一句话职责 |
| --- | --- |
| `registry.py` | 将 architecture 名称解析为可检查、可延迟加载的模型类 |
| `llama.py` | 定义推理专用 Llama 模型结构、forward 和权重映射 |
| `linear.py` | 定义可量化、支持融合权重加载的张量并行 Linear |
| `vocab_parallel_embedding.py` | 定义按词表维度切分的 embedding 与输出权重 |

## 当前阶段的检查清单

- [ ] 能从 `LlamaForCausalLM` architecture 找到实际 Python 类。
- [ ] 能画出 Llama 的完整 forward 层级。
- [ ] 能说明 QKV 为什么使用一个融合 Linear。
- [ ] 能区分 Column Parallel 与 Row Parallel 的切分和通信。
- [ ] 能说明 embedding 在 TP rank 上如何处理非本地 token。
- [ ] 能找到 hidden states 转换成 logits 的入口。
- [ ] 能说明独立 Hugging Face 权重如何加载到融合参数。

## 一句话总结

模型代码阶段负责把配置中的 architecture 解析为具体推理模型，并用融合、张量并行的模型层高效完成 Llama forward、权重加载和 logits 投影。

## 补充专题

- `05-model-loading-and-quantization.md`：从 checkpoint、loader、TP shard 追踪到量化参数。
- `06-lora.md`：理解 Adapter 管理、激活和 Model Runner 集成。
