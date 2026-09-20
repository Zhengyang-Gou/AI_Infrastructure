# Nano-vLLM 源码学习笔记

以本目录保存的 Nano-vLLM 源码为例，按依赖关系阅读。原始笔记中的讲解和代码片段已按主题保留。

建议先复习 [推理基础](../../01-LLM-Inference-Basics/README.md) 中的 Prefill / Decode、KV Cache 和采样，再依次阅读：

| 顺序 | 笔记 | 重点 |
| --- | --- | --- |
| 01 | [入口配置与生成主循环](01-Entry-and-Generation.md) | `example.py`、`llm.py`、`config.py`、`sampling_params.py`、`llm_engine.py` |
| 02 | [请求状态与生命周期](02-Sequence-and-Lifecycle.md) | `sequence.py` |
| 03 | [分页 KV Cache 与前缀缓存](03-Paged-KV-Cache-and-Prefix-Caching.md) | `block_manager.py` |
| 04 | [调度与批处理](04-Scheduler-and-Batching.md) | `scheduler.py` |
| 05 | [模型执行器与输入组织](05-ModelRunner-and-Context.md) | `context.py`、`model_runner.py` |
| 06 | [Qwen3 模型与基础算子](06-Qwen3-and-Basic-Layers.md) | `qwen3.py`、`layernorm.py`、`activation.py`、`rotary_embedding.py` |
| 07 | [Attention 与 KV Cache 读写](07-Attention-and-Cache-Access.md) | `attention.py` |
| 08 | [张量并行与权重加载](08-Tensor-Parallel-and-Weight-Loading.md) | `linear.py`、`embed_head.py`、`loader.py` |
| 09 | [采样与输出 Token](09-Sampling.md) | `sampler.py` |
| 10 | [一次 Step 的完整数据流](10-End-to-End-Step.md) | 串起一次完整推理步骤 |

第一遍先掌握单卡生成主线；第二遍再细看前缀缓存、Chunked Prefill、张量并行和 CUDA Graph。每篇开头提供对应源码链接，结尾可跳转到前后篇。

[返回阶段目录](../README.md) · [Nano-vLLM 源码](../source/nano-vllm/)
