| 模块 | 一级方向 | 二级方向 | 学习内容 |
|---|---|---|---|
| 大模型推理的背景和意义 | 背景认知 | LLM 原理 | 理解大语言模型基本结构、Transformer、Attention、Prefill、Decode |
| 大模型推理的背景和意义 | 背景认知 | 推理相关指标 | Latency、Throughput、TTFT、TPOT、QPS、显存占用、吞吐瓶颈 |
| 基础能力 | 算法基础 | LLM 原理 | Transformer、Attention、KV Cache、Prefill / Decode 流程 |
| 基础能力 | 算法基础 | 推理相关指标 | 延迟、吞吐、显存、带宽、计算量、算子性能 |
| 基础能力 | 编程语言基础 | C++ | C++ 基础、内存管理、多线程、性能优化基础 |
| 基础能力 | 编程语言基础 | Python | Python 基础、PyTorch、推理框架使用与调试 |
| 大模型推理优化手段 | 算法层面 | 模型轻量化 | 量化 |
| 大模型推理优化手段 | 算法层面 | 模型轻量化 | 蒸馏 |
| 大模型推理优化手段 | 算法层面 | 模型轻量化 | 稀疏 |
| 大模型推理优化手段 | 算法层面 | 优化相关 Attention 变种 | Sage Attention |
| 大模型推理优化手段 | 算法层面 | 优化相关 Attention 变种 | Lightning Attention |
| 大模型推理优化手段 | 算法层面 | 优化相关 Attention 变种 | MQA、GQA、MLA 等 |
| 大模型推理优化手段 | 算法层面 | 其他模块优化 | MoE |
| 大模型推理优化手段 | 算法层面 | 其他模块优化 | FlashAttention |
| 大模型推理优化手段 | 算法层面 | 其他模块优化 | Online Softmax |
| 大模型推理优化手段 | 算法层面 | 其他模块优化 | 其他融合算子 |
| 大模型推理优化手段 | 框架层面 | 常见推理框架 | vLLM |
| 大模型推理优化手段 | 框架层面 | 常见推理框架 | SGLang |
| 大模型推理优化手段 | 框架层面 | KV Cache 相关 | PagedAttention |
| 大模型推理优化手段 | 框架层面 | KV Cache 相关 | RadixAttention |
| 大模型推理优化手段 | 框架层面 | KV Cache 相关 | Prefix Cache |
| 大模型推理优化手段 | 框架层面 | Server 部分 | Continuous Batching |
| 大模型推理优化手段 | 框架层面 | Server 部分 | Chunked Prefill |
| 大模型推理优化手段 | 框架层面 | Engine 部分 | 解码策略 |
| 大模型推理优化手段 | 框架层面 | Engine 部分 | EPLB |
| 大模型推理优化手段 | 框架层面 | Engine 部分 | MTP |
| 大模型推理优化手段 | 框架层面 | Engine 部分 | PD 分离架构 |
| 大模型推理优化手段 | 框架层面 | 并行策略 | TP |
| 大模型推理优化手段 | 框架层面 | 并行策略 | DP |
| 大模型推理优化手段 | 框架层面 | 并行策略 | EP |
| 大模型推理优化手段 | 框架层面 | 并行策略 | PP |
| 大模型推理优化手段 | 框架层面 | 并行策略 | 各种并行策略的组合 |
| 大模型推理优化手段 | 框架层面 | 高性能算子实现 | CUDA Kernel、算子融合、矩阵乘优化、Reduce 优化 |
| 大模型推理优化手段 | 硬件层面 | 通信原语 | AllReduce、AllGather、ReduceScatter、Broadcast、P2P 通信 |
| 大模型推理优化手段 | 硬件层面 | 推理硬件架构 | GPU 架构、显存层级、带宽、NVLink、PCIe、算力瓶颈分析 |
| 手撕题 | LeetCode | Hot100 | 常见算法题、数组、链表、树、动态规划、图、回溯 |
| 手撕题 | LeetGPU | Reduce | GPU Reduce Kernel 手写与优化 |
| 手撕题 | LeetGPU | 矩阵乘 | GEMM、Tiling、Shared Memory、Tensor Core 优化 |
| 手撕题 | 算法手撕 | MHA 及其变体手撕 | MHA、MQA、GQA、MLA 前向过程实现 |
| 手撕题 | 算法手撕 | Decode Layer 手撕 | Decode 阶段 Attention、KV Cache 读取、Logits 计算 |
| 手撕题 | 算法手撕 | MoE 手撕 | Router、Top-k Expert、Expert Parallel、负载均衡 |