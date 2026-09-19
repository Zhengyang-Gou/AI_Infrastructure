# LLM 推理学习路线

按照 **基础 → 最小引擎 → 引擎机制 → vLLM 源码 → 性能分析 → GPU → Triton → CUDA → 集成作品** 推进。01 已有正文，02–09 保留空目录，后续学习时逐步添加内容。

前八阶段共 **20～23 周**，第九阶段持续迭代。已掌握 01 后，可从 02 开始，后续阶段 02–08 预计 **18～21 周**。

| 阶段 | 时间 | 学什么 | 目标 |
| --- | ---: | --- | --- |
| [01. LLM 推理基础](01-LLM-Inference-Basics/README.md) | 2 周 | Prefill、Decode、KV Cache、MHA / GQA / MQA、Sampling | 能解释一次生成请求如何执行 |
| [02. Mini Inference Engine](02-Mini-Inference-Engine/) | 2 周 | KV Cache、Batch Decode、Request Lifecycle | 自己写一个最小推理引擎 |
| [03. Engine 核心机制](03-Engine-Core-Mechanisms/) | 3 周 | Scheduler、Continuous Batching、Paged KV Cache、Prefix Cache、Chunked Prefill | 理解现代推理引擎为什么这样设计 |
| [04. vLLM 源码](04-vLLM-Source/) | 3 周 | Scheduler、KV Cache Manager、Model Runner、Attention Backend | 能追踪 Request 到 Kernel 的完整链路 |
| [05. Benchmark / Profiling](05-Benchmark-and-Profiling/) | 1～2 周 | TTFT、TPOT、吞吐、并发、显存 | 能判断系统哪里慢 |
| [06. GPU Architecture](06-GPU-Architecture/) | 2 周 | SM、Warp、HBM、L2、Shared Memory、Tensor Core | 建立 GPU 性能模型 |
| [07. Triton](07-Triton/) | 3 周 | RMSNorm、Softmax、Matmul、RoPE、Attention | 能自己写 LLM Kernel |
| [08. CUDA](08-CUDA/) | 4～6 周 | Memory Coalescing、Shared Memory、Occupancy、GEMM、Attention | 能进一步深入 Kernel 优化 |
| [09. Engine × Kernel](09-Engine-Kernel-Integration/) | 持续 | 替换 / 优化 vLLM Kernel | 形成求职作品集 |

## 目录安排

```text
Inference/
├── Roadmap.md
├── 01-LLM-Inference-Basics/       # 已有基础笔记
├── 02-Mini-Inference-Engine/
├── 03-Engine-Core-Mechanisms/
├── 04-vLLM-Source/
├── 05-Benchmark-and-Profiling/
├── 06-GPU-Architecture/
├── 07-Triton/
├── 08-CUDA/
├── 09-Engine-Kernel-Integration/
├── References/                  # 参考索引与扩展主题
├── Phase2/                      # 已有 Softmax / FlashAttention 资料
└── Phase3/                      # nano-vLLM / vLLM 源码快照
```

## 阶段衔接

- **02 → 03**：先实现连续 KV Cache 和固定 Batch，再逐步加入动态调度、分页管理与前缀复用。
- **03 → 04**：带着自己实现中的问题阅读 vLLM，对照模块职责和数据流。
- **04 → 05**：能追踪执行链路后，建立可复现的基准和瓶颈报告。
- **05 → 06**：用 GPU 性能模型解释已采集的 Profile。
- **06 → 07 → 08**：先用 Triton 实现算子，再深入 CUDA 的执行和访存优化。
- **08 → 09**：把有证据的 Kernel 优化接入引擎，以正确性和端到端收益完成作品。

## 使用方式

01 保留现有笔记与导航，02–09 只保留目录。学习时按需新增笔记；代码实验可放在对应阶段的 `code/`，配置与报告放在 `experiments/`，有实际内容时再创建。

阶段时间是学习预算，进入下一阶段前优先检查验收项。统一记录硬件、软件版本、运行命令、正确性误差和基准口径，让后续阶段复用前面的结果。

现有源码与算法笔记通过 [参考索引](References/README.md) 连接；量化、MoE、分布式推理等原有主题保留在 [扩展主题备忘](References/Extended-Topics.md)，按项目需要选学。
