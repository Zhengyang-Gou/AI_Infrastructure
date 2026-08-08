# AI 基础设施学习

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

这是一个面向学习的 AI 基础设施仓库，目标是从 Transformer 基础、GPU 执行原理出发，逐步理解分布式训练与高性能大模型推理系统。

仓库包含概念笔记、论文解读、小型实验与源码学习记录。当前深度笔记主要使用中文；英文主页与本页共同提供双语项目导航。

## 学习地图

```text
Transformer 基础
      │
      ├── GPU 与 CUDA 基础
      ├── 分布式训练
      │     ├── 数据并行与模型并行
      │     ├── 流水线并行 / PipeDream
      │     ├── Megatron-LM
      │     └── ZeRO
      └── 大模型推理
            ├── Prefill、Decode 与 KV Cache
            ├── Online Softmax 与 FlashAttention
            ├── Attention 变体
            ├── nano-vLLM 实现解析
            └── vLLM 架构与源码精读
```

## 仓库导航

| 学习方向 | 主要内容 | 学习入口 |
| --- | --- | --- |
| Transformers | 最小 GPT 训练实现与 Notebook | [`Transformers/miniGPT`](Transformers/miniGPT) |
| CUDA | GPU 架构、执行模型、存储层次与性能基础 | [`CUDA/Phase1/overview.md`](CUDA/Phase1/overview.md) |
| 分布式训练 | 并行基础、Megatron-LM、PipeDream 与 ZeRO | [`Distributed_Training/Phase1/Introduction.md`](Distributed_Training/Phase1/Introduction.md) |
| 推理系统 | 推理生命周期、Attention 优化与推理引擎内部实现 | [`Inference/Phase1/Overview.md`](Inference/Phase1/Overview.md) |

## 推理学习路线

推理部分按照“基础概念 → 核心优化 → 工程实现”的顺序展开：

1. **基础概念**：请求生命周期、Tokenizer、Prefill、Decode、采样、批处理与 KV Cache。
2. **Attention 优化**：Stable/Online Softmax、FlashAttention 与常见 Attention 变体。
3. **引擎实现**：先通过 nano-vLLM 理解精简实现，再系统阅读 vLLM 源码。

vLLM 源码学习进一步拆分为以下专题：

- 请求调用链与引擎初始化；
- 调度器与 KV Cache 管理；
- GPU 执行、模型代码与 Attention Backend；
- 采样、输出处理与在线服务；
- 多进程与分布式执行；
- 推测解码与多模态处理；
- 可观测性、性能分析与 Benchmark。

可以从 [vLLM 学习总览](Inference/Phase3/vllm/notes/01-request-call-chain/00-overview.md)开始，也可以直接浏览完整的 [vLLM 笔记目录](Inference/Phase3/vllm/notes)。

## 使用方式

- 希望系统学习时，可以按照各方向的 Phase 顺序阅读。
- 阅读笔记时，通过文内链接在概念、代码路径和相关论文之间跳转。
- 仓库中的第三方项目与源码快照主要服务于对应笔记；运行或部署前，请以其上游文档为准。
- 建议在独立 Python 环境中运行实验，并避免将模型权重、Checkpoint、日志和生成结果提交到版本控制。

运行 miniGPT 练习：

```bash
cd Transformers/miniGPT
python train.py
```

不同学习方向的环境要求并不相同。请根据实验内容和本地硬件，分别安装 Python、PyTorch、CUDA 与多 GPU 相关依赖。

## 项目状态

这是一个持续更新的个人学习项目。随着学习深入，笔记可能继续补充、调整结构或修正内容。欢迎通过 Issue 或 Pull Request 改进技术准确性、文字说明与项目导航。

## 致谢

本仓库学习和参考了 PyTorch、CUDA、Megatron-LM、PipeDream、ZeRO、nano-vLLM 与 vLLM 等项目及相关论文。权威信息请以对应的上游仓库和原始论文为准。
