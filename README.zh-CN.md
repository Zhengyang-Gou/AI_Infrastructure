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
            ├── 01 基础 → 02 最小引擎 → 03 核心机制
            ├── 04 vLLM 源码 → 05 Benchmark / Profiling
            ├── 06 GPU 架构 → 07 Triton → 08 CUDA
            └── 09 Engine × Kernel 作品集
```

## 仓库导航

| 学习方向 | 主要内容 | 学习入口 |
| --- | --- | --- |
| Transformers | 最小 GPT 训练实现与 Notebook | [`Transformers/miniGPT`](Transformers/miniGPT) |
| CUDA | GPU 架构、执行模型、存储层次与性能基础 | [`Inference/06-GPU-Architecture/`](Inference/06-GPU-Architecture/) |
| 分布式训练 | 并行基础、Megatron-LM、PipeDream 与 ZeRO | [`Distributed_Training/Phase1/Introduction.md`](Distributed_Training/Phase1/Introduction.md) |
| 推理系统 | 推理生命周期、Attention 优化与推理引擎内部实现 | [`Inference/Roadmap.md`](Inference/Roadmap.md) |

## 推理学习路线

按以下九阶段推进；01 已有笔记，02–09 暂为空目录。

1. [LLM 推理基础](Inference/01-LLM-Inference-Basics/README.md)
2. [Mini Inference Engine](Inference/02-Mini-Inference-Engine/)
3. [Engine 核心机制](Inference/03-Engine-Core-Mechanisms/)
4. [vLLM 源码](Inference/04-vLLM-Source/)
5. [Benchmark / Profiling](Inference/05-Benchmark-and-Profiling/)
6. [GPU Architecture](Inference/06-GPU-Architecture/)
7. [Triton](Inference/07-Triton/)
8. [CUDA](Inference/08-CUDA/)
9. [Engine × Kernel](Inference/09-Engine-Kernel-Integration/)

[查看完整路线与时间安排](Inference/Roadmap.md)。01 已有正文，02–09 的内容将在学习时逐步添加。

## 使用方式

- 希望系统学习推理时，按照 01–09 的编号顺序阅读。
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
