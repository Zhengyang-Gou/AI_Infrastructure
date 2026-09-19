# AI Infrastructure Learning

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

A learning-oriented repository for understanding the systems behind modern AI workloads—from Transformer fundamentals and GPU execution to distributed training and high-performance LLM inference.

The repository combines conceptual notes, paper walkthroughs, small experiments, and source-code studies. Most in-depth notes are currently written in Chinese; this page and the Chinese homepage provide bilingual project navigation.

## Learning Map

```text
Transformer foundations
        │
        ├── GPU & CUDA fundamentals
        ├── Distributed training
        │     ├── Data and model parallelism
        │     ├── Pipeline parallelism / PipeDream
        │     ├── Megatron-LM
        │     └── ZeRO
        └── LLM inference
              ├── 01 Basics → 02 Mini engine → 03 Core mechanisms
              ├── 04 vLLM source → 05 Benchmark / Profiling
              ├── 06 GPU architecture → 07 Triton → 08 CUDA
              └── 09 Engine × Kernel portfolio
```

## Repository Guide

| Track | Topics | Start here |
| --- | --- | --- |
| Transformers | A minimal GPT training implementation and notebook | [`Transformers/miniGPT`](Transformers/miniGPT) |
| CUDA | GPU architecture, execution model, memory hierarchy, and performance fundamentals | [`Inference/06-GPU-Architecture/`](Inference/06-GPU-Architecture/) |
| Distributed Training | Parallelism fundamentals, Megatron-LM, PipeDream, and ZeRO | [`Distributed_Training/Phase1/Introduction.md`](Distributed_Training/Phase1/Introduction.md) |
| Inference | The inference lifecycle, attention optimization, and inference-engine internals | [`Inference/Roadmap.md`](Inference/Roadmap.md) |

## Inference Roadmap

Follow nine stages. Stage 01 contains existing notes; stages 02–09 are empty directories for future work.

1. [LLM Inference Basics](Inference/01-LLM-Inference-Basics/README.md)
2. [Mini Inference Engine](Inference/02-Mini-Inference-Engine/)
3. [Engine Core Mechanisms](Inference/03-Engine-Core-Mechanisms/)
4. [vLLM Source](Inference/04-vLLM-Source/)
5. [Benchmark / Profiling](Inference/05-Benchmark-and-Profiling/)
6. [GPU Architecture](Inference/06-GPU-Architecture/)
7. [Triton](Inference/07-Triton/)
8. [CUDA](Inference/08-CUDA/)
9. [Engine × Kernel](Inference/09-Engine-Kernel-Integration/)

See the [full roadmap and schedule](Inference/Roadmap.md). Stage 01 has existing notes; content for stages 02–09 will be added as learning progresses.

## How to Use This Repository

- Follow inference stages 01–09 in order for a structured curriculum.
- Use the links in each note to move between concepts, code paths, and related papers.
- Treat vendored projects and source snapshots as reading material tied to the notes; check their upstream documentation before running or deploying them.
- Run experiments in an isolated Python environment and keep model weights, checkpoints, logs, and generated outputs outside version control.

For the miniGPT exercise:

```bash
cd Transformers/miniGPT
python train.py
```

Requirements vary by track. Python, PyTorch, CUDA, and multi-GPU dependencies should be installed according to the experiment you plan to run and your local hardware.

## Project Status

This is an evolving personal learning project. Notes may be expanded, reorganized, or corrected as the study progresses. Issues and pull requests that improve technical accuracy, explanations, or navigation are welcome.

## Acknowledgements

The repository studies ideas and implementations from projects and papers including PyTorch, CUDA, Megatron-LM, PipeDream, ZeRO, nano-vLLM, and vLLM. Their respective upstream repositories and publications remain the authoritative sources.
