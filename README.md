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
              ├── Prefill, decode, and KV cache
              ├── Online Softmax and FlashAttention
              ├── Attention variants
              ├── nano-vLLM implementation study
              └── vLLM architecture and source walkthrough
```

## Repository Guide

| Track | Topics | Start here |
| --- | --- | --- |
| Transformers | A minimal GPT training implementation and notebook | [`Transformers/miniGPT`](Transformers/miniGPT) |
| CUDA | GPU architecture, execution model, memory hierarchy, and performance fundamentals | [`CUDA/Phase1/overview.md`](CUDA/Phase1/overview.md) |
| Distributed Training | Parallelism fundamentals, Megatron-LM, PipeDream, and ZeRO | [`Distributed_Training/Phase1/Introduction.md`](Distributed_Training/Phase1/Introduction.md) |
| Inference | The inference lifecycle, attention optimization, and inference-engine internals | [`Inference/Phase1/Overview.md`](Inference/Phase1/Overview.md) |

## Inference Roadmap

The inference track progresses from concepts to production-engine internals:

1. **Foundations** — request lifecycle, tokenization, prefill, decode, sampling, batching, and KV cache.
2. **Attention optimization** — stable/online Softmax, FlashAttention, and common attention variants.
3. **Engine implementation** — nano-vLLM as a compact implementation, followed by a structured vLLM source study.

The vLLM walkthrough is organized into focused chapters covering:

- request call chains and engine initialization;
- scheduling and KV-cache management;
- GPU execution, model code, and attention backends;
- sampling, output processing, and online serving;
- multiprocessing and distributed execution;
- speculative decoding and multimodal processing;
- observability, profiling, and benchmarking.

Start with the [vLLM study overview](Inference/Phase3/vllm/notes/01-request-call-chain/00-overview.md), or browse the complete [vLLM notes directory](Inference/Phase3/vllm/notes).

## How to Use This Repository

- Follow one track in phase order if you want a structured curriculum.
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
