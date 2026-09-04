# AI Infrastructure Inference Optimization Notes

A long-term knowledge base for learning LLM inference systems, reading vLLM source code, running reproducible performance experiments, and preparing for interviews. The main path progresses from fundamentals to serving systems, source code, optimization, distributed inference, and hands-on projects.

## Learning Objectives

- Build an end-to-end mental model from Transformer computation to online LLM serving.
- Trace a vLLM request through scheduling, KV cache management, workers, model execution, and kernels.
- Diagnose bottlenecks with metrics and profilers, then design evidence-based optimizations.
- Produce reproducible projects, paper notes, and interview-ready explanations.

## Prerequisites

- Python, PyTorch, and basic Linux command-line skills.
- Linear algebra, probability, neural networks, and Transformer fundamentals.
- Basic C/C++ reading ability; CUDA knowledge can be developed during Phase 2.

## Recommended Learning Order

1. **Phase 1: LLM Inference Basics** — Transformer, attention, prefill/decode, KV cache, sampling, and metrics.
2. **Phase 2: GPU and CUDA Basics** — GPU architecture, execution model, memory hierarchy, Tensor Cores, and profiling.
3. **Phase 3: LLM Serving Systems** — Request flow, continuous batching, dynamic scheduling, latency, and throughput.
4. **Phase 4: vLLM Source Code** — Architecture, engines, scheduler, KV cache, workers, model execution, and kernels.
5. **Phase 5: Inference Optimization** — Quantization, attention/KV cache optimization, fusion, CUDA Graphs, and speculative decoding.
6. **Phase 6: Distributed Inference** — Tensor, pipeline, and expert parallelism, NCCL, and multi-GPU design.
7. **Phase 7: Hands-on Projects** — Benchmarking, source modification, and a mini LLM serving system.

After the main path, use 08-Interview-Preparation for focused review and 09-Research-Papers for primary-source study.

## Key Questions

- Why do prefill and decode exhibit different compute and memory-access characteristics?
- How should a serving system balance TTFT, TPOT, throughput, fairness, and GPU memory?
- How do PagedAttention and continuous batching solve different but related problems?
- How can an observed metric regression be traced to scheduling, model execution, or a GPU kernel?
- When should a system move from single-GPU optimization to distributed or disaggregated serving?

## Interview Focus

- Explain the complete lifecycle of one request from tokenization through streaming output.
- Quantify the relationships among memory capacity, compute, bandwidth, parallelism, and batch size.
- Prepare one optimization project with a baseline, bottleneck evidence, solution, controlled experiment, and attribution.
- Practice 30-second, 3-minute, and 5-minute versions of each core explanation.

## Maintenance Workflow

- Track progress in [Study Checklist](10-Study-Management/01-Checklist.md).
- Keep terminology consistent in the [Glossary](10-Study-Management/02-Glossary.md).
- Record the vLLM version or commit in every source note.
- Record environment, commands, raw metrics, and conclusions for every experiment.
