# 05 · Inference Optimization

## Learning Objectives

- Optimize inference across algorithms, precision, memory, kernels, execution graphs, and system policies.
- Apply a repeatable measure → hypothesize → experiment → attribute → regress workflow.

## Prerequisites

- LLM inference, GPU/CUDA, serving metrics, and key vLLM execution paths.

## Recommended Learning Order

1. Quantization → Attention Optimization → KV Cache Optimization.
2. Kernel Fusion → CUDA Graphs.
3. Speculative Decoding → Inference Cost Optimization.

## Key Questions

- Does an optimization reduce computation, memory traffic, launch overhead, or precision?
- Does higher throughput trade off TTFT, TPOT, tail latency, or quality?

## Interview Focus

- Use Roofline analysis and profiler data to justify an optimization.
- Design controlled experiments covering quality, performance, cost, and stability.
