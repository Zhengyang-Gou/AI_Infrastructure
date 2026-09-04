# 06 · Distributed Inference

## Learning Objectives

- Understand tensor, pipeline, and expert parallelism, including communication and memory trade-offs.
- Design a multi-GPU inference plan for a model, hardware topology, and SLO.

## Prerequisites

- Transformer computation, GPU/CUDA, collective communication, and single-GPU profiling.

## Recommended Learning Order

1. NCCL Communication → Tensor Parallelism.
2. Pipeline Parallelism → Expert Parallelism.
3. Multi-GPU Inference Architecture and hybrid-parallel experiments.

## Key Questions

- What communication, bubbles, and memory overhead does each strategy introduce?
- How do NVLink, PCIe, and inter-node networking change the optimal design?

## Interview Focus

- Derive All-Reduce and All-Gather locations and costs in tensor parallelism.
- Design a strategy for a model that exceeds one GPU and explain the trade-offs.
