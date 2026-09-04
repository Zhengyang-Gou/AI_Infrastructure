# 07 · Attention Kernels

## Learning Objectives

- Understand attention backend abstractions, kernel selection, and PagedAttention GPU implementation.

## Prerequisites

- Attention math, CUDA/Triton, GPU memory hierarchy, and PagedAttention.

## Recommended Learning Order

1. Attention Backend.
2. Triton Kernels → CUDA Kernels.
3. PagedAttention Kernel Analysis with profiler validation.

## Key Questions

- How do hardware, dtype, head size, and requested features constrain backend selection?
- How do tiling, online Softmax, and paged addressing map to threads and memory accesses?

## Interview Focus

- Explain attention-kernel memory bottlenecks and common optimizations.
- Analyze a kernel through interfaces, layouts, thread mapping, and measurements.
