# 02 · GPU and CUDA Basics

## Learning Objectives

- Map GPU hardware to the CUDA execution model and memory hierarchy.
- Determine whether a workload is compute-bound or memory-bound and verify it with Nsight.

## Prerequisites

- Basic C/C++, parallel-computing intuition, and the LLM forward path.

## Recommended Learning Order

1. GPU Architecture → CUDA Execution Model → Thread/Warp/Block/SM.
2. GPU Memory Hierarchy → Tensor Cores.
3. Compute Bound vs Memory Bound → Nsight Profiling.

## Key Questions

- How do warp scheduling, occupancy, and latency hiding interact?
- How do coalescing, shared memory, and register use affect kernel performance?
- How can Roofline analysis and profiler evidence identify a bottleneck?

## Interview Focus

- Explain thread hierarchy, hardware mapping, warp divergence, and bank conflicts.
- Compare HBM, L2, shared memory, and registers.
- Interpret key Nsight metrics and propose testable optimizations.
