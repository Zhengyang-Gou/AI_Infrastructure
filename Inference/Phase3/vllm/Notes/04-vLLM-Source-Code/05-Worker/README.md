# 05 · Worker

## Learning Objectives

- Understand worker initialization, model loading, input preparation, batch execution, and result return.

## Prerequisites

- PyTorch execution, CUDA streams/events, and vLLM scheduler output.

## Recommended Learning Order

1. Worker Execution Flow.
2. Model Runner.
3. Batch Execution Flow, correlated with CPU and GPU timelines.

## Key Questions

- How is scheduler output transformed into executable tensors and metadata?
- How does a worker handle distributed synchronization, sampling, and CUDA Graphs?

## Interview Focus

- Describe the worker-side stages of one engine step.
- Locate CPU scheduling, host-to-device, and GPU execution bottlenecks.
