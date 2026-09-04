# 03 · Mini LLM Serving

## Learning Objectives

- Build a minimal serving system combining request queues, scheduling, KV cache, and inference.

## Prerequisites

- Python concurrency, PyTorch inference, continuous batching, and KV cache.

## Recommended Learning Order

1. Scheduler Design: define states and budgets.
2. KV Cache Implementation: test allocation and reclamation.
3. Inference Service Implementation: add streaming, cancellation, and metrics.

## Key Questions

- How are request state, cache mapping, and batch results kept consistent?
- How should the minimal system handle backpressure, failure, cancellation, and limits?

## Interview Focus

- Explain boundaries, state machines, interfaces, and concurrency on a whiteboard.
- Compare the mini implementation with vLLM and outline an evolution path.
