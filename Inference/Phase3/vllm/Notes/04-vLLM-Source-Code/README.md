# 04 · vLLM Source Code

## Learning Objectives

- Trace an API request through engines, scheduling, KV cache, workers, models, and kernels.
- Develop a source-reading method that remains useful as vLLM evolves.

## Prerequisites

- Python/PyTorch, LLM serving, CUDA basics, async programming, and multiprocessing.

## Recommended Learning Order

1. Architecture → Engine → Scheduler.
2. KV Cache → Worker → Model Executor.
3. Attention Kernels, followed by an end-to-end request trace.

## Key Questions

- How are control-plane and execution-plane responsibilities divided?
- How does scheduler output become GPU input while KV blocks remain consistent?
- How are attention backends selected behind a common interface?

## Interview Focus

- Draw the core vLLM components and request path on a whiteboard.
- Identify important scheduler, PagedAttention, and worker source locations by commit.
- Explain the scope, validation, and performance impact of a source change.
