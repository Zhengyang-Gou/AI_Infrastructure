# 06 · Model Executor

## Learning Objectives

- Understand model registration, weight loading, parallel layers, and forward execution.

## Prerequisites

- Llama/Qwen architecture, PyTorch modules, tensor parallelism, and quantization basics.

## Recommended Learning Order

1. Forward Pass Analysis.
2. Llama Model Execution.
3. Qwen Model Execution and model-specific differences.

## Key Questions

- How are Hugging Face weights mapped into vLLM model layers?
- How do attention, MLP, normalization, and sampling form the execution path?

## Interview Focus

- Trace one Transformer layer and its tensor shapes.
- Explain extension points for model adaptation, sharding, and quantized loading.
