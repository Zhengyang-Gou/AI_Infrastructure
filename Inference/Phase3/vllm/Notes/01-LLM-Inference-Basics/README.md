# 01 · LLM Inference Basics

## Learning Objectives

- Understand Decoder-only Transformer computation and autoregressive generation.
- Distinguish prefill from decode and master KV caching, sampling, and inference metrics.

## Prerequisites

- Linear algebra, Softmax, neural-network forward passes, and basic PyTorch.

## Recommended Learning Order

1. Transformer Architecture → Attention Mechanism.
2. Prefill and Decode → KV Cache Principles.
3. Sampling Strategies → LLM Inference Metrics.

## Key Questions

- Why can decode not be fully parallelized like training?
- How do KV cache shape and capacity scale with context length?
- How do sampling parameters affect quality, determinism, and performance?

## Interview Focus

- Derive attention tensor shapes and computational complexity.
- Explain TTFT, TPOT, throughput, and end-to-end latency.
- Compare decoding cost with and without KV cache.
