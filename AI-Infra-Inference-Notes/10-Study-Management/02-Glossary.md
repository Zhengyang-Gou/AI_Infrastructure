# AI Infrastructure Glossary

> Definitions use the LLM inference-system context. Extend each entry with formulas, diagrams, source links, and experimental observations.

## Prefill

The phase that processes all prompt tokens in parallel, produces the state required for the first generated token, and initializes the KV cache. It usually exposes more parallel computation than decode, is often closer to compute-bound, and directly affects TTFT.

## Decode

The autoregressive phase that produces tokens one step at a time. Each step reads historical KV cache and appends new keys and values. Its small per-step matrix shapes and high memory traffic often make it closer to memory-bound. It directly affects TPOT.

## KV Cache

Cached key and value tensors for historical tokens at every Transformer layer. It avoids recomputing the entire context during decode. Capacity depends on layer count, KV-head count, head dimension, token count, batch size, and dtype.

## PagedAttention

A memory-management approach that maps logical KV sequences to non-contiguous physical blocks and lets kernels access them through block tables. It reduces fragmentation and supports dynamic growth and sharing. It is not a different attention equation.

## Continuous Batching

A scheduling approach that adds new requests and removes completed requests at every iteration so the GPU batch remains populated. It reduces waiting and batch bubbles compared with static batching, but requires more sophisticated scheduling and KV cache management.

## Tensor Parallelism

A strategy that partitions tensor dimensions and the computation of individual operators across devices, then combines partial results with collectives such as All-Reduce or All-Gather. It addresses capacity or compute limits but is sensitive to communication volume and topology.

## Memory Bound

A condition in which performance is mainly limited by memory bandwidth or data movement. Adding compute units does not directly improve performance. Arithmetic intensity, Roofline analysis, and profiler metrics help establish the diagnosis.

## Compute Bound

A condition in which performance is mainly limited by arithmetic throughput. Optimization often focuses on reducing FLOPs, using efficient precision and Tensor Cores, or improving parallel execution efficiency.

## Throughput

The amount of work completed per unit time. Common LLM serving units include requests/s, input tokens/s, output tokens/s, and total tokens/s. Every comparison should state the workload, length distribution, and SLO.

## Latency

The elapsed time for a request or stage. Important LLM serving measurements include end-to-end latency, TTFT, TPOT/ITL, and P50/P95/P99 percentiles rather than only the mean.
