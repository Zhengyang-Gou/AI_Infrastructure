# 03 · LLM Serving Systems

## Learning Objectives

- Understand the complete online path from request arrival to streamed output.
- Master batching, scheduling, resource management, and latency-throughput trade-offs.

## Prerequisites

- LLM inference stages, KV cache, concurrency, and basic service architecture.

## Recommended Learning Order

1. LLM Serving Architecture → Online Inference Workflow.
2. Continuous Batching → Dynamic Scheduling.
3. Latency and Throughput → Inference System Optimization.

## Key Questions

- How should unknown-length requests be batched while preserving fairness?
- How do static, dynamic, and continuous batching differ?
- How can queueing, prefill, decode, and network latency be isolated?

## Interview Focus

- Design a high-throughput serving system that satisfies an SLO.
- Explain scheduling effects on TTFT, TPOT, throughput, and tail latency.
- Discuss overload control, priorities, cancellation, and tenant isolation.
