# 03 · Scheduler

## Learning Objectives

- Understand request states, token budgets, KV block allocation, and continuous-batching decisions.

## Prerequisites

- Continuous batching, prefill/decode, queues, and KV cache.

## Recommended Learning Order

1. Scheduler Design → Sequence State Management.
2. Schedule Function Analysis.
3. Continuous Batching Source Analysis and scheduling experiments.

## Key Questions

- How do prefill and decode compete for scheduling budget?
- How do preemption, priority, and chunked prefill affect SLOs?

## Interview Focus

- Simulate one scheduling iteration for a given request queue.
- Explain effects on throughput, TTFT, fairness, and memory.
