# 01 · vLLM Performance Benchmark

## Learning Objectives

- Build a reproducible vLLM benchmark and analyze effects from workload, configuration, and hardware.

## Prerequisites

- vLLM deployment, inference metrics, basic statistics, and GPU profiling.

## Recommended Learning Order

1. Experiment Design: define baseline, variables, workload, and metrics.
2. Benchmark Results: preserve raw data and environment details.
3. Performance Analysis: locate and validate bottlenecks.

## Key Questions

- Do request lengths, concurrency, and arrival rate represent the target workload?
- How should warm-up, repetitions, confidence intervals, and outliers be handled?

## Interview Focus

- Explain how TTFT, TPOT, throughput, and P99 were measured.
- Show reproducible commands, controlled charts, and sound attribution.
