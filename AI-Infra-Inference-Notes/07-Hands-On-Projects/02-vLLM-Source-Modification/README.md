# 02 · vLLM Source Modification

## Learning Objectives

- Implement a scheduler optimization and validate correctness, performance, and regressions.

## Prerequisites

- vLLM build/test workflow, scheduler internals, Git, and benchmarking.

## Recommended Learning Order

1. Establish an unmodified baseline.
2. Document design, implementation, and tests in Scheduler Optimization.
3. Summarize results and applicability boundaries.

## Key Questions

- Which bottleneck is targeted, and which invariants or fairness properties are at risk?
- How can noise be excluded and gains attributed to the intended change?

## Interview Focus

- Explain source entry points, data structures, modification scope, and test coverage.
- Prepare speedup, regression cases, and next-step improvements.
