# 01 · Architecture

## Learning Objectives

- Understand vLLM component boundaries, process topology, core data structures, and request lifecycle.

## Prerequisites

- Basic vLLM usage, online serving flow, and Python asynchronous programming.

## Recommended Learning Order

1. vLLM Architecture.
2. Request Lifecycle.
3. Core Call Chain, verified with breakpoints or logs.

## Key Questions

- How do the API layer, engine core, executor, and workers communicate?
- Where do request state transitions and cross-process transfers occur?

## Interview Focus

- Explain the end-to-end path by component responsibility.
- Identify critical entry points, boundary objects, and extension points.
