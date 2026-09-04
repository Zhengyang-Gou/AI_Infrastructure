# 02 · Engine

## Learning Objectives

- Understand synchronous and asynchronous engine responsibilities, lifecycles, and request interfaces.

## Prerequisites

- vLLM architecture, generation parameters, async/await, and producer-consumer patterns.

## Recommended Learning Order

1. LLMEngine.
2. AsyncLLMEngine.
3. Request Processing Flow and streamed output.

## Key Questions

- How does an engine add, advance, cancel, and return requests?
- How do synchronous and asynchronous control paths differ?

## Interview Focus

- Explain the interface boundaries among engine, scheduler, and executor.
- Trace the critical methods from request admission to final output.
