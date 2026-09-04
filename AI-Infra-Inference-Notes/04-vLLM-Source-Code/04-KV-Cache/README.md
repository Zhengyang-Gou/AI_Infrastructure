# 04 · KV Cache

## Learning Objectives

- Understand mappings among logical tokens, physical blocks, block tables, and GPU cache.

## Prerequisites

- Attention, KV cache sizing, memory paging, and basic data structures.

## Recommended Learning Order

1. PagedAttention Principles → Block Manager.
2. KV Cache Lifecycle → GPU Memory Management.
3. Observe block changes during admission, completion, sharing, and preemption.

## Key Questions

- How does PagedAttention reduce internal and external fragmentation?
- How are blocks allocated, shared, reference-counted, and reclaimed?

## Interview Focus

- Calculate KV cache memory for a model, precision, and context length.
- Compare contiguous allocation with paged allocation.
