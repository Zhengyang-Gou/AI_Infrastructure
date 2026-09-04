# 01 · Quantization

## Learning Objectives

- Understand low-precision representation for weights, activations, and KV cache, including error and kernel support.

## Prerequisites

- Floating-point formats, matrix multiplication, distributions, calibration, and Tensor Cores.

## Recommended Learning Order

1. FP16 and BF16 → INT8 → INT4.
2. GPTQ → AWQ.
3. Compare quality, memory, throughput, and latency on the same model.

## Key Questions

- How do symmetric/asymmetric and per-tensor/per-channel/group-wise schemes differ?
- Why does weight compression not always produce proportional speedup?

## Interview Focus

- Compare PTQ, QAT, weight-only quantization, and W8A8.
- Explain GPTQ and AWQ intuition, error handling, and deployment constraints.
