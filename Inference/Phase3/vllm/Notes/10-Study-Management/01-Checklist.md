# AI Infrastructure Study Checklist

> Change [ ] to [x] when an item is complete. Add a date and a link to the relevant note or experiment. Review this checklist weekly.

## Completed Chapters

- [ ] Phase 1: LLM Inference Basics
- [ ] Phase 2: GPU and CUDA Basics
- [ ] Phase 3: LLM Serving Systems
- [ ] Phase 4: vLLM Source Code
- [ ] Phase 5: Inference Optimization
- [ ] Phase 6: Distributed Inference
- [ ] Phase 7: Hands-on Projects
- [ ] Interview Preparation
- [ ] Research Papers

## Experiment Completion

- [ ] Record a standard environment, model, dataset, and vLLM commit
- [ ] Establish prefill/decode latency and throughput baselines
- [ ] Compare concurrency, input/output lengths, and batch configurations
- [ ] Measure KV cache capacity and block utilization
- [ ] Profile with Nsight Systems and Nsight Compute
- [ ] Compare quantization quality, memory, latency, and throughput
- [ ] Evaluate CUDA Graph and kernel optimizations
- [ ] Measure multi-GPU communication and scaling efficiency
- [ ] Project 1: vLLM Performance Benchmark
- [ ] Project 2: vLLM Source Modification
- [ ] Project 3: Mini LLM Serving

## Source Code Reading Progress

- [ ] Pin and record a vLLM version or commit
- [ ] Architecture: components, request lifecycle, and call chain
- [ ] Engine: synchronous/asynchronous engines and request handling
- [ ] Scheduler: state management, schedule function, and continuous batching
- [ ] KV Cache: block manager, PagedAttention, and lifecycle
- [ ] Worker: model runner and batch execution
- [ ] Model Executor: Llama, Qwen, and forward pass
- [ ] Attention Kernels: backend, Triton, CUDA, and PagedAttention
- [ ] Draw critical call graphs and verify them with breakpoints or logs

## Interview Preparation Status

- [ ] Complete the first pass of LLM fundamentals questions
- [ ] Complete the first pass of vLLM source questions
- [ ] Complete the first pass of CUDA and GPU optimization questions
- [ ] Complete the first pass of system-design questions
- [ ] Prepare 1-minute, 3-minute, and 5-minute project introductions
- [ ] Complete at least two timed mock interviews
- [ ] Maintain a mistake and weak-area list
- [ ] Support every critical conclusion with source code, experiments, or papers

## Weekly Review

| Week | Completed | Key Takeaways | Blockers | Next Week |
| --- | --- | --- | --- | --- |
| YYYY-WW |  |  |  |  |
