# Tracing and Profiling

## 1. 文件定位

- Tracing：`vllm/tracing/__init__.py`、`otel.py`、`utils.py`。
- Profiling：`vllm/profiler/wrapper.py`、`layerwise_profile.py`。
- 职责：分别追踪请求跨组件生命周期，以及捕获 Worker 内 CPU/CUDA operator 时间线。

## 2. Tracing 数据路径

```text
客户端 trace headers
→ API Server 提取 trace context
→ 请求对象携带上下文
→ API / Engine / Scheduler 创建 spans
→ 设置 request、model、token 等 attributes
→ OTLP exporter 发送到 tracing backend
```

异步函数和跨进程边界需要显式传播 context；否则 spans 会存在但无法组成一条完整请求链。

## 3. Profiling 用法

- Torch Profiler 捕获 CPU ops、CUDA kernels、shape、stack 和 memory 等信息。
- CUDA profiler wrapper 控制外部 Nsight 等工具的采集区间。
- Layerwise profile 将 operator events 重新组织为模型模块层次统计。
- Profile 会引入明显开销，应限制步数、先 warmup，并避免把采集结果当成正常服务延迟。

## 4. 关键代码解析

### `init_tracer()`

### `maybe_init_worker_tracer()`

### `instrument()`

### `instrument_manual()`

### `init_otel_tracer()`

### `extract_trace_context()`

### `WorkerProfiler.start()`

### `WorkerProfiler.step()`

### `WorkerProfiler.stop()`

### `layerwise_profile.__enter__()`

## 5. 工具选择

| 问题 | 首选手段 |
| --- | --- |
| 服务是否正在退化 | Metrics |
| 某请求时间花在哪个组件 | Distributed trace |
| 某个 forward 内哪个 op/kernel 最慢 | Profiler |
| 改动是否改善真实负载 | Benchmark |

## 6. 当前结论

Tracing 解释跨组件因果顺序，profiling 解释进程内部执行细节；两者采集粒度和运行开销不同，应由指标先缩小调查范围。
