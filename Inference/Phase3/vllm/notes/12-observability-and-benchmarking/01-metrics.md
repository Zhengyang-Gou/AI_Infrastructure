# Metrics

## 1. 文件定位

- 路径：`vllm/v1/metrics/stats.py`、`loggers.py`、`prometheus.py`。
- 职责：把 Scheduler、Engine 和请求生命周期事件聚合为日志与 Prometheus 指标。

## 2. 指标层次

- Counter：累计 prompt/generated tokens、finished requests、preemptions 等事件。
- Gauge：当前 running/waiting requests、KV cache usage、sleep state 等瞬时值。
- Histogram：TTFT、TPOT、ITL、E2EL、queue time、prefill time 等分布。
- Cache metrics：prefix cache、多模态 cache 和 KV transfer 的命中与活动。

请求延迟应看分位数和分布，平均值容易掩盖排队尖峰和长尾。

## 3. 统计链路

```text
EngineCoreOutput + Request events
→ IterationStats.update_from_output()
→ update_from_events()
→ finished request 汇总
→ StatLoggerManager.record()
→ LoggingStatLogger / PrometheusStatLogger
→ 定期 log 或 /metrics scrape
```

多进程 Prometheus 模式需要独立 registry 和子进程文件清理，否则旧进程时间序列可能污染当前结果。

## 4. 关键代码解析

### `IterationStats.update_from_output()`

### `IterationStats.update_from_events()`

### `IterationStats.update_from_finished_request()`

### `PrometheusStatLogger.record()`

### `PrometheusStatLogger.log_engine_initialized()`

### `StatLoggerManager.record()`

### `StatLoggerManager.log()`

### `setup_multiprocess_prometheus()`

### `get_prometheus_registry()`

## 5. 分析建议

- 先用 waiting/running、queue time 判断是否饱和。
- 再用 TTFT 区分排队和 prefill 压力，用 TPOT/ITL 观察 decode 稳定性。
- 同时记录 token 长度分布与请求速率，避免比较不同工作负载。
- 指标异常时用 request ID 或时间窗口关联 trace。

## 6. 当前结论

Metrics 提供低成本、持续性的系统视角；它适合发现回归和容量边界，但通常不足以单独解释某个请求为何变慢。
