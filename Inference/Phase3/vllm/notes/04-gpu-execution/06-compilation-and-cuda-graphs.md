# Compilation and CUDA Graphs

## 1. 文件定位

- 主要路径：`vllm/compilation/`、`vllm/v1/cudagraph_dispatcher.py`。
- 所属层次：GPU 执行优化层。
- 核心职责：编译模型计算图、运行图优化 pass，并为合适的 batch 捕获和重放 CUDA Graph。

## 2. 核心组件

| 组件 | 作用 |
| --- | --- |
| `support_torch_compile` | 把模型类接入 vLLM 编译包装器 |
| `VllmBackend` | 接收 Dynamo FX graph 并执行分区、缓存和编译 |
| Piecewise Backend | 在不兼容算子处分图，分别编译可捕获片段 |
| `CUDAGraphWrapper` | 捕获并重放静态 GPU 执行图 |
| `CudagraphDispatcher` | 根据 batch 特征选择 eager、piecewise 或 full graph |
| Compilation Passes | 完成 fusion、functionalization 和无效节点清理 |

## 3. 主执行流程

```text
模型类使用 support_torch_compile
→ Dynamo 捕获 forward graph
→ VllmBackend
→ 图切分与自定义 passes
→ Inductor 编译
→ 缓存编译产物
→ Model Runner 调用编译后的模型
```

```text
BatchDescriptor
→ CudagraphDispatcher.dispatch()
→ 选择 eager / PIECEWISE / FULL
→ CUDAGraphWrapper 捕获或命中已有 graph
→ replay
```

CUDA Graph 要求地址和执行形状稳定，因此 Model Runner 会准备固定输入 buffer，并把实际 batch padding 到可捕获的 graph size。

## 4. 输入与输出

### 输入

- 模型 forward、动态 shape 标记和 `CompilationConfig`。
- 当前 batch 的 token 数、请求数、LoRA 情况和 Attention backend 能力。

### 输出

- 编译后的 callable 和持久化编译缓存。
- 可供不同 batch descriptor 复用的 CUDA Graph。
- eager/graph 模式的运行时分发结果。

### 状态变化

- 初始化期间完成编译和 graph capture。
- Dispatcher 记录可用 graph key 和 padding 规则。
- 后续请求复用编译产物及已捕获 graph。

## 5. 关键代码解析

### `support_torch_compile()`

### `VllmBackend.__call__()`

### `CUDAGraphWrapper.__call__()`

### `CUDAGraphWrapper.clear_graphs()`

### `CudagraphDispatcher.initialize_cudagraph_keys()`

### `CudagraphDispatcher.dispatch()`

### `CudagraphDispatcher.get_capture_descs()`

## 6. 与其他文件的关系

- 配置：`CompilationConfig` 决定编译级别、分图算子和 CUDA Graph 模式。
- Model Runner：准备稳定输入 buffer，并在初始化时触发 warm-up/capture。
- Attention：后端是否支持 full graph 会限制最终模式。
- 分布式：collective fusion 与 sequence parallel pass 会修改跨 rank 计算图。

## 7. 当前结论

`torch.compile` 优化算子图，CUDA Graph 降低重复 kernel launch 和 CPU 调度开销；Dispatcher 根据 batch 与后端能力选择执行模式。
