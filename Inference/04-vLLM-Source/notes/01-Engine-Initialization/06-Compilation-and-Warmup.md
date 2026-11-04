# 06 · 编译、预热与执行准备

KV 存储建立后，Worker 继续准备执行所需的算子和采样缓冲，再把控制交回 EngineCore。

## 1. 模型和缓存已经准备好，为什么还要运行一次

模型加载好，只表示参数已经在相应对象和设备上；有些执行工作在第一次调用时才会发生。例如，某个算子第一次使用时可能要编译、选择实现，或申请额外缓冲。

底层计算函数在第一次使用时才进行编译，通常称为即时编译（JIT）。后面提到的 JIT 监测，就是检查运行中是否又出现了这种编译工作。

将这些工作尽量放到初始化阶段，可以减少第一批真实请求碰到的额外等待。预热使用引擎准备的模拟输入。

先分清下面几个词：

| 名称 | 具体意思 | 对应调用 |
| --- | --- | --- |
| Profiling | 用模拟运行测量资源使用等开销 | 上一篇的显存测算；也可能触发运行准备 |
| 编译 | 将相应计算准备成可执行且可能更快的实现 | 根据 CompilationConfig 决定是否启用 |
| Kernel 预热 | 提前调用底层计算函数，做需要的准备或调优 | kernel_warmup 等 |
| CUDA Graph 捕获 | 记录适合重复运行的一组 GPU 操作，后续按记录再次执行 | capture_model 等，受模式和模型配置限制 |
| 采样预热 | 提前准备 logits 与采样所需的缓冲和执行过程；logits 是模型给各候选 token 的分数 | _dummy_run、_dummy_sampler_run |

这几种工作可能交叉发生，但作用并不相同。尤其是 profiling 用于测量，预热主要用于准备后续执行；同一次模拟运行有时会同时发挥两种作用。

## 2. 从核心通知 Worker 做执行准备

源码定位：[v1/engine/core.py](../../vllm/vllm/v1/engine/core.py) · `EngineCore._initialize_kv_caches`。

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：_initialize_kv_caches
# 源码范围：L328–L332（连续节选）
vllm_config.validate_block_size()

self.model_executor.initialize_from_config(kv_cache_configs)
if not envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
    self.model_executor.compile_or_warm_up_model()
```

`_initialize_kv_caches` 的后半段继续进行执行准备：先建立 KV 缓存，再通知 Worker 编译或预热。

Executor 将调用继续交给 Worker：

```python
# 文件：vllm/v1/executor/abstract.py
# 类：Executor
# 函数：compile_or_warm_up_model
# 源码范围：L122–L126（连续节选）
def compile_or_warm_up_model(self) -> None:
    """Compile/warm up the model and capture cudagraphs on workers."""
    compilation_times: list[CompilationTimes] = self.collective_rpc(
        "compile_or_warm_up_model"
    )
```

执行完后，Executor 汇总 Worker 返回的编译耗时，并写回配置。

## 3. enforce_eager 怎样影响执行准备

源码定位：[config/vllm.py](../../vllm/vllm/config/vllm.py) · `VllmConfig.__post_init__`。

```python
# 文件：vllm/config/vllm.py
# 类：VllmConfig
# 函数：__post_init__
# 源码范围：L1193–L1199（连续节选）
if self.model_config is not None and self.model_config.enforce_eager:
    logger.warning_once(
        "Enforce eager set, disabling torch.compile and CUDAGraphs. "
        "This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none"
    )
    self.compilation_config.mode = CompilationMode.NONE
    self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
```

`enforce_eager=True` 时，VllmConfig 将编译模式和 CUDA Graph 模式设为 NONE。Worker 随后执行算子和采样预热，顺序如下：

```text
Worker.compile_or_warm_up_model
    → 编译尺寸的预热循环：本章模式下不进入
    → kernel_warmup：进入调用，内部按算子与配置决定具体工作
    → CUDA Graph 捕获：本章跳过
    → V1 生成模型的模拟前向与采样预热
    → 重设随机种子，完成运行检查等准备
    → 返回初始化耗时
```

## 4. 编译尺寸、算子预热与图捕获的顺序

源码定位：[v1/worker/gpu_worker.py](../../vllm/vllm/v1/worker/gpu_worker.py) · `Worker.compile_or_warm_up_model`。

```python
# 文件：vllm/v1/worker/gpu_worker.py
# 类：Worker
# 函数：compile_or_warm_up_model
# 源码范围：L706–L717（连续节选）
for size in sorted(warmup_sizes, reverse=True):
    logger.info("Compile and warming up model for size %d", size)
    self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

# Warmup and tune the kernels used during model execution before
# cuda graph capture.
kernel_warmup(self)

cuda_graph_memory_bytes = 0
if not self.model_config.enforce_eager:
    cuda_graph_memory_bytes = self.model_runner.capture_model()
```

启用 VLLM_COMPILE 时，前面的代码收集 `warmup_sizes`：把用户希望编译、又没有被图捕获尺寸覆盖的规模补进来，并检查编译范围是否得到覆盖。随后按尺寸运行模拟输入。

`size` 指这次模拟执行的 token 规模。一条长 Prompt 就可以占用多个 token 位置。

接着 `kernel_warmup` 准备底层计算函数。`capture_model` 根据 CUDA Graph 配置记录对应规模的 GPU 操作；图捕获占用的内存也会影响上一篇计算的 KV 预算。

## 5. 采样为什么也要单独预热

Worker 调用 Model Runner 做一次模拟前向，再准备采样：

```python
# 文件：vllm/v1/worker/gpu_worker.py
# 类：Worker
# 函数：compile_or_warm_up_model
# 源码范围：L802–L816（连续节选）
max_num_reqs = min(
    self.scheduler_config.max_num_seqs,
    self.scheduler_config.max_num_batched_tokens,
)

# We skip EPLB here since we don't want to record dummy metrics
hidden_states, last_hidden_states = self.model_runner._dummy_run(
    num_tokens=max_num_reqs,
    skip_eplb=True,
    cudagraph_runtime_mode=CUDAGraphMode.NONE,
)
if self.model_runner.is_pooling_model:
    self.model_runner._dummy_pooler_run(hidden_states)
else:
    self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)
```

`max_num_reqs` 由运行请求数和 token 预算共同限制。模拟前向取得模型计算出的中间向量（hidden states），再进入 `_dummy_sampler_run`。

这一步准备 logits 和采样等需要的最大形状缓冲，减少后续反复申请内存的开销。

采样预热安排在图捕获步骤之后，这样图捕获过程中的缓存清理就不会影响刚准备好的采样缓冲。

最后重设随机种子，让正式请求从设定的随机起点开始生成。

## 6. 执行准备怎样收尾

方法末尾还完成后续运行需要的检查与监测准备：

```python
# 文件：vllm/v1/worker/gpu_worker.py
# 类：Worker
# 函数：compile_or_warm_up_model
# 源码范围：L832–L848（连续节选）
# All warmup is done — start monitoring for unexpected JIT
# compilations that would cause latency spikes during inference.
from vllm.utils.jit_monitor import activate as activate_jit_monitor

activate_jit_monitor(
    mode=self.observability_config.jit_monitor_mode,
    verbose=self.observability_config.jit_monitor_verbose,
)

# Freeze the worker heap so the GC won't scan static objects
# (model weights, KV caches, CUDA graphs) during inference.
freeze_gc_heap()
maybe_attach_gc_debug_callback()

# Warmup / first-compile is done — activate the `VLLM_GPU_SYNC_CHECK`
# gate so subsequent `execute_model` / `sample_tokens` calls enforce it.
enable_gpu_sync_check()
```

这里涉及三个容易混在一起的点：

- JIT 监测关注运行中意外发生的新编译，可用于定位延迟突然增加。
- `freeze_gc_heap` 让 Python 的垃圾回收减少反复扫描长期存在的对象。
- GPU 同步检查的启用位置，用于后续执行阶段的相应检查；实际行为由配置和检查开关决定。

启用编译后端时，方法还会提前触发相应的惰性初始化，把首次使用所需的准备工作放在启动阶段。

返回后，`EngineCore._initialize_kv_caches()` 得到可以供 Scheduler 使用的缓存配置，并将控制交回 EngineCore 构造函数。下一篇继续创建 Scheduler。

## 7. 用三个阶段检查理解

| 时刻 | 输入来自哪里 | 运行目的 |
| --- | --- | --- |
| 模型与权重刚加载完 | 模型文件 | 建立模型及参数 |
| 显存 profiling、编译或预热 | 引擎准备的模拟输入 | 测量资源开销，准备执行环境 |
| generate 提交后 | 用户的一批 prompts | 按调度计划生成结果 |

阅读检查：

- 显存 profiling、编译、算子预热和图捕获分别解决什么问题？
- enforce_eager=True 后，为什么还会走到 compile_or_warm_up_model？
- 为什么缓存初始化必须在适用的正式图捕获和执行准备之前完成？
- 为什么采样也需要预热，为什么最后重设随机种子？

[章节目录](README.md) · [上一篇：确定 KV Cache 容量并初始化缓存](05-KV-Cache-Capacity-and-Initialization.md) · [下一篇：创建调度器并完成初始化](07-Scheduler-and-Initialization-Completion.md)
