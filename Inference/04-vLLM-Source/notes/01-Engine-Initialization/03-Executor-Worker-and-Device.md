# 03 · 启动 Executor、Worker 与执行环境

这一篇进入 EngineCore 的第一项主要工作：创建 Executor，由它建立 Worker、准备 GPU，再创建 Model Runner。结尾进入模型加载，下一篇跟着权重继续走。

## 1. 从核心创建执行器

源码定位：[v1/engine/core.py](../../vllm/vllm/v1/engine/core.py) · `EngineCore.__init__`。

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：__init__
# 源码范围：L131–L135（连续节选）
# Setup Model.
self.model_executor = executor_class(vllm_config)
self._pooler_config_logged = False
if executor_fail_callback is not None:
    self.model_executor.register_failure_callback(executor_fail_callback)
```

第一篇已经选定 `executor_class=UniProcExecutor`。现在加上括号并传入配置，开始建立具体执行器。Executor 的公共构造函数先保存各项配置，再调用具体实现：

```python
# 文件：vllm/v1/executor/abstract.py
# 类：Executor
# 函数：__init__
# 源码范围：L99–L111（连续节选）
self.vllm_config = vllm_config
self.model_config = vllm_config.model_config
self.cache_config = vllm_config.cache_config
self.lora_config = vllm_config.lora_config
self.load_config = vllm_config.load_config
self.parallel_config = vllm_config.parallel_config
self.scheduler_config = vllm_config.scheduler_config
self.device_config = vllm_config.device_config
self.speculative_config = vllm_config.speculative_config
self.observability_config = vllm_config.observability_config
self._init_executor()
self.is_sleeping = False
self.sleeping_tags: set[str] = set()
```

`self` 是正在建立的 UniProcExecutor，所以 `self._init_executor()` 会进入 UniProcExecutor 自己定义的方法。这也是为什么公共基类只写一处初始化入口，就能让不同执行器建立各自的 Worker。

## 2. Executor、Wrapper、Worker、Runner 分别做什么

| 对象 | 这里的工作 | 后续执行时的工作 |
| --- | --- | --- |
| Executor | 组织 Worker 的建立与初始化调用 | 把核心的执行任务交给 Worker |
| WorkerWrapperBase | 创建实际 Worker，补充配置上下文与通用转发方法 | 统一承接 Executor 发来的方法调用 |
| GPU Worker | 选择设备、建立运行环境，管理模型与缓存初始化 | 管理 GPU 上的执行资源和执行入口 |
| GPUModelRunner | 建立输入缓冲等执行状态，随后持有模型与 KV 引用 | 将请求计划变成模型输入，运行前向与采样 |

Wrapper 是实际 Worker 外面的一层通用处理。`driver_worker` 保存 WorkerWrapperBase，实际 GPU Worker 保存在它的 `worker` 属性中。

## 3. UniProcExecutor 怎样建立一个 Worker

源码定位：[v1/executor/uniproc_executor.py](../../vllm/vllm/v1/executor/uniproc_executor.py) · `UniProcExecutor._init_executor`。

```python
# 文件：vllm/v1/executor/uniproc_executor.py
# 类：UniProcExecutor
# 函数：_init_executor
# 源码范围：L48–L69（连续节选）
self.driver_worker = WorkerWrapperBase(rpc_rank=0)
distributed_init_method, rank, local_rank = self._distributed_args()
kwargs = dict(
    vllm_config=self.vllm_config,
    local_rank=local_rank,
    rank=rank,
    distributed_init_method=distributed_init_method,
    is_driver_worker=True,
    shared_worker_lock=Lock(),
)

# Set net device env vars for the worker if VLLM_GPU_NIC_PCIE_MAPPING is set
set_worker_net_device(local_rank, self.vllm_config)

self.driver_worker.init_worker(all_kwargs=[kwargs])
self.driver_worker.init_device()

if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
    self.driver_worker.elastic_ep_execute("load_model")
else:
    self.driver_worker.load_model()
current_platform.update_block_size_for_backend(self.vllm_config)
```

初始化顺序是：

1. 建立一个 `WorkerWrapperBase`，`rpc_rank=0`。
2. 准备配置、设备编号、rank、初始化地址等参数。
3. `init_worker` 创建实际 Worker。
4. `init_device` 选择 GPU 并创建 Model Runner。
5. `load_model` 加载模型，第 04 篇继续。
6. 根据选中的执行后端更新相应块大小设置。

`rank` 表示执行成员编号，`local_rank` 用于定位本机设备。执行器把这些编号与通信初始化地址一起交给 Worker。

## 4. 从 Worker 类名到 Worker 实例

源码定位：[v1/worker/worker_base.py](../../vllm/vllm/v1/worker/worker_base.py) · `WorkerWrapperBase.init_worker`。

CUDA 平台在配置检查中会把自动 Worker 选择补成 `vllm.v1.worker.gpu_worker.Worker`，可定位到 [platforms/cuda.py](../../vllm/vllm/platforms/cuda.py) 的 `check_and_update_config`。

Wrapper 根据配置中的类路径找到实际的类：

```python
# 文件：vllm/v1/worker/worker_base.py
# 类：WorkerWrapperBase
# 函数：init_worker
# 源码范围：L249–L259（连续节选）
parallel_config = vllm_config.parallel_config
if isinstance(parallel_config.worker_cls, str):
    worker_class: type[WorkerBase] = resolve_obj_by_qualname(
        parallel_config.worker_cls
    )
else:
    raise ValueError(
        "passing worker_cls is no longer supported. "
        "Please pass keep the class in a separate module "
        "and pass the qualified name of the class as a string."
    )
```

随后在当前配置环境中创建它：

```python
# 文件：vllm/v1/worker/worker_base.py
# 类：WorkerWrapperBase
# 函数：init_worker
# 源码范围：L317–L319（连续节选）
with set_current_vllm_config(self.vllm_config):
    # To make vLLM config available during worker initialization
    self.worker = worker_class(**kwargs)
```

`set_current_vllm_config(...)` 让被调用代码能够取得这套运行配置，退出 `with` 后恢复原来的配置环境。

WorkerWrapperBase 的 `__getattr__` 还会把自己没有提供的方法继续交给实际 Worker，所以 `driver_worker.load_model()` 最终能调用到 GPU Worker 的 `load_model()`。

## 5. init_device 具体准备什么

源码定位：[v1/worker/gpu_worker.py](../../vllm/vllm/v1/worker/gpu_worker.py) · `Worker.init_device`。

CUDA 路线先把逻辑设备编号对应到可见 GPU，选中设备并检查模型精度是否受到支持，然后进行环境初始化和显存记录：

```python
# 文件：vllm/v1/worker/gpu_worker.py
# 类：Worker
# 函数：init_device
# 源码范围：L364–L400（连续节选）
visible_device_index = (
    current_platform.logical_device_id_to_visible_device_id(self.local_rank)
)
self.device = torch.device(f"cuda:{visible_device_index}")
torch.accelerator.set_device_index(self.device)

current_platform.check_if_supports_dtype(self.model_config.dtype)

# Initialize the distributed environment BEFORE taking
# memory snapshot
# This ensures NCCL buffers are allocated before we measure
# available memory
init_worker_distributed_environment(
    self.vllm_config,
    self.rank,
    self.distributed_init_method,
    self.local_rank,
    current_platform.dist_backend,
)

if self.use_v2_model_runner:
    logger.info_once("Using V2 Model Runner")

# Set random seed.
set_random_seed(self.model_config.seed)

# Now take memory snapshot after NCCL is initialized
gc.collect()
torch.accelerator.empty_cache()

# take current memory snapshot
self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
self.requested_memory = request_memory(init_snapshot, self.cache_config)
logger.debug("worker init memory snapshot: %r", self.init_snapshot)
logger.debug(
    "worker requested memory: %sGiB", format_gib(self.requested_memory)
)
```

按因果关系理解这段代码：

- 先选 GPU，后续张量才能放到预期设备。
- 先初始化通信环境，再记录显存起点，才能把通信缓冲等已经占用的资源考虑进去。
- 设置随机种子，让后续需要随机数的操作有明确起点。
- 记录 `init_snapshot`，保存当时的总显存和空闲显存等信息。
- 计算 `requested_memory`，得到引擎计划使用的显存预算；它还不是 KV Cache 的最终大小。

方法接着准备执行工作区，再创建 Model Runner：

```python
# 文件：vllm/v1/worker/gpu_worker.py
# 类：Worker
# 函数：init_device
# 源码范围：L419–L423（连续节选）
from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner as GPUModelRunnerV1,
)

self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)
```

`GPUModelRunner` 构造时会保存配置、设备、输入与采样相关缓冲、请求管理等执行侧状态。它的 `model` 要在随后的 `load_model()` 中取得，持久 KV 张量要等第 05 篇确定容量后才建立。

因此，创建 Model Runner、加载模型、分配 KV Cache 是依次发生的几项工作。

## 6. collective_rpc 怎样调用 Worker

Executor 的公共方法通过 `collective_rpc("方法名")` 调用 Worker。UniProcExecutor 在内部执行 `run_method(self.driver_worker, ...)`，直接调用 Wrapper 上的相应方法。

取得显存或 KV 需求时，结果放进一个只有一个元素的列表。这样 EngineCore 可以始终按“每个 Worker 一份信息”的形式处理返回值。

例如：

```text
Executor.get_kv_cache_specs()
    → collective_rpc("get_kv_cache_spec")
    → 本地 Wrapper / Worker / Model Runner
    → 返回 [这一个 Worker 的 KV 需求字典]
```

## 7. 此刻的状态与后续交接

GPU 已选定，Worker 与 Model Runner 已创建，显存测算有了启动时的记录。`UniProcExecutor._init_executor()` 继续调用 `load_model()`，只有模型加载完成，这个执行器的初始化才会返回核心。

检查失败时，可以按调用位置定位：`init_device` 检查设备、精度与启动显存，`load_model` 负责模型文件和权重加载。

阅读检查：

- driver_worker、实际 Worker 和 Model Runner 分别保存在哪里？
- 为什么记录启动显存之前先初始化通信环境？
- collective_rpc 如何把方法调用交给实际 Worker？
- init_device 返回时，模型权重和持久 KV Cache 是否都已经准备好？

[章节目录](README.md) · [上一篇：建立入口与核心之间的连接](02-Entry-and-Core-Connection.md) · [下一篇：创建模型与加载权重](04-Model-Creation-and-Weight-Loading.md)
