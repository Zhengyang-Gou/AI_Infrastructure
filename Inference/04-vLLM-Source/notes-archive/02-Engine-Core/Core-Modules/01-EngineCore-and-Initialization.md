# 01 · EngineCore 与初始化

**这一篇看核心执行之前，需要把哪些组件和资源准备好。** 先创建执行器和缓存资源，再建立能够使用这些资源的调度器，最后选好执行一轮时调用的方法。

## 1. 职责与边界

`EngineCore` 接收 `VllmConfig` 和已经选定的 Executor 类，创建长期复用的核心组件。一条请求进入后，可以使用这套组件执行很多轮；不同生成调用也能复用同一个核心。

```text
EngineCore.__init__
    → 创建 model_executor
    → 准备 KV Cache 配置并初始化执行侧缓存
    → 创建 structured_output_manager 与 scheduler
    → 准备 batch_queue 等状态
    → 选择 step_fn
```

本篇读清创建顺序和组件关系。缓存容量的计算方法见 [KV Cache Management](../../04-KV-Cache-Management/)，Worker 如何加载权重、准备 GPU 执行见 [Model Execution](../../05-Model-Execution/) 和 [Model and Weight Loading](../../06-Model-and-Weight-Loading/)。

## 2. 源码定位与关键对象

源码路径以 `04-vLLM-Source/vllm/` 为起点，代码块行号对应本地快照。

| 文件 | 类 | 关注内容 |
| --- | --- | --- |
| [vllm/v1/engine/core.py](../../../vllm/vllm/v1/engine/core.py) | EngineCore | __init__、_initialize_kv_caches |
| [vllm/v1/executor/abstract.py](../../../vllm/vllm/v1/executor/abstract.py) | Executor | 执行器初始化、缓存配置与执行接口 |

| 属性或数据 | 在核心里做什么 |
| --- | --- |
| model_executor | 联系 Worker，组织模型执行并取得结果 |
| kv_cache_config | 提供给 Scheduler 的缓存布局与容量配置 |
| scheduler | 保存请求，组织本轮计划，更新执行结果 |
| structured_output_manager | 为结构化输出请求管理约束，普通文本主线先认识其位置 |
| batch_queue | 保存已下发批次的 Future 和调度计划，允许多批次在途 |
| step_fn | 指向本次配置选中的执行方法 |

`KVCacheConfig` 描述资源布局；实际的 GPU KV 张量由执行侧准备，调度侧的 `KVCacheManager` 则管理块的使用。它们通过配置和块信息协作。

## 3. 实现分析

### 3.1 创建执行器，进入执行资源的初始化

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

`executor_class` 是传进来的类，调用它就会执行构造过程。返回的实例保存在 `self.model_executor`，供缓存准备和每轮执行复用。

Executor 的构造会继续调用具体实现的 `_init_executor()`，由实现组织 Worker 和模型加载。这也是为什么创建核心需要实际准备执行资源，不能只把这一行理解为保存配置。

### 3.2 先确定缓存资源，再创建调度器

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：__init__
# 源码范围：L142–L147（连续节选）
# Setup KV Caches and update CacheConfig after profiling.
kv_cache_config = self._initialize_kv_caches(vllm_config)
self.structured_output_manager = StructuredOutputManager(vllm_config)

# Setup scheduler.
Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
```

`_initialize_kv_caches()` 的主要路线是：

```text
向 Executor 取得模型的 KV Cache 需求
    → 确定可用于缓存的显存
    → 生成各 Worker 的缓存配置
    → 汇总调度侧的缓存配置
    → 初始化执行侧缓存，并编译或预热模型
    → 返回给 Scheduler 使用的 KVCacheConfig
```

执行侧初始化与预热的调用位置如下：

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

缓存准备会根据结果更新配置中的块数等信息。随后创建调度器：

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：__init__
# 源码范围：L160–L168（连续节选）
self.scheduler: SchedulerInterface = Scheduler(
    vllm_config=vllm_config,
    kv_cache_config=kv_cache_config,
    structured_output_manager=self.structured_output_manager,
    include_finished_set=include_finished_set,
    log_stats=self.log_stats,
    block_size=scheduler_block_size,
    hash_block_size=hash_block_size,
)
```

这里的局部名称 `Scheduler` 来自配置选择，不一定固定为某一个调度器实现。核心把缓存配置等依赖交给它，让它能够在资源约束下安排请求。

### 3.3 选择每轮实际调用的方法

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：__init__
# 源码范围：L231–L236（连续节选）
self.step_fn = (
    self.step if self.batch_queue is None else self.step_with_batch_queue
)
self.async_scheduling = vllm_config.scheduler_config.async_scheduling

self.aborts_queue = queue.Queue[list[str]]()
```

`self.step` 在这里没有括号，表示保存方法引用；调用 `self.step_fn()` 时才执行选中的方法。

| 条件 | step_fn 指向 | 作用 |
| --- | --- | --- |
| 没有 batch_queue | step | 顺序组织一轮调度、执行、结果更新 |
| 存在 batch_queue | step_with_batch_queue | 将计划与 Future 配对保存，允许多批次在途 |

在这份源码中，`max_concurrent_batches > 1` 时创建 batch queue。这个选择与客户端是否跨进程通信是两个维度；同进程和多进程路径都会调用 `step_fn()`。

`aborts_queue` 保存等待核心处理的取消 ID。它用于衔接消息接收与执行结果更新，见 [完成、取消与清理](05-Completion-and-Abort.md)。

## 4. 用一次初始化串起对象关系

假设创建一个普通文本模型的核心：

```text
创建 Executor，组织 Worker 与模型加载
    ↓
根据模型需求和显存情况准备 KV Cache
    ↓
让 Worker 初始化缓存，完成编译或预热
    ↓
Scheduler 得到缓存配置，建立请求管理与调度能力
    ↓
选好 step_fn，核心可以接收请求并执行
```

请求 A 到来后使用这些资源；A 结束后请求 B 也能继续复用。请求各自的状态在接收时创建，核心组件在初始化时准备。

## 5. 阅读检查

- 为什么要先准备缓存资源，再创建 Scheduler？
- `self.step_fn = self.step` 与 `self.step_fn()` 分别做什么？
- Executor、缓存配置与 KVCacheManager 分别承担什么职责？

[章节目录](../README.md) · [整体框架](../Overview.md) · [下一篇：Request 与请求接收](02-Request-and-Admission.md)
