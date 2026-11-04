# 07 · 创建调度器并完成初始化

这一篇在模型、KV 存储和预热都完成后返回 EngineCore，创建 Scheduler，选择核心执行方法，再逐层返回 LLM。到本篇结束，第二章需要复用的引擎环境已经准备好。

## 1. 回到 EngineCore 构造函数

源码定位：[v1/engine/core.py](../../vllm/vllm/v1/engine/core.py) · `EngineCore.__init__`。

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

`_initialize_kv_caches` 返回之前，上一篇的预热已经完成。此时先建立 `StructuredOutputManager`，再根据配置取得 Scheduler 类。

`StructuredOutputManager` 保存结构化输出约束所需的管理状态，随后作为依赖交给调度器。`SchedulerConfig.get_scheduler_cls()` 取得 Scheduler 类，下面继续创建实例。

随后解决调度块大小与哈希块大小，并创建实例：

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：__init__
# 源码范围：L156–L168（连续节选）
scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
    kv_cache_config, vllm_config
)

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

调度器取得实际测算后的缓存容量，后续据此判断请求需要的块能否安排。

## 2. Scheduler 初始化时建立哪些状态

源码定位：[v1/core/sched/scheduler.py](../../vllm/vllm/v1/core/sched/scheduler.py) · `Scheduler.__init__`。

```python
# 文件：vllm/v1/core/sched/scheduler.py
# 类：Scheduler
# 函数：__init__
# 源码范围：L177–L190（连续节选）
# req_id -> Request
self.requests: dict[str, Request] = {}
# Scheduling policy
try:
    self.policy = SchedulingPolicy(self.scheduler_config.policy)
except ValueError as e:
    raise ValueError(
        f"Unknown scheduling policy: {self.scheduler_config.policy}"
    ) from e
# Priority queues for requests.
self.waiting = create_request_queue(self.policy)
# requests skipped in waiting flow due async deps or constraints.
self.skipped_waiting = create_request_queue(self.policy)
self.running: list[Request] = []
```

| 状态 | 现在是什么 | 请求到来后怎样用 |
| --- | --- | --- |
| requests | 空字典 | 按 request_id 找到持续保存的核心 Request |
| waiting | 按策略建立的空队列 | 放入等待首次接纳或恢复的请求 |
| skipped_waiting | 按策略建立的空队列 | 保存暂时受依赖或其他条件限制的等待请求 |
| running | 空列表 | 保存已经接纳的运行请求 |

字典负责按 ID 查找，队列和列表负责组织执行顺序。未来一条 Request 可以同时被请求表和相应队列引用，它们使用的是同一个对象。

Scheduler 还从配置取出运行请求数上限、每轮 token 预算和最大长度等约束，准备完成 ID、抢占 ID、统计和可选扩展状态。

这里先建立保存状态的位置。第二章提交一批输入，第三章进入核心后，才会逐条建立并登记 Request；第四章开始用这些状态形成本轮执行计划。

## 3. 调度侧的 KV 管理对象此时建立

```python
# 文件：vllm/v1/core/sched/scheduler.py
# 类：Scheduler
# 函数：__init__
# 源码范围：L272–L290（连续节选）
# Create the KV cache manager.
if hash_block_size is None:
    hash_block_size = block_size
self.hash_block_size = hash_block_size
self.kv_cache_manager = KVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=self.max_model_len,
    max_in_flight_tokens=vllm_config.max_in_flight_tokens,
    enable_caching=self.cache_config.enable_prefix_caching,
    use_eagle=self.use_eagle,
    log_stats=self.log_stats,
    enable_kv_cache_events=self.enable_kv_cache_events,
    dcp_world_size=self.dcp_world_size,
    pcp_world_size=1,
    scheduler_block_size=self.block_size,
    hash_block_size=hash_block_size,
    metrics_collector=self.kv_metrics_collector,
    watermark=self.scheduler_config.watermark,
)
```

KVCacheManager 记录哪些块空闲、哪些块由请求使用、哪些内容可供复用。它管理第 05 篇已经建立的 GPU 存储。

| 初始化位置 | 建立的东西 |
| --- | --- |
| Model Runner 初始化 KV Cache | 实际存储与模型层对应关系 |
| Scheduler 创建 KVCacheManager | 块的管理记录、缓存复用与分配接口 |

它们共享对容量和块布局的认识。正式调度中，管理器返回块编号，执行侧再用这些编号访问已有存储。

## 4. 选定以后每轮调用哪个方法

EngineCore 接着准备批次队列：

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：__init__
# 源码范围：L206–L212（连续节选）
self.batch_queue_size = vllm_config.max_concurrent_batches
self.batch_queue: (
    deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
) = None
if self.batch_queue_size > 1:
    logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
    self.batch_queue = deque(maxlen=self.batch_queue_size)
```

`max_concurrent_batches` 的值来自 [config/vllm.py](../../vllm/vllm/config/vllm.py)。值为 1 时，`batch_queue` 保持 None；大于 1 时建立队列，保存尚未处理结果的执行批次。

一批在途工作指已经下发、结果尚未处理的一次执行，批次内包含这轮选中的多条请求。

接着保存执行方法与取消队列等状态：

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：__init__
# 源码范围：L220–L236（连续节选）
self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
    caching_hash_fn = get_hash_fn_by_name(
        vllm_config.cache_config.prefix_caching_hash_algo
    )
    init_none_hash(caching_hash_fn)

    self.request_block_hasher = get_request_block_hasher(
        hash_block_size, caching_hash_fn
    )

self.step_fn = (
    self.step if self.batch_queue is None else self.step_with_batch_queue
)
self.async_scheduling = vllm_config.scheduler_config.async_scheduling

self.aborts_queue = queue.Queue[list[str]]()
```

`self.step` 没有括号，表示保存方法，供以后调用。`batch_queue` 为 None 时选中 `step`，使用批次队列时选中 `step_with_batch_queue`。

`request_block_hasher` 保存计算请求块哈希的方法。哈希是按规则给一段 token 内容算出的标识，后续可用于查找相同前缀。

`aborts_queue` 用于保存在执行期间收到、等待核心处理的取消 ID。普通 step 如何使用它，后续跟请求执行时再展开。

## 5. 控制如何逐层回到用户入口

现在 EngineCore 已建立，构造调用开始返回：

```text
EngineCore.__init__ 返回
    → InprocClient 创建完成
    → EngineCoreClient.make_client 返回客户端
    → LLMEngine 完成日志、清理引用与缓存等收尾
    → LLMEngine.from_engine_args 返回 LLMEngine
    → LLM.__init__ 保存组件引用并完成入口侧准备
    → 用户的 llm = LLM(...) 返回
```

本地 LLMEngine 在同进程模式下还会暴露兼容用的 model_executor 引用，并登记一个在引擎销毁时清理模型相关状态的回调（finalizer）。开启统计时创建日志管理对象，构造末尾清理初始化过程中可能留下的多模态缓存。

最后回到 [entrypoints/llm.py](../../vllm/vllm/entrypoints/llm.py) 的 `LLM.__init__`：

```python
# 文件：vllm/entrypoints/llm.py
# 类：LLM
# 函数：__init__
# 源码范围：L339–L356（连续节选）
self.llm_engine = LLMEngine.from_engine_args(
    engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
)
self.model_config = self.llm_engine.model_config
self.engine_class = type(self.llm_engine)

self.request_counter = Counter()
self.default_sampling_params: dict[str, Any] | None = None

supported_tasks = self.llm_engine.get_supported_tasks()
self.supported_tasks = supported_tasks

self.runner_type = self.model_config.runner_type
self.renderer = self.llm_engine.renderer
self.chat_template = load_chat_template(chat_template)
self.input_processor = self.llm_engine.input_processor

self.renderer.warmup(ChatParams(chat_template=self.chat_template))
```

从这段代码看，可以分为三组状态：

- `model_config`、`supported_tasks` 等帮助入口知道这套引擎支持什么。
- `request_counter` 和默认采样参数缓存等，为后续离线调用准备入口状态。
- `renderer`、`input_processor` 等属性引用 LLMEngine 已有组件，两处名称指向同一个对象。

最后的 `renderer.warmup(...)` 准备输入处理侧可能需要的工作。它与第 06 篇的 GPU 模型预热作用不同。默认采样参数字段先是 None，后续需要时才补齐相应默认值。

在构造函数的最后，还会完成下面两项辅助状态准备：

```python
# 文件：vllm/entrypoints/llm.py
# 类：LLM
# 函数：__init__
# 源码范围：L373–L376（连续节选）
PoolingOfflineMixin.__init__(self)

# Cache for __repr__ to avoid repeated collective_rpc calls
self._cached_repr: str | None = None
```

PoolingOfflineMixin 提供池化任务等接口所复用的方法。初始化时保存支持的池化任务、输入输出处理配置和 renderer 的执行器引用，定位见 [entrypoints/pooling/offline.py](../../vllm/vllm/entrypoints/pooling/offline.py)。

`_cached_repr` 用来缓存引擎的字符串说明，避免每次显示对象时都重复查询核心。构造函数中前面的 renderer 线程池提示，也只是在说明相关选项对当前同步入口是否生效。

走到这里，LLM 构造函数已经结束，控制返回用户。

## 6. 构造返回时逐项核对

| 项目 | 此时应得到什么 |
| --- | --- |
| 运行配置 | 已确定的模型、设备、执行、调度与缓存设置；测算结果已写回相应配置 |
| 入口组件 | Renderer、InputProcessor、OutputProcessor，以及请求计数等入口状态 |
| 核心连接 | InprocClient 连接已经建立的 EngineCore |
| 模型执行侧 | UniProcExecutor、Wrapper、GPU Worker、V1 Model Runner |
| 实际模型 | 已创建结构、加载权重并完成相应加载后处理 |
| KV 存储与管理 | 执行侧张量已建立，调度侧管理器已建立 |
| 执行准备 | profiling、算子与采样预热完成 |
| 核心调度状态 | Scheduler 已创建，真实请求集合为空，step_fn 已选定 |

这些资源会被后续多次 generate 复用。每次新调用增加的是这次请求及其状态；例如新的 Prompt、生成参数、核心 Request 和入口 RequestState。

## 7. 接上第二章的一批请求

下一章从 `llm.generate(prompts)` 开始。我们已经知道它将复用哪套 tokenizer、输入处理器、核心、模型、缓存和输出处理器，可以直接跟踪 A、B、C 等请求怎样进入这些已建立的对象。

第二章从批量输入对齐开始，接着逐条提交；第三章进入核心登记请求，第四章按轮次选择请求执行。

阅读检查：

- 为什么 Scheduler 在实际缓存容量确定后才创建？
- GPU KV 张量与调度侧 KVCacheManager 各自在哪里创建？
- batch_queue 如何影响 step_fn 的选择？
- renderer.warmup 与 Worker.compile_or_warm_up_model 分别准备什么？
- LLM 构造返回后，哪些资源可长期复用，哪些状态仍要等每条请求提交时建立？

[章节目录](README.md) · [上一篇：编译、预热与执行准备](06-Compilation-and-Warmup.md) · [下一章：提交一批离线请求](../02-Offline-Request-Submission/README.md)
