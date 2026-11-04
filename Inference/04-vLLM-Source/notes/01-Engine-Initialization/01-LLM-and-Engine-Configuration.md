# 01 · 从 LLM 构造到运行配置

创建 `LLM` 时，入口先整理参数，生成统一运行配置，再选定执行器类型。这些配置随后交给 LLMEngine，继续创建执行环境。

## 1. 从构造函数进入配置生成

`LLM.__init__` 通过引擎工厂生成配置，再进入 LLMEngine 的构造函数：

```text
LLM.__init__
    → 整理用户传入的配置
    → 创建 EngineArgs
    → LLMEngine.from_engine_args
        → EngineArgs.create_engine_config → VllmConfig
        → Executor.get_class → UniProcExecutor 这个类
        → 构造 LLMEngine（从下一篇继续）
```

## 2. 三种配置表示各负责什么

| 表示 | 用普通话理解 | 保存的典型内容 |
| --- | --- | --- |
| LLM 的构造参数 | 用户这次希望建立怎样的引擎 | 模型路径、精度、并行数量、显存使用比例 |
| EngineArgs | 先把入口收到的参数收在一起 | 用户选项与需要继续确定的默认值 |
| VllmConfig | 各组件共同使用的一套运行配置 | model_config、cache_config、parallel_config、scheduler_config 等 |

`EngineArgs` 把入口选项转成各组件可用的配置。`VllmConfig` 中的一部分数值会在初始化中继续补齐，例如缓存块数需要等显存测算后确定。

## 3. 从 LLM 参数进入引擎工厂

源码定位：[entrypoints/llm.py](../../vllm/vllm/entrypoints/llm.py) · `LLM.__init__`。

构造函数先把几种输入形式整理成相应配置对象。例如，字典形式的 Attention 配置会变成 `AttentionConfig`，整数形式的编译模式会变成 `CompilationConfig`。这样下游组件就能读取同一种配置表示。

随后创建 `EngineArgs`。下面截取其中一段，其他参数沿同一个构造调用继续传入：

```python
# 文件：vllm/entrypoints/llm.py
# 类：LLM
# 函数：__init__
# 源码范围：L295–L335（连续节选）
engine_args = EngineArgs(
    model=model,
    runner=runner,
    convert=convert,
    tokenizer=tokenizer,
    tokenizer_mode=tokenizer_mode,
    skip_tokenizer_init=skip_tokenizer_init,
    trust_remote_code=trust_remote_code,
    allowed_local_media_path=allowed_local_media_path,
    allowed_media_domains=allowed_media_domains,
    tensor_parallel_size=tensor_parallel_size,
    dtype=dtype,
    quantization=quantization,
    revision=revision,
    tokenizer_revision=tokenizer_revision,
    seed=seed,
    gpu_memory_utilization=gpu_memory_utilization,
    kv_cache_memory_bytes=kv_cache_memory_bytes,
    cpu_offload_gb=cpu_offload_gb,
    offload_group_size=offload_group_size,
    offload_num_in_group=offload_num_in_group,
    offload_prefetch_step=offload_prefetch_step,
    offload_params=offload_params or set(),
    enforce_eager=enforce_eager,
    enable_return_routed_experts=enable_return_routed_experts,
    disable_custom_all_reduce=disable_custom_all_reduce,
    hf_token=hf_token,
    hf_overrides=hf_overrides,
    mm_processor_kwargs=mm_processor_kwargs,
    pooler_config=pooler_config,
    structured_outputs_config=structured_outputs_instance,
    profiler_config=profiler_config_instance,
    attention_config=attention_config_instance,
    compilation_config=compilation_config_instance,
    quantization_config=quantization_config,
    logits_processors=logits_processors,
    spec_method=spec_method,
    spec_model=spec_model,
    spec_tokens=spec_tokens,
    **kwargs,
)
```

`gpu_memory_utilization` 和 `kv_cache_memory_bytes` 随参数进入缓存配置，后续用于计算 KV 预算。第 05 篇继续追踪这两个值怎样影响显存分配。

接着，`LLM.__init__` 调用 `LLMEngine.from_engine_args(...)`。工厂中的关键顺序是：

```python
# 文件：vllm/v1/engine/llm_engine.py
# 类：LLMEngine
# 函数：from_engine_args
# 源码范围：L171–L186（连续节选）
vllm_config = engine_args.create_engine_config(usage_context)
executor_class = Executor.get_class(vllm_config)

if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
    logger.debug("Enabling multiprocessing for LLMEngine.")
    enable_multiprocessing = True

# Create the LLMEngine.
return cls(
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=not engine_args.disable_log_stats,
    usage_context=usage_context,
    stat_loggers=stat_loggers,
    multiprocess_mode=enable_multiprocessing,
)
```

这段代码做了三件事：

1. `create_engine_config` 生成统一配置，并检查选项之间是否能一起使用。
2. `Executor.get_class` 根据已经确定的执行后端选中执行器的类。
3. `return cls(...)` 创建 LLMEngine，继续进入下一层初始化。

此处选中的是一个**类**。真正执行 `executor_class(vllm_config)`、建立执行器实例，要等进入 EngineCore。

## 4. create_engine_config 怎样把配置串起来

源码定位：[engine/arg_utils.py](../../vllm/vllm/engine/arg_utils.py) · `EngineArgs.create_engine_config`；配置对象定义在 [config/](../../vllm/vllm/config/)。

这个方法按下面的顺序建立配置：

| 步骤 | 做什么 | 为什么在这里做 |
| --- | --- | --- |
| 确认平台与设备 | 建立 DeviceConfig，检查环境选项 | 后面要知道使用 CUDA 等哪一种执行环境 |
| 读取模型信息 | create_model_config 读取模型配置，确定结构、精度、长度与任务类型等 | 许多运行选项需要先知道模型长什么样 |
| 生成缓存配置 | 保存缓存精度、块大小相关选项、显存预算与前缀缓存设置 | 为后面的缓存测算和初始化提供依据 |
| 生成并行配置 | 保存 TP、PP 等数量及执行后端 | 决定使用哪种 Executor、Worker 怎样组织 |
| 生成调度配置 | 保存每轮 token 预算、运行请求数、分块与调度策略 | 决定后续请求可以怎样共享执行资源 |
| 生成加载、编译及其他配置 | 汇集加载格式、编译、观测等选项 | 给各个初始化步骤提供完整参数 |
| 构造 VllmConfig | 将各项配置组合起来，再做共同检查和必要调整 | 确保最终运行方式彼此兼容 |

读取模型的 `config.json` 等描述文件，取得层数、隐藏维度等信息；第 04 篇再按这些结构信息创建模型并填入权重。

调度配置在请求到来前就会建立：

```python
# 文件：vllm/engine/arg_utils.py
# 类：EngineArgs
# 函数：create_engine_config
# 源码范围：L2266–L2285（连续节选）
scheduler_config = SchedulerConfig(
    runner_type=model_config.runner_type,
    max_num_batched_tokens=self.max_num_batched_tokens,
    max_num_scheduled_tokens=self.max_num_scheduled_tokens,
    max_num_seqs=self.max_num_seqs,
    max_model_len=model_config.max_model_len,
    enable_chunked_prefill=self.enable_chunked_prefill,
    disable_chunked_mm_input=self.disable_chunked_mm_input,
    is_multimodal_model=model_config.is_multimodal_model,
    is_encoder_decoder=model_config.is_encoder_decoder,
    policy=self.scheduling_policy,
    scheduler_cls=self.scheduler_cls,
    long_prefill_token_threshold=self.long_prefill_token_threshold,
    scheduler_reserve_full_isl=self.scheduler_reserve_full_isl,
    watermark=self.watermark,
    prefill_schedule_interval=self.prefill_schedule_interval,
    disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
    async_scheduling=self.async_scheduling,
    stream_interval=self.stream_interval,
)
```

例如 `max_num_seqs` 限制运行请求数量，`max_num_batched_tokens` 与 `max_num_scheduled_tokens` 涉及每轮计算预算。这里先确定数值和开关；第四章会跟着整批请求解释它们如何参与一次调度。

最终这些配置被放进同一个 `VllmConfig`：

```python
# 文件：vllm/engine/arg_utils.py
# 类：EngineArgs
# 函数：create_engine_config
# 源码范围：L2457–L2486（连续节选）
config = VllmConfig(
    model_config=model_config,
    cache_config=cache_config,
    parallel_config=parallel_config,
    scheduler_config=scheduler_config,
    device_config=device_config,
    load_config=load_config,
    offload_config=offload_config,
    attention_config=attention_config,
    mamba_config=mamba_config,
    kernel_config=kernel_config,
    lora_config=lora_config,
    speculative_config=speculative_config,
    diffusion_config=diffusion_config,
    structured_outputs_config=self.structured_outputs_config,
    observability_config=observability_config,
    compilation_config=compilation_config,
    kv_transfer_config=self.kv_transfer_config,
    kv_events_config=self.kv_events_config,
    ec_transfer_config=self.ec_transfer_config,
    reasoning_config=self.reasoning_config,
    profiler_config=self.profiler_config,
    additional_config=self.additional_config,
    optimization_level=self.optimization_level,
    performance_mode=self.performance_mode,
    weight_transfer_config=self.weight_transfer_config,
    shutdown_timeout=self.shutdown_timeout,
)

return config
```

`VllmConfig` 创建过程中还会检查配置是否兼容，平台补齐 Worker 类、执行后端所需块大小等信息。各组件随后读取这些已经确定的值。

并行配置中的 TP 指把同一层的计算和参数分到多张 GPU，PP 指把不同模型层放到不同执行阶段。

## 5. 选择执行器

源码定位：[v1/executor/abstract.py](../../vllm/vllm/v1/executor/abstract.py) · `Executor.get_class`。

`Executor.get_class` 根据执行后端返回具体的执行器类，`uni` 对应 UniProcExecutor：

```python
# 文件：vllm/v1/executor/abstract.py
# 类：Executor
# 函数：get_class
# 源码范围：L73–L76（连续节选）
elif distributed_executor_backend == "uni":
    from vllm.v1.executor.uniproc_executor import UniProcExecutor

    executor_class = UniProcExecutor
```

工厂取得这个类，随后把它作为 `executor_class` 传给 LLMEngine。

`LLM.from_engine_args` 只是另一种用户入口，它会把参数展开后调用 `LLM.__init__`。`LLMEngine.from_engine_args` 则负责创建运行配置和 LLMEngine。

## 6. 到这里已经准备好什么

配置已经能说明使用哪个模型、哪张设备、什么精度、什么执行器，以及后续请求受到哪些运行限制。模型权重、实际 KV 张量和调度器实例仍要沿后面的构造过程建立。

下面用一个变化检查理解：如果在创建 LLM 时修改 `max_num_seqs`，变化先进入 EngineArgs，再写进 SchedulerConfig；它也可能影响 Model Runner 的输入缓冲和预热规模。一个配置可以被多个组件使用。

阅读检查：

- EngineArgs、VllmConfig 和实际模型对象分别在哪一步出现？
- 为什么 Executor.get_class 返回类之后，还没有完成模型加载？
- 模型配置已经读到，为什么还需要后续的权重加载？
- 同一个 max_num_seqs 配置会被哪些组件使用？

[章节目录](README.md) · [下一篇：建立入口与核心之间的连接](02-Entry-and-Core-Connection.md)
