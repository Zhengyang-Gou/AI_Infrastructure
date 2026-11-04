# 04 · 创建模型与加载权重

Worker.load_model 将加载任务交给 Model Runner；加载器创建模型结构，再把权重文件中的数值写入模型参数。

## 1. 先把三件事分开

| 工作 | 得到什么 | 例子 |
| --- | --- | --- |
| 读取模型配置 | 模型结构与运行需求的描述 | 层数、隐藏维度、KV 头数、模型实现名称 |
| 创建模型对象 | 实际的模型层、参数和辅助对象 | Embedding、Decoder Layer、RMSNorm、lm_head |
| 加载权重 | 将训练好的数值写入这些参数 | 将文件里的某个 weight 张量放到对应参数中 |

第一篇已经读取配置。本篇先按配置创建模型，再填入权重，随后执行必要的加载后处理。它们一起完成后，Model Runner 才持有可用于推理的模型。

## 2. Worker 将加载任务交给 Model Runner

源码定位：[v1/worker/gpu_worker.py](../../vllm/vllm/v1/worker/gpu_worker.py) · `Worker.load_model`。

```python
# 文件：vllm/v1/worker/gpu_worker.py
# 类：Worker
# 函数：load_model
# 源码范围：L435–L442（连续节选）
def load_model(self, *, load_dummy_weights: bool = False) -> None:
    with (
        self._maybe_get_memory_pool_context(tag="weights"),
        set_current_vllm_config(self.vllm_config),
        # 20 MiB is the minimum PyTorch allows for max_split_size_mb.
        self._scoped_allocator_max_split(max_split_size_mb=20),
    ):
        self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
```

外围的 `with` 用于控制分配权重时的内存环境、当前配置及分配器设置。真正选择加载器并保存模型的是 [gpu_model_runner.py](../../vllm/vllm/v1/worker/gpu_model_runner.py) 的 `GPUModelRunner.load_model`：

```python
# 文件：vllm/v1/worker/gpu_model_runner.py
# 类：GPUModelRunner
# 函数：load_model
# 源码范围：L5319–L5326（连续节选）
with DeviceMemoryProfiler() as m:
    time_before_load = time.perf_counter()
    if load_dummy_weights:
        self.load_config.load_format = "dummy"
    model_loader = get_model_loader(self.load_config)
    self.model = model_loader.load_model(
        vllm_config=self.vllm_config, model_config=self.model_config
    )
```

`self.model = ...` 将加载完成的模型保存在 Model Runner 中。

`DeviceMemoryProfiler` 记录加载前后的内存变化，写入 `model_memory_usage`，供后续计算缓存预算使用。这份记录包含加载阶段带来的内存变化，与磁盘上的权重文件大小分别统计。

## 3. 加载器怎样选，模型类怎样找

源码定位：[model_loader/__init__.py](../../vllm/vllm/model_executor/model_loader/__init__.py) · `get_model_loader`。

```python
# 文件：vllm/model_executor/model_loader/__init__.py
# 类：无
# 函数：get_model_loader
# 源码范围：L122–L127（连续节选）
def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""
    load_format = load_config.load_format
    if load_format not in _LOAD_FORMAT_TO_MODEL_LOADER:
        raise ValueError(f"Load format `{load_format}` is not supported")
    return _LOAD_FORMAT_TO_MODEL_LOADER[load_format](load_config)
```

映射表将 `safetensors` 对应到 `DefaultModelLoader`，接着创建默认加载器。

默认加载器复用 [base_loader.py](../../vllm/vllm/model_executor/model_loader/base_loader.py) 的 `BaseModelLoader.load_model`，先决定参数放到哪个设备、采用什么默认精度：

```python
# 文件：vllm/model_executor/model_loader/base_loader.py
# 类：BaseModelLoader
# 函数：load_model
# 源码范围：L47–L65（连续节选）
device_config = vllm_config.device_config
load_config = vllm_config.load_config
load_device = (
    device_config.device if load_config.device is None else load_config.device
)
target_device = torch.device(load_device)
with set_default_torch_dtype(model_config.dtype):
    with target_device:
        model = initialize_model(
            vllm_config=vllm_config,
            model_config=model_config,
            prefix=prefix,
        )

    log_model_inspection(model)

    logger.debug("Loading weights on %s ...", load_device)
    self.load_weights(model, model_config)

```

`initialize_model` 在目标 GPU 上创建模型对象，`self.load_weights` 再把权重填进去。

模型类的选择可以继续到 [model_loader/utils.py](../../vllm/vllm/model_executor/model_loader/utils.py)：

```python
# 文件：vllm/model_executor/model_loader/utils.py
# 类：无
# 函数：_get_model_architecture
# 源码范围：L207–L212（连续节选）
architectures = getattr(model_config.hf_config, "architectures", None) or []

model_cls, arch = model_config.registry.resolve_model_cls(
    architectures,
    model_config=model_config,
)
```

`architectures` 来自模型配置，注册表把名称对应到具体实现。`LlamaForCausalLM` 对应 [models/llama.py](../../vllm/vllm/model_executor/models/llama.py) 中的同名类。

取得类以后，创建模型：

```python
# 文件：vllm/model_executor/model_loader/utils.py
# 类：无
# 函数：initialize_model
# 源码范围：L49–L64（连续节选）
if model_config is None:
    model_config = vllm_config.model_config
if model_class is None:
    model_class, _ = get_model_architecture(model_config)

if vllm_config.quant_config is not None:
    configure_quant_config(vllm_config.quant_config, model_class)

signatures = inspect.signature(model_class.__init__)
all_params = [param.name for param in signatures.parameters.values()]
if "vllm_config" in all_params and "prefix" in all_params:
    # new-style model class
    with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
        model = model_class(vllm_config=vllm_config, prefix=prefix)
        record_metadata_for_reloading(model)
        return model
```

`prefix` 用来标识模型或子模块名称。`set_current_vllm_config` 让嵌套创建的模型层能读取当前配置，并参与相应的编译等准备。

## 4. 模型对象内部建立了什么

在 `LlamaForCausalLM.__init__` 中，可以看到主体模型与输出层的建立：

```python
# 文件：vllm/model_executor/models/llama.py
# 类：LlamaForCausalLM
# 函数：__init__
# 源码范围：L478–L492（连续节选）
self.model = self._init_model(
    vllm_config=vllm_config,
    prefix=maybe_prefix(prefix, "model"),
    layer_type=layer_type,
)

if get_pp_group().is_last_rank:
    self.lm_head = ParallelLMHead(
        config.vocab_size,
        config.hidden_size,
        quant_config=quant_config,
        prefix=maybe_prefix(prefix, "lm_head"),
    )
    if config.tie_word_embeddings:
        self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
```

`_init_model` 创建主体，`lm_head` 创建输出层。`LlamaModel.__init__` 继续建立输入 Embedding、配置数量的 Decoder Layer 和最终 RMSNorm。

在对象创建阶段，先记住它们的用途：Embedding 把 token ID 变成向量；Decoder Layer 逐层处理这些向量；RMSNorm 调整数值尺度；lm_head 将最后的向量转成各候选 token 的分数。具体计算在第六章沿前向展开。

如果模型配置要求输入 Embedding 和输出层共享权重，`tie_weights` 会建立相应共享关系；后续加载也会考虑它。

Attention 层创建时会选择具体执行 Attention 计算的后端，KV 初始化时再建立配套状态。第六章接着解释这些层在前向中怎样计算。

## 5. 从 safetensors 文件到逐个权重

源码定位：[default_loader.py](../../vllm/vllm/model_executor/model_loader/default_loader.py) · `_prepare_weights`、`_get_weights_iterator`、`get_all_weights`、`load_weights`。

```text
本地模型目录
    → 按 safetensors 格式筛选权重文件
    → 使用索引等信息过滤重复文件
    → 建立权重迭代器
    → 逐项取得 (权重名称, 张量)
    → 交给模型自己的 load_weights
```

`_prepare_weights` 在模型目录中按 `*.safetensors` 查找权重文件，并根据索引过滤重复文件。找不到可用权重时会报错。

迭代器可以理解为“每次取出一个名称和一份张量的读取过程”。普通 safetensors 分支中调用的是：

```python
# 文件：vllm/model_executor/model_loader/default_loader.py
# 类：DefaultModelLoader
# 函数：_get_weights_iterator
# 源码范围：L287–L298（连续节选）
weights_iterator = safetensors_weights_iterator(
    hf_weights_files,
    self.load_config.use_tqdm_on_load,
    self.load_config.safetensors_load_strategy,
    local_expert_ids=self.local_expert_ids,
    safetensors_prefetch_num_threads=(
        self.load_config.safetensors_prefetch_num_threads
    ),
    safetensors_prefetch_block_size=(
        self.load_config.safetensors_prefetch_block_size
    ),
)
```

调用方可以逐项接收权重。文件读取和预取由迭代器内部的加载策略处理。

最后默认加载器把这些权重交给模型：

```python
# 文件：vllm/model_executor/model_loader/default_loader.py
# 类：DefaultModelLoader
# 函数：load_weights
# 源码范围：L425–L427（连续节选）
self._init_ep_weight_filter(model_config)

loaded_weights = model.load_weights(self.get_all_weights(model_config, model))
```

这里出现两个同名 `load_weights`：外层属于加载器，负责准备数据；内层属于模型，负责把外部名称对应到模型内部的模块与参数。

## 6. 外部参数名为什么可能需要改写

Llama 文件中的 Q、K、V 投影可能分别保存，vLLM 模型把它们组织到同一个 `qkv_proj` 中。当前源码用类属性说明如何对应：

```python
# 文件：vllm/model_executor/models/llama.py
# 类：LlamaModel
# 函数：无（类属性）
# 源码范围：L345–L354（连续节选）
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_stacked={
        # weight_name: (param_name, shard_id)
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }
)
```

例如下面是名称变化示意：

```text
文件中：model.layers.0.self_attn.q_proj.weight
模型内：model.layers.0.self_attn.qkv_proj.weight 的 q 部分

文件中：gate_proj、up_proj
模型内：gate_up_proj 的相应部分
```

`"q"`、`"k"`、`"v"` 或 0、1 标记需要写入合并参数的哪一部分。

`LlamaForCausalLM.load_weights` 与 `LlamaModel.load_weights` 使用 [models/utils.py](../../vllm/vllm/model_executor/models/utils.py) 中的 `AutoWeightsLoader` 逐层找到子模块和参数。子模块可以提供自己的 `load_weights`，具体参数也可以提供自己的 `weight_loader`。

普通参数的最终调用位置是：

```python
# 文件：vllm/model_executor/models/utils.py
# 类：AutoWeightsLoader
# 函数：_load_param
# 源码范围：L280–L285（连续节选）
weight_loader = getattr(param, "weight_loader", default_weight_loader)
weight_loader(param, weight_data)

logger.debug("Loaded weight %s with shape %s", weight_qualname, param.shape)

yield weight_qualname
```

没有专门加载函数时，落到 [weight_utils.py](../../vllm/vllm/model_executor/model_loader/weight_utils.py) 的默认加载函数：

```python
# 文件：vllm/model_executor/model_loader/weight_utils.py
# 类：无
# 函数：default_weight_loader
# 源码范围：L1231–L1249（连续节选）
def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # reshape to match before copying
            param.data.copy_(loaded_weight.view(param.shape))
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise
```

`copy_` 将读出的数值写入已经建立的目标参数。合并投影等参数使用专用加载函数，把各部分写入对应位置。

## 7. 加载后还要完成哪些工作

`BaseModelLoader.load_model` 在填入权重后还会调用 `process_weights_after_loading`，完成相应模块需要的整理，然后返回 `model.eval()`。

```python
# 文件：vllm/model_executor/model_loader/base_loader.py
# 类：BaseModelLoader
# 函数：load_model
# 源码范围：L77–L80（连续节选）
if _has_online_quant(model):
    finalize_layerwise_processing(model, model_config)

process_weights_after_loading(model, model_config, target_device)
```

加载后处理会整理实现需要的权重格式，完成 Attention 等模块的准备。`eval()` 将模型切换为评估模式，梯度记录则由执行上下文控制。

默认加载器根据已加载参数名检查是否遗漏权重，检查时也会考虑共享参数和允许跳过的权重。

回到 Model Runner 后，`self.model` 保存模型，`self.model_memory_usage` 保存加载内存记录，接着完成通信缓冲等执行准备。

到此，`UniProcExecutor._init_executor` 完成模型加载并返回 EngineCore。下一项主要工作是确定剩余资源能够建立多大的 KV Cache。

阅读检查：

- 读到模型结构信息、创建模型对象、加载参数分别发生在哪里？
- DefaultModelLoader 与模型自己的 load_weights 怎样分工？
- q_proj 为什么不一定能直接按原名复制进一个独立参数？
- 模型加载完成后，实际模型由谁持有？下一步为什么还要测算显存？

[章节目录](README.md) · [上一篇：启动 Executor、Worker 与执行环境](03-Executor-Worker-and-Device.md) · [下一篇：确定 KV Cache 容量并初始化缓存](05-KV-Cache-Capacity-and-Initialization.md)
