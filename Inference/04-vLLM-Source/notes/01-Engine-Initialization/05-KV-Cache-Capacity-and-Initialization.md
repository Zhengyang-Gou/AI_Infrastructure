# 05 · 确定 KV Cache 容量并初始化缓存

这一篇接在模型加载之后：先问模型需要怎样的 KV 存储，再测出能使用多少显存，最后把容量配置变成 GPU 上的缓存张量。

## 1. 一次初始化要得到两种东西

| 需要的东西 | 谁使用 | 作用 |
| --- | --- | --- |
| KVCacheConfig 等配置 | 执行侧与调度侧 | 描述块数、分组、层对应关系和存储大小 |
| 实际的 KV 张量 | Model Runner 与 Attention 层 | 正式执行时保存各请求计算出的 K、V |

`KVCacheConfig` 是存储计划，张量是实际存储空间。后续 Scheduler 创建的 KVCacheManager 再负责记录这些块怎样被请求使用。

初始化先建立整套存储，再建立块的管理记录。请求进入调度后，会从这些已有存储中取得所需的块。

## 2. 第一步：从模型取得 KV 需求

源码定位：[v1/engine/core.py](../../vllm/vllm/v1/engine/core.py) · `EngineCore._initialize_kv_caches`。

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：_initialize_kv_caches
# 源码范围：L253–L258（连续节选）
# register all kvcache specs in enginecore process.
register_all_kvcache_specs(vllm_config)

# Get all kv cache needed by the model
kv_cache_specs = self.model_executor.get_kv_cache_specs()

```

Executor 将调用交给 Worker 的 `get_kv_cache_spec`，再进入 [gpu_model_runner.py](../../vllm/vllm/v1/worker/gpu_model_runner.py) 的同名方法。它从已创建的 Attention 层中收集需求：

```python
# 文件：vllm/v1/worker/gpu_model_runner.py
# 类：GPUModelRunner
# 函数：get_kv_cache_spec
# 源码范围：L7802–L7829（连续节选）
kv_cache_spec: dict[str, KVCacheSpec] = {}
layer_type = cast(type[Any], AttentionLayerBase)
attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
for layer_name, attn_module in attn_layers.items():
    if isinstance(attn_module, Attention) and (
        kv_tgt_layer := attn_module.kv_sharing_target_layer_name
    ):
        # The layer doesn't need its own KV cache and will use that of
        # the target layer. We skip creating a KVCacheSpec for it, so
        # that KV cache management logic will act as this layer does
        # not exist, and doesn't allocate KV cache for the layer. This
        # enables the memory saving of cross-layer kv sharing, allowing
        # a given amount of memory to accommodate longer context lengths
        # or enable more requests to be processed simultaneously.
        self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
        continue
    # Skip modules that don't need KV cache (eg encoder-only attention)
    if spec := attn_module.get_kv_cache_spec(self.vllm_config):
        if isinstance(spec, AttentionSpec):
            backend = attn_module.get_attn_backend()
            # indexes_kv_by_block_stride() -> get_kv_cache_stride_order()
            # -> get_kv_cache_layout() needs the current vLLM config.
            with set_current_vllm_config(self.vllm_config):
                indexes = backend.indexes_kv_by_block_stride()
            spec = replace(spec, indexes_kv_by_block_stride=indexes)
        kv_cache_spec[layer_name] = spec

return kv_cache_spec
```

后续循环取得各层的 `KVCacheSpec`。它说明一层缓存的存储要求，例如块覆盖多少 token、KV 头数、每个头的维度、数据类型等；具体字段随缓存种类不同。

循环逐层取得缓存规格，并记录需要共享其他层 KV 的对应关系。

返回到核心时，`kv_cache_specs` 是每个 Worker 一份需求字典组成的列表。

## 3. 第二步：确定可以给 KV 使用的显存

源码定位：[v1/worker/gpu_worker.py](../../vllm/vllm/v1/worker/gpu_worker.py) · `Worker.determine_available_memory`。

先回忆第三篇的 `requested_memory`。它在 [v1/worker/utils.py](../../vllm/vllm/v1/worker/utils.py) 中计算：

```python
# 文件：vllm/v1/worker/utils.py
# 类：无
# 函数：request_memory
# 源码范围：L414–L429（连续节选）
requested_memory = math.ceil(
    init_snapshot.total_memory * cache_config.gpu_memory_utilization
)

if init_snapshot.free_memory < requested_memory:
    raise ValueError(
        f"Free memory on device {init_snapshot.device_} "
        f"({format_gib(init_snapshot.free_memory)}/"
        f"{format_gib(init_snapshot.total_memory)} GiB) on startup "
        f"is less than desired GPU memory utilization "
        f"({cache_config.gpu_memory_utilization}, "
        f"{format_gib(requested_memory)} GiB). Decrease GPU memory "
        f"utilization or reduce GPU memory used by other processes."
    )

return requested_memory
```

`gpu_memory_utilization` 乘以**设备总显存**，得到引擎计划使用的预算。启动时先检查空闲显存能否满足预算，随后再扣除模型等占用，得到 KV 可用的部分。

模型加载后，要测量权重、临时中间结果和其他内存占用。Worker 使用模拟输入跑一次计算，记录内存变化：

```python
# 文件：vllm/v1/worker/gpu_worker.py
# 类：Worker
# 函数：determine_available_memory
# 源码范围：L500–L504（连续节选）
with memory_profiling(
    self.init_snapshot,
    weights_memory=int(self.model_runner.model_memory_usage),
) as profile_result:
    self.model_runner.profile_run()
```

`profile_run` 用模拟输入测量运行开销，也会提前触发相应计算所需的准备工作。

在本地源码中，核心计算式是：

```python
# 文件：vllm/v1/worker/gpu_worker.py
# 类：Worker
# 函数：determine_available_memory
# 源码范围：L544–L548（连续节选）
self.available_kv_cache_memory_bytes = (
    self.requested_memory
    - profile_result.non_kv_cache_memory
    - cudagraph_memory_estimate_applied
)
```

这几个统计值共同决定 KV 预算：

```text
KV 可用预算
    = 引擎计划使用的显存
      - 测算到的非 KV 占用
      - 当前启用的 CUDA Graph 内存估算
```

`cudagraph_memory_estimate_applied` 保存计入预算的图内存估算；关闭图执行时这个值为 0。

设置 `kv_cache_memory_bytes` 后，Worker 直接采用给定的 KV 字节数，并继续执行 `profile_run` 完成运行准备。

## 4. 第三步：从字节数得到块数和布局

核心将需求和可用字节数交给 [kv_cache_utils.py](../../vllm/vllm/v1/core/kv_cache_utils.py) 的 `get_kv_cache_configs`：

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：_initialize_kv_caches
# 源码范围：L301–L316（连续节选）
# Track max_model_len before KV cache config to detect auto-fit changes
max_model_len_before = vllm_config.model_config.max_model_len

kv_cache_configs = get_kv_cache_configs(
    vllm_config, kv_cache_specs, available_gpu_memory
)

# If auto-fit reduced max_model_len, sync the new value to workers.
# This is needed because workers were spawned before memory profiling
# and have the original (larger) max_model_len cached.
max_model_len_after = vllm_config.model_config.max_model_len
if max_model_len_after != max_model_len_before:
    self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
```

这个过程合并并检查各 Worker 的需求，组织缓存分组，确认容量能够支持长度要求，再产生各 Worker 的存储配置。

这里还可以看到：如果允许根据容量自动调整最大长度，调整后必须同步到已经创建好的 Worker。配置中的一部分数值需要等实际测算后才能最终确定。

普通同类 Attention 的块数计算，可以从 `get_num_blocks` 看出基本关系：

```python
# 文件：vllm/v1/core/kv_cache_utils.py
# 类：无
# 函数：get_num_blocks
# 源码范围：L1008–L1010（连续节选）
num_blocks = int(available_memory // page_size // num_layers)
num_blocks = max(num_blocks, 0)
return may_override_num_blocks(vllm_config, num_blocks)
```

`page_size` 是一个层的一块存储字节数。多层共同使用同一个块编号时，需要一起计入这些层的存储量。

### 用具体数字算一次

以 32 层、8 个 KV 头、每头 128 维、每个数占 2 字节为例，计算每个块编号对应的存储量：

```text
层数 L = 32
每层 KV 头数 Hkv = 8
每个头的维度 D = 128
每个数占 s = 2 字节
每块覆盖 B = 16 个 token

一个 token 在所有层的 KV 字节数
    = 2 × L × Hkv × D × s
    = 131,072 字节 = 128 KiB

一个块编号在所有层对应的存储量
    = B × 128 KiB
    = 2 MiB

若测算后的 KV 预算为 16 GiB：
    块数 = 16 GiB / 2 MiB = 8,192
    理想 token 容量 = 8,192 × 16 = 131,072
```

最前面的 2 表示 K 和 V 两份。有些模型让多个 Query 头共享同一组 K、V，这称为 GQA；计算 KV 存储时要使用实际 KV 头数。这里的容量由所有请求共同使用；单请求长度和每轮计算预算分别由相应配置控制。

`KVCacheConfig` 的关键字段定义在 [kv_cache_interface.py](../../vllm/vllm/v1/kv_cache_interface.py)：

| 字段 | 说明 |
| --- | --- |
| num_blocks | 可供块管理使用的块数量 |
| kv_cache_tensors | 需要实际分配哪些存储，每份有多少字节、由哪些层使用 |
| kv_cache_groups | 哪些层共用一套块表，以及对应的缓存规格；块表记录一条请求依次使用哪些 KV 块 |

`generate_scheduler_kv_cache_config` 再生成供 Scheduler 使用的配置；核心还会更新 CacheConfig 中的块数、块大小与容量信息。

## 5. 第四步：让存储真正出现在 GPU 上

Executor 的初始化入口会通知 Worker：

```python
# 文件：vllm/v1/executor/abstract.py
# 类：Executor
# 函数：initialize_from_config
# 源码范围：L118–L120（连续节选）
def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
    """Initialize the KV caches on the underlying workers."""
    self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))
```

Wrapper 取出对应 Worker 的那一份配置，GPU Worker 更新本地块数，并调用 Model Runner：

```python
# 文件：vllm/v1/worker/gpu_worker.py
# 类：Worker
# 函数：initialize_from_config
# 源码范围：L653–L665（连续节选）
# Update local config with adjusted num blocks after profiling,
# so that it's available to the warmup stage.
self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

# Init kv cache connector here, because it requires
# `kv_cache_config`.
# NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
# because `initialize_kv_cache` will inject kv cache groups not
# related to kv cache connector (e.g. kv cache sharing layers).
ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

with self._maybe_get_memory_pool_context(tag="kv_cache"):
    self.model_runner.initialize_kv_cache(kv_cache_config)
```

Model Runner 先准备 Attention 后端和缓存布局所需的执行信息，再建立实际张量：

```python
# 文件：vllm/v1/worker/gpu_model_runner.py
# 类：GPUModelRunner
# 函数：initialize_kv_cache
# 源码范围：L7627–L7653（连续节选）
kv_cache_config = deepcopy(kv_cache_config)
self.kv_cache_config = kv_cache_config
self._mamba_bufs = None
self.may_add_encoder_only_layers_to_kv_cache_config()
self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
self.initialize_attn_backend(kv_cache_config, is_profiling=is_profiling)
initialize_mamba_ssu_backend(
    self.vllm_config.mamba_config, self.kv_cache_config
)
# The kernel block size for all KV cache groups. For example, if
# kv_cache_manager uses block_size 256 for a given group, but the attention
# backends for that group only supports block_size 64, we will return
# kernel_block_size 64 and split the 256-token-block to 4 blocks with 64
# tokens each.
kernel_block_sizes = prepare_kernel_block_sizes(
    kv_cache_config, self.attn_groups
)
self._kernel_block_sizes = kernel_block_sizes

# create metadata builders
self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

# Reinitialize need to after initialize_attn_backend
self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
kv_caches = self.initialize_kv_cache_tensors(
    kv_cache_config, kernel_block_sizes
)
```

这里的 metadata builder 是“后面运行 Attention 时，用来整理长度、块位置等信息的对象”。它在初始化时准备好，具体一轮的数据到第五章准备输入时再填。

`prepare_kernel_block_sizes` 建立调度块大小与执行后端块大小的对应关系。一个调度块可以进一步拆成多个后端使用的块。

通用张量分配路线中的一段是：

```python
# 文件：vllm/v1/worker/gpu_model_runner.py
# 类：GPUModelRunner
# 函数：_allocate_kv_cache_tensors
# 源码范围：L7317–L7334（连续节选）
kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
packed_backing: torch.Tensor | None = None
for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
    if kv_cache_tensor.block_stride > 0:
        # Allocate once; all packed tensors alias the same backing.
        if packed_backing is None:
            packed_backing = torch.zeros(
                kv_cache_tensor.size,
                dtype=torch.int8,
                device=self.device,
            )
        tensor = packed_backing
    else:
        tensor = torch.zeros(
            kv_cache_tensor.size, dtype=torch.int8, device=self.device
        )
    for layer_name in kv_cache_tensor.shared_by:
        kv_cache_raw_tensors[layer_name] = tensor
```

`int8` 在这里用于按字节申请原始存储。随后再按后端要求，将这份存储解释为实际 KV 精度和形状。

最后把缓存与模型层对应起来：

```python
# 文件：vllm/v1/worker/gpu_model_runner.py
# 类：GPUModelRunner
# 函数：initialize_kv_cache_tensors
# 源码范围：L7580–L7586（连续节选）
bind_kv_cache(
    kv_caches,
    self.compilation_config.static_forward_context,
    self.kv_caches,
    num_attn_module,
)
return kv_caches
```

## 6. 初始化缓存与运行时分配的关系

可以把此时的结果理解为：整套可重复使用的 KV 存储已经建立，模型层知道去哪里访问它。后续来了一批请求，调度器再安排这些请求分别使用哪些块。

请求结束后释放块引用，存储便可以再次使用。第四章会在调度过程中继续解释前缀复用、块引用与抢占。

如果模型权重能加载，但 KV 容量不足以支持要求的最大长度，初始化仍可能在容量检查处失败；若预算高估，实际分配时也可能出现显存不足。这些失败发生的位置不同，要沿本篇的需求、预算、配置、分配四步定位。

阅读检查：

- CacheConfig、KVCacheConfig、实际 KV 张量与 KVCacheManager 各自是什么？
- gpu_memory_utilization 的比例以总显存还是空闲显存为基数？
- 为什么模型权重已经加载，仍需用模拟输入做显存测量？
- 计算示例中的 KV 头数、块数、token 容量分别是什么单位？
- 原始字节存储怎样变成实际 KV 精度和形状的张量？

[章节目录](README.md) · [上一篇：创建模型与加载权重](04-Model-Creation-and-Weight-Loading.md) · [下一篇：编译、预热与执行准备](06-Compilation-and-Warmup.md)
