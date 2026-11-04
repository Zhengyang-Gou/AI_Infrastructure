# 04 · 显存探测与 KV 张量初始化

## 1. 职责与边界

本篇包含两个独立接口：先探测并交出可用于 KV 的字节预算，再消费外部交回的 KVCacheConfig 创建存储。中间怎样按 page 和分组计算块数属于 04 章，本篇不重复推导。

实际 GPU 张量分配和绑定在本篇；Attention 物理维度、排列顺序与 kernel 如何读写在 07。本篇只要求对应后端返回可使用的视图。

## 2. 源码定位与字段

| 文件 | 重点 |
| --- | --- |
| [worker/utils.py](../../../vllm/vllm/v1/worker/utils.py) | request_memory、bind_kv_cache |
| [gpu_worker.py](../../../vllm/vllm/v1/worker/gpu_worker.py) | determine_available_memory、initialize_from_config |
| [mem_utils.py](../../../vllm/vllm/utils/mem_utils.py) | MemorySnapshot、memory_profiling |
| [gpu_model_runner.py](../../../vllm/vllm/v1/worker/gpu_model_runner.py) | profile_run、get_kv_cache_spec、initialize_kv_cache、initialize_kv_cache_tensors |

| 字段 | 含义 |
| --- | --- |
| init_snapshot | Worker 设备环境建立后的初始显存快照 |
| requested_memory | 根据总显存和利用率确定的预算 |
| model_memory_usage | 模型加载阶段记录的显存消耗；不等于最终全部非 KV 成本 |
| non_kv_cache_memory | 本地探测口径下的非 KV 消耗与瞬时峰值余量 |
| available_kv_cache_memory_bytes | 可交给外部容量规划的字节预算 |

## 3. 实现分析

### 3.1 利用率乘的是总显存

```python
# 文件：vllm/v1/worker/utils.py
# 类：无（模块级函数）
# 函数：request_memory
# 源码范围：L409–L429（连续节选）
def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    """
    Calculate the amount of memory required by vLLM, then validate
    that the current amount of free memory is sufficient for that.
    """
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

requested_memory 取总显存乘 gpu_memory_utilization 后向上取整；不是“当前空闲显存的 80%”。初始空闲显存不足该预算时会拒绝启动。

### 3.2 用 dummy 工作测量实际成本

```python
# 文件：vllm/v1/worker/gpu_worker.py
# 类：Worker
# 函数：determine_available_memory
# 源码范围：L498–L504（连续节选）
# Execute a forward pass with dummy inputs to profile the memory usage
# of the model.
with memory_profiling(
    self.init_snapshot,
    weights_memory=int(self.model_runner.model_memory_usage),
) as profile_result:
    self.model_runner.profile_run()
```

profile_run 调用代表配置上限规模的 dummy 工作并覆盖所需的末端处理，然后同步和清理临时对象。该调用是探测，不是用户请求的一轮；内部输入与前向细节留在 06 的接口实现。

本地 memory_profiling 的最终统计采用真实快照消耗加瞬时峰值余量：

```python
# 文件：vllm/utils/mem_utils.py
# 类：无（模块级函数）
# 函数：memory_profiling
# 源码范围：L317–L326（连续节选）
result.total_consumed = (
    result.before_create.free_memory - result.after_profile.free_memory
)

# total_consumed already covers persistent torch allocations; add only the
# transient peak headroom to avoid double-counting.
result.transient_peak_headroom = (
    result.after_profile.torch_peak - result.after_profile.torch_allocated
)
result.non_kv_cache_memory = result.total_consumed + result.transient_peak_headroom
```

weights_memory、torch/non-torch 增量仍用于记录和诊断，但本地最终 non_kv_cache_memory 不能照抄旧式“再分别加一遍权重”的公式。total_consumed 已包括持续保留的分配，避免重复计数。

固定 eager、普通文本场景不扣图捕获开销，Worker 使用：

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

函数返回端还考虑可能的多模态 IPC 预留；普通文本场景不展开该分支。显式 kv_cache_memory_bytes 则直接指定预算，跳过这套自动预算计算，但源码仍调用 profile_run；不能写成“显式预算后完全不跑模型”。

### 3.3 规格交出，配置交回

GPUModelRunner.get_kv_cache_spec 从已登记的 Attention 模块取得各层规格；它只汇集需求。外部交回 KVCacheConfig 后，Worker 更新 num_gpu_blocks 并调用 model_runner.initialize_kv_cache。

Runner 深拷贝配置、调用后端准备与元数据构造接口，再初始化 KV 张量。后端选择、物理维度和元数据结构到接口为止；本篇只追踪存储的分配与绑定。

### 3.4 分配字节存储，再绑定可用视图

通用分配分支按配置 size 分配 raw buffer，并将 shared_by 层名映射到存储：

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

raw buffer 使用 int8 表示按字节计的底层存储，不表示 KV 计算 dtype 必然是 INT8。重解释后的 dtype/shape 由规格和后端约定决定。

initialize_kv_cache_tensors 可选择统一跨层分配分支；上面的通用分支是实际存在的另一条路径，不能声称所有配置都逐层独立 torch.zeros。得到后端视图后调用：

```python
# 文件：vllm/v1/worker/gpu_model_runner.py
# 类：GPUModelRunner
# 函数：initialize_kv_cache_tensors
# 源码范围：L7577–L7586（连续节选）
num_attn_module = (
    2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
)
bind_kv_cache(
    kv_caches,
    self.compilation_config.static_forward_context,
    self.kv_caches,
    num_attn_module,
)
return kv_caches
```

bind_kv_cache 将存储引用交给 Runner 和相应模型层。绑定是资源初始化，不是本次 token 的 KV 写入，也没有构造请求 block table。

## 4. 最小预算推演

以下是假设探测值，只验证本篇口径，不进行块容量计算：

```text
总显存 20 GiB，gpu_memory_utilization=0.8 → requested_memory=16 GiB
快照测得 total_consumed=7 GiB，瞬时峰值余量=1 GiB
non_kv_cache_memory=7+1=8 GiB
固定 eager 无图预算、无额外 IPC 预留 → 可交出的 KV 预算=16−8=8 GiB
```

接收外部 KVCacheConfig 时，本篇只按照 size 等配置创建张量，不重新用这个 8 GiB 推算块数。

## 5. 设计原因与自测

- gpu_memory_utilization 为什么不能直接等同于 KV 所占总显存比例？
- 为什么减 non_kv_cache_memory 后不能再减一遍权重？
- 指定 kv_cache_memory_bytes 后是否还会调用 profile_run？
- raw int8 存储是否表示模型使用 INT8 KV 计算？
- 字节预算、KVCacheConfig、实际 tensor 分别在哪道接口出现？

[章节目录](../README.md) · [本章概览](../Overview.md)
