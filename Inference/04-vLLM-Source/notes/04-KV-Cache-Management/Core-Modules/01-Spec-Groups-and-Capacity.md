# 01 · 规格、分组与容量

## 1. 职责与边界

输入是各层 KVCacheSpec 和可用于 KV 的字节预算；输出是 KVCacheConfig。本篇负责存储需求和容量规则，不探测 GPU 显存，也不创建 GPU 张量。

page 在这里指**一层、一个缓存块**对应的字节量。缓存分组表示哪些层可以共用同一套请求块表，不表示它们共享相同的 K/V 数值。

## 2. 源码定位与配置

| 文件 | 重点对象或函数 |
| --- | --- |
| [kv_cache_interface.py](../../../vllm/vllm/v1/kv_cache_interface.py) | FullAttentionSpec、KVCacheTensor、KVCacheGroupSpec、KVCacheConfig |
| [kv_cache_utils.py](../../../vllm/vllm/v1/core/kv_cache_utils.py) | get_kv_cache_groups、get_kv_cache_config_from_groups、get_num_blocks、generate_scheduler_kv_cache_config |

| 配置 | 作用 |
| --- | --- |
| block_size | 单个缓存块覆盖的 token 数 |
| page_size_bytes | 每层每块占用字节，包含规格要求的填充等成本 |
| num_blocks | 共享块池的块 ID 容量 |
| kv_cache_tensors | 每个存储分配的 size、shared_by，特殊布局还含 offset/block_stride |
| kv_cache_groups | 层名与对应的组规格 |

## 3. 实现分析

### 3.1 每层 page 的成本

FullAttentionSpec 的非量化分支允许 K/V 的 head size 不同：

```python
# 文件：vllm/v1/kv_cache_interface.py
# 类：FullAttentionSpec
# 函数：real_page_size_bytes
# 源码范围：L335–L350（连续节选）
@property
def real_page_size_bytes(self) -> int:
    if self.kv_quant_mode.is_nvfp4:
        # Packed layout per head: fp4 data + fp8 block scales.
        # fp4 data: head_size//2 bytes (2 fp4 values per byte)
        # fp8 block scale: head_size//16 bytes (1 scale per 16 elements)
        last_dim = nvfp4_kv_cache_full_dim(
            self.head_size
        ) + nvfp4_kv_cache_full_dim(self.head_size_v)
    elif self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        last_dim = self.head_size // 2 + self.head_size_v // 2
    else:
        last_dim = self.head_size + self.head_size_v
    return (
        self.block_size * self.num_kv_heads * last_dim * get_dtype_size(self.dtype)
    )
```

因此普通 K/V 等宽、无量化、无额外 page padding 时：

```text
每层每块字节 = 2 × block_size × num_kv_heads × head_size × dtype字节数
```

这是规格公式，不是 Attention 张量的维度顺序。量化比例尺、对齐填充、MLA 和混合缓存需要使用各自 Spec 的实际 page_size_bytes，不能沿用上述简式。

### 3.2 从分组得到配置

get_kv_cache_groups 按规格建立兼容的组。本章主线是所有层相同的 FullAttentionSpec，形成一个组。get_kv_cache_config_from_groups 再从给定预算生成存储描述。

普通等规格分支按每个池中块覆盖的层数扣除 page 成本，实际除法如下：

```python
# 文件：vllm/v1/core/kv_cache_utils.py
# 类：无（模块级函数）
# 函数：get_num_blocks
# 源码范围：L1008–L1010（连续节选）
num_blocks = int(available_memory // page_size // num_layers)
num_blocks = max(num_blocks, 0)
return may_override_num_blocks(vllm_config, num_blocks)
```

其中 num_layers 在调用处使用 group_size。单个等规格分组时，它就是该组层数；多组共享存储池的情形不能直接用“全模型层数”替代。

最终配置包含三项：

```python
# 文件：vllm/v1/core/kv_cache_utils.py
# 类：无（模块级函数）
# 函数：get_kv_cache_config_from_groups
# 源码范围：L1439–L1443（连续节选）
return KVCacheConfig(
    num_blocks=num_blocks,
    kv_cache_tensors=kv_cache_tensors,
    kv_cache_groups=kv_cache_groups,
)
```

kv_cache_tensors 只描述要分配多少字节、绑定哪些层，此时还没有创建 GPU 张量。Worker 按此配置分配是 05 的接口职责。

### 3.3 各种容量值不能混为一谈

- num_gpu_blocks_override 可覆盖按预算计算出的块数，覆盖后的存储量仍需实际容纳。
- BlockPool 会取出一个 null block，普通请求能使用的块数通常为 num_blocks−1。
- get_kv_cache_capacity 的报告还会考虑分组和 max_model_len；报告的名义容量不是对某个负载的并发保证。
- 多 Worker 的配置协调与共同块数由 get_kv_cache_configs 完成；分布式拓扑不在本篇展开。generate_scheduler_kv_cache_config 使用已经统一好的配置创建调度侧视图。

## 4. 手工推演：32 层与 8192 块

以下是容量算例，不指定模型名称，也不参与调度案例 A。给定可用于 KV 的预算为 **16 GiB**，32 层，每层 8 个 KV heads，head size=128，FP16，每块 16 token：

| 项目 | 计算 | 结果 |
| --- | --- | ---: |
| 每层每块 | 2×16×8×128×2 | 65,536 B = 64 KiB |
| 一个池中块 ID 覆盖全部 32 层 | 32×64 KiB | 2 MiB |
| 原始块池容量 | 16 GiB / 2 MiB | 8192 块 |
| 保留 null block 后 | 8192−1 | 8191 个可借块 |
| 可借 token 槽上限 | 8191×16 | 131,056 |

这里的 8192 是池中 ID 数；不应再把请求的块数乘 32。16 GiB 已经是外部交来的 KV 预算，不能再减一遍模型权重。

## 5. 设计原因与自测

规格与实际分配分开，使容量规划只处理字节和布局，运行期间只处理块引用。

- 为什么使用 num_kv_heads，而不是 query heads？
- 为什么 32 层只有一套请求逻辑块位置？
- 8192 是请求可借块数还是原始池容量？
- 给定预算变为 8 GiB，其他条件相同，原始块数是多少？

[章节目录](../README.md) · [本章概览](../Overview.md)
