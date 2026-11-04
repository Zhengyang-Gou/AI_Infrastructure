# 01 · Worker 与模型构造

## 1. 职责与边界

本篇负责本地设备和 Runner 的建立，以及模型配置到 nn.Module 实例的转换。到模型对象与参数容器存在为止；checkpoint 的读取与参数填充值由后两篇负责。

单卡主线也调用设备/通信环境初始化接口，本篇不展开进程组和跨 rank 拓扑。外部 Executor 如何创建并调用 Worker 属于 02/10 的接口组织，不在这里重讲。

## 2. 源码定位与关键对象

| 文件 | 重点 |
| --- | --- |
| [gpu_worker.py](../../../vllm/vllm/v1/worker/gpu_worker.py) | Worker.init_device |
| [model_loader/utils.py](../../../vllm/vllm/model_executor/model_loader/utils.py) | get_model_architecture、initialize_model |
| [models/registry.py](../../../vllm/vllm/model_executor/models/registry.py) | _ModelRegistry.resolve_model_cls |

| 对象 | 区别 |
| --- | --- |
| model_config.hf_config | 模型结构与架构名等元数据，不是模型权重 |
| model_class | 由注册表解析的实现类 |
| model | 调用模型构造函数得到的实例 |
| self.model_runner | Worker 选择的资源与执行对象，和 model 不是一层 |

## 3. 实现分析

### 3.1 准备设备与初始快照

Worker.init_device 选择本地 CUDA 设备、检查 dtype，调用通信环境初始化，然后设置随机种子、清理缓存并记录初始显存快照。顺序保证已建立的设备环境在探测基准中可见。

快照与预算的持有位置是：

```python
# 文件：vllm/v1/worker/gpu_worker.py
# 类：Worker
# 函数：init_device
# 源码范围：L394–L396（连续节选）
# take current memory snapshot
self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
self.requested_memory = request_memory(init_snapshot, self.cache_config)
```

本篇只说明这些字段在哪里准备；request_memory 的预算口径由 Core 04 唯一展开。

固定配置选择 V1 GPUModelRunner：

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

构造 Runner 并不等于已经加载 checkpoint；model 的加载接口稍后单独调用。

### 3.2 架构解析选择实现类

get_model_architecture 调用注册表前，读取 HF 配置中的 architectures：

```python
# 文件：vllm/model_executor/model_loader/utils.py
# 类：无（模块级函数）
# 函数：_get_model_architecture
# 源码范围：L204–L212（连续节选）
def _get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    from vllm.model_executor.models.adapters import as_embedding_model, as_seq_cls_model

    architectures = getattr(model_config.hf_config, "architectures", None) or []

    model_cls, arch = model_config.registry.resolve_model_cls(
        architectures,
        model_config=model_config,
    )
```

注册表按 model_impl 和受支持架构解析实现，必要时可选择 Transformers 后端；不能把这条路径一概写成 AutoModelForCausalLM.from_pretrained。主线选择受支持的 vLLM 普通生成模型。

get_model_architecture 还按模型、runner/convert 类型、model_impl 与 architectures 等构成的键缓存解析结果。它缓存的是类解析，不是整个模型实例或权重。

### 3.3 在统一配置上下文中构造模型

```python
# 文件：vllm/model_executor/model_loader/utils.py
# 类：无（模块级函数）
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

模型类接收 vllm_config 与 prefix。prefix 用于模块命名上下文；当前 dtype/device 上下文由加载模板设置，模型构造在其中创建层和参数。

源码还兼容旧式构造签名，本篇沿新式签名，不逐个推导兼容参数。量化参数布局和跨 rank 参数切分也不在主线展开。

## 4. 最小推演：配置到实例

```text
给定 HF architectures=[某个受支持的 CausalLM 架构]
    → registry 返回实现类 C
    → initialize_model 在当前配置/device/dtype 上下文调用 C(...)
    → 得到模块实例 M，拥有各层参数容器
```

此时不能仅凭参数对象存在断言 checkpoint 已写入。下一道加载接口消费的就是这个 M。

## 5. 设计原因与自测

- 架构解析缓存能否让第二个引擎直接复用第一个引擎的权重？
- Worker、Runner、model 分别是什么层次的对象？
- HF config 与 checkpoint 各提供什么？
- 模型类为什么接收统一的 vllm_config 与 prefix？

[章节目录](../README.md) · [本章概览](../Overview.md)
