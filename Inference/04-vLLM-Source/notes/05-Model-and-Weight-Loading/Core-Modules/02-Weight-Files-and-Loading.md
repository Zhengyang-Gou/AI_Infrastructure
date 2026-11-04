# 02 · 权重文件与加载流程

## 1. 职责与边界

本篇从模型位置与 load_format 开始，解释文件怎样成为权重迭代器，并交给已经创建的模型。模型类选择与参数布局算法分别由 Core 01、03 负责；本篇只说明调用契约。

## 2. 源码定位与中间表示

| 文件 | 重点 |
| --- | --- |
| [gpu_model_runner.py](../../../vllm/vllm/v1/worker/gpu_model_runner.py) | load_model 的加载器调用 |
| [model_loader/__init__.py](../../../vllm/vllm/model_executor/model_loader/__init__.py) | loader 注册表、get_model_loader |
| [base_loader.py](../../../vllm/vllm/model_executor/model_loader/base_loader.py) | BaseModelLoader.load_model 共用模板 |
| [default_loader.py](../../../vllm/vllm/model_executor/model_loader/default_loader.py) | _prepare_weights、_get_weights_iterator、get_all_weights、load_weights |
| [weight_utils.py](../../../vllm/vllm/model_executor/model_loader/weight_utils.py) | 分片索引过滤、safetensors_weights_iterator |

DefaultModelLoader.Source 描述模型路径、revision、prefix、subfolder 等来源信息。迭代器逐项产出 `(name, tensor)`；它不是按 GPU 执行批次组织的数据。

## 3. 实现分析

### 3.1 选择 loader，不直接在 Runner 中读取每个参数

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

get_model_loader 按 load_format 查注册表。auto/hf/safetensors 等可映射到同一个 DefaultModelLoader，但其内部文件策略仍有区别。

BaseModelLoader.load_model 在目标 dtype/device 上下文中调用 initialize_model，再调用自身 load_weights，之后执行共用后处理，最后返回 model.eval()。构造算法与参数写入算法到各自接口为止。

### 3.2 本地文件发现与分片过滤

_prepare_weights 先判断本地目录。主线直接使用该目录，按 load_format 生成允许的文件模式；显式 safetensors 分支是：

```python
# 文件：vllm/model_executor/model_loader/default_loader.py
# 类：DefaultModelLoader
# 函数：_prepare_weights
# 源码范围：L168–L174（连续节选）
elif (
    load_format == "safetensors"
    or load_format == "fastsafetensors"
    or load_format == "instanttensor"
):
    use_safetensors = True
    allow_patterns = ["*.safetensors"]
```

随后从模式列表找到候选文件；有 safetensors 索引时，过滤索引之外的重复文件，避免把 consolidated 和 sharded 两套相同权重同时加载。无权重文件时抛错，不继续返回一个空模型。

远端下载、ModelScope、PT、npcache 和特殊 loader 只是可替换来源，不在本地 safetensors 主线展开。auto 在本地代码中还检测 Mistral 格式，不能简单理解为“始终直接选第一份 safetensors”。

### 3.3 迭代文件内容，按需选择读取策略

_get_weights_iterator 先调用 _prepare_weights。普通 safetensors 路径选择 safetensors_weights_iterator；多线程、fastsafetensors 等属于其他读取分支。

默认 safe_open 分支逐个交出条目：

```python
# 文件：vllm/model_executor/model_loader/weight_utils.py
# 类：无（模块级函数）
# 函数：safetensors_weights_iterator
# 源码范围：L967–L972（连续节选）
with safe_open(st_file, framework="pt") as f:
    for name in f.keys():  # noqa: SIM118
        if should_skip_weight(name, local_expert_ids):
            continue
        param = f.get_tensor(name)
        yield name, param
```

具体使用 mmap、eager 或预取取决于 safetensors_load_strategy。逐项迭代不等于所有策略都只占用一个 tensor 的内存，也不保证文件中的 tensor 已经是 GPU 参数对象。

Source.prefix 在迭代器输出端附加到名称；get_all_weights 先读取主来源，再依次读取模型声明的 secondary_weights。普通单来源模型不走后者。

### 3.4 模型负责消费参数流

```python
# 文件：vllm/model_executor/model_loader/default_loader.py
# 类：DefaultModelLoader
# 函数：load_weights
# 源码范围：L427–L433（连续节选）
loaded_weights = model.load_weights(self.get_all_weights(model_config, model))

self.counter_after_loading_weights = time.perf_counter()
logger.info_once(
    "Loading weights took %.2f seconds",
    self.counter_after_loading_weights - self.counter_before_loading_weights,
)
```

model.load_weights 返回已装载参数集合或实现约定的结果。DefaultModelLoader 对支持跟踪的非量化模型默认检查缺失参数；是否允许忽略某些权重由模型规则决定，不能把未知名字一律静默跳过。

## 4. 最小推演：本地两个分片

```text
目录：model-00001-of-00002.safetensors
      model-00002-of-00002.safetensors
      model.safetensors.index.json
    → 按格式发现候选文件，并按索引过滤
    → 迭代输出 (name_1, tensor_1)、(name_2, tensor_2)…
    → model.load_weights 消费同一参数流
    → 返回加载跟踪结果，交给共用后处理
```

本篇在装载调用处结束，不继续展开 qkv 融合或参数拷贝。

## 5. 设计原因与自测

- load_format 与模型 architectures 为什么是不同的选择？
- 分片索引在过滤哪些文件，为什么不能将目录里所有权重文件都读一遍？
- `(name, tensor)` 与模型中的 Parameter 是同一个对象吗？
- 本地显式 safetensors 与 auto 有什么区别？

[章节目录](../README.md) · [本章概览](../Overview.md)
