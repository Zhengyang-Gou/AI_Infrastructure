# 03 · 权重映射与后处理

## 1. 职责与边界

输入是 `(checkpoint_name, tensor)` 参数流和已构造的模型；输出是写入目标参数、完成必要格式处理的模型。本篇解释名称路由、融合片段与参数拷贝，不展开 TP 切分或前向 kernel。

下面选择本地 Qwen2 实现作为具体例子。其他模型可覆盖 load_weights、WeightsMapper 或单参数 weight_loader，不能把本篇的名称表套用到所有模型。

## 2. 源码定位与职责

| 文件 | 重点 |
| --- | --- |
| [models/qwen2.py](../../../vllm/vllm/model_executor/models/qwen2.py) | Qwen2Model.hf_to_vllm_mapper、load_weights |
| [models/utils.py](../../../vllm/vllm/model_executor/models/utils.py) | WeightsMapper、AutoWeightsLoader |
| [layers/linear.py](../../../vllm/vllm/model_executor/layers/linear.py) | QKVParallelLinear.load_weights 的参数加载路由 |
| [model_loader/weight_utils.py](../../../vllm/vllm/model_executor/model_loader/weight_utils.py) | default_weight_loader |
| [model_loader/utils.py](../../../vllm/vllm/model_executor/model_loader/utils.py) | process_weights_after_loading |

checkpoint 名称、模型参数路径、融合片段 shard_id 分别承担不同角色。这里的 q/k/v shard_id 标识融合参数的片段，不等于 GPU rank。

## 3. 实现分析

### 3.1 模块树路由后，进入模型自己的映射

外层 AutoWeightsLoader 按名称前缀遍历模型子模块；子模块可提供自己的 load_weights。Qwen2Model 明确传入自己的 mapper：

```python
# 文件：vllm/model_executor/models/qwen2.py
# 类：Qwen2Model
# 函数：load_weights
# 源码范围：L436–L438（连续节选）
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    loader = AutoWeightsLoader(self)
    return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
```

该类的融合映射为：

```python
# 文件：vllm/model_executor/models/qwen2.py
# 类：Qwen2Model
# 函数：无（类属性 hf_to_vllm_mapper）
# 源码范围：L323–L332（连续节选）
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

三条 checkpoint 投影权重汇入 qkv_proj，gate/up 汇入 gate_up_proj。映射只改变目标名字并附加片段身份，不在这里运行投影计算。

### 3.2 片段身份随 tensor 传递

```python
# 文件：vllm/model_executor/models/utils.py
# 类：WeightsMapper
# 函数：apply
# 源码范围：L137–L147（连续节选）
def apply(
    self, weights: Iterable[tuple[str, torch.Tensor]]
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, data in weights:
        result = self._map_name_with_shard(name)
        if result is None:
            continue
        out_name, shard_id = result
        if shard_id is not None:
            data.shard_id = shard_id
        yield out_name, data
```

AutoWeightsLoader 再按映射后的路径找到模块或参数。QKVParallelLinear.load_weights 读取 tensor 上的 shard_id，并调用 param.weight_loader；该参数加载器知道如何写入对应的融合区域。

单参数未定义自有加载器时使用默认回退：

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

这是普通参数回退，不应绕过 QKV 模块的专用 load_weights 路线，直接把完整 q_proj 张量复制到整个 qkv_proj 参数。

### 3.3 默认拷贝仍需检查形状

```python
# 文件：vllm/model_executor/model_loader/weight_utils.py
# 类：无（模块级函数）
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

普通张量要求形状一致，再 copy_ 到参数数据；标量单独处理。融合、量化或切分参数可使用专用加载器。形状不匹配是定位映射/配置错误的信号，不应通过无条件 reshape 掩盖。

### 3.4 装载后仍有格式处理

BaseModelLoader 在 load_weights 后调用 process_weights_after_loading。后者遍历模块，调用量化方法、Attention 等模块提供的后处理接口，并可调用模型级 hook。

其作用可能包括重排、打包或初始化所需的缓存格式；非量化主线也不能简单将“文件拷贝完成”视为所有后处理都可省略。具体算子布局和 kernel 算法到模块 hook 接口为止，本篇不展开。

## 4. 手工推演：一条 q_proj 权重

```text
checkpoint：model.layers.0.self_attn.q_proj.weight
    → 外层加载器进入 model 子模块，交出剩余路径
    → Qwen2Model mapper：.q_proj → .qkv_proj，shard_id="q"
    → 模块树进入 layers.0.self_attn.qkv_proj
    → QKV 模块调用 param.weight_loader(..., "q")
    → 写入融合参数的 Q 区域
```

K、V 权重随后分别写入各自区域；名称路由和参数装载不产生用户 token。若 tied embedding 允许跳过 lm_head 的独立装载，属于模型显式声明的规则，不是所有缺失参数都可忽略。

## 5. 设计原因与自测

- 为什么 checkpoint 的 q_proj 名称不必等于模型 Parameter 名称？
- q/k/v shard_id 与 TP rank 有什么区别？
- 为什么模块级 load_weights 与单参数 weight_loader 都存在？
- 为什么后处理必须发生在装载后、资源交付前？
- 普通参数拷贝和 QKV 融合装载能否交换使用？

[章节目录](../README.md) · [本章概览](../Overview.md)
