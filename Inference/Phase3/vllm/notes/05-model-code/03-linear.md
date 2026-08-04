# Tensor-Parallel Linear Layers

## 1. 文件定位

- 文件路径：`vllm/model_executor/layers/linear.py`
- 所属层次：模型执行基础层
- 核心职责：定义可插拔、可量化的 Linear 抽象，以及 replicated、column-parallel、row-parallel、融合 MLP 和融合 QKV 等具体实现。
- 在调用链中的位置：被 Llama 等具体模型用于 Attention、MLP 和输出投影。

本文件同时负责运行时矩阵乘法和 checkpoint 权重切片。理解时应把“forward 如何通信”和“weight loader 如何选取本 rank 参数”分开追踪。

## 2. 核心类与组件

| 类 / 组件 | 作用 | Llama 中的用途 |
| --- | --- | --- |
| `LinearMethodBase` | 定义权重创建和矩阵乘法接口 | 量化与非量化实现的统一边界 |
| `UnquantizedLinearMethod` | 普通未量化 GEMM | 第一轮主线 |
| `LinearBase` | 保存尺寸、dtype、量化与 TP 信息 | 所有 Linear 的公共基类 |
| `ReplicatedLinear` | 每个 rank 保存完整权重 | 无需张量并行切分的投影 |
| `ColumnParallelLinear` | 按输出维度切分权重 | 并行产生局部输出特征 |
| `MergedColumnParallelLinear` | 融合多个 column-parallel 权重 | Llama `gate_up_proj` |
| `QKVParallelLinear` | 按 attention head 组织融合 QKV | Llama `qkv_proj` |
| `RowParallelLinear` | 按输入维度切分权重 | Llama `o_proj`、`down_proj` |

## 3. Linear 方法抽象

```text
LinearBase.__init__()
→ 根据 quant_config 选择 quant_method
├─ 无量化配置 → UnquantizedLinearMethod
└─ 有量化配置 → QuantizationConfig.get_quant_method()

具体 Linear.__init__()
→ quant_method.create_weights()

具体 Linear.forward()
→ quant_method.apply(layer, input, bias)
```

非量化主线中，`UnquantizedLinearMethod.create_weights()` 创建带有 input/output 维度元数据和 `weight_loader` 的参数；`apply()` 通过平台选择的 GEMM 实现完成矩阵乘法。

## 4. Column Parallel

数学形式可写为：

```text
Y = X A
A = [A₀, A₁, ..., Aₚ₋₁]
rank i 计算 Yᵢ = X Aᵢ
```

`ColumnParallelLinear` 按输出维度切分权重，每个 rank 接收完整输入并产生局部输出特征。默认不 gather；仅当 `gather_output=True` 时对局部输出执行 tensor-parallel all-gather。

```text
ColumnParallelLinear.forward()
→ quant_method.apply(input)
→ output_parallel
→ 可选 all-gather
→ output
```

权重加载时，根据 `tp_rank * shard_size` 从 checkpoint 的输出维度取出当前 rank 的切片。

## 5. Row Parallel

数学形式可写为：

```text
       [A₀]
A =    [A₁]      X = [X₀, X₁, ..., Xₚ₋₁]
       [...]
       [Aₚ₋₁]

Y = Σ Xᵢ Aᵢ
```

`RowParallelLinear` 按输入维度切分权重。如果输入尚未切分，先按最后一维切分；每个 rank 计算局部结果后，默认通过 tensor-parallel all-reduce 求和。

```text
RowParallelLinear.forward()
→ 必要时切分 input
→ quant_method.apply(input_parallel)
→ output_parallel
→ 可选 all-reduce
→ output
```

只有 rank 0 将 bias 融合进局部 GEMM，避免 TP all-reduce 时重复加入 bias。

## 6. 融合 MLP Linear

`MergedColumnParallelLinear` 将多个逻辑权重沿输出维度拼接，但每个逻辑权重仍分别按 TP size 切分。

Llama 中的对应关系为：

```text
gate_proj weight ┐
                 ├→ gate_up_proj weight
up_proj weight   ┘
```

加载权重时，`loaded_shard_id` 指定 checkpoint 权重应写入融合参数的哪一段；已经在 checkpoint 中融合的权重也可以先拆分，再递归调用 shard loader。

## 7. 融合 QKV Linear

`QKVParallelLinear` 根据 query head 和 KV head 数计算当前 rank 的本地尺寸：

- Q heads 总是按 TP size 切分。
- KV heads 多于或等于 TP size 时按 rank 切分。
- KV heads 少于 TP size 时在多个 rank 上复制。

```text
q_proj weight ┐
k_proj weight ├→ qkv_proj weight
v_proj weight ┘
```

`loaded_shard_id` 使用 `"q"`、`"k"` 或 `"v"`。loader 为不同 shard 计算偏移、大小和 KV 复制 rank，再从 checkpoint 中取出当前 rank 对应权重。

## 8. 输入与输出

### 输入

- forward 输入：hidden states Tensor。
- 初始化输入：全局 input/output size、TP 配置、dtype、bias 和量化配置。
- 权重加载输入：checkpoint Tensor，以及融合权重使用的 shard ID。

### 输出

- `ColumnParallelLinear` 通常输出最后一维的本地分片。
- `RowParallelLinear` 默认 all-reduce 后输出完整 hidden dimension。
- 根据 `return_bias` 与 `skip_bias_add`，forward 可返回 Tensor 或 `(Tensor, bias)`。

### 状态变化

- 初始化阶段为当前 TP rank 创建权重分片。
- `update_param_tp_status()` 把层的 TP rank 和 size 同步到参数对象。
- 权重加载阶段将 checkpoint 的正确 shard 写入本地参数。

## 9. 关键代码解析

### `UnquantizedLinearMethod.create_weights()`

### `UnquantizedLinearMethod.apply()`

### `LinearBase.__init__()`

### `ReplicatedLinear.forward()`

### `ColumnParallelLinear.__init__()`

### `ColumnParallelLinear.weight_loader()`

### `ColumnParallelLinear.forward()`

### `MergedColumnParallelLinear.__init__()`

### `MergedColumnParallelLinear.weight_loader()`

### `MergedColumnParallelLinear.weight_loader_v2()`

### `QKVParallelLinear.__init__()`

### `QKVParallelLinear.weight_loader()`

### `QKVParallelLinear.weight_loader_v2()`

### `RowParallelLinear.__init__()`

### `RowParallelLinear.weight_loader()`

### `RowParallelLinear.forward()`

## 10. 与其他文件的关系

- 上游模型：`vllm/model_executor/models/llama.py` 等模型实现。
- 分布式通信：依赖 TP rank、all-gather、all-reduce 和输入切分操作。
- 量化：通过 `QuantizationConfig` 选择具体 `LinearMethodBase` 实现。
- 权重加载：参数对象绑定本文件提供的 `weight_loader`，由 Model Loader 间接调用。
- 与 embedding 的关系：LM Head 的权重切分逻辑定义在 `vocab_parallel_embedding.py`，不使用普通 Linear forward。

## 11. 当前结论

`linear.py` 把模型中的 Dense 投影改造成张量并行、融合且可量化的推理层；Column Parallel 产生局部特征，Row Parallel 汇总局部计算，而专用 loader 保证每个 rank 得到正确的 checkpoint 分片。
