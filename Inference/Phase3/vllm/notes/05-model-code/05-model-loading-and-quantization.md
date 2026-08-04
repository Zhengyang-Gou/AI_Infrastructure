# Model Loading and Quantization

## 1. 文件定位

- 主要路径：`vllm/model_executor/model_loader/`、`vllm/model_executor/layers/quantization/`。
- 所属层次：模型实例化、权重加载和参数表示层。
- 核心职责：选择 loader，流式读取 checkpoint，把权重写入 TP/融合参数，并接入具体量化方法。

## 2. 核心对象

| 对象 | 作用 |
| --- | --- |
| `BaseModelLoader` | 定义 download、load weights 和 load model 接口 |
| `DefaultModelLoader` | 加载 Hugging Face、safetensors 和 PyTorch checkpoint |
| `initialize_model` | 根据 Registry 解析出的模型类创建空模型结构 |
| `weight_loader` | 把 checkpoint tensor 写入当前 rank 的参数 shard |
| `QuantizationConfig` | 解析量化格式并为各层返回 quant method |
| `QuantizeMethodBase` | 定义量化权重创建、后处理和 forward apply 接口 |

## 3. 主执行流程

```text
LoadConfig
→ get_model_loader()
→ initialize_model()
→ Registry 解析模型类
→ 构造当前 TP rank 的模型参数
→ weights iterator 流式读取 checkpoint
→ model.load_weights()
→ 参数 weight_loader 选择并写入 shard
→ process_weights_after_loading()
→ 可执行模型
```

量化模型在初始化时就依据 `QuantizationConfig` 创建目标格式参数，避免先保存一份完整浮点模型。

## 4. 输入与输出

### 输入

- Hugging Face config、checkpoint 路径和 revision。
- Load format、TP/EP rank、dtype 与量化配置。
- checkpoint 中的参数名和 tensor。

### 输出

- 已按当前 rank 分片并完成后处理的模型。
- 与 Linear、Embedding、MoE 等层绑定的 quant method。

### 状态变化

- 模型先创建参数结构，再逐项填充权重。
- 融合层把多个 checkpoint 参数写入同一目标参数的不同 shard。
- 量化方法在加载后生成 scale、packed weight 或 kernel metadata。

## 5. 关键代码解析

### `get_model_loader()`

### `BaseModelLoader.load_model()`

### `DefaultModelLoader._prepare_weights()`

### `DefaultModelLoader._get_weights_iterator()`

### `DefaultModelLoader.load_weights()`

### `initialize_model()`

### `configure_quant_config()`

### `QuantizationConfig.get_quant_method()`

### `QuantizeMethodBase.process_weights_after_loading()`

## 6. 与其他文件的关系

- Registry：决定实际 Python 模型类。
- 模型实现：`load_weights()` 定义名称映射和融合参数规则。
- TP Layers：各参数的 `weight_loader` 负责 rank-local 切片。
- GPU Worker：Model Runner 初始化期间调用 model loader。

## 7. 当前结论

模型加载主线是先按最终并行与量化形态建模，再流式写入当前 rank 所需权重。
