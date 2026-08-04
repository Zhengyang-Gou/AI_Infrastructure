# LoRA

## 1. 文件定位

- 主要路径：`vllm/lora/`、`vllm/v1/worker/lora_model_runner_mixin.py`。
- 所属层次：模型参数适配与请求级模型扩展层。
- 核心职责：加载、缓存和激活 LoRA Adapter，并让同一 batch 的不同请求使用不同 Adapter。

## 2. 核心对象

| 对象 | 作用 |
| --- | --- |
| `LoRARequest` | 描述请求选择的 Adapter ID、名称和路径 |
| `LoRAModelManager` | 管理 Adapter 的注册、激活、移除与容量 |
| `LoRAModelRunnerMixin` | 把 Adapter 管理接入 Model Runner 生命周期 |
| LoRA Layer Wrapper | 在 Linear、Embedding、LM Head 或 MoE 上叠加低秩更新 |
| Punica Wrapper | 批量执行不同请求对应的 LoRA A/B 矩阵运算 |

## 3. 主执行流程

```text
请求携带 LoRARequest
→ Engine / Scheduler 保留 LoRA ID
→ Worker 确保 Adapter 已加载
→ Model Runner 设置 active LoRAs
→ LoRA Layer Wrapper 建立 request-to-adapter mapping
→ base output + LoRA delta
```

Adapter 权重与基础模型权重分开保存。每轮只更新 batch 映射，不需要复制基础模型。

## 4. 输入与输出

### 输入

- Adapter checkpoint、rank、scale 和目标 module 信息。
- 本轮 batch 中每个请求对应的 LoRA ID。

### 输出

- 已缓存并可激活的 `LoRAModel`。
- 基础模型输出叠加请求级 LoRA delta 后的 tensor。

### 状态变化

- Adapter 可动态加入、移除、固定或通过 LRU 淘汰。
- Model Runner 每轮更新 active adapter 集合及 token mapping。

## 5. 关键代码解析

### `LoRAModelManager.add_adapter()`

### `LoRAModelManager.activate_adapter()`

### `LoRAModelManager.remove_adapter()`

### `LoRAModelRunnerMixin.load_lora_model()`

### `LoRAModelRunnerMixin.set_active_loras()`

### `LoRAModelRunnerMixin.add_lora()`

### `LoRAModelRunnerMixin.remove_lora()`

## 6. 与其他文件的关系

- 在线服务：API 层可以动态加载、卸载并按模型名选择 Adapter。
- Scheduler：需要限制一轮激活的 LoRA 数量。
- Model Runner：准备请求到 Adapter 的映射并调用 Punica。
- 模型层：Linear、Embedding 和 MoE wrapper 负责应用 delta。

## 7. 当前结论

LoRA 通过共享基础模型、动态 Adapter 缓存和 batch 内映射实现低开销的请求级模型定制。
