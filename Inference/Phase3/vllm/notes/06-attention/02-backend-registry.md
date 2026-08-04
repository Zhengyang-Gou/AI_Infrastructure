# Attention Backend Registry

## 1. 学习目标

理解 vLLM 如何用枚举名称表示 Attention 后端、如何延迟解析后端类，以及内置后端和自定义覆盖如何共用同一套注册机制。

## 2. 文件定位

- 文件路径：`vllm/v1/attention/backends/registry.py`
- 所属层次：Attention 后端注册与类路径解析层
- 核心职责：维护后端枚举到完整类路径的映射，支持导入后端类、注册自定义实现、覆盖和清除覆盖。
- 在调用链中的位置：服务于后端选择与初始化，不参与每一轮 Attention Tensor 计算。

## 3. 核心类与组件

| 类 / 组件 | 作用 |
| --- | --- |
| `_AttentionBackendEnumMeta` | 在通过名称访问枚举失败时给出合法后端列表 |
| `AttentionBackendEnum` | 保存标准 Attention 后端名称及其默认类路径 |
| `MambaAttentionBackendEnum` | 保存 Mamba、短卷积和线性 Attention 等后端名称及类路径 |
| `_ATTN_OVERRIDES` | 保存标准 Attention 后端的运行时覆盖路径 |
| `_MAMBA_ATTN_OVERRIDES` | 保存 Mamba 类后端的运行时覆盖路径 |
| `register_backend()` | 以装饰器或显式类路径方式注册、覆盖后端 |

枚举值保存字符串类路径，只有调用 `get_class()` 时才通过完整限定名导入对象，从而避免注册表加载时立即导入所有重型后端依赖。

## 4. 主执行流程

### 内置后端解析

```text
后端名称
→ AttentionBackendEnum[名称]
→ get_path()
→ 优先读取运行时 override，否则使用默认枚举值
→ get_class()
→ resolve_obj_by_qualname()
→ AttentionBackend 子类
```

### 自定义或覆盖后端

```text
register_backend(enum_member, class_path)
或 @register_backend(enum_member)
→ 写入对应 override 字典
→ enum_member.get_path()
→ enum_member.get_class()
→ 返回覆盖后的实现类
```

`CUSTOM` 没有默认类路径，使用前必须先注册。`clear_override()` 会删除覆盖，使内置枚举成员回到默认路径。

## 5. 输入与输出

### 输入

- 后端枚举成员或后端名称。
- 可选的完整类路径字符串。
- 通过装饰器传入的后端类。
- `is_mamba`：指定覆盖应写入标准还是 Mamba 后端表。

### 输出

- `get_path()` 返回后端实现的完整限定类路径。
- `get_class()` 返回解析后的 `AttentionBackend` 子类。
- `register_backend()` 返回装饰器或保持传入类不变的函数。

### 状态变化

- 注册与覆盖会更新 `_ATTN_OVERRIDES` 或 `_MAMBA_ATTN_OVERRIDES`。
- `clear_override()` 删除当前枚举成员的覆盖记录。

## 6. 关键代码解析

### `_AttentionBackendEnumMeta.__getitem__()`

### `AttentionBackendEnum.get_path()`

### `AttentionBackendEnum.get_class()`

### `AttentionBackendEnum.is_overridden()`

### `AttentionBackendEnum.clear_override()`

### `MambaAttentionBackendEnum.get_path()`

### `MambaAttentionBackendEnum.get_class()`

### `register_backend()`

## 7. 与其他文件的关系

- 上游：配置解析与 `vllm/v1/attention/selector.py` 使用后端枚举或类路径。
- 下游：枚举中列出的 `vllm/v1/attention/backends/*.py` 具体后端类。
- 接口约束：解析出的类遵循 `vllm/v1/attention/backend.py` 中的 `AttentionBackend` 接口。
- 模型层关系：`Attention` 保存选定后端，并通过 `get_impl_cls()` 创建具体实现。

注册表回答“某个后端名称对应哪个类”，但不独立回答“当前硬件和配置应该选择哪个后端”。后一个问题属于 selector 与 platform 的职责。

## 8. 当前结论

`registry.py` 是轻量的名称与类路径注册中心。它通过延迟导入隔离不同后端依赖，并允许在不修改调用方的情况下覆盖内置实现或注册第三方后端。
