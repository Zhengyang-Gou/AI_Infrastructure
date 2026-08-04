# Model Registry

## 1. 文件定位

- 文件路径：`vllm/model_executor/models/registry.py`
- 所属层次：模型架构注册与解析层
- 核心职责：维护 architecture 到模型模块、类名的映射，检查模型能力，并在需要时延迟导入具体模型类。
- 在调用链中的位置：位于 `ModelConfig` / Model Loader 与具体模型实现之间。

当前源码中，注册表类名为 `_ModelRegistry`，模块底部创建的全局实例名为 `ModelRegistry`。上层代码通常直接使用这个实例。

## 2. 核心类与数据

| 类 / 数据 | 作用 | 主要内容 |
| --- | --- | --- |
| `_VLLM_MODELS` | 汇总全部内置 architecture 映射 | architecture → `(module, class)` |
| `_ModelInfo` | 描述模型能力 | generation、pooling、多模态、PP 等标志 |
| `_BaseRegisteredModel` | 注册项公共接口 | inspect 与 load |
| `_RegisteredModel` | 保存已导入的模型类 | 适合直接注册 Python class |
| `_LazyRegisteredModel` | 保存模块名与类名 | 需要时再 import |
| `_ModelRegistry` | 注册、检查和解析模型 | architecture → 注册项 |
| `ModelRegistry` | 模块级全局注册表实例 | 内置模型的统一入口 |

`"LlamaForCausalLM": ("llama", "LlamaForCausalLM")` 最终会通过 `_resolve_module_name()` 展开为 `vllm.model_executor.models.llama`。

## 3. 注册流程

### 内置模型

```text
各模型分类字典
→ 合并为 _VLLM_MODELS
→ 为每个条目创建 _LazyRegisteredModel
→ 构造全局 ModelRegistry
```

内置模型采用 lazy registration，导入 `registry.py` 时不需要立刻导入所有模型模块，可降低启动开销，也避免检查阶段意外初始化 CUDA。

### 外部模型

```text
ModelRegistry.register_model(model_arch, model_cls)
├─ model_cls 是 "module:class" 字符串 → _LazyRegisteredModel
└─ model_cls 是 nn.Module 子类       → _RegisteredModel
```

同名 architecture 可以被后续注册覆盖，为插件或自定义模型替换实现提供入口。

## 4. 模型检查流程

```text
ModelRegistry.inspect_model_cls(architectures, model_config)
→ 根据 model_impl 决定 vLLM / Transformers / Terratorch 路径
→ 标准化 architecture
→ _try_inspect_model_cls()
→ registered_model.inspect_model_cls()
→ 返回 (_ModelInfo, matched_architecture)
```

`_LazyRegisteredModel.inspect_model_cls()` 优先读取模型信息缓存；缓存失效时，在子进程中导入模型类并调用 `_ModelInfo.from_model_cls()`，避免主进程检查模型能力时初始化 CUDA。模型模块哈希用于判断缓存是否仍与源码一致。

## 5. 模型类解析流程

```text
ModelRegistry.resolve_model_cls(architectures, model_config)
→ 选择兼容的 architecture
→ _try_load_model_cls()
→ _LazyRegisteredModel.load_model_cls()
→ importlib.import_module(module_name)
→ getattr(module, class_name)
→ 返回 (model_cls, matched_architecture)
```

若本地注册表无法处理且 `model_impl` 允许，Registry 会尝试兼容的 Transformers backend；所有候选都失败时，统一生成不支持架构的错误信息。

## 6. 输入与输出

### 输入

- Hugging Face config 提供的一个或多个 architecture 名称。
- `ModelConfig` 中的 `model_impl`、`trust_remote_code` 和转换配置。
- 外部注册时提供的模型类或 `"module:class"` 字符串。

### 输出

- `inspect_model_cls()` 返回模型能力 `_ModelInfo` 与匹配的 architecture。
- `resolve_model_cls()` 返回可实例化的 `nn.Module` 类与匹配的 architecture。
- 能力查询方法返回 generation、pooling、多模态、PP 等布尔结果。

### 状态变化

- `register_model()` 新增或覆盖 `models` 中的 architecture 条目。
- lazy inspection 可在 Cache 目录写入模型信息 JSON。
- `_try_load_model_cls()` 与 `_try_inspect_model_cls()` 使用 LRU cache 保存解析结果。

## 7. 关键代码解析

### `_ModelInfo.from_model_cls()`

### `_RegisteredModel.from_model_cls()`

### `_LazyRegisteredModel.inspect_model_cls()`

### `_LazyRegisteredModel.load_model_cls()`

### `_ModelRegistry.register_model()`

### `_ModelRegistry._normalize_arch()`

### `_ModelRegistry.inspect_model_cls()`

### `_ModelRegistry.resolve_model_cls()`

### `_resolve_module_name()`

### `_run_in_subprocess()`

## 8. 与其他文件的关系

- 上游：`ModelConfig` 和 `vllm/model_executor/model_loader`。
- 下游：`vllm/model_executor/models/llama.py` 等具体模型模块。
- 能力判断：依赖模型 interface 与 interface base 工具。
- Transformers fallback：在配置允许时解析 Transformers backend 模型。
- 插件扩展：外部代码可使用全局 `ModelRegistry.register_model()` 注册模型。

## 9. 当前结论

Model Registry 是模型选择总入口：它将配置中的 architecture 名称映射到具体类，同时通过延迟导入、子进程检查和能力缓存控制模型模块的加载时机。
