# Model and Weight Loading Overview

**本章把配置、checkpoint 和存储描述变成可执行资源。** 不处理某条用户请求的调度、token 输入或输出文本。

## 1. 三条资源路线

```text
模型路线
VllmConfig / HF config → 解析模型架构 → 实例化 vLLM 模型

权重路线
本地文件 → (name, tensor) 迭代器 → 映射到参数 → 写入与后处理

缓存路线
显存探测 → 可用 KV 字节预算交出
外部容量规划交回 KVCacheConfig → 分配 GPU KV 张量 → 绑定模型层
```

中间的容量规划属于 04；本章不从预算推导块数。核心怎样安排这些接口调用属于 02，本页只说明本章各资源的输入与结果。

## 2. 对象怎样分工

| 对象 | 本章职责 |
| --- | --- |
| Worker | 准备本地设备，调用加载、显存探测、缓存初始化和基础预热 |
| GPUModelRunner | 持有模型与 KV，组织加载器调用，提供 profile/dummy 接口 |
| ModelRegistry | 将架构名解析为受支持的实现类 |
| BaseModelLoader | 模型构造、权重装载、后处理的共用模板 |
| DefaultModelLoader | 文件发现、迭代器选择和调用 model.load_weights |
| AutoWeightsLoader / WeightsMapper | 模块树定位、名称与融合片段映射、参数写入路由 |
| MemorySnapshot / memory_profiling | 记录显存快照和探测消耗 |

Runner 在这里是资源持有者；它怎样为每轮请求准备输入不在本章展开。

## 3. 资源不是同一时刻全部就绪

| 状态 | 已具备 | 尚不能据此断定 |
| --- | --- | --- |
| Worker/Runner 已创建 | 设备、配置与运行对象 | checkpoint 已装载 |
| 模型已构造 | 模块树与参数对象 | 参数值已来自 checkpoint |
| loader 返回 | 权重装载及共用后处理完成，eval 模式 | 已有最终容量的 KV 存储 |
| profile 完成 | 可交出的 KV 字节预算 | 块数规划与实际 KV 分配已完成 |
| KV 初始化完成 | 存储与层引用绑定 | 所有基础预热已完成 |
| 预热完成 | 基础执行路径及必要缓冲就绪 | 已产生任何用户请求结果 |

## 4. 阅读路线与停止位置

按 [01 构造](Core-Modules/01-Worker-and-Model-Construction.md) → [02 文件](Core-Modules/02-Weight-Files-and-Loading.md) → [03 参数](Core-Modules/03-Weight-Mapping-and-Postprocessing.md) → [04 显存与 KV](Core-Modules/04-Memory-Profiling-and-KV-Initialization.md) → [05 预热](Core-Modules/05-Eager-Warmup-and-Readiness.md) 阅读。

本章最终只交出资源就绪状态，不继续进入 schedule 或真实请求的 execute_model。

[章节目录](README.md)
