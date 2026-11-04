# 05 · Model and Weight Loading

**本章回答：模型、权重和执行资源怎样从配置变成可用状态？**

状态：Overview 与 5 篇 Core-Modules 已完成，依据本地源码快照。

## 1. 职责边界

| 项目 | 契约 |
| --- | --- |
| 输入 | VllmConfig、模型配置、权重文件；外部交回的 KVCacheConfig |
| 输出 | 就绪的 Worker、加载权重的模型、可用 KV 字节预算、已分配并绑定的 KV 张量与基础预热结果 |
| 本章负责 | Worker 本地设备准备、模型注册与实例化、权重发现/读取/映射/装载/后处理、显存探测、KV 张量初始化、基础 eager 预热 |
| 终点 | 资源可供执行接口使用；不推演用户请求的逐轮运行 |

| 内容 | 唯一归属 |
| --- | --- |
| EngineCore 如何按依赖组织初始化调用 | 02 Engine Core |
| KV 分组、容量公式、块数规划与运行期间块管理 | 04 KV Cache Management |
| 逐轮输入、block table、slot_mapping 与模型执行 | 06 Model Execution |
| Attention 物理布局、元数据和 kernel | 07 Attention Backend |
| 采样算法 | 08 Sampling |
| torch.compile、图捕获/重放与其专用预热 | 09 Runtime Optimizations |
| 权重跨 rank 切分、进程组拓扑与协调 | 10 Distributed Execution |

固定主线为单卡、普通文本、非量化、本地 safetensors、V1 GPUModelRunner、enforce_eager=True。权重读取沿 DefaultModelLoader；需要消除 auto 格式选择差异时显式指定 load_format="safetensors"。Qwen2 仅用于说明名称映射，不要求所有模型采用相同加载规则。

## 2. 正文范围与阅读顺序

先读 [Overview](Overview.md)，再进入以下正文：

| 顺序 | 笔记 | 唯一负责的内容 |
| --- | --- | --- |
| 01 | [Worker 与模型构造](Core-Modules/01-Worker-and-Model-Construction.md) | 本地设备/Runner、架构解析、模型类实例化 |
| 02 | [权重文件与加载流程](Core-Modules/02-Weight-Files-and-Loading.md) | loader 选择、文件过滤、迭代器与装载调用 |
| 03 | [权重映射与后处理](Core-Modules/03-Weight-Mapping-and-Postprocessing.md) | 名称/融合片段映射、参数写入与装载后的格式处理 |
| 04 | [显存探测与 KV 张量初始化](Core-Modules/04-Memory-Profiling-and-KV-Initialization.md) | 测得字节预算，消费外部配置，分配并绑定存储 |
| 05 | [基础预热与就绪检查](Core-Modules/05-Eager-Warmup-and-Readiness.md) | eager dummy 调用、sampler 缓冲、随机种子恢复与验收 |

01 不读取 checkpoint；02 不进入参数布局算法；03 不展开前向 kernel；04 不重算块容量；05 不重讲初始化编排或图优化。

## 3. 验收

- 区分 HF 配置、vLLM 模型类和实际模型实例。
- 解释目录/分片索引怎样得到 (name, tensor) 迭代器。
- 跟踪一条 Qwen2 q_proj 权重如何写入融合参数，区分融合片段与 TP rank。
- 解释本地 memory_profiling 的实际扣减口径，避免重复减权重。
- 区分 KVCacheConfig 的存储描述与实际张量、绑定引用。
- 说明 enforce_eager=True 为什么仍需 profile 和基础预热。

[源码纵览](../00-Overview.md) · [返回模块索引](../README.md)
