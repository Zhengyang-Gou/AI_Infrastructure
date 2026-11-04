# vLLM 源码笔记

**先读 [00 · 源码纵览](00-Overview.md)**，认识模块关系，再按 01 → 08 阅读。09–11 是基础主线的扩展。

## 1. 固定运行配置

以下配置适用于 01–08 的普通文本主线。扩展章节另列与此处不同的设置。

| 项目 | 设置 |
| --- | --- |
| 接口 | 离线 `LLM.generate(prompts)`，每条输入生成一个候选（`n=1`） |
| 模型 | 本地 decoder-only 普通文本模型，safetensors 权重 |
| 设备 | 单张 CUDA GPU，TP=1，PP=1 |
| 核心连接 | `VLLM_ENABLE_V1_MULTIPROCESSING=0`，使用 `InprocClient` |
| 执行器 | `UniProcExecutor` → Worker → `GPUModelRunner` |
| Model Runner | 沿 `v1/worker/gpu_model_runner.py`；选择支持该路径的模型，设置 `VLLM_USE_V2_MODEL_RUNNER=0` |
| 调度 | `async_scheduling=False`，基础 `Scheduler`，单批在途 |
| 前缀缓存 / Prompt 分块 | `enable_prefix_caching=True`、`enable_chunked_prefill=True` |
| 编译与图执行 | `enforce_eager=True` |
| 长度与预算 | `max_model_len=2048`、`max_num_batched_tokens=256`、`max_num_seqs=4` |
| 显存 | `gpu_memory_utilization=0.8`、`block_size=16` |

这些是源码阅读场景；运行时确认模型、设备与后端支持这组配置，并记录实际使用的版本。

## 2. 模块边界规则

1. **一个机制只有一个正文归属。** 以负责的对象和状态为准，不按调用链把下游实现复制到上游。
2. **跨模块只写接口。** 可保留调用点、输入、输出和本模块的调用顺序；到接口为止，不进入被调用函数的算法、字段更新或逐轮推演。
3. **总览与正文分工。** 00 串联全部模块；各章 Overview 只概览本章；Core-Modules 只展开该篇分配到的实现。
4. **推演在所属模块闭合。** 调度案例只在 03 展开，块账本只在 04 展开；入口、核心不复演调度轮次。
5. **链接用于定位。** 当前章节需要的接口契约在本页给全，不要求来回跳转才能读懂当前职责。

## 3. 模块索引与唯一归属

“边界页”表示职责已定义、正文尚未展开；“部分正文”表示仅列出的主题已完成。

| 模块 | 本章负责 | 本章不展开 | 状态 |
| --- | --- | --- | --- |
| [00 · Overview](00-Overview.md) | 模块关系、启动与运行主线 | 源码、算法、数字推演 | 已完成 |
| [01 · Inference Entry](01-Inference-Entry/README.md) | 离线入口、输入转换与校验、请求 ID、输出状态、解码与停止字符串 | 客户端内部、核心循环、调度、采样算法 | 正文 6 篇 |
| [02 · Engine Core](02-Engine-Core/README.md) | 核心接收、初始化顺序、同步 step、客户端与进程循环、取消转交 | Request 进度、调度判停、KV 算法、执行器内部、批次重叠 | 正文 5 篇 |
| [03 · Scheduler](03-Scheduler/README.md) | Request 与队列、预算与接纳、抢占选择、结果更新、核心判停与请求退场 | 块池、张量准备、文本停止字符串、异步占位算法 | Overview + 部分正文 2 篇 |
| [04 · KV Cache Management](04-KV-Cache-Management/README.md) | 缓存布局与容量规则、块池、引用计数、前缀复用、淘汰与实际回收 | 抢占谁、GPU 张量初始化过程、slot_mapping 构造、kernel | Overview + 正文 5 篇 |
| [05 · Model and Weight Loading](05-Model-and-Weight-Loading/README.md) | 模型构造、权重加载、显存探测、KV 张量初始化、基础预热 | 运行期间块管理、逐轮输入准备、图捕获与重放 | Overview + 正文 5 篇 |
| [06 · Model Execution](06-Model-Execution/README.md) | Executor/Worker 执行调用、执行侧请求状态、张量准备、block table 与 slot_mapping、前向和采样交接 | 调度策略、块池、kernel、采样算法、加载流程 | 边界页 |
| [07 · Attention Backend](07-Attention-Backend/README.md) | 后端选择、Attention 元数据、KV 物理布局、缓存读写与 kernel 调用 | 缓存块所有权、整批输入拼装、调度与采样 | 边界页 |
| [08 · Sampling](08-Sampling/README.md) | logits 变换、约束处理、采样与 logprobs | EOS/长度判停、停止字符串、用户结果、推测验证 | 边界页 |
| [09 · Runtime Optimizations](09-Runtime-Optimizations/README.md) | 编译、CUDA Graph、异步占位、batch queue、推测解码与验证 | 基础调度重讲、分布式拓扑、HTTP 接入 | 正文 1 篇 + 概览 1 篇 |
| [10 · Distributed Execution](10-Distributed-Execution/README.md) | TP/PP/DP/EP、Worker 拓扑、通信与跨 rank 协调 | 单核心消息循环、单卡基础执行、HTTP 流式交付 | 边界页 |
| [11 · Online Serving](11-Online-Serving/README.md) | HTTP、AsyncLLM、异步结果交付、连接/取消与背压 | 分词/解码算法、核心循环、AsyncScheduler 与 GPU 重叠 | 边界页 |

## 4. 推演案例的归属

- **案例 A：调度与批处理 → [03 · Scheduler §6](03-Scheduler/Overview.md#6-用三条请求走完四轮)。** 在逐轮驱动中加入 B、A、C，观察预算与批次成员变化。完整条件、数字和四轮表只在 03 维护。
- **案例 B：KV 块账本 → [04 · KV Cache Management](04-KV-Cache-Management/README.md#3-案例-b块账本)。** 对照已给定的计算位置与块数，不推导调度策略；slot_mapping 构造由 06 负责。

其他篇可用最小输入演示本篇对象的变化，不重复跨模块的完整生命周期。

## 5. 源码阅读与维护约定

- 源码依据仓库内 `04-vLLM-Source/vllm/`；源码块路径相对于这个目录。
- 每个源码块标注文件、类、函数与本地行范围；模块级函数标“类：无”。
- 节选保持连续，只调整共同缩进；示意与源码分开。
- 同一段源码只在负责的 Core-Modules 中展开；其他位置只保留接口说明。
- 待写章节先遵守边界页，不以补正文为由复制邻章实现。
- 已有笔记的检查与迁移记录见 [边界检查](Boundary-Review.md)。

[返回推理学习路线](../../Roadmap.md)
