# vLLM 源码笔记

按组件模块组织，每个模块对应源码里一组真实的对象与文件。固定场景、推演案例和源码约定只在本页维护一次，各模块直接引用。

**先读 [主线走查](Trace.md)**：一条请求从 `generate` 走到结果返回，每步链接到对应模块。它是这套笔记的骨架，也是进度表。

## 1. 固定运行配置

以下设定适用于全部模块正文，各篇不再复述。

| 项目 | 设置 |
| --- | --- |
| 接口 | 离线 `LLM.generate(prompts)`，一次提交一批输入，每条生成一个候选（`n=1`） |
| 模型 | 本地 decoder-only 普通文本模型，safetensors 权重 |
| 设备 | 单张 CUDA GPU，TP=1，PP=1 |
| 核心连接 | `InprocClient`，入口与核心同进程 |
| 执行器 | `UniProcExecutor` → Worker → `GPUModelRunner` |
| 调度 | 基础 `Scheduler`，同步调度，单批在途 |
| 前缀缓存 / Prompt 分块 | 均启用 |
| 编译与图执行 | `enforce_eager=True`，关闭 torch.compile 与 CUDA Graph |
| 长度与预算 | `max_model_len=2048`、`max_num_batched_tokens=256`、`max_num_seqs=4` |
| 显存 | `gpu_memory_utilization=0.8`，块大小 `block_size=16` |

## 2. 两个推演案例

正文里的数字全部来自这两个案例，不再各造一套。

**案例 A：调度与批处理（手算用）。** 把预算缩到 **4**、`max_num_seqs` 缩到 **2**，只为首算得动；其余设置不变。B（2 token，先入队）、A（6 token）、C（2 token，第 3 轮前到达），各最多生成 2 个 token。四轮走完，同时覆盖 Chunked Prefill 与 Continuous Batching。完整四轮表见 [03 · Scheduler §6](03-Scheduler/Overview.md)。

**案例 B：KV 块账本。** 使用真实预算 256 和 `block_size=16`，单条 600-token 请求：分 256 / 256 / 88 三轮算完，累计占用 16 / 32 / 38 块。用于讲块分配、`slot_mapping` 与块表。

两者尺度不同不是笔误：案例 A 要把预算压到能手算，案例 B 要看真实块数。切换时正文会标明用哪一个。

## 3. 模块索引

状态：**正文** = 有完整 Core-Modules；**大纲** = 有 README + Overview；**待写** = 只有占位。

| 模块 | 要回答的问题 | 状态 | 前置 |
| --- | --- | --- | --- |
| [01 · Inference Entry](01-Inference-Entry/README.md) | 一次用户调用怎样变成引擎请求，又怎样取回结果 | 正文 6 篇 | — |
| [02 · Engine Core](02-Engine-Core/README.md) | 请求进入核心后，一轮轮调度、执行、回写怎样闭环 | 正文 5 篇 | 01 |
| [03 · Scheduler](03-Scheduler/README.md) | 每轮选哪些请求、各算多少 token、资源不足怎样取舍 | 大纲 | 02 |
| [04 · KV Cache Management](04-KV-Cache-Management/) | 块从哪来、怎样借还、前缀怎样复用、不足时怎样抢占 | 待写 | 03 |
| [05 · Model and Weight Loading](05-Model-and-Weight-Loading/) | 模型结构、权重、KV 张量与预热怎样就绪 | 待写 | 02 |
| [06 · Model Execution](06-Model-Execution/) | 执行计划怎样变成输入张量、KV 写到哪里、前向与采样怎样执行 | 待写 | 03、04、05 |
| [07 · Attention Backend](07-Attention-Backend/) | 后端怎样选择、KV 布局与 kernel 块大小怎样对应 | 待写 | 06 |
| [08 · Sampling](08-Sampling/) | logits 怎样变成 token、结束条件在哪里判定 | 待写 | 06 |

模块顺序即阅读顺序，它与依赖一致：先入口与核心骨架，再调度与缓存，然后模型加载与执行，最后注意力和采样。

### 扩展专题

以下三项是主线之外的变体，读完 08 之后按需展开：

| 专题 | 内容 |
| --- | --- |
| [09 · Runtime Optimizations](09-Runtime-Optimizations/) | 图捕获与重放、异步调度、推测解码、批次重叠 |
| [10 · Distributed Execution](10-Distributed-Execution/) | TP / PP / EP 的切分方式与进程拓扑 |
| [11 · Online Serving](11-Online-Serving/) | `AsyncLLM`、流式输出与 HTTP 层 |

这三项目前只有占位目录。若长期不写，建议合并成一节说明后删除目录，避免读者点进空文件夹。

## 4. 怎么读

1. 本页第 1、2 节 → 知道固定场景和案例数字。
2. [主线走查](Trace.md) → 建立“谁在什么时候改了什么”的整体图。
3. 按 01 → 08 进入模块；每个模块先读 `Overview.md`，再读 `Core-Modules`。
4. 读完一个模块，回到主线走查对应步骤，确认能自己讲出那一步的状态变化。

模块之间有交叉的地方，用各模块 README 的“相关章节的分工”表确认边界，不要在两个模块里重复推导同一件事。

## 5. 源码阅读约定

- 源码依据仓库内 `04-vLLM-Source/vllm/`，代码块开头的路径相对于这个目录。
- 每个代码块标注 `# 文件`、`# 类`、`# 函数` 和本地行范围；模块级函数标“类：无”。
- 节选保持连续，只调整共同缩进；流程、公式与计算示意单独标注，不与源码混排。
- 术语首次出现时用一句简明中文解释；同一术语不在后续模块重复定义。

## 6. 当前状态与待办

- 01、02 已有完整正文，03 有大纲，04–08 待写，09–11 为占位。
- 02 章节 §5 的推演仍是两条请求的简化版，与案例 A 不一致，改造时统一到案例 A。
- 从旧八章版本（提交 `c10fdeb`）可取回两块素材并入模块版：KV 容量的 32 层 / 8192 块算例（→ 04）、一轮调度的流程图（→ 03）。

[返回推理学习路线](../../Roadmap.md)
